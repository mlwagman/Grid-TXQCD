#pragma once
// Fused DTXQCD aux+cross+clover hermop kernel.
//
// Mirrors the TXQCD pattern (Grid/qcd/action/txqcd/TXQCDFusedAuxClover.h) for
// the doubled DTXQCD operator. The DTXQCD doubled operator has TWO blocks
// (upper, lower) which separately need an aux+cross+clover insertion on top of
// the per-block Wilson hop. The legacy path
// (DTXQCDWilsonCloverFermionEO::AddDiagAndCrossAndClover) calls 6 separate
// kernels:
//
//   DtxqcdApplyDeltaDiag     (σ, π, s, p; in.upper)   -> delta_u
//   DtxqcdApplyDeltaDiagLower(σ, π, s, p; in.lower)   -> delta_l (transposed σ, π)
//   DtxqcdApplyDnCross       (d, n; in.lower)         -> cross_u (M_UR)
//   DtxqcdApplyDnCross       (d, n; in.upper, conj)   -> cross_l (M_LL)
//   DtxqcdApplyCloverUpper   (csw, F; in.upper)       -> clov_u  (-csw/2 · F · σ_μν)
//   DtxqcdApplyCloverLower   (csw, F; in.lower)       -> clov_l  (+csw/2 · F^T · σ_μν)
//
// Each pre-rotates γ5 and (for clover) the 6 σ_μν fermion lanes independently.
// The fused path computes those rotations ONCE per fused call (12 σ_μν Lattice
// expressions + 4 γ5 pre-rotations) and accumulates everything into the caller's
// out_accum in TWO accelerator_for kernels:
//
//   Pass A (flavor-mixing color-diagonal):
//     out.upper[a] += Σ_b σ_{ab} · in.upper[b] + π_{ab} · γ5·in.upper[b]
//                   + s·in.upper[a] + p·γ5·in.upper[a]
//     out.lower[a] += Σ_b σ^T_{ab} · in.lower[b] + π^T_{ab} · γ5·in.lower[b]
//                   + s·in.lower[a] + p·γ5·in.lower[a]
//     (σ^T, π^T = joint color×flavor transpose: read [bj,ai] instead of [ai,bj])
//
//   Pass B (color-mixing flavor-diagonal: cross + clover):
//     out.upper[a] += Σ_b d_{ab}·γ5·in.lower[b] + n_{ab}·in.lower[b]
//                   + Σ_k F_eff_U(k)(i,j) · (σ_μν_k · in.upper[a])
//     out.lower[a] += Σ_b conj(d)_{ab}·γ5·in.upper[b] + conj(n)_{ab}·in.upper[b]
//                   + Σ_k F_eff_L(k)(i,j) · (σ_μν_k · in.lower[a])
//   where:
//     F_eff_U(k)(i,j) = -csw/2 · F_k(i,j)
//     F_eff_L(k)(i,j) = +csw/2 · F_k(j,i)            (color transpose at use)
//
// Why two passes:
//   The TXQCD prototype's single-pass version regressed at production volume
//   (16³×48) because the doubled live-state inflated register usage past the
//   occupancy cliff. Splitting into Pass A (σ/π+singlet) and Pass B (cross+
//   clover, with σ_μν views streamed) preserves occupancy while still saving
//   4 launches (6 → 2) per M-apply.
//
// Activated by env knob DTXQCD_FUSED_HERMOP=1 inside
// DTXQCDWilsonCloverFermionEO::M.  Default OFF.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdFusedAuxClover {

// Read the env knob once and cache.
inline bool Enabled() {
  static int cached = -1;
  if (cached == -1) {
    const char *s = std::getenv("DTXQCD_FUSED_HERMOP");
    cached = (s && std::atoi(s) != 0) ? 1 : 0;
  }
  return cached != 0;
}

// Split Pass B into separate upper / lower kernels. Halves the per-thread
// live state of the (cross+clover) pass at the cost of one extra kernel
// launch. Empirically a win at 32^3x64 where the combined Pass B hits the
// occupancy cliff; loss at 16^3x48 where the launch saving dominates.
inline bool SplitPassB() {
  static int cached = -1;
  if (cached == -1) {
    const char *s = std::getenv("DTXQCD_FUSED_SPLIT_PASSB");
    cached = (s && std::atoi(s) != 0) ? 1 : 0;
  }
  return cached != 0;
}

}  // namespace DtxqcdFusedAuxClover

namespace DtxqcdFusedAuxCloverDetail {

// Pass A: flavor-mixing σ + π for upper AND lower in one kernel.
//   out.upper[a] += Σ_b σ_{ab}·in.upper[b] + π_{ab}·g5·in.upper[b]
//                 + s·in.upper[a] + p·g5·in.upper[a]
//   out.lower[a] += Σ_b σ^T_{ab}·in.lower[b] + π^T_{ab}·g5·in.lower[b]
//                 + s·in.lower[a] + p·g5·in.lower[a]
//
// σ^T, π^T per DTXQCD v2 convention: joint color×flavor transpose, i.e. read
// element [bj][ai] instead of [ai][bj].
inline void PassFlavorMixAccumulate(
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi    &pi,
    const LatticeDtxqcdS     &s_field,
    const LatticeDtxqcdP     &p_field,
    const DTXQCDFermionDoubled &in,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_u,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_l,
    DTXQCDFermionDoubled &out_accum) {
  static_assert(DtxqcdNf == 2, "fused fast path only for Nf=2");
  GridBase *grid = in.Grid();

  autoView(sigmav, sigma,   AcceleratorRead);
  autoView(piv,    pi,      AcceleratorRead);
  autoView(sv,     s_field, AcceleratorRead);
  autoView(pv,     p_field, AcceleratorRead);
  autoView(inu0v,  in.upper.f[0], AcceleratorRead);
  autoView(inu1v,  in.upper.f[1], AcceleratorRead);
  autoView(inl0v,  in.lower.f[0], AcceleratorRead);
  autoView(inl1v,  in.lower.f[1], AcceleratorRead);
  autoView(gu0v,   g5_in_u[0], AcceleratorRead);
  autoView(gu1v,   g5_in_u[1], AcceleratorRead);
  autoView(gl0v,   g5_in_l[0], AcceleratorRead);
  autoView(gl1v,   g5_in_l[1], AcceleratorRead);
  autoView(outu0v, out_accum.upper.f[0], AcceleratorWrite);
  autoView(outu1v, out_accum.upper.f[1], AcceleratorWrite);
  autoView(outl0v, out_accum.lower.f[0], AcceleratorWrite);
  autoView(outl1v, out_accum.lower.f[1], AcceleratorWrite);

  const int Nsimd = LatticeFermion::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto sigma_lane = sigmav(ss);
    auto pi_lane    = piv(ss);
    auto s_lane     = sv(ss);
    auto p_lane     = pv(ss);
    auto inu0_lane  = inu0v(ss);
    auto inu1_lane  = inu1v(ss);
    auto inl0_lane  = inl0v(ss);
    auto inl1_lane  = inl1v(ss);
    auto gu0_lane   = gu0v(ss);
    auto gu1_lane   = gu1v(ss);
    auto gl0_lane   = gl0v(ss);
    auto gl1_lane   = gl1v(ss);
    auto outu0_acc  = coalescedRead(outu0v[ss]);
    auto outu1_acc  = coalescedRead(outu1v[ss]);
    auto outl0_acc  = coalescedRead(outl0v[ss]);
    auto outl1_acc  = coalescedRead(outl1v[ss]);

    // For each flavor a of the upper block: contribution
    //   acc_u(α,i) = Σ_b,j σ_{ab}(i,j)·in.upper[b](α,j) + π_{ab}(i,j)·g5·in.upper[b](α,j)
    //              + s·in.upper[a](α,i) + p·g5·in.upper[a](α,i)
    // For each flavor a of the lower block: contribution
    //   acc_l(α,i) = Σ_b,j σ_{ba}(j,i)·in.lower[b](α,j) + π_{ba}(j,i)·g5·in.lower[b](α,j)
    //              + s·in.lower[a](α,i) + p·g5·in.lower[a](α,i)
    //
    // Lattice storage: sigma()(a,b)(i,j).  σ^T read order: sigma()(b,a)(j,i).

    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          // Upper accumulator (σ, π, s, p forward).
          decltype(sigma_lane()(0,0)(i,0) * inu0_lane()(alpha)(0)) valu;
          zeroit(valu);
          // Lower accumulator (σ^T, π^T, s, p).
          decltype(sigma_lane()(0,0)(i,0) * inl0_lane()(alpha)(0)) vall;
          zeroit(vall);
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto in_u_b   = (b == 0) ? inu0_lane : inu1_lane;
            auto g5_u_b   = (b == 0) ? gu0_lane  : gu1_lane;
            auto in_l_b   = (b == 0) ? inl0_lane : inl1_lane;
            auto g5_l_b   = (b == 0) ? gl0_lane  : gl1_lane;
            for (int j = 0; j < Nc; ++j) {
              // σ_{ab}(i,j) and π_{ab}(i,j) for upper.
              auto sig_ab_ij = sigma_lane()(a, b)(i, j);
              auto  pi_ab_ij =    pi_lane()(a, b)(i, j);
              valu = valu + sig_ab_ij * in_u_b()(alpha)(j)
                          +  pi_ab_ij * g5_u_b()(alpha)(j);
              // σ^T: σ_{ba}(j,i); π^T: π_{ba}(j,i) for lower.
              auto sig_ba_ji = sigma_lane()(b, a)(j, i);
              auto  pi_ba_ji =    pi_lane()(b, a)(j, i);
              vall = vall + sig_ba_ji * in_l_b()(alpha)(j)
                          +  pi_ba_ji * g5_l_b()(alpha)(j);
            }
          }
          // Singlet s, p (same on both blocks since s, p are real scalars).
          auto sscalar = s_lane()()();
          auto pscalar = p_lane()()();
          auto in_u_a  = (a == 0) ? inu0_lane : inu1_lane;
          auto g5_u_a  = (a == 0) ? gu0_lane  : gu1_lane;
          auto in_l_a  = (a == 0) ? inl0_lane : inl1_lane;
          auto g5_l_a  = (a == 0) ? gl0_lane  : gl1_lane;
          valu = valu + sscalar * in_u_a()(alpha)(i) + pscalar * g5_u_a()(alpha)(i);
          vall = vall + sscalar * in_l_a()(alpha)(i) + pscalar * g5_l_a()(alpha)(i);
          if (a == 0) {
            outu0_acc()(alpha)(i) = outu0_acc()(alpha)(i) + valu;
            outl0_acc()(alpha)(i) = outl0_acc()(alpha)(i) + vall;
          } else {
            outu1_acc()(alpha)(i) = outu1_acc()(alpha)(i) + valu;
            outl1_acc()(alpha)(i) = outl1_acc()(alpha)(i) + vall;
          }
        }
      }
    }
    coalescedWrite(outu0v[ss], outu0_acc);
    coalescedWrite(outu1v[ss], outu1_acc);
    coalescedWrite(outl0v[ss], outl0_acc);
    coalescedWrite(outl1v[ss], outl1_acc);
  });
}

// WhichBlock controls which half of Pass B runs.  Used by DTXQCD_FUSED_SPLIT_PASSB
// to split a combined upper+lower Pass B into two smaller kernels (one upper,
// one lower) when the combined live state is too large for the device's
// occupancy window.
enum PassBBlock { kPassBBoth = 0, kPassBUpperOnly = 1, kPassBLowerOnly = 2 };

// Pass B: color-mixing flavor-diagonal cross (d, n) + clover (F).
//   out.upper[a] += Σ_b,j d_{ab}(i,j)·g5·in.lower[b](α,j) + n_{ab}(i,j)·in.lower[b](α,j)
//                 + Σ_k F_eff_U(k)(i,j) · (σ_μν_k·in.upper[a])(α,j)
//   out.lower[a] += Σ_b,j conj(d)_{ab}(i,j)·g5·in.upper[b](α,j) + conj(n)_{ab}(i,j)·in.upper[b](α,j)
//                 + Σ_k F_eff_L(k)(i,j) · (σ_μν_k·in.lower[a])(α,j)
// where F_eff_U(k)(i,j) = -csw/2·F_k(i,j) and F_eff_L(k)(i,j) = +csw/2·F_k(j,i).
template <bool HaveClover, PassBBlock Which = kPassBBoth>
inline void PassColorMixAccumulate(
    RealD csw,
    const LatticeDtxqcdD &d_field,
    const LatticeDtxqcdN &n_field,
    const LatticeColourMatrix &F0r,
    const LatticeColourMatrix &F1r,
    const LatticeColourMatrix &F2r,
    const LatticeColourMatrix &F3r,
    const LatticeColourMatrix &F4r,
    const LatticeColourMatrix &F5r,
    const DTXQCDFermionDoubled &in,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_u,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_l,
    const std::vector<LatticeFermion> &smn_in_u,   // [6*Nf]
    const std::vector<LatticeFermion> &smn_in_l,   // [6*Nf]
    DTXQCDFermionDoubled &out_accum) {
  static_assert(DtxqcdNf == 2, "fused fast path only for Nf=2");
  GridBase *grid = in.Grid();

  autoView(dv, d_field, AcceleratorRead);
  autoView(nv, n_field, AcceleratorRead);
  autoView(inu0v, in.upper.f[0], AcceleratorRead);
  autoView(inu1v, in.upper.f[1], AcceleratorRead);
  autoView(inl0v, in.lower.f[0], AcceleratorRead);
  autoView(inl1v, in.lower.f[1], AcceleratorRead);
  autoView(gu0v,  g5_in_u[0], AcceleratorRead);
  autoView(gu1v,  g5_in_u[1], AcceleratorRead);
  autoView(gl0v,  g5_in_l[0], AcceleratorRead);
  autoView(gl1v,  g5_in_l[1], AcceleratorRead);
  // 6 σ_μν · in.upper[a] views, indexed [a*6+k] for a∈{0,1}, k∈{0..5}.
  autoView(su00v, smn_in_u[0],  AcceleratorRead);
  autoView(su01v, smn_in_u[1],  AcceleratorRead);
  autoView(su02v, smn_in_u[2],  AcceleratorRead);
  autoView(su03v, smn_in_u[3],  AcceleratorRead);
  autoView(su04v, smn_in_u[4],  AcceleratorRead);
  autoView(su05v, smn_in_u[5],  AcceleratorRead);
  autoView(su10v, smn_in_u[6],  AcceleratorRead);
  autoView(su11v, smn_in_u[7],  AcceleratorRead);
  autoView(su12v, smn_in_u[8],  AcceleratorRead);
  autoView(su13v, smn_in_u[9],  AcceleratorRead);
  autoView(su14v, smn_in_u[10], AcceleratorRead);
  autoView(su15v, smn_in_u[11], AcceleratorRead);
  autoView(sl00v, smn_in_l[0],  AcceleratorRead);
  autoView(sl01v, smn_in_l[1],  AcceleratorRead);
  autoView(sl02v, smn_in_l[2],  AcceleratorRead);
  autoView(sl03v, smn_in_l[3],  AcceleratorRead);
  autoView(sl04v, smn_in_l[4],  AcceleratorRead);
  autoView(sl05v, smn_in_l[5],  AcceleratorRead);
  autoView(sl10v, smn_in_l[6],  AcceleratorRead);
  autoView(sl11v, smn_in_l[7],  AcceleratorRead);
  autoView(sl12v, smn_in_l[8],  AcceleratorRead);
  autoView(sl13v, smn_in_l[9],  AcceleratorRead);
  autoView(sl14v, smn_in_l[10], AcceleratorRead);
  autoView(sl15v, smn_in_l[11], AcceleratorRead);
  autoView(F0v,   F0r, AcceleratorRead);
  autoView(F1v,   F1r, AcceleratorRead);
  autoView(F2v,   F2r, AcceleratorRead);
  autoView(F3v,   F3r, AcceleratorRead);
  autoView(F4v,   F4r, AcceleratorRead);
  autoView(F5v,   F5r, AcceleratorRead);
  autoView(outu0v, out_accum.upper.f[0], AcceleratorWrite);
  autoView(outu1v, out_accum.upper.f[1], AcceleratorWrite);
  autoView(outl0v, out_accum.lower.f[0], AcceleratorWrite);
  autoView(outl1v, out_accum.lower.f[1], AcceleratorWrite);

  const RealD neg_csw_half = -0.5 * csw;
  const RealD pos_csw_half = +0.5 * csw;
  const int Nsimd = LatticeFermion::vector_object::Nsimd();

  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto d_lane = dv(ss);
    auto n_lane = nv(ss);

    auto F0_lane = F0v(ss);
    auto F1_lane = F1v(ss);
    auto F2_lane = F2v(ss);
    auto F3_lane = F3v(ss);
    auto F4_lane = F4v(ss);
    auto F5_lane = F5v(ss);

    // F_eff per (μ,ν), upper and lower (only the blocks we need given Which).
    typedef typename std::remove_cv<typename std::remove_reference<
        decltype(d_lane()(0, 0)(0, 0))>::type>::type ScalarLane;
    constexpr bool need_upper = (Which == kPassBBoth || Which == kPassBUpperOnly);
    constexpr bool need_lower = (Which == kPassBBoth || Which == kPassBLowerOnly);
    ScalarLane feffU[6][Nc][Nc];
    ScalarLane feffL[6][Nc][Nc];
    if (HaveClover) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          if (need_upper) {
            feffU[0][i][j] = neg_csw_half * F0_lane()()(i, j);
            feffU[1][i][j] = neg_csw_half * F1_lane()()(i, j);
            feffU[2][i][j] = neg_csw_half * F2_lane()()(i, j);
            feffU[3][i][j] = neg_csw_half * F3_lane()()(i, j);
            feffU[4][i][j] = neg_csw_half * F4_lane()()(i, j);
            feffU[5][i][j] = neg_csw_half * F5_lane()()(i, j);
          }
          if (need_lower) {
            // F^T at use for lower: F_eff_L(i,j) = +csw/2 · F(j,i).
            feffL[0][i][j] = pos_csw_half * F0_lane()()(j, i);
            feffL[1][i][j] = pos_csw_half * F1_lane()()(j, i);
            feffL[2][i][j] = pos_csw_half * F2_lane()()(j, i);
            feffL[3][i][j] = pos_csw_half * F3_lane()()(j, i);
            feffL[4][i][j] = pos_csw_half * F4_lane()()(j, i);
            feffL[5][i][j] = pos_csw_half * F5_lane()()(j, i);
          }
        }
      }
    }

    auto outu0_acc = coalescedRead(outu0v[ss]);
    auto outu1_acc = coalescedRead(outu1v[ss]);
    auto outl0_acc = coalescedRead(outl0v[ss]);
    auto outl1_acc = coalescedRead(outl1v[ss]);

    for (int a = 0; a < DtxqcdNf; ++a) {
      // Cross: out.upper[a] += Σ_b,j d_{ab}(i,j)·g5·in.lower[b](α,j) + n_{ab}(i,j)·in.lower[b](α,j)
      // Cross: out.lower[a] += Σ_b,j conj(d)_{ab}(i,j)·g5·in.upper[b](α,j) + conj(n)_{ab}(i,j)·in.upper[b](α,j)
      //
      // Clover upper: out.upper[a] += Σ_k F_eff_U(k)(i,j) · (σ_μν_k · in.upper[a])(α,j)
      // Clover lower: out.lower[a] += Σ_k F_eff_L(k)(i,j) · (σ_μν_k · in.lower[a])(α,j)
      auto smU0 = (a == 0) ? su00v(ss) : su10v(ss);
      auto smU1 = (a == 0) ? su01v(ss) : su11v(ss);
      auto smU2 = (a == 0) ? su02v(ss) : su12v(ss);
      auto smU3 = (a == 0) ? su03v(ss) : su13v(ss);
      auto smU4 = (a == 0) ? su04v(ss) : su14v(ss);
      auto smU5 = (a == 0) ? su05v(ss) : su15v(ss);
      auto smL0 = (a == 0) ? sl00v(ss) : sl10v(ss);
      auto smL1 = (a == 0) ? sl01v(ss) : sl11v(ss);
      auto smL2 = (a == 0) ? sl02v(ss) : sl12v(ss);
      auto smL3 = (a == 0) ? sl03v(ss) : sl13v(ss);
      auto smL4 = (a == 0) ? sl04v(ss) : sl14v(ss);
      auto smL5 = (a == 0) ? sl05v(ss) : sl15v(ss);

      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          decltype(d_lane()(0,0)(i,0) * smU0()(alpha)(0)) valu;
          decltype(d_lane()(0,0)(i,0) * smL0()(alpha)(0)) vall;
          if (need_upper) zeroit(valu);
          if (need_lower) zeroit(vall);

          // Cross contributions.
          for (int b = 0; b < DtxqcdNf; ++b) {
            for (int j = 0; j < Nc; ++j) {
              auto d_ab_ij = d_lane()(a, b)(i, j);
              auto n_ab_ij = n_lane()(a, b)(i, j);
              if (need_upper) {
                auto gl_b   = (b == 0) ? gl0v(ss) : gl1v(ss);
                auto inl_b  = (b == 0) ? inl0v(ss) : inl1v(ss);
                // M_UR (lower → upper).
                valu = valu + d_ab_ij * gl_b()(alpha)(j) + n_ab_ij * inl_b()(alpha)(j);
              }
              if (need_lower) {
                auto gu_b   = (b == 0) ? gu0v(ss) : gu1v(ss);
                auto inu_b  = (b == 0) ? inu0v(ss) : inu1v(ss);
                // M_LL (upper → lower) with conj(d), conj(n).
                auto d_conj = conjugate(d_ab_ij);
                auto n_conj = conjugate(n_ab_ij);
                vall = vall + d_conj * gu_b()(alpha)(j) + n_conj * inu_b()(alpha)(j);
              }
            }
          }

          // Clover contributions.
          if (HaveClover) {
            for (int j = 0; j < Nc; ++j) {
              if (need_upper) {
                valu = valu + feffU[0][i][j] * smU0()(alpha)(j)
                            + feffU[1][i][j] * smU1()(alpha)(j)
                            + feffU[2][i][j] * smU2()(alpha)(j)
                            + feffU[3][i][j] * smU3()(alpha)(j)
                            + feffU[4][i][j] * smU4()(alpha)(j)
                            + feffU[5][i][j] * smU5()(alpha)(j);
              }
              if (need_lower) {
                vall = vall + feffL[0][i][j] * smL0()(alpha)(j)
                            + feffL[1][i][j] * smL1()(alpha)(j)
                            + feffL[2][i][j] * smL2()(alpha)(j)
                            + feffL[3][i][j] * smL3()(alpha)(j)
                            + feffL[4][i][j] * smL4()(alpha)(j)
                            + feffL[5][i][j] * smL5()(alpha)(j);
              }
            }
          }

          if (a == 0) {
            if (need_upper) outu0_acc()(alpha)(i) = outu0_acc()(alpha)(i) + valu;
            if (need_lower) outl0_acc()(alpha)(i) = outl0_acc()(alpha)(i) + vall;
          } else {
            if (need_upper) outu1_acc()(alpha)(i) = outu1_acc()(alpha)(i) + valu;
            if (need_lower) outl1_acc()(alpha)(i) = outl1_acc()(alpha)(i) + vall;
          }
        }
      }
    }
    if (need_upper) {
      coalescedWrite(outu0v[ss], outu0_acc);
      coalescedWrite(outu1v[ss], outu1_acc);
    }
    if (need_lower) {
      coalescedWrite(outl0v[ss], outl0_acc);
      coalescedWrite(outl1v[ss], outl1_acc);
    }
  });
}

}  // namespace DtxqcdFusedAuxCloverDetail

// Apply Delta(diag) + cross + clover for both upper and lower blocks in two
// fused kernels (one for σ+π+s+p flavor-mix, one for d+n+clover color-mix).
//
//   accumulate=true:  out_accum.upper += ...  out_accum.lower += ...
//   accumulate=false: out_accum.upper  = ...  out_accum.lower  = ...
//
// Matches DTXQCDWilsonCloverFermionEO::AddDiagAndCrossAndClover to FP
// precision.  Nf=2 fast path; generic Nf falls back to the per-piece
// Lattice arithmetic (slow but correct).
inline void ApplyFusedAuxCrossClover(
    RealD csw,
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi    &pi,
    const LatticeDtxqcdD     &d,
    const LatticeDtxqcdN     &n,
    const LatticeDtxqcdS     &s,
    const LatticeDtxqcdP     &p,
    const std::vector<LatticeColourMatrix> &FS,
    const DTXQCDFermionDoubled &in,
    DTXQCDFermionDoubled       &out_accum,
    bool accumulate = true) {
  GridBase *grid = in.Grid();
  int cb_u = in.upper.f[0].Checkerboard();
  int cb_l = in.lower.f[0].Checkerboard();
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_accum.upper.f[a].Checkerboard() = cb_u;
    out_accum.lower.f[a].Checkerboard() = cb_l;
  }

  if constexpr (DtxqcdNf != 2) {
    // Fallback to legacy per-piece path.
    DTXQCDFermionNf delta_u(grid), delta_l(grid);
    DtxqcdApplyDeltaDiag     (sigma, pi, s, p, in.upper, delta_u);
    DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, in.lower, delta_l);
    DTXQCDFermionNf cross_u(grid), cross_l(grid);
    DtxqcdApplyDnCross(d, n, in.lower, cross_u, /*apply_conj=*/false);
    DtxqcdApplyDnCross(d, n, in.upper, cross_l, /*apply_conj=*/true);
    if (csw != 0.0 && FS.size() == 6) {
      DTXQCDFermionNf clov_u(grid), clov_l(grid);
      DtxqcdApplyCloverUpper(csw, FS, in.upper, clov_u);
      DtxqcdApplyCloverLower(csw, FS, in.lower, clov_l);
      for (int a = 0; a < DtxqcdNf; ++a) {
        delta_u.f[a] = delta_u.f[a] + clov_u.f[a];
        delta_l.f[a] = delta_l.f[a] + clov_l.f[a];
      }
    }
    for (int a = 0; a < DtxqcdNf; ++a) {
      if (accumulate) {
        out_accum.upper.f[a] = out_accum.upper.f[a] + delta_u.f[a] + cross_u.f[a];
        out_accum.lower.f[a] = out_accum.lower.f[a] + delta_l.f[a] + cross_l.f[a];
      } else {
        out_accum.upper.f[a] = delta_u.f[a] + cross_u.f[a];
        out_accum.lower.f[a] = delta_l.f[a] + cross_l.f[a];
      }
    }
    return;
  }

  if (!accumulate) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_accum.upper.f[a] = Zero();
      out_accum.upper.f[a].Checkerboard() = cb_u;
      out_accum.lower.f[a] = Zero();
      out_accum.lower.f[a].Checkerboard() = cb_l;
    }
  }

  const bool have_clover = (csw != 0.0) && (FS.size() == 6);

  Gamma g5(Gamma::Algebra::Gamma5);

  // γ5 pre-rotations: 2 (upper) + 2 (lower) = 4 GPU Lattice expressions.
  std::array<LatticeFermion, DtxqcdNf> g5_in_u =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  std::array<LatticeFermion, DtxqcdNf> g5_in_l =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  for (int a = 0; a < DtxqcdNf; ++a) {
    g5_in_u[a] = g5 * in.upper.f[a];
    g5_in_l[a] = g5 * in.lower.f[a];
  }

  // σ_μν spinor pre-rotations: 6 (μν) × 2 (a) per block.  Done ONCE per call
  // (the legacy DtxqcdApplyCloverUpper / DtxqcdApplyCloverLower each does its
  // own 12).
  constexpr int Npairs = 6;
  std::vector<LatticeFermion> smn_in_u;
  std::vector<LatticeFermion> smn_in_l;
  smn_in_u.reserve(Npairs * DtxqcdNf);
  smn_in_l.reserve(Npairs * DtxqcdNf);
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        smn_in_u.emplace_back(grid);
        smn_in_u.back() = smn * in.upper.f[a];
        smn_in_l.emplace_back(grid);
        smn_in_l.back() = smn * in.lower.f[a];
      }
    }
  }

  // FS view targets: when csw=0 we still need 6 LatticeColourMatrix references
  // to satisfy the kernel signature.  Reinterpret-cast a compatible Lattice
  // (any LatticeColourMatrix-like; we pick FS[0] when have_clover, dummy
  // otherwise; the kernel branch is templated so the dummy is never read).
  static thread_local std::unique_ptr<LatticeColourMatrix> dummy_holder;
  if (!have_clover) {
    if (!dummy_holder || dummy_holder->Grid() != grid) {
      dummy_holder = std::make_unique<LatticeColourMatrix>(grid);
      *dummy_holder = Zero();
    }
  }
  const LatticeColourMatrix &dummy = have_clover ? FS[0] : *dummy_holder;
  const LatticeColourMatrix &F0r = have_clover ? FS[0] : dummy;
  const LatticeColourMatrix &F1r = have_clover ? FS[1] : dummy;
  const LatticeColourMatrix &F2r = have_clover ? FS[2] : dummy;
  const LatticeColourMatrix &F3r = have_clover ? FS[3] : dummy;
  const LatticeColourMatrix &F4r = have_clover ? FS[4] : dummy;
  const LatticeColourMatrix &F5r = have_clover ? FS[5] : dummy;

  // Pass A: flavor-mixing σ + π + singlet s + p (upper + lower).
  DtxqcdFusedAuxCloverDetail::PassFlavorMixAccumulate(
      sigma, pi, s, p, in, g5_in_u, g5_in_l, out_accum);

  // Pass B: color-mixing d + n + clover (upper + lower).
  // When DTXQCD_FUSED_SPLIT_PASSB=1, split into two smaller kernels (one
  // upper-only, one lower-only) to halve the per-thread live state.  Useful
  // at large per-rank volumes where the combined Pass B overflows the
  // occupancy window (observed at 32^3x64 b6.5 production scale).
  const bool split_b = DtxqcdFusedAuxClover::SplitPassB();
  if (have_clover) {
    if (split_b) {
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<true,
          DtxqcdFusedAuxCloverDetail::kPassBUpperOnly>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<true,
          DtxqcdFusedAuxCloverDetail::kPassBLowerOnly>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
    } else {
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<true>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
    }
  } else {
    if (split_b) {
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<false,
          DtxqcdFusedAuxCloverDetail::kPassBUpperOnly>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<false,
          DtxqcdFusedAuxCloverDetail::kPassBLowerOnly>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
    } else {
      DtxqcdFusedAuxCloverDetail::PassColorMixAccumulate<false>(
          csw, d, n, F0r, F1r, F2r, F3r, F4r, F5r,
          in, g5_in_u, g5_in_l, smn_in_u, smn_in_l, out_accum);
    }
  }
}

NAMESPACE_END(Grid);
