#pragma once
// M-wrap.3: DTXQCD aux-only (no clover) fused per-site kernel on the doubled
// fermion in Grid SIMD layout.
//
// Used by the QUDA Mat wrapper (M-wrap.4+) to add the DTXQCD aux contribution
// *after* QUDA's stock Wilson-clover Mat has populated out with M_WC·in.
// QUDA already handles csw·F·σ_μν natively in its clover field — this kernel
// is the missing aux-only piece (σ + π + s + p + d + n) for the DTXQCD v2
// roster (see reference_dtxqcd_v2_no_t_tensor memory: no t-tensor in v2).
//
// Differences from the existing DTXQCDFusedAuxClover.h:
//   - NO clover (no F views, no σ_μν pre-rotations, no csw)
//   - Runtime `transpose_aux` and `use_dn_conj` flags (vs constexpr in the
//     production-locked path). Outer dispatcher routes to one of 4 fully
//     instantiated template specialisations so the hot path still folds the
//     flag to a literal for occupancy.
//
// Per-site arithmetic mirrors:
//   Pass A (flavor-mixing color-diagonal):
//     out.upper[a] += Σ_b σ_{ab}·in.upper[b] + π_{ab}·γ5·in.upper[b]
//                   + s·in.upper[a] + p·γ5·in.upper[a]
//     out.lower[a] += Σ_b σ_{ba|ab}·in.lower[b] + π_{ba|ab}·γ5·in.lower[b]
//                   + s·in.lower[a] + p·γ5·in.lower[a]
//   (lower uses σ_{ba}(j,i) when TransposeAux, σ_{ab}(i,j) otherwise.)
//
//   Pass B (cross color-mixing):
//     out.upper[a] += Σ_b,j d_{ab}(i,j)·γ5·in.lower[b](α,j) + n_{ab}(i,j)·in.lower[b](α,j)
//     out.lower[a] += Σ_b,j (d'_{ab})(i,j)·γ5·in.upper[b](α,j) + (n'_{ab})(i,j)·in.upper[b](α,j)
//   where d', n' = conj(d), conj(n) when UseDnConj, else d, n.
//
// Both passes scaled by DtxqcdOffdiagFactor() (= 1.0 in production) for the
// cross piece (Pass A's off-diagonal factor is absorbed in the σ/π definition).

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdQudaAuxKernel {

namespace dtxqcd_kernel_dispatch_inner {

// Pass A: σ + π + s + p (flavor-mixing color-diagonal) for upper AND lower.
// TransposeAux controls whether lower reads σ_{ba}(j,i) (T=true, production
// SIGMA_PI_HERMITIAN_ONLY path) or σ_{ab}(i,j) (T=false, naive same-as-upper).
template <bool TransposeAux>
inline void PassFlavorMix(
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

    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          decltype(sigma_lane()(0,0)(i,0) * inu0_lane()(alpha)(0)) valu;
          zeroit(valu);
          decltype(sigma_lane()(0,0)(i,0) * inl0_lane()(alpha)(0)) vall;
          zeroit(vall);
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto in_u_b = (b == 0) ? inu0_lane : inu1_lane;
            auto g5_u_b = (b == 0) ? gu0_lane  : gu1_lane;
            auto in_l_b = (b == 0) ? inl0_lane : inl1_lane;
            auto g5_l_b = (b == 0) ? gl0_lane  : gl1_lane;
            for (int j = 0; j < Nc; ++j) {
              auto sig_ab_ij = sigma_lane()(a, b)(i, j);
              auto  pi_ab_ij =    pi_lane()(a, b)(i, j);
              valu = valu + sig_ab_ij * in_u_b()(alpha)(j)
                          +  pi_ab_ij * g5_u_b()(alpha)(j);

              if constexpr (TransposeAux) {
                auto sig_ba_ji = sigma_lane()(b, a)(j, i);
                auto  pi_ba_ji =    pi_lane()(b, a)(j, i);
                vall = vall + sig_ba_ji * in_l_b()(alpha)(j)
                            +  pi_ba_ji * g5_l_b()(alpha)(j);
              } else {
                vall = vall + sig_ab_ij * in_l_b()(alpha)(j)
                            +  pi_ab_ij * g5_l_b()(alpha)(j);
              }
            }
          }
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

// Pass B: d + n cross (color-mixing flavor-diagonal).
// UseDnConj=true (production DTXQCD_DN_COMPLEX_SYMMETRIC path) applies conj(d),
// conj(n) for the upper→lower direction; false uses d, n verbatim.
// Scale: DtxqcdOffdiagFactor() (= 1.0 in production).
template <bool UseDnConj>
inline void PassCrossDn(
    const LatticeDtxqcdD &d_field,
    const LatticeDtxqcdN &n_field,
    const DTXQCDFermionDoubled &in,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_u,
    const std::array<LatticeFermion, DtxqcdNf> &g5_in_l,
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
  autoView(outu0v, out_accum.upper.f[0], AcceleratorWrite);
  autoView(outu1v, out_accum.upper.f[1], AcceleratorWrite);
  autoView(outl0v, out_accum.lower.f[0], AcceleratorWrite);
  autoView(outl1v, out_accum.lower.f[1], AcceleratorWrite);

  const RealD scale = DtxqcdOffdiagFactor();   // = 1.0
  const int Nsimd = LatticeFermion::vector_object::Nsimd();

  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto d_lane = dv(ss);
    auto n_lane = nv(ss);
    auto inu0_lane = inu0v(ss);
    auto inu1_lane = inu1v(ss);
    auto inl0_lane = inl0v(ss);
    auto inl1_lane = inl1v(ss);
    auto gu0_lane  = gu0v(ss);
    auto gu1_lane  = gu1v(ss);
    auto gl0_lane  = gl0v(ss);
    auto gl1_lane  = gl1v(ss);
    auto outu0_acc = coalescedRead(outu0v[ss]);
    auto outu1_acc = coalescedRead(outu1v[ss]);
    auto outl0_acc = coalescedRead(outl0v[ss]);
    auto outl1_acc = coalescedRead(outl1v[ss]);

    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          decltype(d_lane()(0,0)(i,0) * inl0_lane()(alpha)(0)) valu;
          decltype(d_lane()(0,0)(i,0) * inu0_lane()(alpha)(0)) vall;
          zeroit(valu);
          zeroit(vall);
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto inl_b = (b == 0) ? inl0_lane : inl1_lane;
            auto gl_b  = (b == 0) ? gl0_lane  : gl1_lane;
            auto inu_b = (b == 0) ? inu0_lane : inu1_lane;
            auto gu_b  = (b == 0) ? gu0_lane  : gu1_lane;
            for (int j = 0; j < Nc; ++j) {
              auto d_ab_ij = d_lane()(a, b)(i, j);
              auto n_ab_ij = n_lane()(a, b)(i, j);
              // Upper gets cross from lower (M_UR): no conj.
              valu = valu + d_ab_ij * gl_b()(alpha)(j) + n_ab_ij * inl_b()(alpha)(j);
              // Lower gets cross from upper (M_LL): conj if UseDnConj.
              if constexpr (UseDnConj) {
                auto d_conj = conjugate(d_ab_ij);
                auto n_conj = conjugate(n_ab_ij);
                vall = vall + d_conj * gu_b()(alpha)(j) + n_conj * inu_b()(alpha)(j);
              } else {
                vall = vall + d_ab_ij * gu_b()(alpha)(j) + n_ab_ij * inu_b()(alpha)(j);
              }
            }
          }
          if (a == 0) {
            outu0_acc()(alpha)(i) = outu0_acc()(alpha)(i) + scale * valu;
            outl0_acc()(alpha)(i) = outl0_acc()(alpha)(i) + scale * vall;
          } else {
            outu1_acc()(alpha)(i) = outu1_acc()(alpha)(i) + scale * valu;
            outl1_acc()(alpha)(i) = outl1_acc()(alpha)(i) + scale * vall;
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

template <bool TransposeAux, bool UseDnConj>
inline void DispatchTpl(const LatticeDtxqcdSigma &sigma,
                       const LatticeDtxqcdPi    &pi,
                       const LatticeDtxqcdD     &d,
                       const LatticeDtxqcdN     &n,
                       const LatticeDtxqcdS     &s,
                       const LatticeDtxqcdP     &p,
                       const DTXQCDFermionDoubled &in,
                       const std::array<LatticeFermion, DtxqcdNf> &g5_in_u,
                       const std::array<LatticeFermion, DtxqcdNf> &g5_in_l,
                       DTXQCDFermionDoubled &out_accum) {
  PassFlavorMix<TransposeAux>(sigma, pi, s, p, in, g5_in_u, g5_in_l, out_accum);
  PassCrossDn<UseDnConj>(d, n, in, g5_in_u, g5_in_l, out_accum);
}

}  // namespace dtxqcd_kernel_dispatch_inner

// Public entry point.
//   accumulate=true:  out_accum.upper += (X(σ,π,s,p) + cross(d,n))·in
//   accumulate=false: out_accum.upper  = (X(σ,π,s,p) + cross(d,n))·in
//                     out_accum.lower  = (X^T?(σ,π,s,p) + cross(conj(d),conj(n))?)·in
// Runtime flags transpose_aux + use_dn_conj dispatch to one of 4 templated paths.
inline void ApplyFusedDtxqcdAuxKernel(
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi    &pi,
    const LatticeDtxqcdD     &d,
    const LatticeDtxqcdN     &n,
    const LatticeDtxqcdS     &s,
    const LatticeDtxqcdP     &p,
    const DTXQCDFermionDoubled &in,
    DTXQCDFermionDoubled       &out_accum,
    bool transpose_aux = true,    // production constexpr default
    bool use_dn_conj   = true,    // production constexpr default
    bool accumulate    = true) {
  static_assert(DtxqcdNf == 2, "fused fast path only for Nf=2");
  GridBase *grid = in.Grid();
  int cb_u = in.upper.f[0].Checkerboard();
  int cb_l = in.lower.f[0].Checkerboard();
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_accum.upper.f[a].Checkerboard() = cb_u;
    out_accum.lower.f[a].Checkerboard() = cb_l;
  }

  if (!accumulate) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_accum.upper.f[a] = Zero();
      out_accum.upper.f[a].Checkerboard() = cb_u;
      out_accum.lower.f[a] = Zero();
      out_accum.lower.f[a].Checkerboard() = cb_l;
    }
  }

  // γ5 pre-rotations: 4 GPU Lattice expressions, computed once per call.
  Gamma g5(Gamma::Algebra::Gamma5);
  std::array<LatticeFermion, DtxqcdNf> g5_in_u =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  std::array<LatticeFermion, DtxqcdNf> g5_in_l =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  for (int a = 0; a < DtxqcdNf; ++a) {
    g5_in_u[a] = g5 * in.upper.f[a];
    g5_in_l[a] = g5 * in.lower.f[a];
  }

  // Runtime dispatch -> compile-time specialised inner kernels.
  if (transpose_aux && use_dn_conj)
    dtxqcd_kernel_dispatch_inner::DispatchTpl<true,  true >(sigma, pi, d, n, s, p, in, g5_in_u, g5_in_l, out_accum);
  else if (transpose_aux)
    dtxqcd_kernel_dispatch_inner::DispatchTpl<true,  false>(sigma, pi, d, n, s, p, in, g5_in_u, g5_in_l, out_accum);
  else if (use_dn_conj)
    dtxqcd_kernel_dispatch_inner::DispatchTpl<false, true >(sigma, pi, d, n, s, p, in, g5_in_u, g5_in_l, out_accum);
  else
    dtxqcd_kernel_dispatch_inner::DispatchTpl<false, false>(sigma, pi, d, n, s, p, in, g5_in_u, g5_in_l, out_accum);
}

}  // namespace DtxqcdQudaAuxKernel
NAMESPACE_END(Grid);
