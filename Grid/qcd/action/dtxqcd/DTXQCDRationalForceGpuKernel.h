#pragma once
// GPU kernel that replaces DTXQCDWilsonCloverRationalEOAction::Accumulate-
// SiteForces' per-site CPU thread_for (unvectorize X/Y -> build 48-vectors ->
// symmetrised bilinear -> AuxForceAt + CloverSigmaAt -> Wirtinger transpose ->
// AddEvenOddToFull).
//
// Reads the per-pole, per-CB doubled CG solutions X, Y (DTXQCDFermionDoubled,
// already device-resident on the CB grid) directly via getlane and writes the
// six aux-force RB-CB Lattice outputs (sigma, pi, d, n, s, p) plus the 6
// clover_sigma colour matrices.  Caller (AccumulateSiteForces) then accumulates
// each into the full-grid dSdU.* with the per-pole coefficient (2 alpha_k) via
// the same pickCheckerboard/+=/setCheckerboard chain the CPU path used.
//
// Bit-for-bit reference: the CPU thread_for in AccumulateSiteForces.  This
// kernel transcribes DtxqcdSiteForceKernel::{AuxForceAt,CloverSigmaAt} exactly,
// folding the locked compile-time conventions:
//   sigpi_T = DtxqcdSigmaPiHermitianOnly() == true
//   dn_cs   = DtxqcdDnComplexSymmetric()   == true
//   sqrt2   = DtxqcdOffdiagFactor()        == 1.0
// and the Wirtinger transpose T (a<->b, i<->j) on sigma/pi/d/n (s/p/clover
// are stored untransposed, matching the CPU code).
//
// nvcc notes (cf. DTXQCDLogDetCloverEOAction / TXQCDLogDetGpuKernel):
//   - getlane/putlane + an explicit simt_lane (NOT coalescedRead) so the body
//     compiles identically on CPU (mode-gated, never run) and GPU.
//   - gamma5 / sigma_munu flattened to POD std::array captured BY VALUE; no
//     Eigen, no peekSite inside the accelerator_for.
//   - the per-site index helper is accelerator_inline (the host DtxqcdSiteIdx24
//     is host-only and would not link in device code).

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdRatForceGpu {

static constexpr int kDim24 = kDtxqcdSiteDim24;  // 24
static constexpr int kDim48 = kDtxqcdSiteDim48;  // 48

// Device-callable (a, alpha, i) -> [0,24) packer (host DtxqcdSiteIdx24 is
// host-only; replicate the identical arithmetic here for the kernel).
accelerator_inline int Idx24Dev(int a, int alpha, int i) {
  return a * (Ns * Nc) + alpha * Nc + i;
}

// std::array-of-LatticeView helper (LatticeView has no default ctor; the
// index_sequence trick aggregate-initialises the array).  Mirrors the EO
// operator's MakeFermViewsRead.
template <std::size_t N, class FieldT, std::size_t... Is>
static auto MakeReadViews(const FieldT &fld, std::index_sequence<Is...>)
    -> std::array<decltype(fld.f[0].View(AcceleratorRead)), N> {
  return {{ fld.f[Is].View(AcceleratorRead)... }};
}

// ----------------------------------------------------------------------
// Extract the per-CB rational aux + clover-sigma forces into RB-CB Lattice
// outputs.  Outputs carry the UNSCALED kernel values (the caller multiplies by
// coef = 2 alpha_k when accumulating), matching the CPU fsig/.../fcs arrays.
//
// F_cs is always written (cheap); the caller skips accumulating it when
// csw == 0 (where every entry is -0.5*csw*... = 0 anyway).
// ----------------------------------------------------------------------
inline void Extract(const DTXQCDFermionDoubled &X,
                    const DTXQCDFermionDoubled &Y,
                    int cb,
                    RealD csw,
                    const DtxqcdSpinMatrices &spin,
                    LatticeDtxqcdSigma &F_sig,
                    LatticeDtxqcdPi    &F_pi,
                    LatticeDtxqcdD     &F_d,
                    LatticeDtxqcdN     &F_n,
                    LatticeDtxqcdS     &F_s,
                    LatticeDtxqcdP     &F_p,
                    std::array<LatticeColourMatrix, 6> &F_cs) {
  // Flatten gamma5 (4x4) and 6 sigma_{mu,nu} (4x4) into POD arrays by value.
  std::array<ComplexD, Ns * Ns> g5f{};
  for (int a = 0; a < Ns; ++a)
    for (int b = 0; b < Ns; ++b)
      g5f[a * Ns + b] = ComplexD(spin.gamma5(a, b).real(),
                                 spin.gamma5(a, b).imag());
  std::array<ComplexD, 6 * Ns * Ns> smf{};
  for (int p = 0; p < 6; ++p)
    for (int a = 0; a < Ns; ++a)
      for (int b = 0; b < Ns; ++b)
        smf[p * Ns * Ns + a * Ns + b] =
            ComplexD(spin.sigma_munu[p](a, b).real(),
                     spin.sigma_munu[p](a, b).imag());

  const RealD sqrt2 = DtxqcdOffdiagFactor();  // == 1.0 (constexpr)
  const RealD csw_l = csw;

  F_sig.Checkerboard() = cb;
  F_pi.Checkerboard()  = cb;
  F_d.Checkerboard()   = cb;
  F_n.Checkerboard()   = cb;
  F_s.Checkerboard()   = cb;
  F_p.Checkerboard()   = cb;
  for (int k = 0; k < 6; ++k) F_cs[k].Checkerboard() = cb;

  GridBase *rbgrid = X.upper.f[0].Grid();
  uint64_t oSites = rbgrid->oSites();
  constexpr int Nsimd = LatticeFermion::vector_object::Nsimd();

  auto Xup = MakeReadViews<DtxqcdNf>(X.upper, std::make_index_sequence<DtxqcdNf>{});
  auto Xlo = MakeReadViews<DtxqcdNf>(X.lower, std::make_index_sequence<DtxqcdNf>{});
  auto Yup = MakeReadViews<DtxqcdNf>(Y.upper, std::make_index_sequence<DtxqcdNf>{});
  auto Ylo = MakeReadViews<DtxqcdNf>(Y.lower, std::make_index_sequence<DtxqcdNf>{});

  autoView(sigv, F_sig, AcceleratorWrite);
  autoView(piv,  F_pi,  AcceleratorWrite);
  autoView(dv,   F_d,   AcceleratorWrite);
  autoView(nv,   F_n,   AcceleratorWrite);
  autoView(sv,   F_s,   AcceleratorWrite);
  autoView(pv,   F_p,   AcceleratorWrite);
  autoView(cs0, F_cs[0], AcceleratorWrite);
  autoView(cs1, F_cs[1], AcceleratorWrite);
  autoView(cs2, F_cs[2], AcceleratorWrite);
  autoView(cs3, F_cs[3], AcceleratorWrite);
  autoView(cs4, F_cs[4], AcceleratorWrite);
  autoView(cs5, F_cs[5], AcceleratorWrite);

  accelerator_for(s, oSites, Nsimd, {
#if defined(GRID_CUDA) || defined(GRID_HIP) || defined(GRID_SYCL)
    int simt_lane = static_cast<int>(lane);
#else
    int simt_lane = 0;  // CPU: SIMT-packed kernel is mode-gated, never run
#endif
    // ---- gather the doubled 48-vectors X_x, Y_x for this (site, lane) ----
    ComplexD X_x[kDim48], Y_x[kDim48];
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          int r = Idx24Dev(a, alpha, i);
          X_x[r]          = getlane(Xup[a][s]()(alpha)(i), simt_lane);
          X_x[kDim24 + r] = getlane(Xlo[a][s]()(alpha)(i), simt_lane);
          Y_x[r]          = getlane(Yup[a][s]()(alpha)(i), simt_lane);
          Y_x[kDim24 + r] = getlane(Ylo[a][s]()(alpha)(i), simt_lane);
        }
      }
    }

    // Bil(R,C) = 0.5*(conj(Y[C])X[R] + conj(X[C])Y[R])  (symmetrised Wirtinger).
    auto Inv = [&](int R, int C) -> ComplexD {
      return ComplexD(0.5, 0.0) *
          (DtxqcdConj(Y_x[C]) * X_x[R] + DtxqcdConj(X_x[C]) * Y_x[R]);
    };

    // Kernel outputs in the ORIGINAL (kernel) index convention, pre-transpose.
    ComplexD sig_k[DtxqcdNf][DtxqcdNf][Nc][Nc];
    ComplexD pi_k [DtxqcdNf][DtxqcdNf][Nc][Nc];
    ComplexD d_k  [DtxqcdNf][DtxqcdNf][Nc][Nc];
    ComplexD n_k  [DtxqcdNf][DtxqcdNf][Nc][Nc];

    // ---- sigma (sigpi_T = true) ----
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            ComplexD val(0.0, 0.0);
            for (int alpha = 0; alpha < Ns; ++alpha) {
              int ra = Idx24Dev(a, alpha, i);
              int cb_ = Idx24Dev(b, alpha, j);
              val = val + Inv(cb_, ra) + Inv(kDim24 + ra, kDim24 + cb_);
            }
            sig_k[a][b][i][j] = ComplexD(-val.real(), -val.imag());
          }

    // ---- pi (sigpi_T = true) ----
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            ComplexD val(0.0, 0.0);
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD g5 = g5f[alpha * Ns + beta];
                if (g5.real() == 0.0 && g5.imag() == 0.0) continue;
                int ra = Idx24Dev(a, alpha, i);
                int cb_ = Idx24Dev(b, beta, j);
                val = val + g5 * (Inv(cb_, ra) + Inv(kDim24 + ra, kDim24 + cb_));
              }
            pi_k[a][b][i][j] = ComplexD(-val.real(), -val.imag());
          }

    // ---- d (dn_cs = true) ----
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            ComplexD val(0.0, 0.0);
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD g5 = g5f[alpha * Ns + beta];
                if (g5.real() == 0.0 && g5.imag() == 0.0) continue;
                int ra = Idx24Dev(a, alpha, i);
                int cb_ = Idx24Dev(b, beta, j);
                val = val + g5 * (DtxqcdConj(Inv(kDim24 + cb_, ra))
                                  + Inv(cb_, kDim24 + ra));
              }
            ComplexD v = ComplexD(-sqrt2, 0.0) * val;
            d_k[a][b][i][j] = v;
          }

    // ---- n (dn_cs = true) ----
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            ComplexD val(0.0, 0.0);
            for (int alpha = 0; alpha < Ns; ++alpha) {
              int ra = Idx24Dev(a, alpha, i);
              int cb_ = Idx24Dev(b, alpha, j);
              val = val + DtxqcdConj(Inv(kDim24 + cb_, ra))
                        + Inv(cb_, kDim24 + ra);
            }
            ComplexD v = ComplexD(-sqrt2, 0.0) * val;
            n_k[a][b][i][j] = v;
          }

    // ---- s singlet ----
    ComplexD s_val(0.0, 0.0);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int alpha = 0; alpha < Ns; ++alpha)
        for (int i = 0; i < Nc; ++i) {
          int r = Idx24Dev(a, alpha, i);
          s_val = s_val + Inv(r, r) + Inv(kDim24 + r, kDim24 + r);
        }
    ComplexD s_out(-s_val.real(), -s_val.imag());

    // ---- p singlet ----
    ComplexD p_val(0.0, 0.0);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int alpha = 0; alpha < Ns; ++alpha)
        for (int beta = 0; beta < Ns; ++beta) {
          ComplexD g5 = g5f[alpha * Ns + beta];
          if (g5.real() == 0.0 && g5.imag() == 0.0) continue;
          for (int i = 0; i < Nc; ++i) {
            int ra = Idx24Dev(a, alpha, i);
            int cb_ = Idx24Dev(a, beta, i);
            p_val = p_val + g5 * (Inv(ra, cb_) + Inv(kDim24 + ra, kDim24 + cb_));
          }
        }
    ComplexD p_out(-p_val.real(), -p_val.imag());

    // ---- Wirtinger transpose (a<->b, i<->j) into the output scalar objects,
    //      then putlane.  s/p direct.  ----
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            putlane(sigv[s]()(a, b)(i, j), sig_k[b][a][j][i], simt_lane);
            putlane(piv [s]()(a, b)(i, j), pi_k [b][a][j][i], simt_lane);
            putlane(dv  [s]()(a, b)(i, j), d_k  [b][a][j][i], simt_lane);
            putlane(nv  [s]()(a, b)(i, j), n_k  [b][a][j][i], simt_lane);
          }
    putlane(sv[s]()()(), s_out, simt_lane);
    putlane(pv[s]()()(), p_out, simt_lane);

    // ---- clover_sigma (untransposed) ----
    for (int p_idx = 0; p_idx < 6; ++p_idx) {
      const ComplexD *smn_p = &smf[p_idx * Ns * Ns];
      for (int ic = 0; ic < Nc; ++ic)
        for (int jc = 0; jc < Nc; ++jc) {
          ComplexD val(0.0, 0.0);
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD smn = smn_p[alpha * Ns + beta];
                if (smn.real() == 0.0 && smn.imag() == 0.0) continue;
                int row_u = Idx24Dev(a, beta,  jc);
                int col_u = Idx24Dev(a, alpha, ic);
                int row_l = Idx24Dev(a, beta,  ic);
                int col_l = Idx24Dev(a, alpha, jc);
                val = val + smn * Inv(row_u, col_u);
                val = val - smn * Inv(kDim24 + row_l, kDim24 + col_l);
              }
          ComplexD cs_val = ComplexD(-0.5 * csw_l, 0.0) * DtxqcdConj(val);
          switch (p_idx) {
            case 0: putlane(cs0[s]()()(ic, jc), cs_val, simt_lane); break;
            case 1: putlane(cs1[s]()()(ic, jc), cs_val, simt_lane); break;
            case 2: putlane(cs2[s]()()(ic, jc), cs_val, simt_lane); break;
            case 3: putlane(cs3[s]()()(ic, jc), cs_val, simt_lane); break;
            case 4: putlane(cs4[s]()()(ic, jc), cs_val, simt_lane); break;
            case 5: putlane(cs5[s]()()(ic, jc), cs_val, simt_lane); break;
          }
        }
    }
  });

  for (int a = 0; a < DtxqcdNf; ++a) {
    Xup[a].ViewClose();
    Xlo[a].ViewClose();
    Yup[a].ViewClose();
    Ylo[a].ViewClose();
  }
}

// ----------------------------------------------------------------------
// All-sites variant for the NON-EO full action
// (DTXQCDWilsonCloverRationalFullAction::AccumulateSiteForcesAll).  Same
// per-site body as Extract() above -- it is byte-for-byte the device
// transcription of DtxqcdSiteForceKernel::{AuxForceAt,CloverSigmaAt} plus
// the Wirtinger transpose -- but X, Y and the F_* outputs all live on the
// FULL Cartesian grid (no checkerboard).  We delegate to Extract() with the
// caller's full-grid lattices: Extract() reads oSites/grid from
// X.upper.f[0].Grid() (which is the full grid here), iterates ALL local
// sites, and the F_*.Checkerboard() tags it sets are inert on a full grid.
// Keeping a single device-kernel source guarantees the all-sites path stays
// bit-comparable to the CB path's body forever.
inline void ExtractAll(const DTXQCDFermionDoubled &X,
                       const DTXQCDFermionDoubled &Y,
                       RealD csw,
                       const DtxqcdSpinMatrices &spin,
                       LatticeDtxqcdSigma &F_sig,
                       LatticeDtxqcdPi    &F_pi,
                       LatticeDtxqcdD     &F_d,
                       LatticeDtxqcdN     &F_n,
                       LatticeDtxqcdS     &F_s,
                       LatticeDtxqcdP     &F_p,
                       std::array<LatticeColourMatrix, 6> &F_cs) {
  // cb is irrelevant on a full grid; pass Even (the Checkerboard() tag it
  // sets on the full-grid outputs is never consulted).
  Extract(X, Y, Even, csw, spin, F_sig, F_pi, F_d, F_n, F_s, F_p, F_cs);
}

}  // namespace DtxqcdRatForceGpu

NAMESPACE_END(Grid);
