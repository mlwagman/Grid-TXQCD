#pragma once
// DTXQCD v2 aux-field wall-wall correlators.
//
// Used by:
//   - Test_dtxqcd_meas_aux.cc        (standalone per-cfg measurement)
//   - Test_dtxqcd_2pt_gencfgs.cc     (per-traj HMC diagnostic via
//                                     DtxqcdDiagnostics::record_aux extension)
//
// Mirrors TXQCD production/aux_correlator.h but adapted to v2's CF-Hermitian
// roster (sigma, pi, d, n are 6×6 colour×flavour Hermitian; s, p are singlets).
//
// Operator decomposition (Nf=2 u,d):
//   Color-trace sigma → 2×2 flavour matrix Pi_ab = Σ_i sigma^{ii}_{ab}.
//   Isovector triplet:
//     pi^+  ∼ Pi_{12} (u̅γ₅d analogue, complex)
//     pi^-  ∼ Pi_{21} = (Pi_{12})*  (same channel under H.c.)
//     pi^0  ∼ (Pi_{11} − Pi_{22}) / √2  (real)
//   Same decomposition for sigma field → a0+, a0-, a00.
//   Isoscalar (4 channels):
//     s, p, Tr(sigma) ≡ Σ_{a,i} sigma^{ii}_{aa}, Tr(pi)
//
// All slice helpers do a Grid sliceSum (collective on the t-direction), so
// MUST be called on every rank.  Post-gather helpers (CorrelatorFromSlice
// etc.) are CPU loops on the gathered slice; safe on any rank.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>

NAMESPACE_BEGIN(Grid);

// ---- Color-traced flavour-matrix lattice type ------------------------------
// 2×2 (Nf×Nf) complex matrix at each site, obtained as Π_ab(x) = Σ_i F^{ii}_ab(x).
template <class vtype>
using DtxqcdSiteFlavorMatrix =
    iScalar<iMatrix<iScalar<vtype>, DtxqcdNf>>;
template <class vtype>
using DtxqcdLatticeFlavorMatrix = Lattice<DtxqcdSiteFlavorMatrix<vtype>>;
typedef DtxqcdLatticeFlavorMatrix<vComplex> LatticeDtxqcdFlavorMat;

// ---- Color trace (CF Hermitian → flavour matrix per site) ------------------
// Π_ab(x) = Σ_i F^{ii}_ab(x).  Manual per-site loop on oSites; SIMD-safe via
// autoView/thread_for.
inline LatticeDtxqcdFlavorMat
DtxqcdColorTrace(const LatticeDtxqcdSigma &F) {
  LatticeDtxqcdFlavorMat out(F.Grid());
  autoView(Fv, F,   CpuRead);
  autoView(ov, out, CpuWrite);
  thread_for(ss, F.Grid()->oSites(), {
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int b = 0; b < DtxqcdNf; ++b) {
        auto acc = Fv[ss]()(a, b)(0, 0);
        for (int i = 1; i < Nc; ++i) acc = acc + Fv[ss]()(a, b)(i, i);
        ov[ss]()(a, b)() = acc;
      }
    }
  });
  return out;
}

// ---- Slice helpers (collective) --------------------------------------------

// Σ_x Π_ab(x) per (a,b,t).  Output [Nf×Nf][T] row-major a*Nf+b.
inline std::vector<std::vector<ComplexD>>
DtxqcdSliceSumFlavorAll(const LatticeDtxqcdFlavorMat &P) {
  GridBase *g = P.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  std::vector<std::vector<ComplexD>> out(DtxqcdNf * DtxqcdNf,
                                         std::vector<ComplexD>(T));
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      LatticeComplex Pab(g);
      autoView(Pv, P, CpuRead);
      autoView(pabv, Pab, CpuWrite);
      thread_for(ss, g->oSites(), {
        pabv[ss]()()() = Pv[ss]()(a, b)();
      });
      std::vector<TComplex> sl;
      sliceSum(Pab, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[a * DtxqcdNf + b][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

// Σ_x scalar(x) per t.  For LatticeDtxqcdS / LatticeDtxqcdP.
inline std::vector<ComplexD>
DtxqcdSliceSumScalar(const LatticeDtxqcdS &S) {
  GridBase *g = S.Grid();
  LatticeComplex sc(g);
  autoView(Sv, S, CpuRead);
  autoView(scv, sc, CpuWrite);
  thread_for(ss, g->oSites(), {
    scv[ss]()()() = Sv[ss]()()();
  });
  std::vector<TComplex> sl;
  sliceSum(sc, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// ---- Post-gather helpers (CPU loops) ---------------------------------------

// Translation-averaged real-part correlator:
//   C(τ) = (1/V_3 T) Σ_t Re[ A(t+τ) · B*(t) ]
// (Period T, divide by V_3 to match standard wall-wall normalization.)
inline std::vector<ComplexD>
DtxqcdCorrelatorFromSlice(const std::vector<ComplexD> &sA,
                          const std::vector<ComplexD> &sB,
                          RealD V4) {
  int T = (int)sA.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += sA[(t0 + dt) % T] * conjugate(sB[t0]);
  for (auto &c : C) c /= V4;
  return C;
}

inline std::vector<ComplexD>
DtxqcdCorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  return DtxqcdCorrelatorFromSlice(s, s, V4);
}

inline void DtxqcdScaleVec(std::vector<ComplexD> &v, RealD c) {
  for (auto &x : v) x *= c;
}

inline std::vector<ComplexD>
DtxqcdAddVec(const std::vector<ComplexD> &a, const std::vector<ComplexD> &b,
              RealD ca = 1.0, RealD cb = 1.0) {
  std::vector<ComplexD> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = ca * a[i] + cb * b[i];
  return r;
}

// ---- Bundled output --------------------------------------------------------
//
// All "C_*" correlators are λ⁴-scaled raw (translation-averaged) C(τ).  Analysis
// subtracts disconnected ⟨W⟩⟨W*⟩ pieces using the ensemble mean of the
// wall_* per-cfg slices.  Wall slices are NOT λ-scaled (raw aux field values).
//
// Channel inventory (T = lattice time extent):
//
//   Isovector pion triplet (∼ pi+, pi-, pi0 mesons, length T each):
//     C_pi_plus      = ⟨Pi_{12}(t) Pi_{12}*(0)⟩    [pi+ pi- correlator]
//     C_pi_minus     = ⟨Pi_{21}(t) Pi_{21}*(0)⟩    [redundant by H.c.; kept for cross-check]
//     C_pi_zero      = ⟨((Pi_{11}-Pi_{22})/√2)² (t)⟩   [neutral pion]
//   Isovector a0 triplet (∼ a0+, a0-, a00 from sigma color-trace):
//     C_a0_plus      = ⟨Sig_{12}(t) Sig_{12}*(0)⟩
//     C_a0_minus     = ⟨Sig_{21}(t) Sig_{21}*(0)⟩
//     C_a0_zero      = ⟨((Sig_{11}-Sig_{22})/√2)² (t)⟩
//   Isoscalar (4 channels):
//     C_s, C_p       = ⟨s(t)s(0)⟩, ⟨p(t)p(0)⟩
//     C_trsig        = ⟨Tr σ(t) Tr σ(0)⟩  with Tr σ = Σ_{a,i} σ^{ii}_{aa}
//     C_trpi         = ⟨Tr π(t) Tr π(0)⟩
//   Mixed isoscalar correlators (for SD identity check):
//     C_trsig_s      = ⟨Tr σ(t) s(0)⟩
//     C_trpi_p       = ⟨Tr π(t) p(0)⟩
//
// SD identity (Gaussian, single-isoscalar dominance at large t):
//     C_trsig : C_trsig_s : C_s ≈ 36 : 6 : 1   (scalar singlet)
//     C_trpi  : C_trpi_p  : C_p ≈ 36 : 6 : 1   (pseudoscalar singlet)
//
// Per-cfg wall slices saved for vac-sub in ensemble analysis (length T each):
//     wall_sig_ab_flat, wall_pi_ab_flat   [Nf² × T row-major; full 2×2 each]
//     wall_s, wall_p
//     wall_trsig, wall_trpi               [convenience; redundant with ab_flat]
struct DtxqcdAuxWallCorrelators {
  int T;
  int Nf2;

  // λ⁴-scaled correlators
  std::vector<ComplexD> C_pi_plus, C_pi_minus, C_pi_zero;
  std::vector<ComplexD> C_a0_plus, C_a0_minus, C_a0_zero;
  std::vector<ComplexD> C_s, C_p;
  std::vector<ComplexD> C_trsig, C_trpi;
  std::vector<ComplexD> C_trsig_s, C_trpi_p;

  // Per-cfg wall slices (raw, NOT λ-scaled)
  std::vector<ComplexD> wall_sig_ab_flat;   // Nf² × T
  std::vector<ComplexD> wall_pi_ab_flat;    // Nf² × T
  std::vector<ComplexD> wall_s, wall_p;
  std::vector<ComplexD> wall_trsig, wall_trpi;
};

// Compute all aux wall-wall correlators for one config.  MUST be called on
// every rank (slice helpers do collectives).
inline DtxqcdAuxWallCorrelators
DtxqcdComputeAuxWallCorrelators(const DTXQCDField &U, RealD lambda) {
  GridBase *g = U.sigma.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= (RealD)g->GlobalDimensions()[mu];
  const RealD lam4 = lambda * lambda * lambda * lambda;
  const int Nf2 = DtxqcdNf * DtxqcdNf;

  DtxqcdAuxWallCorrelators r;
  r.T = T;
  r.Nf2 = Nf2;

  // Color-trace sigma and pi → 2×2 flavour matrices, then sliceSum each
  // component.  Output indexed [a*Nf+b][t].
  LatticeDtxqcdFlavorMat SigCT = DtxqcdColorTrace(U.sigma);
  LatticeDtxqcdFlavorMat PiCT  = DtxqcdColorTrace(U.pi);
  auto sig_ab = DtxqcdSliceSumFlavorAll(SigCT);
  auto pi_ab  = DtxqcdSliceSumFlavorAll(PiCT);

  // Singlet scalar slices.
  auto s_slice = DtxqcdSliceSumScalar(U.s);
  auto p_slice = DtxqcdSliceSumScalar(U.p);

  // ---- Isovector pi triplet ----
  // pi+:  Pi_{12} <-> a=0, b=1
  // pi-:  Pi_{21} <-> a=1, b=0
  // pi0:  (Pi_{11} - Pi_{22}) / √2
  const int i_12 = 0 * DtxqcdNf + 1;
  const int i_21 = 1 * DtxqcdNf + 0;
  const int i_11 = 0 * DtxqcdNf + 0;
  const int i_22 = 1 * DtxqcdNf + 1;
  auto pi0_slice = DtxqcdAddVec(pi_ab[i_11], pi_ab[i_22], 1.0 / std::sqrt(2.0),
                                 -1.0 / std::sqrt(2.0));
  r.C_pi_plus  = DtxqcdCorrelatorFromSlice(pi_ab[i_12], V4);
  r.C_pi_minus = DtxqcdCorrelatorFromSlice(pi_ab[i_21], V4);
  r.C_pi_zero  = DtxqcdCorrelatorFromSlice(pi0_slice,  V4);
  DtxqcdScaleVec(r.C_pi_plus,  lam4);
  DtxqcdScaleVec(r.C_pi_minus, lam4);
  DtxqcdScaleVec(r.C_pi_zero,  lam4);

  // ---- Isovector a0 triplet (sigma color-trace, same decomposition) ----
  auto a00_slice = DtxqcdAddVec(sig_ab[i_11], sig_ab[i_22], 1.0 / std::sqrt(2.0),
                                 -1.0 / std::sqrt(2.0));
  r.C_a0_plus  = DtxqcdCorrelatorFromSlice(sig_ab[i_12], V4);
  r.C_a0_minus = DtxqcdCorrelatorFromSlice(sig_ab[i_21], V4);
  r.C_a0_zero  = DtxqcdCorrelatorFromSlice(a00_slice,    V4);
  DtxqcdScaleVec(r.C_a0_plus,  lam4);
  DtxqcdScaleVec(r.C_a0_minus, lam4);
  DtxqcdScaleVec(r.C_a0_zero,  lam4);

  // ---- Isoscalar: singlet s, p ----
  r.C_s = DtxqcdCorrelatorFromSlice(s_slice, V4); DtxqcdScaleVec(r.C_s, lam4);
  r.C_p = DtxqcdCorrelatorFromSlice(p_slice, V4); DtxqcdScaleVec(r.C_p, lam4);

  // ---- Isoscalar: flavour-trace of color-trace of sigma, pi ----
  // Tr σ = Σ_{a,i} σ^{ii}_{aa} = Pi_{11} + Pi_{22}
  auto trsig_slice = DtxqcdAddVec(sig_ab[i_11], sig_ab[i_22], 1.0, 1.0);
  auto trpi_slice  = DtxqcdAddVec(pi_ab [i_11], pi_ab [i_22], 1.0, 1.0);
  r.C_trsig = DtxqcdCorrelatorFromSlice(trsig_slice, V4);
  r.C_trpi  = DtxqcdCorrelatorFromSlice(trpi_slice,  V4);
  DtxqcdScaleVec(r.C_trsig, lam4);
  DtxqcdScaleVec(r.C_trpi,  lam4);

  // ---- Mixed isoscalar correlators: ⟨Tr σ · s⟩, ⟨Tr π · p⟩ ----
  r.C_trsig_s = DtxqcdCorrelatorFromSlice(trsig_slice, s_slice, V4);
  r.C_trpi_p  = DtxqcdCorrelatorFromSlice(trpi_slice,  p_slice, V4);
  DtxqcdScaleVec(r.C_trsig_s, lam4);
  DtxqcdScaleVec(r.C_trpi_p,  lam4);

  // ---- Wall slices (raw, not λ-scaled) ----
  r.wall_sig_ab_flat.assign(Nf2 * T, ComplexD(0.0));
  r.wall_pi_ab_flat .assign(Nf2 * T, ComplexD(0.0));
  for (int ab = 0; ab < Nf2; ++ab)
    for (int t = 0; t < T; ++t) {
      r.wall_sig_ab_flat[ab * T + t] = sig_ab[ab][t];
      r.wall_pi_ab_flat [ab * T + t] = pi_ab [ab][t];
    }
  r.wall_s     = std::move(s_slice);
  r.wall_p     = std::move(p_slice);
  r.wall_trsig = std::move(trsig_slice);
  r.wall_trpi  = std::move(trpi_slice);

  return r;
}

NAMESPACE_END(Grid);
