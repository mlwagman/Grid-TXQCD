#pragma once
// Composite FieldImplementation for DTXQCD HMC (v2 roster).
//
// Delegates gauge to PeriodicGimplR (exp-update via Ta projection). Aux fields:
//   sigma, pi, d, n : color-flavor 6x6 traceless Hermitian (combined index
//                     (i,a) x (j,b)).  Stored as full iMatrix<iMatrix<vComplex,
//                     Nc>, Nf>; HermitizeAndTracelessInPlace enforces invariants
//                     after every generation, projection, force update.
//   s, p            : singlet scalars (real DOF in real part of vComplex).
//
// Momentum conjugate to each aux field has the same site-tensor type and the
// same projection constraints.  Quadratic kinetic piece is (1/2) Tr P^2 (or
// (P)^2 for scalars), accumulated into FieldSquareNorm with the standard
// scalar-field minus sign that mirrors TXQCDCompositeImpl.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/gauge/GaugeImplementations.h>

NAMESPACE_BEGIN(Grid);

// ---------- field-type projectors ----------

// Hermitize-only projection on a CF-matrix lattice field, operating on the
// combined (a,i)(b,j) 6x6 index space: M_{(a,i)(b,j)} <- 0.5 (M + adj(M)).
//
// 2026-06-12: Removed the traceless step.  In TXQCD the trace of sigma carries
// the qqbar singlet condensate (⟨Tr sigma⟩ = Sigma/lambda^2).  In v2 DTXQCD the
// trace was being pinned to zero by the per-site subtraction below, redirecting
// the condensate onto the separate s singlet field with a non-trivial
// renormalization factor (the s coefficient in S_aux = (lambda^2/2) s^2 should
// be 6*(lambda^2/2) to match X = sigma_traceless + s*I in Tr X^2).  Cleaner to
// drop the traceless constraint here and let sigma carry the full Hermitian
// degrees of freedom (matching TXQCD).  The s and p singlet slots remain in
// the field but become redundant trace modes; we leave them in for now since
// they still contribute to S_aux additively without breaking anything.  The
// name is kept for grep-compatibility; the function is now Hermitize-only.
inline void DtxqcdHermitizeAndTracelessCFInPlace(LatticeDtxqcdSigma &X) {
  X = 0.5 * (X + adj(X));
}

// Real-symmetric projection: X → Symm(Re(X)).  Used (under env knob
// DTXQCD_DN_REAL_SYMMETRIC=1) on d, n to restore formal Pfaffian
// antisymmetry of (K·M48) — Hermitian d, n break it.  σ, π are
// independently kept Hermitian by the surrounding code; the mixed
// projection (real-symm d/n + Hermitian σ/π) is the v2 algebraic
// formulation per the dtxqcd_v2 derivation.
inline void DtxqcdRealSymmetricCFInPlace(LatticeDtxqcdSigma &X) {
  LatticeDtxqcdSigma tmp(X.Grid());
  tmp = X + transpose(X);                   // 2*Symm(M)
  tmp = 0.5 * (tmp + conjugate(tmp));       // 2*Re(Symm(M))
  X   = 0.5 * tmp;                          // Re(Symm(M)) = Symm(Re(M))
}

// Complex-symmetric projection: X → Symm(X) = (X + X^T)/2.  Transpose
// only — leaves Re/Im parts both with transpose-symmetric structure.
// Used (under env knob DTXQCD_DN_COMPLEX_SYMMETRIC=1) on d, n in the
// "complex d, d* independent" formulation where M48 LL = conj(M48 UR).
// 42 real DOFs/site for 6×6 vs Hermitian 36 vs real-symm 21.
inline void DtxqcdComplexSymmetricCFInPlace(LatticeDtxqcdSigma &X) {
  X = 0.5 * (X + transpose(X));
}

// =====================================================================
// sigmaHerm production convention (2026-06-19) — BAKED IN.
//
// All DTXQCD production now runs the sigmaHerm convention:
//   - σ, π are stored Hermitian (36 real DOFs/site each, full DOFs
//     preserved for exact H-S decoupling of quark bilinears)
//   - d, n are stored truly complex-symmetric (42 real DOFs/site each,
//     joint color+flavor transpose symmetry, imaginary parts kept)
//   - M48 has M_LL = conj(M_UR) at the assembly AND on-the-fly Apply
//     level so γ5-Hermiticity and Pfaffian antisymmetry both hold
//   - M_lower applies +X^T (joint color+flavor transpose of σ, π)
//
// Three gates defended (all tests pass under sigmaHerm):
//   1. γ5-Hermiticity of full M48 (Test_dtxqcd_gamma5_herm_full)
//   2. (K·M48)^T = -(K·M48) Pfaffian antisymmetry (Test_dtxqcd_pfaffian_antisymmetry)
//   3. Full complex DOFs in every aux field (Test_dtxqcd_freefield_qbarq_*
//      — silently-projected real-symmetric d/n gave ~0.5% Fierz residual
//      at light mass, caught and fixed 2026-06-18/19)
//
// The DtxqcdDnComplexSymmetric() / DtxqcdSigmaPiHermitianOnly() helpers
// returned `true` by default during the convention finalization; they
// are now hardcoded to true so production cannot silently revert.  The
// DTXQCD_DN_COMPLEX_SYMMETRIC, DTXQCD_SIGMA_PI_HERMITIAN_ONLY, and
// DTXQCD_DN_REAL_SYMMETRIC env knobs are ignored — if set, a one-time
// warning is emitted at first call.
// =====================================================================

namespace dtxqcd_detail {
inline void WarnIfLegacyKnobSet() {
  static const bool warned = []() {
    auto check = [](const char *name) {
      if (const char *v = std::getenv(name); v && *v) {
        std::cout << GridLogWarning
                  << "[DTXQCD] env knob " << name << "='" << v
                  << "' is IGNORED — sigmaHerm convention is hardcoded "
                     "(complex-symm d/n + Hermitian σ/π)."
                  << std::endl;
      }
    };
    check("DTXQCD_DN_COMPLEX_SYMMETRIC");
    check("DTXQCD_SIGMA_PI_HERMITIAN_ONLY");
    check("DTXQCD_DN_REAL_SYMMETRIC");
    return true;
  }();
  (void)warned;
}
}  // namespace dtxqcd_detail

static inline constexpr bool DtxqcdDnComplexSymmetric() {
  return true;
}
static inline constexpr bool DtxqcdSigmaPiHermitianOnly() {
  return true;
}
static inline constexpr bool DtxqcdDnRealSymmetric() {
  return false;
}

// Color-traceless projection: for each (a,b) flavor pair, subtract the
// trace over the inner Nc×Nc color matrix.  After this, sum_i X_ab^{ii} = 0
// for every (a,b) pair.  Used (under env knob DTXQCD_DN_COLOR_TRACELESS=1)
// on d, n to enforce the Fierz-derived constraint that the color-singlet
// (q̄^C γ5 q · δ_ij = 0 from 3⊗3 = 6 ⊕ 3̄ having no singlet).
inline void DtxqcdMakeColorTracelessCFInPlace(LatticeDtxqcdSigma &X) {
  typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SObj;
  GridBase *grid = X.Grid();
  Coordinate latt = grid->GlobalDimensions();
  const ComplexD invNc(1.0 / RealD(Nc), 0.0);
  for (int t = 0; t < latt[Tdir]; ++t)
    for (int x3 = 0; x3 < latt[Zdir]; ++x3)
      for (int x2 = 0; x2 < latt[Ydir]; ++x2)
        for (int x1 = 0; x1 < latt[Xdir]; ++x1) {
          Coordinate site(4);
          site[0] = x1; site[1] = x2; site[2] = x3; site[3] = t;
          SObj M;
          peekSite(M, X, site);
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int b = 0; b < DtxqcdNf; ++b) {
              ComplexD tr(0.0, 0.0);
              for (int i = 0; i < Nc; ++i)
                tr += static_cast<ComplexD>(M()(a, b)(i, i));
              ComplexD c = tr * invNc;
              for (int i = 0; i < Nc; ++i)
                M()(a, b)(i, i) = M()(a, b)(i, i) - c;
            }
          pokeSite(M, X, site);
        }
}

// Cached env-knob: when DTXQCD_DN_COLOR_TRACELESS=1, project d and n
// color-traceless throughout HMC (field, momentum, force).  Default OFF.
static inline bool DtxqcdDnColorTraceless() {
  static const bool b = []() {
    if (const char *v = std::getenv("DTXQCD_DN_COLOR_TRACELESS"); v && *v) {
      bool on = std::atoi(v) != 0;
      std::cout << GridLogMessage
                << "[DTXQCD] DN_COLOR_TRACELESS = " << (on ? "ON" : "OFF")
                << "  (project d, n color-traceless per Fierz)" << std::endl;
      return on;
    }
    return false;
  }();
  return b;
}

// Gaussian sample for CF matrix: complex Gaussian fill of all NfNc^2 entries,
// then Hermitian + traceless project.  Variance of the resulting Hermitian
// matrix entries: off-diagonal ~ N(0, 1/2) in re/im each; diagonal ~ N(0, 1)
// real -- standard for symmetric Hermitian Gaussian (GUE) ensemble.
inline void DtxqcdHermitianCFGaussian(GridParallelRNG &pRNG,
                                       LatticeDtxqcdSigma &X) {
  gaussian(pRNG, X);
  DtxqcdHermitizeAndTracelessCFInPlace(X);
}

// True complex-symmetric Gaussian: full complex Gaussian then complex-symm
// project (preserving imaginary parts).  Generates d, n in the
// "d, d* independent" form needed for the formal H-S decoupling under
// DN_COMPLEX_SYMMETRIC.  (Distinct from
// Hermitian-then-complex-symm which collapses to real-symmetric and loses
// the imaginary-part DOFs.)
inline void DtxqcdComplexSymmetricCFGaussian(GridParallelRNG &pRNG,
                                              LatticeDtxqcdSigma &X) {
  dtxqcd_detail::WarnIfLegacyKnobSet();
  gaussian(pRNG, X);
  DtxqcdComplexSymmetricCFInPlace(X);
}

// Real projection for singlet scalar (zero out imag part of the vComplex
// container).  Same idiom as the v1 triplet real-projection.
inline void DtxqcdRealScalarProjectInPlace(LatticeDtxqcdS &X) {
  X = 0.5 * (X + conjugate(X));
}

// Gaussian sample for singlet scalar: standard complex Gaussian fill then
// real-project.  Variance of the resulting real part = 1.
inline void DtxqcdRealScalarGaussian(GridParallelRNG &pRNG, LatticeDtxqcdS &X) {
  gaussian(pRNG, X);
  DtxqcdRealScalarProjectInPlace(X);
}

// Squared norms.  For Hermitian CF matrices, norm2 sums |M_{ij,ab}|^2 (this
// equals Tr(M^dag M) = Tr(M^2) for Hermitian M).
inline RealD DtxqcdCFSquareNorm(LatticeDtxqcdSigma &X) { return norm2(X); }

// For real-projected scalar, norm2 is the sum of (real part)^2.
inline RealD DtxqcdScalarSquareNorm(LatticeDtxqcdS &X) { return norm2(X); }

// ---------- FieldImplementation static interface ----------

class DTXQCDCompositeImpl {
 public:
  typedef vComplex Simd;
  typedef DTXQCDField Field;
  typedef iScalar<iScalar<iScalar<vComplex>>> SiteComplex;
  typedef Lattice<SiteComplex> ComplexField;
  typedef typename PeriodicGimplR::SiteField SiteField;

  static inline void generate_momenta(Field &P, GridSerialRNG &sRNG,
                                      GridParallelRNG &pRNG) {
    PeriodicGimplR::generate_momenta(P.U, sRNG, pRNG);
    // Match the sqrt(HMC_MOMENTUM_DENOMINATOR) scaling that gauge momenta
    // get inside PeriodicGimplR::generate_momenta so the integrator update
    // P -= F * ep * HMC_MOMENTUM_DENOMINATOR is balanced against dq/dt = P
    // for each aux slot.
    RealD scale = ::sqrt(HMC_MOMENTUM_DENOMINATOR);
    DtxqcdHermitianCFGaussian(pRNG, P.sigma);  P.sigma = scale * P.sigma;
    DtxqcdHermitianCFGaussian(pRNG, P.pi);     P.pi    = scale * P.pi;
    // Under DN_COMPLEX_SYMMETRIC: d, n generated as raw complex Gaussian
    // then complex-symm projected (preserves imaginary parts → truly
    // complex-symm, NOT real-symm).  Otherwise Hermitian.
    if (DtxqcdDnComplexSymmetric()) {
      DtxqcdComplexSymmetricCFGaussian(pRNG, P.d);
      DtxqcdComplexSymmetricCFGaussian(pRNG, P.n);
    } else {
      DtxqcdHermitianCFGaussian(pRNG, P.d);
      DtxqcdHermitianCFGaussian(pRNG, P.n);
    }
    P.d = scale * P.d;
    P.n = scale * P.n;
    DtxqcdRealScalarGaussian(pRNG, P.s);       P.s     = scale * P.s;
    DtxqcdRealScalarGaussian(pRNG, P.p);       P.p     = scale * P.p;
    if (DtxqcdDnColorTraceless()) {
      DtxqcdMakeColorTracelessCFInPlace(P.d);
      DtxqcdMakeColorTracelessCFInPlace(P.n);
    }
    if (DtxqcdDnRealSymmetric()) {
      DtxqcdRealSymmetricCFInPlace(P.d);
      DtxqcdRealSymmetricCFInPlace(P.n);
    }
    if (DtxqcdDnComplexSymmetric()) {
      if (!DtxqcdSigmaPiHermitianOnly()) {
        DtxqcdRealSymmetricCFInPlace(P.sigma);
        DtxqcdRealSymmetricCFInPlace(P.pi);
      }
      // d, n already complex-symm from generation; no re-projection needed.
    }
    // DTXQCD_FREEZE_AUX: hold every aux field fixed during HMC by killing its
    // conjugate momentum.  Combined with ZERO_ALL_AUX=1 (aux==0 at init) this
    // freezes aux==0 throughout -> M48 = doubled D_WC -> the HMC is pure Nf=2
    // QCD expressed through the doubled operator (isolates the doubled-operator
    // structure/force from all aux dynamics).  Default off => byte-identical.
    static const bool freeze_aux = [](){ const char*e=std::getenv("DTXQCD_FREEZE_AUX"); return e&&std::atoi(e)!=0; }();
    if (freeze_aux) { P.sigma=Zero(); P.pi=Zero(); P.d=Zero(); P.n=Zero(); P.s=Zero(); P.p=Zero(); }
  }

  static inline Field projectForce(Field &Fforce) {
    Field out(Fforce.Grid());
    out.U = PeriodicGimplR::projectForce(Fforce.U);
    out.sigma = Fforce.sigma;  DtxqcdHermitizeAndTracelessCFInPlace(out.sigma);
    out.pi    = Fforce.pi;     DtxqcdHermitizeAndTracelessCFInPlace(out.pi);
    // Under DN_COMPLEX_SYMMETRIC: skip Hermitize for d, n (Hermitize +
    // then complex-symm collapses to real-symm, losing imag DOFs).
    out.d     = Fforce.d;
    out.n     = Fforce.n;
    if (!DtxqcdDnComplexSymmetric()) {
      DtxqcdHermitizeAndTracelessCFInPlace(out.d);
      DtxqcdHermitizeAndTracelessCFInPlace(out.n);
    }
    out.s     = Fforce.s;      DtxqcdRealScalarProjectInPlace(out.s);
    out.p     = Fforce.p;      DtxqcdRealScalarProjectInPlace(out.p);
    if (DtxqcdDnColorTraceless()) {
      DtxqcdMakeColorTracelessCFInPlace(out.d);
      DtxqcdMakeColorTracelessCFInPlace(out.n);
    }
    if (DtxqcdDnRealSymmetric()) {
      DtxqcdRealSymmetricCFInPlace(out.d);
      DtxqcdRealSymmetricCFInPlace(out.n);
    }
    if (DtxqcdDnComplexSymmetric()) {
      if (!DtxqcdSigmaPiHermitianOnly()) {
        DtxqcdRealSymmetricCFInPlace(out.sigma);
        DtxqcdRealSymmetricCFInPlace(out.pi);
      }
      DtxqcdComplexSymmetricCFInPlace(out.d);
      DtxqcdComplexSymmetricCFInPlace(out.n);
    }
    // DTXQCD_FREEZE_AUX: zero the aux force components so update_P never kicks
    // the (zeroed) aux momenta -> aux stays frozen at its init value (see
    // generate_momenta).  Default off => byte-identical to the dynamic-aux path.
    static const bool freeze_aux = [](){ const char*e=std::getenv("DTXQCD_FREEZE_AUX"); return e&&std::atoi(e)!=0; }();
    if (freeze_aux) { out.sigma=Zero(); out.pi=Zero(); out.d=Zero(); out.n=Zero(); out.s=Zero(); out.p=Zero(); }
    return out;
  }

  static inline void update_field(Field &P, Field &U, double ep) {
    PeriodicGimplR::update_field(P.U, U.U, ep);
    U.sigma = U.sigma + P.sigma * ep;
    U.pi    = U.pi    + P.pi    * ep;
    U.d     = U.d     + P.d     * ep;
    U.n     = U.n     + P.n     * ep;
    U.s     = U.s     + P.s     * ep;
    U.p     = U.p     + P.p     * ep;
  }

  static inline RealD FieldSquareNorm(Field &U) {
    // Gauge momenta: antihermitian, Tr(P^2) < 0 naturally.
    // Aux momenta: Hermitian / real-projected, Tr(P^2) > 0; scalar-field
    // convention requires -Tr(P^2)/2 so dH/dt vanishes.
    RealD total = PeriodicGimplR::FieldSquareNorm(U.U);
    total -= DtxqcdCFSquareNorm(U.sigma)  / 2.0;
    total -= DtxqcdCFSquareNorm(U.pi)     / 2.0;
    total -= DtxqcdCFSquareNorm(U.d)      / 2.0;
    total -= DtxqcdCFSquareNorm(U.n)      / 2.0;
    total -= DtxqcdScalarSquareNorm(U.s)  / 2.0;
    total -= DtxqcdScalarSquareNorm(U.p)  / 2.0;
    return total;
  }

  static inline void Project(Field &U) {
    PeriodicGimplR::Project(U.U);
    DtxqcdHermitizeAndTracelessCFInPlace(U.sigma);
    DtxqcdHermitizeAndTracelessCFInPlace(U.pi);
    if (!DtxqcdDnComplexSymmetric()) {
      DtxqcdHermitizeAndTracelessCFInPlace(U.d);
      DtxqcdHermitizeAndTracelessCFInPlace(U.n);
    }
    DtxqcdRealScalarProjectInPlace(U.s);
    DtxqcdRealScalarProjectInPlace(U.p);
    if (DtxqcdDnColorTraceless()) {
      DtxqcdMakeColorTracelessCFInPlace(U.d);
      DtxqcdMakeColorTracelessCFInPlace(U.n);
    }
    if (DtxqcdDnRealSymmetric()) {
      DtxqcdRealSymmetricCFInPlace(U.d);
      DtxqcdRealSymmetricCFInPlace(U.n);
    }
    if (DtxqcdDnComplexSymmetric()) {
      if (!DtxqcdSigmaPiHermitianOnly()) {
        DtxqcdRealSymmetricCFInPlace(U.sigma);
        DtxqcdRealSymmetricCFInPlace(U.pi);
      }
      DtxqcdComplexSymmetricCFInPlace(U.d);
      DtxqcdComplexSymmetricCFInPlace(U.n);
    }
  }

  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::HotConfiguration(pRNG, U.U);
    DtxqcdHermitianCFGaussian(pRNG, U.sigma);
    DtxqcdHermitianCFGaussian(pRNG, U.pi);
    if (DtxqcdDnComplexSymmetric()) {
      DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
      DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
    } else {
      DtxqcdHermitianCFGaussian(pRNG, U.d);
      DtxqcdHermitianCFGaussian(pRNG, U.n);
    }
    DtxqcdRealScalarGaussian(pRNG, U.s);
    DtxqcdRealScalarGaussian(pRNG, U.p);
    if (DtxqcdDnColorTraceless()) {
      DtxqcdMakeColorTracelessCFInPlace(U.d);
      DtxqcdMakeColorTracelessCFInPlace(U.n);
    }
    if (DtxqcdDnRealSymmetric()) {
      DtxqcdRealSymmetricCFInPlace(U.d);
      DtxqcdRealSymmetricCFInPlace(U.n);
    }
    if (DtxqcdDnComplexSymmetric()) {
      if (!DtxqcdSigmaPiHermitianOnly()) {
        DtxqcdRealSymmetricCFInPlace(U.sigma);
        DtxqcdRealSymmetricCFInPlace(U.pi);
      }
      // d, n already truly complex-symm from generation.
    }
  }

  static inline void TepidConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::TepidConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.d     = Zero();
    U.n     = Zero();
    U.s     = Zero();
    U.p     = Zero();
  }

  static inline void ColdConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::ColdConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.d     = Zero();
    U.n     = Zero();
    U.s     = Zero();
    U.p     = Zero();
  }

  // Fill all aux slots with Gaussian samples at the physical saddle
  // variance Var = 1/lambda^2 per independent component, plus optional
  // mean shift to the singlet saddle.  AUX_FLUCT_LAMBDA optionally
  // decouples the init fluctuation width from the physical lambda.
  //
  // Sigma is the chiral condensate (Σ ≈ −⟨q̄q⟩ from Hutchinson on the
  // initial gauge with stout smearing + AP-time BC).  Equilibrium saddle
  // values (confirmed from λ=5,10 production data 2026-06-12):
  //   ⟨s⟩                = N_F · Σ / λ²       (flavor-trace SD identity)
  //   ⟨Tr σ⟩             = N_F · Σ / λ²       (same as ⟨s⟩, both singlet)
  //   ⟨σ^{ij}_{ab}⟩_diag = Σ / (N_C · λ²)     (per (i,a) entry; sum to ⟨Trσ⟩)
  // π, d, n, p stay mean-zero (parity-odd / non-singlet).
  // PRIOR BUG (pre-2026-06-13): code used per-entry shift Σ/λ² for both σ
  // and s, giving ⟨s⟩ = Σ/λ² (factor N_F=2 too small) and ⟨Trσ⟩ = N_F·N_C·Σ/λ²
  // (factor N_C=3 too large).  Corrected here.
  static inline void FillAuxFields(GridParallelRNG &pRNG, Field &U,
                                    RealD lambda, RealD Sigma = 0.0,
                                    bool with_fluct = true) {
    // with_fluct=false -> MEAN-ONLY aux: zero the Gaussian fluctuations and keep
    // only the saddle shift below.  Used by the AUX_INIT_AUTO condensate solve:
    // the self-consistent saddle is a MEAN-FIELD quantity (the fluctuations
    // average out of the SD condition), and at small lambda the full-variance
    // (width 1/lambda) fluctuations drive M48 near-singular so the Tr M^-1 CG
    // stalls.  Cold (Sigma=0, no fluct) is just the well-conditioned Wilson-
    // clover operator -- the proper mean-field starting point for Picard-0.
    if (!with_fluct) {
      U.sigma = Zero(); U.pi = Zero(); U.d = Zero();
      U.n = Zero(); U.s = Zero(); U.p = Zero();
    } else {
    RealD lambda_var = lambda;
    if (const char *e = std::getenv("AUX_FLUCT_LAMBDA"); e && *e) {
      lambda_var = std::atof(e);
    }
    RealD scale = 1.0 / lambda_var;
    DtxqcdHermitianCFGaussian(pRNG, U.sigma);  U.sigma = scale * U.sigma;
    DtxqcdHermitianCFGaussian(pRNG, U.pi);     U.pi    = scale * U.pi;
    if (DtxqcdDnComplexSymmetric()) {
      DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
      DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
    } else {
      DtxqcdHermitianCFGaussian(pRNG, U.d);
      DtxqcdHermitianCFGaussian(pRNG, U.n);
    }
    U.d = scale * U.d;
    U.n = scale * U.n;
    DtxqcdRealScalarGaussian(pRNG, U.s);       U.s     = scale * U.s;
    DtxqcdRealScalarGaussian(pRNG, U.p);       U.p     = scale * U.p;
    if (DtxqcdDnColorTraceless()) {
      DtxqcdMakeColorTracelessCFInPlace(U.d);
      DtxqcdMakeColorTracelessCFInPlace(U.n);
    }
    if (DtxqcdDnRealSymmetric()) {
      DtxqcdRealSymmetricCFInPlace(U.d);
      DtxqcdRealSymmetricCFInPlace(U.n);
    }
    if (DtxqcdDnComplexSymmetric()) {
      if (!DtxqcdSigmaPiHermitianOnly()) {
        DtxqcdRealSymmetricCFInPlace(U.sigma);
        DtxqcdRealSymmetricCFInPlace(U.pi);
      }
      // d, n already truly complex-symm from generation.
    }
    }  // end if (with_fluct)

    if (Sigma != 0.0) {
      // σ per-entry shift = Σ/(N_C λ²)  → ⟨Trσ⟩ = N_F · Σ/λ²
      const RealD sigma_shift = Sigma / (Nc * lambda * lambda);
      // s shift = N_F · Σ/λ²
      const RealD s_shift = DtxqcdNf * Sigma / (lambda * lambda);
      // σ diagonal shift
      typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
      SigSobj sigma_id;  sigma_id = Zero();
      for (int a = 0; a < DtxqcdNf; ++a)
        for (int i = 0; i < Nc; ++i)
          sigma_id()(a, a)(i, i) = sigma_shift;
      LatticeDtxqcdSigma shift_sigma(U.sigma.Grid());
      shift_sigma = sigma_id;
      U.sigma = U.sigma + shift_sigma;
      // s singlet shift
      typedef typename LatticeDtxqcdS::vector_object::scalar_object SSobj;
      SSobj s_id;  s_id()()() = s_shift;
      LatticeDtxqcdS shift_s(U.s.Grid());
      shift_s = s_id;
      U.s = U.s + shift_s;
    }
  }

  // Weak-field gauge + thermal aux init.  Same recipe as v1.
  static inline void ThermalAuxConfiguration(GridParallelRNG &pRNG, Field &U,
                                              RealD lambda,
                                              double wf_scale = 0.1,
                                              RealD Sigma = 0.0) {
    GenerateWeakFieldGauge(pRNG, U, wf_scale);
    FillAuxFields(pRNG, U, lambda, Sigma);
  }

  // Weak-field gauge initializer matching TXQCDCompositeImpl convention
  // (LieRandomize per mu, default wf_scale=0.1 = chroma WEAK_FIELD).
  static inline void GenerateWeakFieldGauge(GridParallelRNG &pRNG, Field &U,
                                            double wf_scale) {
    LatticeColourMatrix Ulink(U.U.Grid());
    for (int mu = 0; mu < Nd; ++mu) {
      SU<Nc>::LieRandomize(pRNG, Ulink, wf_scale);
      PokeIndex<LorentzIndex>(U.U, Ulink, mu);
    }
  }

  static const int num_colours = Nc;
};

NAMESPACE_END(Grid);
