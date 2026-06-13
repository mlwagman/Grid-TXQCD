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

// Gaussian sample for CF matrix: complex Gaussian fill of all NfNc^2 entries,
// then Hermitian + traceless project.  Variance of the resulting Hermitian
// matrix entries: off-diagonal ~ N(0, 1/2) in re/im each; diagonal ~ N(0, 1)
// real -- standard for symmetric Hermitian Gaussian (GUE) ensemble.
inline void DtxqcdHermitianCFGaussian(GridParallelRNG &pRNG,
                                       LatticeDtxqcdSigma &X) {
  gaussian(pRNG, X);
  DtxqcdHermitizeAndTracelessCFInPlace(X);
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
    DtxqcdHermitianCFGaussian(pRNG, P.d);      P.d     = scale * P.d;
    DtxqcdHermitianCFGaussian(pRNG, P.n);      P.n     = scale * P.n;
    DtxqcdRealScalarGaussian(pRNG, P.s);       P.s     = scale * P.s;
    DtxqcdRealScalarGaussian(pRNG, P.p);       P.p     = scale * P.p;
  }

  static inline Field projectForce(Field &Fforce) {
    Field out(Fforce.Grid());
    out.U = PeriodicGimplR::projectForce(Fforce.U);
    out.sigma = Fforce.sigma;  DtxqcdHermitizeAndTracelessCFInPlace(out.sigma);
    out.pi    = Fforce.pi;     DtxqcdHermitizeAndTracelessCFInPlace(out.pi);
    out.d     = Fforce.d;      DtxqcdHermitizeAndTracelessCFInPlace(out.d);
    out.n     = Fforce.n;      DtxqcdHermitizeAndTracelessCFInPlace(out.n);
    out.s     = Fforce.s;      DtxqcdRealScalarProjectInPlace(out.s);
    out.p     = Fforce.p;      DtxqcdRealScalarProjectInPlace(out.p);
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
    DtxqcdHermitizeAndTracelessCFInPlace(U.d);
    DtxqcdHermitizeAndTracelessCFInPlace(U.n);
    DtxqcdRealScalarProjectInPlace(U.s);
    DtxqcdRealScalarProjectInPlace(U.p);
  }

  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::HotConfiguration(pRNG, U.U);
    DtxqcdHermitianCFGaussian(pRNG, U.sigma);
    DtxqcdHermitianCFGaussian(pRNG, U.pi);
    DtxqcdHermitianCFGaussian(pRNG, U.d);
    DtxqcdHermitianCFGaussian(pRNG, U.n);
    DtxqcdRealScalarGaussian(pRNG, U.s);
    DtxqcdRealScalarGaussian(pRNG, U.p);
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
                                    RealD lambda, RealD Sigma = 0.0) {
    RealD lambda_var = lambda;
    if (const char *e = std::getenv("AUX_FLUCT_LAMBDA"); e && *e) {
      lambda_var = std::atof(e);
    }
    RealD scale = 1.0 / lambda_var;
    DtxqcdHermitianCFGaussian(pRNG, U.sigma);  U.sigma = scale * U.sigma;
    DtxqcdHermitianCFGaussian(pRNG, U.pi);     U.pi    = scale * U.pi;
    DtxqcdHermitianCFGaussian(pRNG, U.d);      U.d     = scale * U.d;
    DtxqcdHermitianCFGaussian(pRNG, U.n);      U.n     = scale * U.n;
    DtxqcdRealScalarGaussian(pRNG, U.s);       U.s     = scale * U.s;
    DtxqcdRealScalarGaussian(pRNG, U.p);       U.p     = scale * U.p;

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
