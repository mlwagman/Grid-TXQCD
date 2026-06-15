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

// Real-symmetric projection on a CF-matrix lattice field, operating on the
// combined (a,i)(b,j) 6x6 index space.  Real-symmetric in the joint (i,a)
// row index swapped with (j,b) column index:
//
//   M_RS = 0.25 * (M + M^T + M* + adj(M))
//        = 0.5 * (Re(M) + Re(M)^T)
//
// Why real-symmetric and not Hermitian?  The C·K-doubled M48 in our
// formulation is required to satisfy (K·M48)^T = -(K·M48) (the Pfaffian
// antisymmetry of arxiv:2209.13183) for the Pf(K·M) = det(M)^{1/2}
// identification.  Empirically (test_pfaffian_antisymmetry.py 2026-06-13):
//   - Complex Hermitian aux X breaks Pfaffian antisymmetry  (ratio ~ 1.03)
//   - Real symmetric aux X preserves Pfaffian antisymmetry (ratio = 0)
//
// Real-symmetric is a strict subset of Hermitian (Hermitian = symmetric +
// real part), so old DTX2 checkpoints with complex Hermitian aux just lose
// their imaginary off-diagonal pieces on read — fine as a hot restart.
//
// 2026-06-12: Removed traceless step (see git log) -- trace of sigma
// carries the singlet condensate, redirecting it to s introduced spurious
// factor of N_F·N_C in the normalization.
//
// 2026-06-15: REDESIGN per s/σ redundancy resolution: σ and π are again
// traceless via DtxqcdMakeTracelessCFInPlace (called separately after the
// real-symmetric step), and the s, p Gaussian coefficient is halved
// (λ²/4 instead of λ²/2) — see DTXQCDAuxGaussianAction and
// project_dtxqcd_s_sigma_redundancy.md.  d, n stay real-symmetric only
// (no analogous redundancy).
//
// Name kept for grep-compatibility; function is real-symmetric projection only.
// For σ, π: call DtxqcdMakeTracelessCFInPlace after this for full projection.
inline void DtxqcdHermitizeAndTracelessCFInPlace(LatticeDtxqcdSigma &X) {
  // Real-symmetric: 0.5*(Re(M) + Re(M)^T) = 0.25*(M + M^T + M* + (M*)^T)
  LatticeDtxqcdSigma tmp(X.Grid());
  tmp = X + transpose(X);                  // 2*Symm(M)
  tmp = 0.5 * (tmp + conjugate(tmp));      // 2*Re(Symm(M))
  X   = 0.5 * tmp;                         // Re(Symm(M)) = Symm(Re(M))
}

// Subtract the singlet trace mode from a CF Hermitian matrix:
//   X → X − (Tr X / (N_F·N_C)) · I
// where the trace sums over both flavor and color, and I is the 6×6
// identity in (a,i)⊗(b,j) space.  After this, Tr_{a,i}(X) ≡ 0, removing
// the redundancy with the singlet scalar s (resp. p).
//
// Per-site iteration via peek/poke — clear and works for any lattice
// shape in serial; performance fine for the 4^3 × 8 test scout.
inline void DtxqcdMakeTracelessCFInPlace(LatticeDtxqcdSigma &X) {
  typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SObj;
  GridBase *grid = X.Grid();
  Coordinate latt = grid->GlobalDimensions();
  const RealD invN = 1.0 / RealD(DtxqcdNf * Nc);
  for (int t = 0; t < latt[Tdir]; ++t)
    for (int x3 = 0; x3 < latt[Zdir]; ++x3)
      for (int x2 = 0; x2 < latt[Ydir]; ++x2)
        for (int x1 = 0; x1 < latt[Xdir]; ++x1) {
          Coordinate site(4);
          site[0] = x1; site[1] = x2; site[2] = x3; site[3] = t;
          SObj M;
          peekSite(M, X, site);
          ComplexD tr(0.0, 0.0);
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int i = 0; i < Nc; ++i)
              tr += static_cast<ComplexD>(M()(a, a)(i, i));
          ComplexD c = tr * invN;
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int i = 0; i < Nc; ++i)
              M()(a, a)(i, i) = M()(a, a)(i, i) - c;
          pokeSite(M, X, site);
        }
}

// Gaussian sample for CF matrix: complex Gaussian fill of all NfNc^2 entries,
// then real-symmetric project.  Resulting entries:
//   diagonal      ~ N(0, 1) real
//   off-diagonal  ~ N(0, 1/2) real, with M_ji = M_ij
// Tr(X^2) has expectation = NfNc*(NfNc+1)/2 = 21 (the DOF count).
// For σ, π use DtxqcdHermitianTracelessCFGaussian instead.
inline void DtxqcdHermitianCFGaussian(GridParallelRNG &pRNG,
                                       LatticeDtxqcdSigma &X) {
  gaussian(pRNG, X);
  DtxqcdHermitizeAndTracelessCFInPlace(X);
}

// Same as above plus traceless projection — for σ, π.
inline void DtxqcdHermitianTracelessCFGaussian(GridParallelRNG &pRNG,
                                                LatticeDtxqcdSigma &X) {
  gaussian(pRNG, X);
  DtxqcdHermitizeAndTracelessCFInPlace(X);
  DtxqcdMakeTracelessCFInPlace(X);
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
    // σ, π momenta are traceless (matching their position-space constraint —
    // a non-zero trace in the momentum would push σ into a non-traceless
    // configuration during a leapfrog step, breaking the projection).
    // d, n keep their real-symmetric Gaussian.  s, p momenta unchanged
    // (the halved Gaussian coefficient in S affects only the force amplitude,
    // not the kinetic-energy normalization).
    DtxqcdHermitianTracelessCFGaussian(pRNG, P.sigma);  P.sigma = scale * P.sigma;
    DtxqcdHermitianTracelessCFGaussian(pRNG, P.pi);     P.pi    = scale * P.pi;
    DtxqcdHermitianCFGaussian(pRNG, P.d);               P.d     = scale * P.d;
    DtxqcdHermitianCFGaussian(pRNG, P.n);               P.n     = scale * P.n;
    DtxqcdRealScalarGaussian(pRNG, P.s);                P.s     = scale * P.s;
    DtxqcdRealScalarGaussian(pRNG, P.p);                P.p     = scale * P.p;
  }

  static inline Field projectForce(Field &Fforce) {
    Field out(Fforce.Grid());
    out.U = PeriodicGimplR::projectForce(Fforce.U);
    // σ, π forces are projected real-symmetric AND traceless (so the
    // singlet trace mode never gets a force kick, preserving traceless).
    out.sigma = Fforce.sigma;  DtxqcdHermitizeAndTracelessCFInPlace(out.sigma);
                                DtxqcdMakeTracelessCFInPlace(out.sigma);
    out.pi    = Fforce.pi;     DtxqcdHermitizeAndTracelessCFInPlace(out.pi);
                                DtxqcdMakeTracelessCFInPlace(out.pi);
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
    DtxqcdMakeTracelessCFInPlace(U.sigma);
    DtxqcdHermitizeAndTracelessCFInPlace(U.pi);
    DtxqcdMakeTracelessCFInPlace(U.pi);
    DtxqcdHermitizeAndTracelessCFInPlace(U.d);
    DtxqcdHermitizeAndTracelessCFInPlace(U.n);
    DtxqcdRealScalarProjectInPlace(U.s);
    DtxqcdRealScalarProjectInPlace(U.p);
  }

  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::HotConfiguration(pRNG, U.U);
    DtxqcdHermitianTracelessCFGaussian(pRNG, U.sigma);
    DtxqcdHermitianTracelessCFGaussian(pRNG, U.pi);
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
  // initial gauge with stout smearing + AP-time BC).
  //
  // 2026-06-15 redesign — σ, π are traceless, s, p have halved Gaussian
  // coefficient (λ²/4).  Saddle now puts the entire singlet condensate
  // into s alone (no Tr σ contribution since σ is forced traceless):
  //   ⟨s⟩      = 2 · N_F · Σ / λ²    (doubled vs old since coefficient halved)
  //   ⟨Tr σ⟩   = 0                    (traceless by construction)
  // π, d, n, p stay mean-zero.
  //
  // Pre-2026-06-15 (and superseded): both s and Tr σ each shifted by
  // N_F·Σ/λ², doubling the effective singlet coupling.  Their sum
  // matched the SD identity, but the chain explored a redundant
  // (s − Tr σ) DOF that was tightly constrained near zero by the
  // gradient pressure (corr(s, c)=0.999 in data).  Resolved by analytic
  // integration of the redundant variable — see
  // project_dtxqcd_s_sigma_redundancy.md.
  static inline void FillAuxFields(GridParallelRNG &pRNG, Field &U,
                                    RealD lambda, RealD Sigma = 0.0) {
    RealD lambda_var = lambda;
    if (const char *e = std::getenv("AUX_FLUCT_LAMBDA"); e && *e) {
      lambda_var = std::atof(e);
    }
    RealD scale = 1.0 / lambda_var;
    // σ, π: traceless real-symmetric Gaussian
    DtxqcdHermitianTracelessCFGaussian(pRNG, U.sigma);  U.sigma = scale * U.sigma;
    DtxqcdHermitianTracelessCFGaussian(pRNG, U.pi);     U.pi    = scale * U.pi;
    DtxqcdHermitianCFGaussian(pRNG, U.d);               U.d     = scale * U.d;
    DtxqcdHermitianCFGaussian(pRNG, U.n);               U.n     = scale * U.n;
    // s, p: halved-coefficient Gaussian → width × √2 wider per-entry.
    // sample standard Gaussian then scale by √2 / λ to get target variance.
    RealD scale_sp = std::sqrt(2.0) / lambda_var;
    DtxqcdRealScalarGaussian(pRNG, U.s);       U.s     = scale_sp * U.s;
    DtxqcdRealScalarGaussian(pRNG, U.p);       U.p     = scale_sp * U.p;

    if (Sigma != 0.0) {
      // s shift = 2 · N_F · Σ / λ²  (doubled to match halved-coeff saddle).
      // σ traceless → no σ shift; the entire singlet condensate sits in s.
      const RealD s_shift = 2.0 * DtxqcdNf * Sigma / (lambda * lambda);
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
