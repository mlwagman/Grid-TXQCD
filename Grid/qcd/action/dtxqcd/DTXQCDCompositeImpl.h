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

// Hermitize + traceless project on a CF-matrix lattice field.  Operates on the
// combined (a,i)(b,j) 6x6 index space: M_{(a,i)(b,j)} <- 0.5 (M + adj(M)) then
// subtract Tr(M)/NfNc on the diagonal.
inline void DtxqcdHermitizeAndTracelessCFInPlace(LatticeDtxqcdSigma &X) {
  // Step 1: Hermitize on the combined index.  Grid's adj() on
  // iMatrix<iMatrix<vComplex, Nc>, Nf> does the right thing -- it conjugates
  // and transposes both the outer (flavor) and inner (color) iMatrix layers,
  // which is exactly the combined-index Hermitian conjugate.
  X = 0.5 * (X + adj(X));

  // Step 2: subtract the combined-index trace.  Need Tr = sum_{a,i} X(a,a)(i,i).
  // Built per-site with a thread_for; only DtxqcdNfNc diagonal entries.
  autoView(Xv, X, CpuWrite);
  GridBase *grid = X.Grid();
  thread_for(ss, grid->oSites(), {
    auto tr_v = Xv[ss]()(0, 0)(0, 0);
    zeroit(tr_v);
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        tr_v = tr_v + Xv[ss]()(a, a)(i, i);
      }
    }
    auto shift = tr_v * (1.0 / RealD(DtxqcdNfNc));
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        Xv[ss]()(a, a)(i, i) = Xv[ss]()(a, a)(i, i) - shift;
      }
    }
  });
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

  // Fill all aux slots with mean-zero Gaussian samples at the physical
  // saddle variance Var = 1/lambda^2 per independent component.  The
  // CF Hermitian fields each have NfNc^2 - 1 = 35 independent DOFs; the
  // singlet scalars each have 1 DOF.  AUX_FLUCT_LAMBDA optionally
  // decouples the init fluctuation width from the physical lambda (cf.
  // v1 use case in the deleted TXQCD analog).
  static inline void FillAuxFields(GridParallelRNG &pRNG, Field &U,
                                    RealD lambda) {
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
  }

  // Weak-field gauge + thermal aux init.  Same recipe as v1.
  static inline void ThermalAuxConfiguration(GridParallelRNG &pRNG, Field &U,
                                              RealD lambda,
                                              double wf_scale = 0.1) {
    GenerateWeakFieldGauge(pRNG, U, wf_scale);
    FillAuxFields(pRNG, U, lambda);
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
