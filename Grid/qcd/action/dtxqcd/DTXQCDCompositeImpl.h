#pragma once
// Composite FieldImplementation for DTXQCD HMC.
//
// Delegates gauge to PeriodicGimplR (exp-update via Ta projection). Aux fields:
//   sigma^A, pi^A : real-projected vComplex triplets (imag held at 0).
//   t^A_{mu,nu}   : real-projected per Pauli component, antisymmetric in (mu,nu).
//   d, n          : Hermitian-projected color matrices (same as TXQCD's s, p).
//
// Momentum conjugate to each aux field has the same site-tensor type and
// projection constraints; quadratic kinetic piece is (1/2) Tr P^2 (or sum_A
// (P^A)^2 for triplets), accumulated into FieldSquareNorm.  Scalar-field
// momentum convention follows TXQCDCompositeImpl: FieldSquareNorm subtracts
// (1/2) of each Hermitian / real-projected norm so the integrator update
// P -= F * ep * HMC_MOMENTUM_DENOMINATOR is symplectic against dq/dt = P.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/gauge/GaugeImplementations.h>

NAMESPACE_BEGIN(Grid);

// ---------- field-type projectors ----------

// Real projection: X <- 0.5 * (X + conjugate(X)).  For vComplex storage this
// zeros the imaginary part component-wise, leaving the physical real DOF.
template <class LatticeT>
inline void DtxqcdRealProjectInPlace(LatticeT &X) {
  X = 0.5 * (X + conjugate(X));
}

// Real Gaussian: complex Gaussian fill, then real-project.  Variance: gaussian
// fills re,im each ~ N(0,1); after projection the real part has variance 1.
// (Half the RNG work is wasted on the imag part — acceptable for v1.)
template <class LatticeT>
inline void DtxqcdRealGaussian(GridParallelRNG &pRNG, LatticeT &X) {
  gaussian(pRNG, X);
  DtxqcdRealProjectInPlace(X);
}

// Hermitian projection / Gaussian for d, n (iScalar<iScalar<iMatrix<vComplex,Nc>>>).
template <class LatticeMat>
inline void DtxqcdHermitianProjectInPlace(LatticeMat &X) {
  X = 0.5 * (X + adj(X));
}

template <class LatticeMat>
inline void DtxqcdHermitianGaussian(GridParallelRNG &pRNG, LatticeMat &X) {
  gaussian(pRNG, X);
  DtxqcdHermitianProjectInPlace(X);
}

// Antisymmetrize the (mu,nu) tensor of t^A_{mu,nu}: t_{mu,nu} = -t_{nu,mu},
// diagonal zero.  Operates per Pauli component on the real-projected field.
inline void DtxqcdAntisymmetrizeTensor(LatticeDtxqcdT &T) {
  autoView(T_v, T, CpuWrite);
  GridBase *grid = T.Grid();
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      T_v[ss]()(mu, mu) = Zero();
      for (int nu = mu + 1; nu < Nd; ++nu) {
        auto upper = T_v[ss]()(mu, nu);
        auto lower = T_v[ss]()(nu, mu);
        auto anti = 0.5 * (upper - lower);
        T_v[ss]()(mu, nu) =  anti;
        T_v[ss]()(nu, mu) = -anti;
      }
    }
  });
}

// Gaussian draw for t^A_{mu,nu}: complex Gaussian → real-project per Pauli
// component → antisymmetrize in (mu,nu) → rescale by sqrt(2) so that the
// independent DOFs of the antisym representation end up with unit variance
// (the antisym projector halves per-DOF variance).
inline void DtxqcdGaussianAntisymTensor(GridParallelRNG &pRNG,
                                        LatticeDtxqcdT &T) {
  gaussian(pRNG, T);
  DtxqcdRealProjectInPlace(T);
  DtxqcdAntisymmetrizeTensor(T);
  T = T * std::sqrt(2.0);
}

// Squared norms.  For real-projected fields, norm2 collapses to sum of real^2
// (imag=0).  Antisymmetric tensor: (1/2) norm2 (each (mu,nu) counted twice
// in the full mu,nu sum).
template <class LatticeT>
inline RealD DtxqcdTripletSquareNorm(LatticeT &X) { return norm2(X); }

template <class LatticeMat>
inline RealD DtxqcdHermitianSquareNorm(LatticeMat &X) { return norm2(X); }

inline RealD DtxqcdTensorSquareNorm(LatticeDtxqcdT &T) { return norm2(T) / 2.0; }

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
    DtxqcdRealGaussian(pRNG, P.sigma);  P.sigma = scale * P.sigma;
    DtxqcdRealGaussian(pRNG, P.pi);     P.pi    = scale * P.pi;
    DtxqcdGaussianAntisymTensor(pRNG, P.t);  P.t = scale * P.t;
    DtxqcdHermitianGaussian(pRNG, P.d); P.d     = scale * P.d;
    DtxqcdHermitianGaussian(pRNG, P.n); P.n     = scale * P.n;
  }

  static inline Field projectForce(Field &Fforce) {
    Field out(Fforce.Grid());
    out.U = PeriodicGimplR::projectForce(Fforce.U);
    out.sigma = Fforce.sigma;  DtxqcdRealProjectInPlace(out.sigma);
    out.pi    = Fforce.pi;     DtxqcdRealProjectInPlace(out.pi);
    out.t     = Fforce.t;
    DtxqcdRealProjectInPlace(out.t);
    DtxqcdAntisymmetrizeTensor(out.t);
    out.d     = Fforce.d;      DtxqcdHermitianProjectInPlace(out.d);
    out.n     = Fforce.n;      DtxqcdHermitianProjectInPlace(out.n);
    return out;
  }

  static inline void update_field(Field &P, Field &U, double ep) {
    PeriodicGimplR::update_field(P.U, U.U, ep);
    U.sigma = U.sigma + P.sigma * ep;
    U.pi    = U.pi    + P.pi    * ep;
    U.t     = U.t     + P.t     * ep;
    U.d     = U.d     + P.d     * ep;
    U.n     = U.n     + P.n     * ep;
  }

  static inline RealD FieldSquareNorm(Field &U) {
    // Gauge momenta: antihermitian, Tr(P^2) < 0 naturally.
    // Aux momenta: real-projected / Hermitian, Tr(P^2) > 0; scalar-field
    // convention requires -Tr(P^2)/2 so dH/dt vanishes.
    RealD total = PeriodicGimplR::FieldSquareNorm(U.U);
    total -= DtxqcdTripletSquareNorm(U.sigma)  / 2.0;
    total -= DtxqcdTripletSquareNorm(U.pi)     / 2.0;
    total -= DtxqcdTensorSquareNorm(U.t)       / 2.0;
    total -= DtxqcdHermitianSquareNorm(U.d)    / 2.0;
    total -= DtxqcdHermitianSquareNorm(U.n)    / 2.0;
    return total;
  }

  static inline void Project(Field &U) {
    PeriodicGimplR::Project(U.U);
    DtxqcdRealProjectInPlace(U.sigma);
    DtxqcdRealProjectInPlace(U.pi);
    DtxqcdRealProjectInPlace(U.t);
    DtxqcdAntisymmetrizeTensor(U.t);
    DtxqcdHermitianProjectInPlace(U.d);
    DtxqcdHermitianProjectInPlace(U.n);
  }

  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::HotConfiguration(pRNG, U.U);
    DtxqcdRealGaussian(pRNG, U.sigma);
    DtxqcdRealGaussian(pRNG, U.pi);
    DtxqcdGaussianAntisymTensor(pRNG, U.t);
    DtxqcdHermitianGaussian(pRNG, U.d);
    DtxqcdHermitianGaussian(pRNG, U.n);
  }

  static inline void TepidConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::TepidConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.t     = Zero();
    U.d     = Zero();
    U.n     = Zero();
  }

  static inline void ColdConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::ColdConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.t     = Zero();
    U.d     = Zero();
    U.n     = Zero();
  }

  // Fill all aux slots with mean-zero Gaussian samples at the physical
  // saddle variance Var = 1/lambda^2 per independent component.  Same
  // pattern as TXQCDCompositeImpl::FillAuxFields (without the optional
  // Sigma saddle shift -- DTXQCD has no analogue of the TXQCD <q-bar q>
  // saddle structure at this stage; if/when a non-trivial saddle is
  // identified for the diquark-tensor variant, add a Sigma overload here).
  //
  // For sigma^A, pi^A, t^A: stored as triplets, real-projected, so the
  // base DtxqcdRealGaussian gives Var=1 per real DOF and we scale by 1/lambda.
  // For d, n: Hermitian color matrices, DtxqcdHermitianGaussian Var=1/2 per
  // real DOF; we still scale by 1/lambda to match TXQCD's HermitianGaussian
  // convention.
  static inline void FillAuxFields(GridParallelRNG &pRNG, Field &U,
                                    RealD lambda) {
    // Optional fluctuation-scale decoupling, mirroring TXQCD's
    // AUX_FLUCT_LAMBDA: set the env var to use width 1/lambda_var while
    // the action still couples at the physical lambda.  The DTXQCD use
    // case is cold-start initialisation -- at the production lambda (~3)
    // the per-site 48x48 Mee picks up near-zero eigenvalues from outlier
    // aux sites and Mpc^dag Mpc lambda_max explodes (measured 6e6 vs
    // ~30 at lambda=10 on a 4^3 x 8 cold gauge).  Starting at the larger
    // fluctuation-lambda 10-30 keeps the operator well-conditioned for
    // the first few trajectories while the HMC dynamics evolve toward
    // the physical width.
    RealD lambda_var = lambda;
    if (const char *e = std::getenv("AUX_FLUCT_LAMBDA"); e && *e) {
      lambda_var = std::atof(e);
    }
    RealD s = 1.0 / lambda_var;
    DtxqcdRealGaussian(pRNG, U.sigma);  U.sigma = s * U.sigma;
    DtxqcdRealGaussian(pRNG, U.pi);     U.pi    = s * U.pi;
    DtxqcdGaussianAntisymTensor(pRNG, U.t);
    U.t = (s / std::sqrt(2.0)) * U.t;
    DtxqcdHermitianGaussian(pRNG, U.d); U.d     = s * U.d;
    DtxqcdHermitianGaussian(pRNG, U.n); U.n     = s * U.n;
  }

  // Weak-field gauge + thermal aux init.  This is the recommended cold-start
  // for DTXQCD HMC: U links near identity (scale wf_scale ~ 0.1) plus aux
  // fields drawn at their physical Var=1/lambda^2 saddle width.  Avoids the
  // exact-aux=0 spectral degeneracy that makes the doubled M block-diagonal
  // (and breaks multi-shift CG convergence on cold starts).  Direct sibling
  // of TXQCDCompositeImpl::ThermalAuxConfiguration.
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
