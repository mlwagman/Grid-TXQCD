#pragma once
// Composite FieldImplementation for TXQCD HMC.
//
// The Integrator is templated on one FieldImplementation; its static methods
// here delegate to:
//   - PeriodicGimplR for gauge (exp-update via Ta projection)
//   - Hermitian-matrix helpers for sigma/pi/s/p (additive update)
//   - Antisymmetric-color-tensor helper for t_{mu,nu} (additive update)
//
// The momentum conjugate to an auxiliary field X is just another Hermitian
// (resp. antisymmetric-color) matrix of the same type. Its quadratic kinetic
// piece is (1/2) Tr P^2, which the Integrator sums via FieldSquareNorm.

#include <Grid/qcd/action/txqcd/TXQCDField.h>
#include <Grid/qcd/action/gauge/GaugeImplementations.h>

NAMESPACE_BEGIN(Grid);

// ---------- aux-field helpers (Lattice-level, Hermitian + antisym tensor) ----------

// Hermitize in place: X <- 0.5*(X + adj(X)).
template <class LatticeMat>
inline void HermitianProjectInPlace(LatticeMat &X) {
  X = 0.5 * (X + adj(X));
}

// Generate a Hermitian Gaussian matrix field with unit variance per DOF.
// Step 1: fill every complex entry with iid N(0,1).
// Step 2: Hermitize. The resulting diagonal entries have variance 1, off-
//   diagonal entries have independent real/imag parts each with variance 1/2
//   (correct Hermitian GUE-like measure up to normalization).
template <class LatticeMat>
inline void HermitianGaussian(GridParallelRNG &pRNG, LatticeMat &X) {
  gaussian(pRNG, X);
  HermitianProjectInPlace(X);
}

// Antisymmetrize in (mu, nu) for the TXQCD tensor field (color indices are
// independently Hermitized). Enforces t_{mu,nu} = -t_{nu,mu}; diagonal zero.
inline void AntisymmetrizeTensor(LatticeTField &T) {
  autoView(T_v, T, CpuWrite);
  GridBase *grid = T.Grid();
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      T_v[ss]()(mu, mu) = Zero();
      for (int nu = mu + 1; nu < Nd; ++nu) {
        auto upper = T_v[ss]()(mu, nu);
        auto lower = T_v[ss]()(nu, mu);
        auto anti  = 0.5 * (upper - lower);
        T_v[ss]()(mu, nu) =  anti;
        T_v[ss]()(nu, mu) = -anti;
      }
    }
  });
}

// Hermitize color indices of the tensor field.
inline void HermitianProjectTensorColor(LatticeTField &T) {
  autoView(T_v, T, CpuWrite);
  GridBase *grid = T.Grid();
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = 0; nu < Nd; ++nu) {
        auto M = T_v[ss]()(mu, nu);
        auto Mdag = adj(M);
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            T_v[ss]()(mu, nu)(i, j) = 0.5 * (M(i, j) + Mdag(i, j));
      }
    }
  });
}

inline void GaussianAntisymTensor(GridParallelRNG &pRNG, LatticeTField &T) {
  // Fill with complex N(0,1)+i*N(0,1) entries, then:
  //   1. Hermitize color blocks (preserves per-DOF variance 1 on the Hermitian
  //      representation),
  //   2. Antisymmetrize in (mu,nu) -- this halves per-DOF variance because the
  //      projector is (t_{mu,nu} - t_{nu,mu})/2 combining two independent
  //      samples,
  //   3. Rescale by sqrt(2) so the independent DOFs of the antisym-Hermitian
  //      representation end up with unit variance, matching the convention
  //      used by HermitianGaussian (and required so the HMC kinetic term
  //      (1/2) Tr P^2 is normalized correctly).
  gaussian(pRNG, T);
  HermitianProjectTensorColor(T);
  AntisymmetrizeTensor(T);
  T = T * std::sqrt(2.0);
}

// Squared Frobenius norm of a Hermitian site-matrix field: sum_x Tr(X^2).
template <class LatticeMat>
inline RealD HermitianFieldSquareNorm(LatticeMat &X) {
  return TensorRemove(sum(trace(X * X))).real();
}

inline RealD TensorFieldSquareNorm(LatticeTField &T) {
  // (1/2) sum_{mu,nu} Tr(t_{mu,nu}^2) ; factor 1/2 keeps antisymmetric pairs
  // from being double-counted across (mu<nu) vs (nu<mu).
  autoView(T_v, T, CpuRead);
  GridBase *grid = T.Grid();
  RealD total = 0.0;
  // Simple unoptimized reduction; Phase 4a only needs correctness, not speed.
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        auto M = T_v[ss]()(mu, nu);
        // Tr(M^2): sum_i,j M_ij * M_ji
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            auto v = M(i, j) * M(j, i);
            // SIMD reduction: sum vComplex lanes.
            auto vs = Reduce(v);
            total += real(vs);
          }
        }
      }
    }
  });
  // MPI reduction
  grid->GlobalSum(total);
  return total;
}

// ---------- Composite FieldImplementation ----------

class TXQCDCompositeImpl {
 public:
  typedef vComplex Simd;
  typedef TXQCDField Field;

  // HMC framework expectation: INHERIT_FIELD_TYPES expands to typedef
  // Impl::Simd, Impl::ComplexField, Impl::SiteField, Impl::Field.
  // Provide ComplexField/SiteField for completeness (points to gauge).
  typedef iScalar<iScalar<iScalar<vComplex>>> SiteComplex;
  typedef Lattice<SiteComplex> ComplexField;
  typedef typename PeriodicGimplR::SiteField SiteField;

  static inline void generate_momenta(Field &P, GridSerialRNG &sRNG,
                                      GridParallelRNG &pRNG) {
    PeriodicGimplR::generate_momenta(P.U, sRNG, pRNG);
    HermitianGaussian(pRNG, P.sigma);
    HermitianGaussian(pRNG, P.pi);
    HermitianGaussian(pRNG, P.s);
    HermitianGaussian(pRNG, P.p);
    GaussianAntisymTensor(pRNG, P.t);
  }

  static inline Field projectForce(Field &Fforce) {
    Field out(Fforce.Grid());
    out.U = PeriodicGimplR::projectForce(Fforce.U);
    out.sigma = Fforce.sigma;
    HermitianProjectInPlace(out.sigma);
    out.pi = Fforce.pi;
    HermitianProjectInPlace(out.pi);
    out.s = Fforce.s;
    HermitianProjectInPlace(out.s);
    out.p = Fforce.p;
    HermitianProjectInPlace(out.p);
    out.t = Fforce.t;
    HermitianProjectTensorColor(out.t);
    AntisymmetrizeTensor(out.t);
    return out;
  }

  static inline void update_field(Field &P, Field &U, double ep) {
    PeriodicGimplR::update_field(P.U, U.U, ep);
    U.sigma = U.sigma + P.sigma * ep;
    U.pi    = U.pi    + P.pi    * ep;
    U.s     = U.s     + P.s     * ep;
    U.p     = U.p     + P.p     * ep;
    U.t     = U.t     + P.t     * ep;
  }

  static inline RealD FieldSquareNorm(Field &U) {
    RealD total = PeriodicGimplR::FieldSquareNorm(U.U);
    total += HermitianFieldSquareNorm(U.sigma);
    total += HermitianFieldSquareNorm(U.pi);
    total += HermitianFieldSquareNorm(U.s);
    total += HermitianFieldSquareNorm(U.p);
    total += TensorFieldSquareNorm(U.t);
    return total;
  }

  static inline void Project(Field &U) {
    PeriodicGimplR::Project(U.U);
    HermitianProjectInPlace(U.sigma);
    HermitianProjectInPlace(U.pi);
    HermitianProjectInPlace(U.s);
    HermitianProjectInPlace(U.p);
    HermitianProjectTensorColor(U.t);
    AntisymmetrizeTensor(U.t);
  }

  static inline void HotConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::HotConfiguration(pRNG, U.U);
    HermitianGaussian(pRNG, U.sigma);
    HermitianGaussian(pRNG, U.pi);
    HermitianGaussian(pRNG, U.s);
    HermitianGaussian(pRNG, U.p);
    GaussianAntisymTensor(pRNG, U.t);
  }

  static inline void TepidConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::TepidConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.s     = Zero();
    U.p     = Zero();
    U.t     = Zero();
  }

  static inline void ColdConfiguration(GridParallelRNG &pRNG, Field &U) {
    PeriodicGimplR::ColdConfiguration(pRNG, U.U);
    U.sigma = Zero();
    U.pi    = Zero();
    U.s     = Zero();
    U.p     = Zero();
    U.t     = Zero();
  }

  static const int num_colours = Nc;
};

NAMESPACE_END(Grid);
