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
    // Aux momenta need the same sqrt(HMC_MOMENTUM_DENOMINATOR) scaling that
    // gauge momenta receive inside PeriodicGimplR::generate_momenta, so that
    // K = ||P||^2 / HMC_MOMENTUM_DENOMINATOR gives K ~ nDOF/2 per component
    // and the symplectic integrator update  P -= F * ep * HMC_MOMENTUM_DENOMINATOR
    // is balanced against dq/dt = P.
    RealD scale = ::sqrt(HMC_MOMENTUM_DENOMINATOR);
    HermitianGaussian(pRNG, P.sigma);  P.sigma = scale * P.sigma;
    HermitianGaussian(pRNG, P.pi);     P.pi    = scale * P.pi;
    HermitianGaussian(pRNG, P.s);      P.s     = scale * P.s;
    HermitianGaussian(pRNG, P.p);      P.p     = scale * P.p;
    GaussianAntisymTensor(pRNG, P.t);  P.t     = scale * P.t;
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
    // Gauge momenta are antihermitian → Tr(P^2) < 0 naturally, matching
    // the integrator's H = -FieldSquareNorm(P)/denom + S convention.
    // Aux momenta are Hermitian → Tr(P^2) > 0. To get the symplectic
    // structure correct (dH/dt=0 with P -= 2*F*dt, X += P*dt) we need
    // the scalar-field convention: FieldSquareNorm_aux = -Tr(P^2)/2.
    RealD total = PeriodicGimplR::FieldSquareNorm(U.U);
    total -= HermitianFieldSquareNorm(U.sigma) / 2.0;
    total -= HermitianFieldSquareNorm(U.pi)    / 2.0;
    total -= HermitianFieldSquareNorm(U.s)     / 2.0;
    total -= HermitianFieldSquareNorm(U.p)     / 2.0;
    total -= TensorFieldSquareNorm(U.t)        / 2.0;
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

  // Aux drawn from the AuxiliaryFieldGaussianAction's thermal equilibrium
  // (variance 1/lambda^2 per Hermitian DOF).  Pairs with a configurable-scale
  // weak-field gauge: U_mu = exp(i * wf_scale * Σ_a c_a t_a) per link via
  // LieRandomize (bypassing Grid's hard-coded 0.01 in TepidConfiguration).
  // Default wf_scale = 0.1 matches chroma's WEAK_FIELD convention.
  //
  // Optional Sigma_l: light-quark chiral-condensate guess per site
  //   Σ_l ≡ −<q̄q>/N_f  (txqcd_notes.tex eq. above eq. 238).
  // When Sigma_l != 0, σ and s are initialized at their equilibrium means
  //   <σ_ab> = δ_ab · Σ_l / λ²
  //   <s^ij> = δ_ij · N_f · Σ_l / (√2 · N_c · λ²)
  // plus the usual Gaussian fluctuation of width 1/λ.  Skips the mean-drift
  // phase of thermalization at large λ where aux fields start far from <σ>.
  // Sigma_l=0 (default) gives the zero-mean initialization.
  static inline void ThermalAuxConfiguration(GridParallelRNG &pRNG, Field &U,
                                             RealD lambda,
                                             double wf_scale = 0.1,
                                             RealD Sigma_l = 0.0) {
    // Scaled weak-field gauge.
    LatticeColourMatrix Ulink(U.U.Grid());
    for (int mu = 0; mu < Nd; ++mu) {
      SU<Nc>::LieRandomize(pRNG, Ulink, wf_scale);
      PokeIndex<LorentzIndex>(U.U, Ulink, mu);
    }
    RealD s = 1.0 / lambda;
    HermitianGaussian(pRNG, U.sigma); U.sigma = s * U.sigma;
    HermitianGaussian(pRNG, U.pi);    U.pi    = s * U.pi;
    HermitianGaussian(pRNG, U.s);     U.s     = s * U.s;
    HermitianGaussian(pRNG, U.p);     U.p     = s * U.p;
    GaussianAntisymTensor(pRNG, U.t); U.t     = (s / std::sqrt(2.0)) * U.t;

    if (Sigma_l != 0.0) {
      const RealD sigma_mean = Sigma_l / (lambda * lambda);
      const RealD s_mean =
          static_cast<RealD>(TxqcdNf) * Sigma_l /
          (std::sqrt(2.0) * static_cast<RealD>(Nc) * lambda * lambda);
      // σ: shift diagonal flavor entries by sigma_mean.
      TxqcdSiteSigma sigma_id;
      sigma_id = Zero();
      for (int a = 0; a < TxqcdNf; ++a) sigma_id()()(a, a) = sigma_mean;
      LatticeSigmaField shift_sigma(U.sigma.Grid());
      shift_sigma = sigma_id;
      U.sigma = U.sigma + shift_sigma;
      // s: shift diagonal color entries by s_mean.
      TxqcdSiteS s_id;
      s_id = Zero();
      for (int i = 0; i < Nc; ++i) s_id()()(i, i) = s_mean;
      LatticeSFieldC shift_s(U.s.Grid());
      shift_s = s_id;
      U.s = U.s + shift_s;
    }
  }

  static const int num_colours = Nc;
};

NAMESPACE_END(Grid);
