#pragma once
// Kinetic-term action for ALL FIVE TXQCD auxiliary fields:
//
//   S_kin = (Z_σ/2) Σ_{x,μ} Tr[(σ(x+μ̂) − σ(x))²]
//         + (Z_π/2) Σ_{x,μ} Tr[(π(x+μ̂) − π(x))²]
//         + (Z_s/2) Σ_{x,μ} Tr[(s(x+μ̂) − s(x))²]
//         + (Z_p/2) Σ_{x,μ} Tr[(p(x+μ̂) − p(x))²]
//         +  Z_t    Σ_{x,μ,ν<μ} Tr[(t_{μν}(x+ρ̂) − t_{μν}(x))²]   (sum over derivative ρ̂)
//
// Coefficients mirror the quadratic AuxGaussianAction exactly, with λ² → Z:
//   * σ, π, s, p all weighted 1/2·Z·norm2 (same as 1/2·λ²·norm2 in quadratic)
//   * t weighted 1·Z·TensorFieldSquareNorm = 1/2·Z·norm2(t) (same as 1·λ²·TensorFieldSquareNorm
//     = 1/2·λ²·norm2(t) in quadratic, where TensorFieldSquareNorm = norm2/2 absorbs μ<ν
//     antisymmetric double-counting).
//
// This convention is required for Fierz identity preservation at finite lattice
// spacing — every aux field must enter the kinetic action with the same
// coefficient structure it has in the quadratic action.
//
// In momentum space the joint quadratic + kinetic gives propagator
//   G(k) = 1 / (λ² + Z · k̂²),    k̂² = Σ_μ 4 sin²(k_μ/2)
// for each field — preserving the σ/s VEV at k=0 while damping high-k noise.
//
// Force (= dS/dX for HMC):
//
//   dS_kin/dX(y) = −Z_X · Σ_μ [X(y+μ̂) + X(y−μ̂) − 2X(y)]   (lattice −Δ for σ, π, s, p)
//   dS_kin/dt(y) = −2·Z_t · Σ_μ [t(y+μ̂) + t(y−μ̂) − 2t(y)]  (extra 2 mirrors the quadratic 2·λ²·t)
//
// Implemented as an Action<TXQCDField> so it can register at any MD level
// alongside AuxiliaryFieldGaussianAction.  Env knobs read by the production
// drivers (default 0 = inactive for that field):
//   SIGMA_KINETIC_Z, PI_KINETIC_Z, S_KINETIC_Z, P_KINETIC_Z, T_KINETIC_Z

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class AuxiliaryFieldKineticAction : public Action<TXQCDField> {
 public:
  AuxiliaryFieldKineticAction(RealD Z_sigma_, RealD Z_pi_, RealD Z_s_,
                              RealD Z_p_, RealD Z_t_)
      : Z_sigma(Z_sigma_), Z_pi(Z_pi_), Z_s(Z_s_), Z_p(Z_p_), Z_t(Z_t_) {}

  // Back-compat: σ + s only constructor.  Sets π/p/t Z to 0.
  AuxiliaryFieldKineticAction(RealD Z_sigma_, RealD Z_s_)
      : Z_sigma(Z_sigma_), Z_pi(0.0), Z_s(Z_s_), Z_p(0.0), Z_t(0.0) {}

  std::string action_name() override { return "AuxiliaryFieldKineticAction"; }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage
       << "[AuxiliaryFieldKineticAction] Z_sigma = " << Z_sigma
       << "  Z_pi = " << Z_pi
       << "  Z_s = " << Z_s
       << "  Z_p = " << Z_p
       << "  Z_t = " << Z_t << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {}

  RealD S(const TXQCDField &U) override {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    RealD s_total = 0.0;
    if (Z_sigma != 0.0) {
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeSigmaField diff = Cshift(Unc.sigma, mu, +1) - Unc.sigma;
        s_total += 0.5 * Z_sigma * HermitianFieldSquareNorm(diff);
      }
    }
    if (Z_pi != 0.0) {
      for (int mu = 0; mu < Nd; ++mu) {
        LatticePiField diff = Cshift(Unc.pi, mu, +1) - Unc.pi;
        s_total += 0.5 * Z_pi * HermitianFieldSquareNorm(diff);
      }
    }
    if (Z_s != 0.0) {
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeSFieldC diff = Cshift(Unc.s, mu, +1) - Unc.s;
        s_total += 0.5 * Z_s * HermitianFieldSquareNorm(diff);
      }
    }
    if (Z_p != 0.0) {
      for (int mu = 0; mu < Nd; ++mu) {
        LatticePFieldC diff = Cshift(Unc.p, mu, +1) - Unc.p;
        s_total += 0.5 * Z_p * HermitianFieldSquareNorm(diff);
      }
    }
    if (Z_t != 0.0) {
      // Tensor: coefficient 1.0 (not 0.5) to mirror the 2.0 prefactor in the
      // quadratic action (which makes the per-element coefficient λ² instead
      // of λ²/2 once TensorFieldSquareNorm's 1/2 is absorbed).
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeTField diff = Cshift(Unc.t, mu, +1) - Unc.t;
        s_total += Z_t * TensorFieldSquareNorm(diff);
      }
    }
    return s_total;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    dSdU.U     = Zero();
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    // dS/dX = -Z_X · Lap(X) for σ, π, s, p.  dS/dt = -2·Z_t · Lap(t).
    if (Z_sigma != 0.0) {
      LatticeSigmaField lap(U.sigma.Grid());
      lap = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        lap = lap + Cshift(U.sigma, mu, +1) + Cshift(U.sigma, mu, -1)
                  - 2.0 * U.sigma;
      }
      dSdU.sigma = -Z_sigma * lap;
    }
    if (Z_pi != 0.0) {
      LatticePiField lap(U.pi.Grid());
      lap = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        lap = lap + Cshift(U.pi, mu, +1) + Cshift(U.pi, mu, -1) - 2.0 * U.pi;
      }
      dSdU.pi = -Z_pi * lap;
    }
    if (Z_s != 0.0) {
      LatticeSFieldC lap(U.s.Grid());
      lap = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        lap = lap + Cshift(U.s, mu, +1) + Cshift(U.s, mu, -1) - 2.0 * U.s;
      }
      dSdU.s = -Z_s * lap;
    }
    if (Z_p != 0.0) {
      LatticePFieldC lap(U.p.Grid());
      lap = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        lap = lap + Cshift(U.p, mu, +1) + Cshift(U.p, mu, -1) - 2.0 * U.p;
      }
      dSdU.p = -Z_p * lap;
    }
    if (Z_t != 0.0) {
      // Factor 2 mirrors AuxGaussianAction::deriv (dSdU.t = 2c·t).  The HMC
      // integrator's t metric is halved (FieldSquareNorm uses TFSN(P)/2 =
      // norm2/4 instead of norm2/2 for σ/π/s/p), so the force is scaled up
      // by 2 in storage units to compensate.  When BOTH actions follow this
      // convention, they add coherently and the integrator remains symplectic
      // against the halved t metric.  See [[aux-t-factor-2-convention]].
      LatticeTField lap(U.t.Grid());
      lap = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        lap = lap + Cshift(U.t, mu, +1) + Cshift(U.t, mu, -1) - 2.0 * U.t;
      }
      dSdU.t = (-2.0 * Z_t) * lap;
    }
  }

 private:
  RealD Z_sigma;
  RealD Z_pi;
  RealD Z_s;
  RealD Z_p;
  RealD Z_t;
};

NAMESPACE_END(Grid);
