#pragma once
// Fierz-preserving Lap shift for the aux fields when a kinetic term is active.
//
//   σ_eff(x) = σ(x) - (Z_σ/λ²) · Lap_code(σ)(x)
//
// and analogously for π, s, p, t with their own Z's (same Z across all 5 for
// EXACT Fierz cancellation — see [[fierz-lap-shift]] memory and
// fierz_lap_shift.tex for the derivation).
//
// Lap_code is the standard 4-direction lattice Laplacian
//   Lap_code(X)(x) = Σ_μ [X(x+μ̂) + X(x-μ̂) - 2 X(x)]
// with eigenvalue -k̂² in Fourier (i.e., this IS the standard scalar ∂² used
// in the derivation).
//
// Two operations:
//   * Apply(U)    : in-place σ → σ_eff for all 5 aux fields
//   * ApplyToForce(dSdU) : in-place F_σ_eff → F_σ via the chain rule
//                          F_σ(y) = F_σ_eff(y) - (Z/λ²)·Lap_code(F_σ_eff)(y)
//                          (Lap_code is self-adjoint, so forward = backward
//                          chain rule.)
//
// Wrap any Action<TXQCDField> with FierzShiftedAction below to make it see
// σ_eff while exposing the unmodified-σ interface to the HMC integrator.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class AuxFierzShift {
 public:
  // c_X = Z_X / λ²  for each aux field.  Z=0 → no shift on that field.
  AuxFierzShift(RealD lambda, RealD Z_sigma, RealD Z_pi, RealD Z_s,
                RealD Z_p, RealD Z_t, RealD sign = -1.0)
      : c_sigma_(Z_sigma / (lambda * lambda)),
        c_pi_   (Z_pi    / (lambda * lambda)),
        c_s_    (Z_s     / (lambda * lambda)),
        c_p_    (Z_p     / (lambda * lambda)),
        c_t_    (Z_t     / (lambda * lambda)),
        sign_(sign),
        lambda_(lambda),
        Z_sigma_(Z_sigma), Z_pi_(Z_pi), Z_s_(Z_s), Z_p_(Z_p), Z_t_(Z_t) {}

  bool active() const {
    return c_sigma_ != 0.0 || c_pi_ != 0.0 || c_s_ != 0.0
        || c_p_ != 0.0    || c_t_ != 0.0;
  }

  std::string LogParameters() const {
    std::stringstream os;
    os << "[AuxFierzShift] λ=" << lambda_
       << "  sign=" << (sign_ > 0 ? "+1 (DAMP, empirical)" : "-1 (BOOST, derived)")
       << "  Z(σ,π,s,p,t)=(" << Z_sigma_ << "," << Z_pi_ << "," << Z_s_
       << "," << Z_p_ << "," << Z_t_ << ")"
       << "  c=Z/λ²: (" << c_sigma_ << "," << c_pi_ << "," << c_s_
       << "," << c_p_ << "," << c_t_ << ")";
    return os.str();
  }

  // σ → σ + sign·(Z_σ/λ²)·Lap_code(σ)  for each aux field.
  // sign = -1 (default) matches the COV-derivation result (high-k BOOST in σ).
  // sign = +1 (TXQCD_FIERZ_LAP_SIGN=+1) is the empirical sign-flipped form
  // (high-k DAMP in σ).  Set via FromEnv based on TXQCD_FIERZ_LAP_SIGN.
  void Apply(TXQCDField &U) const {
    if (c_sigma_ != 0.0) U.sigma = U.sigma + (sign_ * c_sigma_) * lap(U.sigma);
    if (c_pi_    != 0.0) U.pi    = U.pi    + (sign_ * c_pi_)    * lap(U.pi);
    if (c_s_     != 0.0) U.s     = U.s     + (sign_ * c_s_)     * lap(U.s);
    if (c_p_     != 0.0) U.p     = U.p     + (sign_ * c_p_)     * lap(U.p);
    if (c_t_     != 0.0) U.t     = U.t     + (sign_ * c_t_)     * lap(U.t);
  }

  // Apply the chain rule for the force: F_σ = F_σ_eff + sign·(Z_σ/λ²)·Lap(F_σ_eff).
  // Same sign as Apply() because Lap_code is self-adjoint and the chain rule
  // is symmetric.
  void ApplyToForce(TXQCDField &dSdU) const {
    if (c_sigma_ != 0.0) dSdU.sigma = dSdU.sigma + (sign_ * c_sigma_) * lap(dSdU.sigma);
    if (c_pi_    != 0.0) dSdU.pi    = dSdU.pi    + (sign_ * c_pi_)    * lap(dSdU.pi);
    if (c_s_     != 0.0) dSdU.s     = dSdU.s     + (sign_ * c_s_)     * lap(dSdU.s);
    if (c_p_     != 0.0) dSdU.p     = dSdU.p     + (sign_ * c_p_)     * lap(dSdU.p);
    if (c_t_     != 0.0) dSdU.t     = dSdU.t     + (sign_ * c_t_)     * lap(dSdU.t);
  }

  // Construct from env vars: TXQCD_FIERZ_LAP=1 to enable; reuses
  // SIGMA_KINETIC_Z, PI_KINETIC_Z, S_KINETIC_Z, P_KINETIC_Z, T_KINETIC_Z
  // (the same env vars that turn on AuxKineticAction).  Returns inactive
  // shift if TXQCD_FIERZ_LAP is unset or zero.
  static AuxFierzShift FromEnv(RealD lambda) {
    auto readZ = [](const char *name) -> RealD {
      const char *e = std::getenv(name);
      return (e && *e) ? std::atof(e) : 0.0;
    };
    const char *enabled = std::getenv("TXQCD_FIERZ_LAP");
    bool on = enabled && *enabled && std::atoi(enabled);
    if (!on) return AuxFierzShift(lambda, 0, 0, 0, 0, 0);
    RealD sign = -1.0;
    if (const char *s = std::getenv("TXQCD_FIERZ_LAP_SIGN"); s && *s) {
      sign = (std::atof(s) > 0) ? +1.0 : -1.0;
    }
    return AuxFierzShift(lambda,
                         readZ("SIGMA_KINETIC_Z"),
                         readZ("PI_KINETIC_Z"),
                         readZ("S_KINETIC_Z"),
                         readZ("P_KINETIC_Z"),
                         readZ("T_KINETIC_Z"),
                         sign);
  }

 private:
  template <class LatticeAuxT>
  static LatticeAuxT lap(const LatticeAuxT &X) {
    LatticeAuxT out(X.Grid());
    out = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      out = out + Cshift(X, mu, +1) + Cshift(X, mu, -1) - 2.0 * X;
    }
    return out;
  }

  RealD c_sigma_, c_pi_, c_s_, c_p_, c_t_;  // Z/λ² for each aux
  RealD sign_;                                // +1 (damp) or -1 (boost, default)
  RealD lambda_;                              // kept for logging
  RealD Z_sigma_, Z_pi_, Z_s_, Z_p_, Z_t_;   // kept for logging
};

// Action wrapper: presents itself to the HMC integrator as an Action<TXQCDField>
// taking the bare σ (and other aux), but internally calls the underlying action
// with σ_eff = σ - (Z/λ²)·Lap σ, and applies the chain rule to the resulting
// force.  Works for any underlying Action<TXQCDField> (clover pseudo-fermion,
// log-det, rational, etc.).
class FierzShiftedAction : public Action<TXQCDField> {
 public:
  FierzShiftedAction(Action<TXQCDField> &underlying, const AuxFierzShift &shift)
      : underlying_(underlying), shift_(shift) {}

  std::string action_name() override {
    return "FierzShifted[" + underlying_.action_name() + "]";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << shift_.LogParameters() << " wraps " << underlying_.LogParameters();
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    if (!shift_.active()) {
      underlying_.refresh(U, sRNG, pRNG);
      return;
    }
    TXQCDField U_eff(U.U.Grid());
    U_eff = U;
    shift_.Apply(U_eff);
    underlying_.refresh(U_eff, sRNG, pRNG);
  }

  RealD S(const TXQCDField &U) override {
    if (!shift_.active()) return underlying_.S(U);
    TXQCDField U_eff(U.U.Grid());
    U_eff = U;
    shift_.Apply(U_eff);
    return underlying_.S(U_eff);
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    if (!shift_.active()) {
      underlying_.deriv(U, dSdU);
      return;
    }
    TXQCDField U_eff(U.U.Grid());
    U_eff = U;
    shift_.Apply(U_eff);
    underlying_.deriv(U_eff, dSdU);
    shift_.ApplyToForce(dSdU);
  }

 private:
  Action<TXQCDField> &underlying_;
  AuxFierzShift shift_;
};

NAMESPACE_END(Grid);
