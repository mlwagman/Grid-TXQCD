#pragma once
// Stout-smeared configuration for the TXQCD composite field.
//
// Wraps Grid's SmearedConfiguration<PeriodicGimplR> to apply stout smearing
// to the gauge-link component of TXQCDField while passing auxiliary fields
// through unchanged. The force chain rule (analytic stout derivative) is
// applied only to the gauge force; aux forces are unmodified.
//
// Usage:
//   Smear_Stout<PeriodicGimplR> Stout(rho);
//   TXQCDSmearedConfiguration SmearPolicy(&Grid, Nsmear, Stout);
//   // set is_smeared = true on gauge/fermion actions
//   PF.is_smeared = true;  LogDet.is_smeared = true;  Gauge.is_smeared = true;
//   // pass SmearPolicy to integrator instead of NoSmearing

#include <Grid/qcd/smearing/GaugeConfiguration.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class TXQCDSmearedConfiguration
    : public ConfigurationBase<TXQCDField> {
 public:
  TXQCDSmearedConfiguration(GridCartesian *UGrid, unsigned int Nsmear,
                             Smear_Stout<PeriodicGimplR> &Stout)
      : Nsmear_(Nsmear),
        gauge_smearing_(UGrid, Nsmear, Stout),
        SmearedField_(UGrid),
        ThinLinks_(nullptr),
        last_u_norm_(0.0),
        first_set_(true) {}

  void set_Field(TXQCDField &U) override {
    ThinLinks_ = &U;
    // SMEAR-SKIP (2026-05-23, retry): integrator calls set_Field after every
    // Q-step at any nested level, including aux-only Q steps where U is
    // unchanged.  Re-smearing the gauge chain on those calls is wasted work
    // (~30% of traj wallclock at AUX_MULT=4).  Guard the gauge_smearing_
    // .set_Field call with a fast norm2(U.U) check; if U is unchanged the
    // cached SmearedU_ is still valid.
    //
    // First attempt was wrongly blamed for a 40× hermop slowdown that
    // turned out to be unrelated (also seen with the patch reverted).
    // Disable via env TXQCD_SMEAR_ALWAYS=1 if needed for debugging.
    static const bool always_smear = std::getenv("TXQCD_SMEAR_ALWAYS") != nullptr;
    bool gauge_changed = always_smear || first_set_;
    if (!gauge_changed && Nsmear_ > 0) {
      RealD u_norm = norm2(U.U);
      gauge_changed = std::abs(u_norm - last_u_norm_) > 1e-10 * std::max(u_norm, last_u_norm_);
      if (gauge_changed) last_u_norm_ = u_norm;
    }
    if (gauge_changed && Nsmear_ > 0) {
      gauge_smearing_.set_Field(U.U);
      last_u_norm_ = norm2(U.U);
      first_set_ = false;
    }
    SmearedField_ = U;                                   // updates aux components
    if (Nsmear_ > 0)
      SmearedField_.U = gauge_smearing_.get_SmearedU(); // cached smeared U
  }

  void smeared_force(TXQCDField &dSdU) override {
    gauge_smearing_.smeared_force(dSdU.U);
  }

  TXQCDField &get_SmearedU() override { return SmearedField_; }

  TXQCDField &get_U(bool smeared = false) override {
    if (smeared)
      return SmearedField_;
    return *ThinLinks_;
  }

 private:
  unsigned int Nsmear_;
  SmearedConfiguration<PeriodicGimplR> gauge_smearing_;
  TXQCDField SmearedField_;
  TXQCDField *ThinLinks_;
  RealD last_u_norm_;
  bool first_set_;
};

NAMESPACE_END(Grid);
