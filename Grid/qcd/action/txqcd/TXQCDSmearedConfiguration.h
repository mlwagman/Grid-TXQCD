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
        ThinLinks_(nullptr) {}

  void set_Field(TXQCDField &U) override {
    ThinLinks_ = &U;
    gauge_smearing_.set_Field(U.U);
    SmearedField_ = U;
    if (Nsmear_ > 0)
      SmearedField_.U = gauge_smearing_.get_SmearedU();
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
};

NAMESPACE_END(Grid);
