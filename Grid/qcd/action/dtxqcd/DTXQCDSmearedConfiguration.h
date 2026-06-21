#pragma once
// Stout-smeared configuration for the DTXQCD composite field.
//
// Direct analogue of TXQCDSmearedConfiguration: wraps Grid's
// SmearedConfiguration<PeriodicGimplR> to apply stout smearing to the gauge-link
// component of DTXQCDField while passing the auxiliary fields
// (sigma, pi, d, n, s, p) through unchanged.  The analytic stout force chain
// rule is applied only to the gauge force; aux forces are untouched (the aux
// fields do not enter the stout map).
//
// Usage (mirrors TXQCD):
//   Smear_Stout<PeriodicGimplR> Stout(rho);
//   DTXQCDSmearedConfiguration SmearPolicy(&Grid, Nsmear, Stout);
//   PF.is_smeared = true;  LogDet.is_smeared = true;  Gauge.is_smeared = true;
//   // pass SmearPolicy to the integrator instead of NoSmearing
//
// Nsmear == 0 is an identity (the gauge passes through unchanged), so the same
// policy can drive both smeared production and unsmeared csw=0 scout runs.

#include <Grid/qcd/smearing/GaugeConfiguration.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDSmearedConfiguration
    : public ConfigurationBase<DTXQCDField> {
 public:
  DTXQCDSmearedConfiguration(GridCartesian *UGrid, unsigned int Nsmear,
                             Smear_Stout<PeriodicGimplR> &Stout)
      : Nsmear_(Nsmear),
        gauge_smearing_(UGrid, Nsmear, Stout),
        SmearedField_(UGrid),
        ThinLinks_(nullptr) {}

  void set_Field(DTXQCDField &U) override {
    ThinLinks_ = &U;
    gauge_smearing_.set_Field(U.U);
    SmearedField_ = U;                       // copies gauge + all aux
    if (Nsmear_ > 0)
      SmearedField_.U = gauge_smearing_.get_SmearedU();
  }

  void smeared_force(DTXQCDField &dSdU) override {
    // Stout chain rule on the gauge force only; aux force components untouched.
    gauge_smearing_.smeared_force(dSdU.U);
  }

  DTXQCDField &get_SmearedU() override { return SmearedField_; }

  DTXQCDField &get_U(bool smeared = false) override {
    if (smeared)
      return SmearedField_;
    return *ThinLinks_;
  }

 private:
  unsigned int Nsmear_;
  SmearedConfiguration<PeriodicGimplR> gauge_smearing_;
  DTXQCDField SmearedField_;
  DTXQCDField *ThinLinks_;
};

NAMESPACE_END(Grid);
