#pragma once
// Adapter: turn an Action<LatticeGaugeField> into an Action<DTXQCDField> by
// routing S/deriv/refresh through the gauge slot of the composite field and
// zeroing the aux slots of the returned force.  Lets stock Grid gauge actions
// (WilsonGaugeAction, IwasakiGaugeAction, PlaqPlusRectangleAction, ...) plug
// into the DTXQCD HMC without modification.  Mirror of TXQCD's
// GaugeActionAdapter with the aux roster updated for diquark-tensor variant.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

template <class GaugeAction>
class DTXQCDGaugeActionAdapter : public Action<DTXQCDField> {
 public:
  template <class... Args>
  explicit DTXQCDGaugeActionAdapter(Args &&...args)
      : inner(std::forward<Args>(args)...) {}

  std::string action_name() override {
    return std::string("DTXQCDGaugeActionAdapter[") + inner.action_name() + "]";
  }

  std::string LogParameters() override { return inner.LogParameters(); }

  void refresh(const DTXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    inner.refresh(U.U, sRNG, pRNG);
  }

  RealD S(const DTXQCDField &U) override { return inner.S(U.U); }

  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    inner.deriv(U.U, dSdU.U);
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.d     = Zero();
    dSdU.n     = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
  }

  GaugeAction &underlying() { return inner; }

 private:
  GaugeAction inner;
};

// Non-owning adapter: wraps any Action<LatticeGaugeField> (e.g. a Nf=1
// pseudofermion / LogDet action for a spectator strange quark) into an
// Action<DTXQCDField>.  Routes S/deriv/refresh through the gauge slot and zeros
// the six aux force slots (the QCD strange does not depend on σ,π,d,n,s,p).
// Mirror of TXQCD's QCDActionAdapter with the DTXQCD aux roster.  is_smeared is
// inherited from Action<DTXQCDField> and must be set by the caller to match the
// composite SmearedConfiguration (the strange acts on the stout-smeared gauge).
class DTXQCDQCDActionAdapter : public Action<DTXQCDField> {
 public:
  explicit DTXQCDQCDActionAdapter(Action<LatticeGaugeField> &a) : inner_(a) {}

  std::string action_name() override {
    return std::string("DTXQCDQCDActionAdapter[") + inner_.action_name() + "]";
  }

  std::string LogParameters() override { return inner_.LogParameters(); }

  void refresh(const DTXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    inner_.refresh(U.U, sRNG, pRNG);
  }

  RealD S(const DTXQCDField &U) override { return inner_.S(U.U); }

  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    inner_.deriv(U.U, dSdU.U);
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.d     = Zero();
    dSdU.n     = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
  }

 private:
  Action<LatticeGaugeField> &inner_;
};

NAMESPACE_END(Grid);
