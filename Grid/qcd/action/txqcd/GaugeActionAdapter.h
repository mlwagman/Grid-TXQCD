#pragma once
// Adapter: turn an Action<LatticeGaugeField> into an Action<TXQCDField> by
// routing S/deriv/refresh through the gauge slot of the composite Field and
// zeroing the aux slots of the returned force. Lets stock Grid gauge actions
// (WilsonGaugeAction, IwasakiGaugeAction, ...) plug into the TXQCD HMC without
// modification.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

template <class GaugeAction>
class GaugeActionAdapter : public Action<TXQCDField> {
 public:
  template <class... Args>
  explicit GaugeActionAdapter(Args &&...args)
      : inner(std::forward<Args>(args)...) {}

  std::string action_name() override {
    return std::string("GaugeActionAdapter[") + inner.action_name() + "]";
  }

  std::string LogParameters() override { return inner.LogParameters(); }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    inner.refresh(U.U, sRNG, pRNG);
  }

  RealD S(const TXQCDField &U) override { return inner.S(U.U); }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    inner.deriv(U.U, dSdU.U);
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
  }

  GaugeAction &underlying() { return inner; }

 private:
  GaugeAction inner;
};

// Non-owning adapter: wraps any Action<LatticeGaugeField> (e.g. a
// pseudofermion action for a spectator quark) into an Action<TXQCDField>.
class QCDActionAdapter : public Action<TXQCDField> {
 public:
  explicit QCDActionAdapter(Action<LatticeGaugeField> &a) : inner_(a) {}

  std::string action_name() override {
    return std::string("QCDActionAdapter[") + inner_.action_name() + "]";
  }

  std::string LogParameters() override { return inner_.LogParameters(); }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    inner_.refresh(U.U, sRNG, pRNG);
  }

  RealD S(const TXQCDField &U) override { return inner_.S(U.U); }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    inner_.deriv(U.U, dSdU.U);
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
  }

 private:
  Action<LatticeGaugeField> &inner_;
};

NAMESPACE_END(Grid);
