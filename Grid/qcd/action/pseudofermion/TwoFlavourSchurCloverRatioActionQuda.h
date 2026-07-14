#pragma once
// Wires a QUDA-backed rung solver (QudaMGSchurSolver or QudaCGSchurSolver --
// see QudaRungSolverBase.h) into TwoFlavourSchurCloverRatioAction's
// DerivativeSolver/ActionSolver slots -- both act on Mpc(DenOp), the rung
// being QUDA-accelerated.  HeatbathSolver acts on Vpc(NumOp) (the heavier
// operator) and defaults to a plain Grid CG -- it must NOT go through the
// DenOp-mass QUDA object above, which is built for a different kappa.
// Optionally, the caller may pass a SEPARATE QudaRungSolverBase (built at
// NumOp's mass) as heatbath_cg to accelerate the heatbath solve too (the
// normal-equation problem HeatbathSolver poses -- solve M_1^dag M_1 tmp = b,
// then apply M_1 -- is exactly what QudaCGSchurSolver already computes, no
// new solver logic needed); heatbath_cg still accepts a plain
// OperatorFunction (Grid CG) as before, detected via dynamic_cast.
// Overrides refresh/S/deriv only to keep the QUDA solver's loaded gauge in
// sync (neither concrete solver has direct access to U through the plain
// OperatorFunction interface, unlike TwoFlavourSchurCloverQudaForceActionMP
// which owns its solver outright).
//
// Held behind QudaRungSolverBase& (not a template parameter) so a driver can
// pick MG vs plain-CG per rung at runtime (env-var driven) without needing
// the concrete type at compile time -- existing callers passing a
// QudaMGSchurSolver& bind unchanged via the base-reference upcast.
//
// Additive; generic over FermionOp like TwoFlavourSchurCloverRatioAction
// itself, so it binds to WilsonCloverFermion (gen_qcd_cfgs_2plus1.cc) and
// CompactWilsonCloverFermion (gen_qcd_hasenbusch_tune_compact_schur.cc)
// without duplication.

#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioAction.h>
#include <Grid/algorithms/iterative/QudaRungSolverBase.h>

namespace Grid {

template <class Impl, class FermionOp = WilsonCloverFermion<Impl, CloverHelpers<Impl>>>
class TwoFlavourSchurCloverRatioActionQuda
    : public TwoFlavourSchurCloverRatioAction<Impl, FermionOp> {
 public:
  typedef TwoFlavourSchurCloverRatioAction<Impl, FermionOp> Base;
  typedef typename Impl::GaugeField GaugeField;

  TwoFlavourSchurCloverRatioActionQuda(
      FermionOp &NumOp, FermionOp &DenOp, QudaRungSolverBase &solver,
      OperatorFunction<typename Base::FermionField> &heatbath_cg)
      : Base(NumOp, DenOp, solver, solver, heatbath_cg), solver_(solver),
        heatbath_quda_(dynamic_cast<QudaRungSolverBase *>(&heatbath_cg)) {}

  void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {
    solver_.SetGauge(U);
    if (heatbath_quda_) heatbath_quda_->SetGauge(U);
    Base::refresh(U, sRNG, pRNG);
  }
  RealD S(const GaugeField &U) override {
    solver_.SetGauge(U);
    return Base::S(U);
  }
  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    solver_.SetGauge(U);
    Base::deriv(U, dSdU);
  }

 private:
  QudaRungSolverBase &solver_;
  QudaRungSolverBase *heatbath_quda_;
};

}  // namespace Grid
