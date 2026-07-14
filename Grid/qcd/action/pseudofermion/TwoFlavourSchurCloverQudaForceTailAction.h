#pragma once
// Schur tail determinant with QUDA-fused FORCE ASSEMBLY (Path B).
//
// The tail analog of TwoFlavourSchurCloverRatioQudaForceAction.h: a single
// operator, so ONE call into the shared per-operator engine
// (Grid/util/QudaSchurOpForce.h) with the determinant pairing (X, Y=Mpc·X).
// Path A (TwoFlavourSchurCloverAction::deriv, lines 83-124) accumulates all
// terms with POSITIVE sign and no trailing flip, so Path B is simply
//   dSdU = F_wilson + F_sigma.
//
// Constructor mirrors the driver-local TwoFlavourSchurCloverActionQudaTail
// (QUDA-CG deriv/action solvers behind QudaRungSolverBase, SetGauge kept in
// sync per solve) plus the QudaCloverParams needed for the force-call
// parameter source.  Env knobs identical to the ratio class:
// QUDA_RUNG_FORCE_OFF, QUDA_FORCE_KERNEL_COMPARE, sign/scale knobs.

#ifdef GRID_HAVE_QUDA

#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/algorithms/iterative/QudaRungSolverBase.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaSchurOpForce.h>

#include <cstdlib>
#include <memory>

namespace Grid {

template <class ImplD, class FermOpD_>
class TwoFlavourSchurCloverQudaForceTailAction
    : public TwoFlavourSchurCloverAction<ImplD, FermOpD_> {
 public:
  typedef TwoFlavourSchurCloverAction<ImplD, FermOpD_> Base;
  typedef typename ImplD::GaugeField GaugeField;
  typedef typename ImplD::FermionField FermionField;

  TwoFlavourSchurCloverQudaForceTailAction(typename Base::FermionOperator &opD,
                                           QudaRungSolverBase &DS,
                                           QudaRungSolverBase &AS,
                                           const QudaCloverParams &qp)
      : Base(opD, DS, AS), quda_ds_(DS), quda_as_(AS) {
    Quda::initialize(-1, nullptr, opD.GaugeGrid());
    QudaCloverMultiShiftSpec spec;
    spec.shifts = {0.0};
    spec.tols   = {1e-8};
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
    QudaCloverParams p = qp;  p.use_multigrid = false;
    loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        opD.GaugeGrid(), p, spec);
  }

  std::string action_name() override {
    return Base::action_name() + " [QUDA force assembly]";
  }

  RealD S(const GaugeField &U) override {
    quda_as_.SetGauge(U);
    return Base::S(U);
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    quda_ds_.SetGauge(U);  // loads gauge+clover, refreshes extended gauge
    if (std::getenv("QUDA_RUNG_FORCE_OFF") != nullptr) {
      Base::deriv(U, dSdU);  // QUDA solve, Grid assembly
      return;
    }
    const bool compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;

    this->FermOp.ImportGauge(U);
    GridBase *fcbgrid = this->FermOp.FermionRedBlackGrid();
    SchurDifferentiableOperator<ImplD> Mpc(this->FermOp);

    FermionField X(fcbgrid), Y(fcbgrid);
    X = Zero();
    this->DerivativeSolver(Mpc, this->PhiOdd, X);
    Mpc.Mpc(X, Y);

    GaugeField Fw(dSdU.Grid()), Fs(dSdU.Grid());
    Quda::computeSchurOpForceQuda(this->FermOp, X, Y, *loader_,
                                  Fw, Fs, scratch_);
    dSdU = Fw + Fs;

    if (std::getenv("QUDA_RUNG_FORCE_VERBOSE") != nullptr)
      scratch_.Print(action_name());

    if (!compare) return;

    // COMPARE: Path A pieces from the SAME X, Y (base lines 98-123).
    GaugeField A(dSdU.Grid()), A_total(dSdU.Grid()), t(dSdU.Grid());
    FermionField W(fcbgrid), Z(fcbgrid), t1(fcbgrid);

    Mpc.MpcDeriv(t, Y, X);              A = t;
    Mpc.MpcDagDeriv(t, X, Y);           A = A + t;
    Quda::CompareForcePiece(action_name() + " wilson", A, Fw);
    A_total = A;

    this->FermOp.MooDeriv(t, Y, X, DaggerNo);   A = t;
    this->FermOp.MooDeriv(t, X, Y, DaggerYes);  A = A + t;
    this->FermOp.Meooe(X, t1);     this->FermOp.MooeeInv(t1, W);
    this->FermOp.MeooeDag(Y, t1);  this->FermOp.MooeeInvDag(t1, Z);
    this->FermOp.MeeDeriv(t, Z, W, DaggerNo);   A = A + t;
    this->FermOp.MeeDeriv(t, W, Z, DaggerYes);  A = A + t;
    Quda::CompareForcePiece(action_name() + " sigma", A, Fs);
    A_total = A_total + A;

    Quda::CompareForcePiece(action_name() + " total", A_total, dSdU);
    dSdU = A_total;
  }

  ~TwoFlavourSchurCloverQudaForceTailAction() {
    scratch_.Print(this->action_name());
  }

 private:
  QudaRungSolverBase &quda_ds_;
  QudaRungSolverBase &quda_as_;
  std::unique_ptr<QudaCloverMultiShiftInverter> loader_;
  Quda::QudaSchurForceScratch scratch_;
};

}  // namespace Grid

#endif  // GRID_HAVE_QUDA
