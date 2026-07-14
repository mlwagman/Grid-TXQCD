#pragma once
// Hasenbusch ratio rung with QUDA-fused FORCE ASSEMBLY (Path B).
//
// Extends TwoFlavourSchurCloverRatioActionQuda (which swaps only the SOLVER)
// by also routing the deriv's force assembly — the four hopping derivatives
// and eight clover Cmunu derivatives that dominate at ~20 s/monomial at 48³
// — through QUDA's fused force primitives, via the shared per-operator
// engine in Grid/util/QudaSchurOpForce.h.
//
// Path A (Grid, TwoFlavourSchurCloverRatioAction::deriv) computes, after
// X = (Mpc†Mpc)^{-1} Vpc† Phi and Y = Mpc X:
//   dSdU = (A_den_wilson + A_den_sigma) - (A_num_wilson + A_num_sigma)
// where each operator's positive-orientation piece pair is
//   A_op_wilson = Mpc.MpcDeriv(P,X) + Mpc.MpcDagDeriv(X,P)
//   A_op_sigma  = MooDeriv(P,X,No) + MooDeriv(X,P,Yes)
//               + MeeDeriv(Z,W,No) + MeeDeriv(W,Z,Yes)
// with P = Y for DenOp and P = PhiOdd for NumOp.  Path B computes the same
// four pieces on the GPU:  dSdU = F_den - F_num.
//
// Env behavior (defaults = Path B on, since instantiating this class is
// itself opt-in via the driver's HASEN_QUDA_FORCE_RUNGS knob):
//   QUDA_RUNG_FORCE_OFF=1        fall through to the parent (solver-swap +
//                                Grid assembly) — same-binary A/B.
//   QUDA_FORCE_KERNEL_COMPARE=1  run BOTH paths, print per-piece
//                                cos(Ta(A),B)/factor for den/num ×
//                                wilson/sigma + total, RETURN Path A.
//   QUDA_RUNG_FORCE_WILSON_SIGN / QUDA_RUNG_FORCE_SIGMA_SIGN /
//   QUDA_FORCE_SCHUR_SCALE       see QudaSchurOpForce.h.
//
// Residency: the injected rung solver's SetGauge(U) at deriv entry loads
// gauge+clover and refreshes the extended resident gauge; the force calls
// need nothing further (kappa_den vs kappa_num enter only through scalar
// coefficients — see QudaSchurOpForce.h header).  The two
// QudaCloverMultiShiftInverter members are parameter sources only (kappa,
// csw, ODD_ODD_ASYMMETRIC structs), never solved with, never SetGauge'd;
// QUDA_RUNG_FORCE_RELOAD=1 forces den_loader_->SetGauge(U) for residency
// debugging.

#ifdef GRID_HAVE_QUDA

#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverRatioActionQuda.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaSchurOpForce.h>

#include <cstdlib>
#include <memory>

namespace Grid {

template <class Impl, class FermionOp = WilsonCloverFermion<Impl, CloverHelpers<Impl>>>
class TwoFlavourSchurCloverRatioQudaForceAction
    : public TwoFlavourSchurCloverRatioActionQuda<Impl, FermionOp> {
 public:
  typedef TwoFlavourSchurCloverRatioActionQuda<Impl, FermionOp> Base;
  typedef typename Impl::GaugeField GaugeField;
  typedef typename Impl::FermionField FermionField;

  // qp_den / qp_num: QudaCloverParams at DenOp's / NumOp's mass (csw,
  // antiperiodic-t, etc. as for the rung solver).  use_multigrid is forced
  // off — the loaders are never solved with.
  TwoFlavourSchurCloverRatioQudaForceAction(
      FermionOp &NumOp_, FermionOp &DenOp_, QudaRungSolverBase &solver,
      OperatorFunction<FermionField> &heatbath_cg,
      const QudaCloverParams &qp_den, const QudaCloverParams &qp_num)
      : Base(NumOp_, DenOp_, solver, heatbath_cg), solver2_(solver) {
    Quda::initialize(-1, nullptr, DenOp_.GaugeGrid());
    QudaCloverMultiShiftSpec spec;
    spec.shifts = {0.0};
    spec.tols   = {1e-8};
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
    QudaCloverParams pd = qp_den;  pd.use_multigrid = false;
    QudaCloverParams pn = qp_num;  pn.use_multigrid = false;
    den_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        DenOp_.GaugeGrid(), pd, spec);
    num_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        NumOp_.GaugeGrid(), pn, spec);
  }

  std::string action_name() override {
    return Base::action_name() + " [QUDA force assembly]";
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    if (std::getenv("QUDA_RUNG_FORCE_OFF") != nullptr) {
      Base::deriv(U, dSdU);  // parent does SetGauge + Grid assembly
      return;
    }
    const bool compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;

    solver2_.SetGauge(U);  // loads gauge+clover, refreshes extended gauge
    this->NumOp.ImportGauge(U);
    this->DenOp.ImportGauge(U);
    if (std::getenv("QUDA_RUNG_FORCE_RELOAD") != nullptr)
      den_loader_->SetGauge(U);

    GridBase *fcbgrid = this->NumOp.FermionRedBlackGrid();
    SchurDifferentiableOperator<Impl> Mpc(this->DenOp);
    SchurDifferentiableOperator<Impl> Vpc(this->NumOp);

    // Solve exactly as the Grid base (TwoFlavourSchurCloverRatioAction
    // lines 157-160) — the injected rung solver keeps the MG/CG assignment.
    FermionField X(fcbgrid), Y(fcbgrid);
    Vpc.MpcDag(this->PhiOdd, Y);
    X = Zero();
    this->DerivativeSolver(Mpc, Y, X);
    Mpc.Mpc(X, Y);

    // Path B: per-operator fused force.  DenOp pairs (X, Y); NumOp pairs
    // (X, PhiOdd) — Phi plays Y's role in the numerator terms.
    GaugeField Fdw(dSdU.Grid()), Fds(dSdU.Grid());
    GaugeField Fnw(dSdU.Grid()), Fns(dSdU.Grid());
    Quda::computeSchurOpForceQuda(this->DenOp, X, Y, *den_loader_,
                                  Fdw, Fds, scratch_);
    Quda::computeSchurOpForceQuda(this->NumOp, X, this->PhiOdd, *num_loader_,
                                  Fnw, Fns, scratch_);
    dSdU = Fdw + Fds - Fnw - Fns;

    if (std::getenv("QUDA_RUNG_FORCE_VERBOSE") != nullptr)
      scratch_.Print(action_name());

    if (!compare) return;

    // COMPARE: Path A recomputed inline from the SAME X, Y (no re-solve),
    // split per piece.  Returns Path A's force.
    GaugeField A(dSdU.Grid()), A_total(dSdU.Grid()), t(dSdU.Grid());
    FermionField W(fcbgrid), Z(fcbgrid), t1(fcbgrid);

    Mpc.MpcDeriv(t, Y, X);              A = t;
    Mpc.MpcDagDeriv(t, X, Y);           A = A + t;
    Quda::CompareForcePiece(action_name() + " den.wilson", A, Fdw);
    A_total = A;

    this->DenOp.MooDeriv(t, Y, X, DaggerNo);   A = t;
    this->DenOp.MooDeriv(t, X, Y, DaggerYes);  A = A + t;
    this->DenOp.Meooe(X, t1);     this->DenOp.MooeeInv(t1, W);
    this->DenOp.MeooeDag(Y, t1);  this->DenOp.MooeeInvDag(t1, Z);
    this->DenOp.MeeDeriv(t, Z, W, DaggerNo);   A = A + t;
    this->DenOp.MeeDeriv(t, W, Z, DaggerYes);  A = A + t;
    Quda::CompareForcePiece(action_name() + " den.sigma", A, Fds);
    A_total = A_total + A;

    Vpc.MpcDagDeriv(t, X, this->PhiOdd);  A = t;
    Vpc.MpcDeriv(t, this->PhiOdd, X);     A = A + t;
    Quda::CompareForcePiece(action_name() + " num.wilson", A, Fnw);
    A_total = A_total - A;

    this->NumOp.MooDeriv(t, this->PhiOdd, X, DaggerNo);   A = t;
    this->NumOp.MooDeriv(t, X, this->PhiOdd, DaggerYes);  A = A + t;
    this->NumOp.Meooe(X, t1);                this->NumOp.MooeeInv(t1, W);
    this->NumOp.MeooeDag(this->PhiOdd, t1);  this->NumOp.MooeeInvDag(t1, Z);
    this->NumOp.MeeDeriv(t, Z, W, DaggerNo);   A = A + t;
    this->NumOp.MeeDeriv(t, W, Z, DaggerYes);  A = A + t;
    Quda::CompareForcePiece(action_name() + " num.sigma", A, Fns);
    A_total = A_total - A;

    Quda::CompareForcePiece(action_name() + " total", A_total, dSdU);
    dSdU = A_total;
  }

  ~TwoFlavourSchurCloverRatioQudaForceAction() {
    scratch_.Print(this->action_name());
  }

 private:
  QudaRungSolverBase &solver2_;  // Base's solver_ is private; keep our own ref
  std::unique_ptr<QudaCloverMultiShiftInverter> den_loader_, num_loader_;
  Quda::QudaSchurForceScratch scratch_;
};

}  // namespace Grid

#endif  // GRID_HAVE_QUDA
