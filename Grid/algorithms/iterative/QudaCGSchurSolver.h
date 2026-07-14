#pragma once
// Single-shift QUDA-CG solver for (Mpc^dag Mpc)^-1 on Grid's Odd- or
// Even-parity Schur complement -- the plain-CG sibling of QudaMGSchurSolver.
//
// Built on QudaCloverMultiShiftInverter with N=1 shift=0, reusing the
// ASYMMETRIC matpc convention already validated for the strange rational
// action (OneFlavourSchurCloverQudaRationalActionMP.h): ODD_ODD_ASYMMETRIC
// matches Grid's SchurDiagMooeeOperator form M_pc = M_oo - M_oe.M_ee^-1.M_eo
// exactly, whereas the plain "symmetric" ODD_ODD form gives a ~46x wrong
// force there (comment in that file, verified empirically). QudaMGSchurSolver
// cannot be reused for plain CG despite superficially working the same way:
// its inner QudaCloverInverter hardcodes matpc_type=EVEN_EVEN (irrelevant
// for MG's QUDA_DIRECT_SOLVE, since MG handles EO internally -- but very
// much relevant, and wrong, for CG's QUDA_NORMOP_PC_SOLVE).
//
// Unlike QudaMGSchurSolver, no full-volume zero-padding or gamma5 two-solve
// is needed here: QudaCloverMultiShiftInverter::solve_rb_odd/solve_rb_even
// take and return half-volume (checkerboarded) fields directly via
// QUDA_MATPCDAG_MATPC_SOLUTION + QUDA_NORMOP_PC_SOLVE, which is exactly what
// TwoFlavourSchurCloverRatioAction's DerivativeSolver/ActionSolver already
// operate on (Y, X on FermionRedBlackGrid()) -- one QUDA call per solve.

#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/algorithms/iterative/QudaRungSolverBase.h>
#include <Grid/util/QudaInit.h>

namespace Grid {

class QudaCGSchurSolver : public QudaRungSolverBase {
 public:
  // grid: gauge grid (e.g. FermOp.GaugeGrid()), passed to Quda::initialize
  // so this class is self-sufficient even if nothing else in the driver
  // has touched QUDA yet (mirrors QudaMGSchurSolver's own rationale).
  // qp: use_multigrid is forced false regardless of the caller's setting --
  // this class is plain-CG-only by construction.
  // cb: Grid::Odd or Grid::Even -- which checkerboard the caller's Mpc
  // solves on; selects ODD_ODD_ASYMMETRIC vs EVEN_EVEN_ASYMMETRIC.
  QudaCGSchurSolver(GridBase *grid, QudaCloverParams qp, int cb) : cb_(cb) {
    qp.use_multigrid = false;
    QudaCloverMultiShiftSpec spec;
    spec.shifts = {0.0};  // single shift=0 => plain (Mpc^dag Mpc)^-1, no regularization
    spec.matpc_type = (cb_ == Odd) ? QUDA_MATPC_ODD_ODD_ASYMMETRIC
                                    : QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, grid);
    inv_ = std::make_unique<QudaCloverMultiShiftInverter>(grid, qp, spec);
  }

  void SetGauge(const LatticeGaugeField &U) override { inv_->SetGauge(U); }

  void operator()(LinearOperatorBase<LatticeFermion> &Linop,
                  const LatticeFermion &src, LatticeFermion &sol) override {
    (void)Linop;
    std::vector<LatticeFermion> out(1, src.Grid());
    if (cb_ == Odd) {
      inv_->solve_rb_odd(src, out);
    } else {
      inv_->solve_rb_even(src, out);
    }
    sol = out[0];
    last_iter_ = inv_->LastIter();
    last_secs_ = inv_->LastSecs();
  }

  int    LastIter() const { return last_iter_; }
  double LastSecs() const { return last_secs_; }
  double LastMgSetupSecs() const { return 0.0; }  // no MG in this path

 private:
  int cb_;
  std::unique_ptr<QudaCloverMultiShiftInverter> inv_;
  int last_iter_ = 0;
  double last_secs_ = 0.0;
};

}  // namespace Grid
