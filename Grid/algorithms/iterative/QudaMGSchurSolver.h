#pragma once
// Reusable QUDA-backed Schur-complement solver:  X = (Mpc^dag Mpc)^-1 src
// for a Grid Odd- or Even-parity SchurDifferentiableOperator, backed by
// QUDA's full-volume M^-1 (QudaCloverInverter).  Works with EITHER
// qp.use_multigrid=true (QUDA-MG, biggest win on near-critical rungs) or
// qp.use_multigrid=false (plain QUDA-CG, still faster than Grid's own CG
// via GPU-side kernels/mixed precision) -- the gamma5 two-solve trick below
// is agnostic to which inner solver QudaCloverInverter runs.
//
// Extracted and generalized (checkerboard is now a constructor argument
// instead of hardcoded EVEN) from the two-solve trick already validated in
// TwoFlavourSchurCloverQudaForceActionMP.h (Phase 7a/7b, memory
// mg-light-force-dagger-g5-fix: cos=1.0, ~13x at 48^3).  That file's header
// comment on TwoFlavourSchurCloverActionEven.h notes MpcDeriv/MpcDagDeriv
// (and, by the same Schur algebra, this zero-padding trick) are
// "parity-agnostic -- same kernel applies on either checkerboard", so the
// EVEN->cb substitution here is mechanical, not a re-derivation.
//
// QUDA's multigrid preconditioner is built for the FORWARD operator M only
// -- flipping inv_param.dagger to solve M^dag stalls (30000 iters, observed
// in the original investigation).  Instead use gamma5-hermiticity,
//   M^dag^-1 = gamma5 . M^-1 . gamma5   (exact for full Wilson-clover),
// so both solves below run on the working forward MG (dagger=NO).
//
// This wraps QUDA purely as an INNER SOLVER -- force assembly stays on
// Grid's own Mpc.MpcDeriv/MpcDagDeriv (via whatever action class injects
// this as its DerivativeSolver/ActionSolver), so none of the QUDA force
// KERNEL machinery (dagger/rescale/mom-buffer unpacking, see
// TwoFlavourSchurCloverQudaForceActionMP.h's deriv()) is needed here -- the
// same low-risk pattern already validated for QUDA_LIGHT (CG) in
// gen_qcd_hasenbusch_tune_compact.cc.

#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <Grid/algorithms/iterative/QudaRungSolverBase.h>
#include <Grid/util/QudaInit.h>

namespace Grid {

class QudaMGSchurSolver : public QudaRungSolverBase {
 public:
  // full_grid: the FULL (unchecked-boarded) fermion grid -- e.g.
  // FermOp.GaugeGrid() / FermOp.FermionGrid() -- NOT the redblack grid.
  // qp: use_multigrid=true routes through QUDA-MG; use_multigrid=false routes
  // through plain QUDA-CG -- both go through the same two-solve trick below.
  // cb: which checkerboard the caller's src/sol fields live on --
  // Grid::Odd/Grid::Even (plain `int`, there is no dedicated Checkerboard
  // type) -- Odd for TwoFlavourSchurCloverRatioAction, Even for
  // TwoFlavourSchurCloverActionEven-style classes.
  //
  // Quda::initialize() is idempotent (static bool guard) and MUST be called
  // before any QudaCloverInverter/loadGaugeQuda -- callers that already build a
  // QUDA strange/light force action get this for free from that action's own
  // constructor, but a driver that skips those (e.g. FORCES_SKIP_STRANGE=1,
  // no QUDA_FORCE_LIGHT) would otherwise never initialize QUDA at all, and
  // loadGaugeQuda fails with "QUDA not initialized".  Call it here so this
  // class is self-sufficient regardless of what else is active.
  QudaMGSchurSolver(GridBase *full_grid, const QudaCloverParams &qp, int cb)
      : full_grid_(full_grid), cb_(cb), inv_(full_grid, qp) {
    Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, full_grid);
  }

  // SHARED-MG mode (chroma's shared-SubspaceID pattern): this instance solves
  // at ITS OWN mass (qp.mass) with a GCR outer, preconditioned by the MG
  // setup the DONOR instance built at its (lighter) mass.  Pass qp with
  // use_multigrid=false -- this instance builds NO MG of its own (zero extra
  // setup time / GPU memory); the donor's handle is re-fetched before every
  // solve so hard-failure rebuilds on the donor are picked up automatically.
  // Correctness is owned by the outer GCR (residual at THIS mass); the mass
  // mismatch to the donor's setup only costs outer iterations, growing with
  // the mass distance -- measure LastIter()/LastSecs() to find the crossover
  // where plain QUDA-CG wins.
  QudaMGSchurSolver(GridBase *full_grid, const QudaCloverParams &qp, int cb,
                    QudaMGSchurSolver *mg_donor)
      : full_grid_(full_grid), cb_(cb), inv_(full_grid, qp), mg_donor_(mg_donor) {
    assert(mg_donor_ != nullptr && "shared-MG constructor needs a donor");
    assert(!qp.use_multigrid &&
           "shared-MG instance must not build its own MG (use_multigrid=false)");
    Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, full_grid);
  }

  void SetGauge(const LatticeGaugeField &U) override { inv_.SetGauge(U); }

  // Forwarded from the inner QudaCloverInverter for driver-side timing/comparison
  // logs.  NOTE: operator() below calls inv_ TWICE (gamma5 two-solve trick) --
  // LastIter()/LastSecs() after a call reflect only the SECOND inner solve, not
  // the sum of both.  LastMgSetupSecs() is the one-time (or thin-update) MG
  // build cost from the most recent SetGauge() call -- 0 if use_multigrid=false.
  int    LastIter()        const { return inv_.LastIter(); }
  double LastSecs()        const { return inv_.LastSecs(); }
  double LastMgSetupSecs() const { return inv_.LastMgSetupSecs(); }

  // Linop is ignored (QUDA inverts against whatever gauge SetGauge last
  // loaded) -- same contract as QudaCloverInverter::operator().
  void operator()(LinearOperatorBase<LatticeFermion> &Linop,
                  const LatticeFermion &src, LatticeFermion &sol) override {
    (void)Linop;

    // Shared-MG mode: install the donor's CURRENT preconditioner handle into
    // our inner inverter and flip its outer solver to GCR (mirrors the
    // use_multigrid branch of QudaCloverInverter::setup_params_).  Re-done
    // every call: the donor's handle changes on hard-failure rebuild /
    // rebuild_every, and the field writes are free next to the solve.
    if (mg_donor_ != nullptr) {
      // Our SetGauge (and any interleaved heatbath inverter) REPLACED QUDA's
      // resident gauge/clover; re-sync the donor's MG to them before
      // borrowing the handle, or its internal operators dereference freed
      // fields (checkParitySpinor abort, 2026-07-09).  No-op when current.
      mg_donor_->inv_.EnsureMgCurrent();
      void *h = mg_donor_->inv_.MgPreconditioner();
      // Donor builds its MG on its first SetGauge; monomials are pushed
      // lightest-first so the donor rung solves before any sharee.  A null
      // handle here is a wiring error (donor never SetGauge'd) -- abort
      // loudly rather than silently degrade to unpreconditioned GCR.
      assert(h != nullptr &&
             "shared-MG solve before donor built its MG (donor SetGauge not called?)");
      QudaInvertParam &ip = inv_.InvertParam();
      ip.inv_type              = QUDA_GCR_INVERTER;
      ip.inv_type_precondition = QUDA_MG_INVERTER;
      ip.solve_type            = QUDA_DIRECT_SOLVE;
      ip.schwarz_type          = QUDA_INVALID_SCHWARZ;
      ip.precondition_cycle    = 1;
      ip.tol_precondition      = 1e-1;
      ip.maxiter_precondition  = 1;
      ip.preconditioner        = h;
    }

    LatticeFermion src_full(full_grid_), y_full(full_grid_), X_full(full_grid_);
    LatticeFermion y_cb(src.Grid());
    y_cb.Checkerboard() = cb_;

    QudaInvertParam &iparam = inv_.InvertParam();
    QudaDagType saved_dagger = iparam.dagger;
    iparam.dagger = QUDA_DAG_NO;
    Gamma g5(Gamma::Algebra::Gamma5);

    // Step 1: y_full = M^dag^-1 . src_full = gamma5 . M^-1 . gamma5 . src_full
    src_full = Zero();
    setCheckerboard(src_full, src);
    LatticeFermion src_g5(full_grid_);
    src_g5 = g5 * src_full;
    inv_(Linop, src_g5, y_full);
    y_full = g5 * y_full;

    // Extract the cb-checkerboard half of y_full, re-embed as a fresh
    // full-volume source with the other half zero.
    pickCheckerboard(cb_, y_cb, y_full);
    src_full = Zero();
    setCheckerboard(src_full, y_cb);

    // Step 2: X_full = M^-1 . src_full
    inv_(Linop, src_full, X_full);
    sol.Checkerboard() = cb_;
    pickCheckerboard(cb_, sol, X_full);

    iparam.dagger = saved_dagger;
  }

 private:
  GridBase *full_grid_;
  int cb_;
  QudaCloverInverter inv_;
  QudaMGSchurSolver *mg_donor_ = nullptr;  // shared-MG mode; null = standalone
};

}  // namespace Grid
