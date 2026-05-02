#pragma once
// Drop-in wrapper for Wilson-clover propagator solves that switches between
// Grid's ConjugateGradient<MdagM> (default) and QudaCloverInverter (when
// QUDA_SOLVER=1 in the environment AND the binary was built with --with-quda).
//
// Use:
//   WCF Dw(...); MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
//   QudaPropSolver<WCF> solver(Dw, HermOp, U_smeared,
//                              mass, csw, cg_tol, cg_max);
//   solver.solve(src, x);  // src is the unaugmented source — NOT Mdag·src.
//
// In Grid path: solver internally does Dw.Mdag(src, b); CG(HermOp, b, x).
// In QUDA path: solver calls invertQuda directly on src (M^-1·src ≡ same
// answer modulo solver tol).

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>

#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <memory>
#endif

namespace Grid {

template <class WCF>
class QudaPropSolver {
public:
  QudaPropSolver(WCF &Dw,
                 MdagMLinearOperator<WCF, LatticeFermion> &HermOp,
                 const LatticeGaugeField &Usm,
                 RealD mass, RealD csw,
                 RealD tol, int max_iter,
                 bool anti_periodic_t = true)
    : Dw_(Dw), HermOp_(HermOp), grid_(Usm.Grid()), CG_(tol, max_iter)
  {
#ifdef GRID_HAVE_QUDA
    if (std::getenv("QUDA_SOLVER")) {
      Quda::initialize();
      QudaCloverParams qp;
      qp.mass = mass;
      qp.csw  = csw;
      qp.anti_periodic_t = anti_periodic_t;
      qp.tol = tol;
      qp.max_iter = max_iter;
      qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
      quda_.reset(new QudaCloverInverter(grid_, qp));
      quda_->SetGauge(Usm);
      std::cout << GridLogMessage
                << "[QudaPropSolver] QUDA backend active (mass="
                << mass << ", csw=" << csw << ")" << std::endl;
    }
#endif
  }

  void solve(const LatticeFermion &src, LatticeFermion &x) {
#ifdef GRID_HAVE_QUDA
    if (quda_) {
      (*quda_)(HermOp_, src, x);
      return;
    }
#endif
    LatticeFermion b(grid_);
    Dw_.Mdag(src, b);
    x = Zero();
    CG_(HermOp_, b, x);
  }

private:
  WCF &Dw_;
  MdagMLinearOperator<WCF, LatticeFermion> &HermOp_;
  GridBase *grid_;
  ConjugateGradient<LatticeFermion> CG_;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<QudaCloverInverter> quda_;
#endif
};

}  // namespace Grid
