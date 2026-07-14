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
      // Pass the gauge grid so QUDA inherits Grid's MPI comm + rank map (MPI build).
      Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, grid_);
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

  // Multi-source: solve M·x_i = src_i for i = 0..N-1, all on the same gauge.
  // QUDA path uses invertMultiSrcQuda which shares the operator setup across
  // sources for ~2-3× speedup over sequential invertQuda calls (M.3).
  // Grid path falls back to a sequential loop calling solve().
  void solve_multi(const std::vector<LatticeFermion> &src,
                   std::vector<LatticeFermion> &x) {
    int N = (int)src.size();
    GRID_ASSERT(N > 0);
    GRID_ASSERT((int)x.size() == N);

#ifdef GRID_HAVE_QUDA
    if (quda_) {
      // Pack all sources into EO-host buffers, invert as a batch, unpack.
      int V = Quda::local_volume(grid_);
      std::vector<std::vector<double>> src_eo(N, std::vector<double>(24 * V));
      std::vector<std::vector<double>> sol_eo(N, std::vector<double>(24 * V, 0.0));
      std::vector<void*> src_ptrs(N), sol_ptrs(N);

      for (int i = 0; i < N; ++i) {
        Quda::fermion_to_eo_buffer(src[i], src_eo[i].data());
        src_ptrs[i] = src_eo[i].data();
        sol_ptrs[i] = sol_eo[i].data();
      }

      QudaInvertParam &ip = quda_->InvertParam();
      // QUDA convention for multi-src: tell invert how many sources via num_src.
      ip.num_src = N;
      ip.num_src_per_sub_partition = N;  // single MPI rank → all on this rank

      invertMultiSrcQuda(sol_ptrs.data(), src_ptrs.data(), &ip);

      for (int i = 0; i < N; ++i) {
        Quda::eo_buffer_to_fermion(sol_eo[i].data(), x[i]);
      }

      std::cout << GridLogMessage << "[QudaPropSolver::solve_multi] N=" << N
                << " batched invert done" << std::endl;
      return;
    }
#endif
    // Grid fallback: sequential loop.
    for (int i = 0; i < N; ++i) solve(src[i], x[i]);
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
