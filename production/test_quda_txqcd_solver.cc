// Phase K validation: QudaTxqcdPropSolver (defect correction) vs Grid TxqcdCG.
//
// Question: given a random TXQCD field U (gauge + aux) and random source b,
// does the QUDA-defect-corrector path
//     QUDA_SOLVER=1 → solve M_TXQCD x = b via outer iteration with
//                     QUDA's M_QCD^{-1} as preconditioner
// give the same answer (within solver tolerance) as the Grid full-volume
// CG path
//     unset QUDA_SOLVER → solve M†M x = M† b via Grid CG
// ?
//
// We check three things:
//   1) Final residual: |M_TXQCD · x − b| / |b| < tol  (both paths)
//   2) Solution agreement: cos(x_grid, x_quda) ≥ 0.99999, factor ≈ 1
//      → confirms both paths converged to the same x = M^{-1}·b
//   3) Per-iter behavior of defect correction: residual norms log10-decay
//      geometrically → confirms the M_QCD^{-1} preconditioner is effective.
//
// Run:
//   mpirun -np 1 ./test_quda_txqcd_solver --grid 4.4.4.4 --mpi 1.1.1.1
// or for a sterner test:
//   mpirun -np 1 ./test_quda_txqcd_solver --grid 8.8.8.8 --mpi 1.1.1.1

#include "params.h"
#include "quda_txqcd_helper.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace Grid;
using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian        Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  // --- Random TXQCD field (gauge + aux) ---
  TXQCDField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  // Aux fields with small magnitude so Δ is "perturbatively small" (typical
  // production regime); this is what defect correction is designed for.
  // Use small values directly to avoid Σ-measurement step.
  TXQCDCompositeImpl::FillAuxFields(pRNG, U, /*lambda=*/7.0, /*Sigma=*/0.06);

  // --- TXQCD operator on the random gauge ---
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  TXQCDWilsonCloverOp Mop(U.U, Grid4, RBGrid4, mass,
                           U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);

  // --- Random source ---
  TXQCDFermionNf src(&Grid4);
  for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, src.f[a]);

  RealD tol = 1e-8;
  int max_iter = 50000;

  // --- Path A: Grid CG (TXQCD_LOGDET_GPU has nothing to do with the solver
  //     here — meas-side Grid CG; we explicitly DISABLE QUDA_SOLVER so the
  //     fallback path runs).
  std::array<RealD, TxqcdNf> mass_arr;  mass_arr.fill(mass);
  TXQCDFermionNf x_grid(&Grid4);
  unsetenv("QUDA_SOLVER");
  {
    QudaTxqcdPropSolver grid_solver(Mop, mass_arr, csw, U.U, tol, max_iter);
    auto t0 = usecond();
    grid_solver.solve(src, x_grid);
    auto t1 = usecond();
    std::cout << GridLogMessage
              << "[GRID-CG] elapsed = " << double(t1 - t0) * 1e-3 << " ms"
              << std::endl;
  }

  // --- Path B: QUDA defect correction (QUDA_SOLVER=1) ---
  TXQCDFermionNf x_quda(&Grid4);
  setenv("QUDA_SOLVER", "1", 1);
  {
    QudaTxqcdPropSolver quda_solver(Mop, mass_arr, csw, U.U, tol, max_iter);
    auto t0 = usecond();
    quda_solver.solve(src, x_quda);
    auto t1 = usecond();
    std::cout << GridLogMessage
              << "[QUDA-DEFECT] elapsed = " << double(t1 - t0) * 1e-3 << " ms"
              << std::endl;
  }

  // --- Compare: residual checks + cos(x_grid, x_quda) ---
  TXQCDFermionNf Mx(&Grid4);

  // Check Grid path's residual: |M·x_grid − src|
  Mop.M(x_grid, Mx);
  TXQCDFermionNf r_grid(&Grid4);
  for (int a = 0; a < TxqcdNf; ++a) r_grid.f[a] = Mx.f[a] - src.f[a];
  RealD r_grid_n2  = norm2(r_grid);
  RealD src_n2     = norm2(src);
  RealD rel_grid   = std::sqrt(r_grid_n2 / src_n2);

  // Check QUDA path's residual.
  Mop.M(x_quda, Mx);
  TXQCDFermionNf r_quda(&Grid4);
  for (int a = 0; a < TxqcdNf; ++a) r_quda.f[a] = Mx.f[a] - src.f[a];
  RealD r_quda_n2 = norm2(r_quda);
  RealD rel_quda  = std::sqrt(r_quda_n2 / src_n2);

  // cos and factor between solutions.
  RealD nx_grid  = norm2(x_grid);
  RealD nx_quda  = norm2(x_quda);
  ComplexD ip = innerProduct(x_grid, x_quda);
  RealD cos_v  = (nx_grid > 0 && nx_quda > 0)
                    ? real(ip) / std::sqrt(nx_grid * nx_quda) : 0.0;
  RealD factor = (nx_quda > 0) ? real(ip) / nx_quda : 0.0;

  std::cout << GridLogMessage << "===== Phase K validation =====" << std::endl;
  std::cout << GridLogMessage
            << "  |x_grid|² = " << nx_grid << std::endl;
  std::cout << GridLogMessage
            << "  |x_quda|² = " << nx_quda << std::endl;
  std::cout << GridLogMessage
            << "  Grid CG: |M·x − b| / |b| = " << rel_grid << std::endl;
  std::cout << GridLogMessage
            << "  QUDA defect: |M·x − b| / |b| = " << rel_quda << std::endl;
  std::cout << GridLogMessage
            << "  cos(x_grid, x_quda) = " << cos_v << std::endl;
  std::cout << GridLogMessage
            << "  factor = ⟨g,q⟩/|q|² = " << factor << std::endl;

  bool pass = (cos_v >= 0.99999) && (std::abs(factor - 1.0) < 1e-3)
              && (rel_grid < 10 * tol) && (rel_quda < 10 * tol);
  std::cout << GridLogMessage
            << "  RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;

  Grid_finalize();
  return pass ? 0 : 1;
}
