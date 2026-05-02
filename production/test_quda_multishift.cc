// Phase 4 smoke test: QudaCloverMultiShiftInverter.
//
// Verifies QUDA's multi-shift CG converges per-shift and produces
// distinct solutions for distinct shifts.  The deeper validation
// (compare against Grid's ConjugateGradientMultiShiftMixedPrec
// shift-by-shift) runs in Phase 5 via the force-FD harness.
//
// Run:
//   mpirun -np 1 ./test_quda_multishift --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>

using namespace Grid;
typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  Quda::initialize();

  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  QudaCloverParams qp;
  qp.mass = mass;
  qp.csw  = csw;
  qp.anti_periodic_t = true;
  qp.tol = 1e-9;
  qp.max_iter = 5000;
  qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  // Three test shifts spanning ~2 orders of magnitude.
  QudaCloverMultiShiftSpec spec;
  spec.shifts = {0.05, 0.5, 5.0};
  spec.tols   = {1e-9, 1e-9, 1e-9};

  QudaCloverMultiShiftInverter quda_ms(&Grid4, qp, spec);
  quda_ms.SetGauge(U);

  LatticeFermion src(&Grid4);
  random(pRNG, src);
  std::vector<LatticeFermion> sols(spec.shifts.size(), LatticeFermion(&Grid4));
  for (auto &s : sols) s = Zero();

  // Linop is unused by QUDA inverter but needed by the OperatorMultiFunction
  // signature.  Construct with a no-cost dummy operator.
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  GridRedBlackCartesian RB4(&Grid4);
  WCF D(U, Grid4, RB4, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(D);

  std::cout << GridLogMessage << "Calling QudaCloverMultiShiftInverter on "
            << spec.shifts.size() << " shifts..." << std::endl;
  quda_ms(HermOp, src, sols);

  std::cout << GridLogMessage << "Multishift converged in "
            << quda_ms.LastIter() << " iter, "
            << quda_ms.LastSecs() << " s" << std::endl;

  bool all_pass = true;
  for (int k = 0; k < (int)spec.shifts.size(); ++k) {
    RealD nf = norm2(sols[k]);
    RealD true_res = quda_ms.LastResPerShift()[k];
    bool conv = (true_res < 10 * spec.tols[k]);
    std::cout << GridLogMessage
              << "  shift[" << k << "] = " << spec.shifts[k]
              << "  norm2(x) = " << nf
              << "  true_res = " << true_res
              << (conv ? "  ✓" : "  ✗ FAIL") << std::endl;
    if (!conv) all_pass = false;
  }

  // Sanity: largest shift suppresses solution norm; smallest shift's
  // norm should exceed largest shift's by a clear margin.
  RealD n_small = norm2(sols.front());
  RealD n_large = norm2(sols.back());
  bool monotone = (n_small > n_large);
  std::cout << GridLogMessage
            << "Norm(x[smallest_shift])/Norm(x[largest_shift]) = "
            << std::sqrt(n_small / n_large)
            << (monotone ? "  ✓ (small shift → bigger solution as expected)"
                         : "  ✗ FAIL (norm ordering wrong)")
            << std::endl;

  bool overall = all_pass && monotone;
  std::cout << GridLogMessage << "Phase 4 multishift smoke: "
            << (overall ? "PASS" : "FAIL") << std::endl;

  Quda::finalize();
  Grid_finalize();
  return overall ? 0 : 1;
}
