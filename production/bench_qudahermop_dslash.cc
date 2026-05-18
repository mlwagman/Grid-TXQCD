// Phase I.1 bench: QUDA internal dirac->Dslash (device-resident) vs Grid Meooe.
//
// Public dslashQuda was 15× SLOWER than Grid (per bench_dslash) due to
// per-call host-buffer roundtrip.  Here we call the internal C++ API
// dirac->Dslash(out, in, parity) on device-resident ColorSpinorFields —
// the same code path computeCloverForceWithGridY uses internally.
//
// Gate for proceeding with Phase I (TXQCD hermop using QUDA Dslash):
//   QUDA internal Dslash ≤ 0.25 ms/call on 16³×48 (≥40% faster than
//   Grid Meooe at ~0.42 ms).  If not, Grid's Wilson hop is the floor.
//
// Run: mpirun -np 1 ./bench_qudahermop_dslash --grid 16.16.16.48 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

#include <quda.h>

// Internal QUDA C++ headers (same set used by QudaForcePrimitives.h).
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dirac_quda.h>

extern ::quda::GaugeField  *::gaugePrecise;
extern ::quda::CloverField *::cloverPrecise;

using namespace Grid;

typedef WilsonFermion<WilsonImplR> WilsonOp;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  Quda::initialize();

  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WilsonOp Dw(U, Grid4, RBGrid4, mass, impl_p);

  // Grid Meooe baseline.
  LatticeFermion src_full(&Grid4);
  random(pRNG, src_full);
  LatticeFermion src_e(&RBGrid4), out_grid_o(&RBGrid4);
  pickCheckerboard(Even, src_e, src_full);
  out_grid_o.Checkerboard() = Odd;
  Dw.Meooe(src_e, out_grid_o);

  // QUDA setup.
  QudaCloverParams qp;
  qp.mass = mass;
  qp.csw  = csw;
  qp.anti_periodic_t = true;
  qp.tol = 1e-10;
  qp.max_iter = 5000;
  qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  QudaCloverInverter quda_loader(&Grid4, qp);
  quda_loader.SetGauge(U);

  QudaInvertParam &inv_param = quda_loader.InvertParam();
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.dagger     = QUDA_DAG_NO;
  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  // Pack src_e (RB EVEN) → host buffer for one-time CSF load.
  int V_eo = Quda::local_volume(&Grid4) / 2;
  std::vector<double> src_buf(24 * V_eo, 0.0);
  {
    using SiteSpinor = typename LatticeFermion::scalar_object;
    std::vector<SiteSpinor> sv;
    unvectorizeToLexOrdArray(sv, src_e);
    std::memcpy(src_buf.data(), sv.data(), V_eo * 24 * sizeof(double));
  }

  // ----- Internal QUDA Dirac setup (mirrors QudaForcePrimitives.h:175). -----
  using namespace ::quda;
  if (!::gaugePrecise) {
    std::cerr << "ERROR: ::gaugePrecise not loaded — QudaCloverInverter::SetGauge\n"
              << "did not populate libquda's resident gauge.  Aborting.\n";
    return 1;
  }

  // Build a fParam to derive the spinor params from (parity-projected).
  // Use the public gauge_param.X for lattice dims.
  QudaGaugeParam gauge_param = quda_loader.GaugeParam();
  GaugeFieldParam gParam(gauge_param, nullptr, QUDA_GENERAL_LINKS);

  ColorSpinorParam qParam(nullptr, inv_param, gParam.x, /*pc=*/false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(gParam.Precision(), gParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, /*pc_solve=*/true);
  Dirac *dirac = Dirac::create(diracParam);

  // Allocate device-resident full-volume CSFs.
  ColorSpinorField x_dev(qParam);
  ColorSpinorField y_dev(qParam);

  // Load src_e (EVEN parity) into x_dev[EVEN].
  {
    ColorSpinorParam cpuParam(src_buf.data(), inv_param, gParam.x, /*pc=*/true,
                              QUDA_CPU_FIELD_LOCATION);
    ColorSpinorField cpuSrc(cpuParam);
    x_dev[QUDA_EVEN_PARITY] = cpuSrc;
  }

  // ----- Warm-up + timing. -----
  std::cout << GridLogMessage
            << "QUDA internal Dslash warm-up + timing on lattice "
            << latt_size[0] << "." << latt_size[1] << "." << latt_size[2] << "."
            << latt_size[3] << "..." << std::endl;

  for (int i = 0; i < 10; ++i) {
    dirac->Dslash(y_dev[QUDA_ODD_PARITY], x_dev[QUDA_EVEN_PARITY], QUDA_ODD_PARITY);
  }
  // Synchronize to ensure warm-up actually ran before we start the timer.
  qudaDeviceSynchronize();

  const int reps = 1000;
  auto t0 = usecond();
  for (int i = 0; i < reps; ++i) {
    dirac->Dslash(y_dev[QUDA_ODD_PARITY], x_dev[QUDA_EVEN_PARITY], QUDA_ODD_PARITY);
  }
  qudaDeviceSynchronize();
  auto t1 = usecond();
  double quda_internal_ms = double(t1 - t0) * 1e-3 / reps;

  // Time Grid Meooe.
  for (int i = 0; i < 10; ++i) Dw.Meooe(src_e, out_grid_o);
  auto t2 = usecond();
  for (int i = 0; i < reps; ++i) Dw.Meooe(src_e, out_grid_o);
  auto t3 = usecond();
  double grid_ms = double(t3 - t2) * 1e-3 / reps;

  // ----- Correctness sanity check (cos vs Grid). -----
  // Pull QUDA's result back to host and into a Grid RB Lattice.
  std::vector<double> out_buf(24 * V_eo, 0.0);
  {
    ColorSpinorParam cpuParam(out_buf.data(), inv_param, gParam.x, /*pc=*/true,
                              QUDA_CPU_FIELD_LOCATION);
    ColorSpinorField cpuOut(cpuParam);
    cpuOut = y_dev[QUDA_ODD_PARITY];
  }
  LatticeFermion out_quda_o(&RBGrid4);
  out_quda_o.Checkerboard() = Odd;
  {
    using SiteSpinor = typename LatticeFermion::scalar_object;
    std::vector<SiteSpinor> sv(V_eo);
    std::memcpy(sv.data(), out_buf.data(), V_eo * 24 * sizeof(double));
    vectorizeFromLexOrdArray(sv, out_quda_o);
  }

  RealD n_grid = norm2(out_grid_o);
  RealD n_quda = norm2(out_quda_o);
  ComplexD ip  = innerProduct(out_grid_o, out_quda_o);
  RealD cos_v  = (n_grid > 0 && n_quda > 0)
                     ? real(ip) / std::sqrt(n_grid * n_quda) : 0.0;
  RealD factor = (n_quda > 0) ? real(ip) / n_quda : 0.0;

  std::cout << GridLogMessage << "===== QUDA internal Dslash benchmark =====" << std::endl;
  std::cout << GridLogMessage << "  |out_grid|² = " << n_grid << std::endl;
  std::cout << GridLogMessage << "  |out_quda|² = " << n_quda << std::endl;
  std::cout << GridLogMessage << "  cos(grid, quda)  = " << cos_v << std::endl;
  std::cout << GridLogMessage << "  factor           = " << factor << std::endl;
  std::cout << GridLogMessage << "  Grid Meooe:           " << grid_ms << " ms/call" << std::endl;
  std::cout << GridLogMessage << "  QUDA internal Dslash: " << quda_internal_ms << " ms/call" << std::endl;
  std::cout << GridLogMessage << "  speedup: " << grid_ms / quda_internal_ms << "×" << std::endl;
  std::cout << GridLogMessage << "  GATE: QUDA ≤ 0.25 ms (40% faster than Grid)" << std::endl;
  std::cout << GridLogMessage << "  RESULT: "
            << (quda_internal_ms <= 0.25 ? "PASS — proceed to Phase I.2"
                                          : "FAIL — Grid Wilson hop is the floor; shelve")
            << std::endl;

  delete dirac;
  Quda::finalize();
  Grid_finalize();
  return 0;
}
