// Phase E.2 cos comparator + timing benchmark: Grid Meooe vs QUDA dslashQuda.
//
// Pre-condition for going further with Phase E (QUDA Dslash inside the TXQCD
// multishift CG): QUDA's parity-projected Wilson hop must give the same
// result as Grid's WilsonFermion::Meooe up to the kappa factor we already
// negotiated for the force primitives.
//
// What this prints:
//   - cos(out_grid, out_quda)   — angle (gate: ≥ 0.99999)
//   - factor = ⟨grid,quda⟩/|quda|^2   — relative scale
//   - Grid Meooe   ms/call  (over 100 reps after warm-up)
//   - QUDA dslashQuda ms/call (over 100 reps after warm-up)
//
// Run: mpirun -np 1 ./bench_dslash --grid 8.8.8.8 --mpi 1.1.1.1
//
// If cos < 0.99999 we either need a different parity convention or have a
// kappa-scaling mismatch — the factor printout disambiguates.

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

#include <quda.h>

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

  // Plain Wilson hop comparison.  Csw set to production value to satisfy
  // QudaCloverInverter's clover-loader requirement; Dslash is off-diagonal
  // only so it never sees the clover term.
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WilsonOp Dw(U, Grid4, RBGrid4, mass, impl_p);

  // Grid Meooe: input on EVEN, output on ODD.
  LatticeFermion src_full(&Grid4);
  random(pRNG, src_full);
  LatticeFermion src_e(&RBGrid4), out_grid_o(&RBGrid4);
  pickCheckerboard(Even, src_e, src_full);
  out_grid_o.Checkerboard() = Odd;
  Dw.Meooe(src_e, out_grid_o);

  // QUDA setup via QudaCloverInverter: it loads gauge with the right anti-
  // periodic time BC, kappa, etc.
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

  // Pack src_e into a host buffer in QUDA EO order (full lattice volume,
  // half lives in EVEN sites).  QUDA dslashQuda expects parity-projected
  // input/output buffers — i.e. half-volume.
  int V = Quda::local_volume(&Grid4);
  std::vector<double> src_buf(V * 24, 0.0);
  std::vector<double> out_buf(V * 24, 0.0);

  // Pack src_e (RB grid, EVEN parity) → buffer in EO order.
  Quda::fermion_rb_to_eo_buffer_half(src_e, 0 /*EVEN*/, src_buf.data());

  // The dslashQuda public API: out = D · in where parity = parity-of-output.
  // Apply to ODD output from EVEN input.
  // For half-volume buffers, dslashQuda expects pointers offset to the
  // appropriate half — but the public signature takes a full buffer of
  // size V*24 and uses parity to select which half is the "input".
  //
  // Following PyQUDA convention: pass full-volume buffer with the EVEN half
  // populated, parity=ODD writes the ODD half.

  // Warm-up + timing.
  inv_param.input_location  = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  std::cout << GridLogMessage << "QUDA Dslash warm-up..." << std::endl;
  for (int i = 0; i < 5; ++i) {
    dslashQuda(out_buf.data(), src_buf.data(), &inv_param, QUDA_ODD_PARITY);
  }
  // Time it.
  const int reps = 100;
  auto t0 = usecond();
  for (int i = 0; i < reps; ++i) {
    dslashQuda(out_buf.data(), src_buf.data(), &inv_param, QUDA_ODD_PARITY);
  }
  auto t1 = usecond();
  double quda_ms_per = double(t1 - t0) * 1e-3 / reps;

  // Time Grid Meooe.
  for (int i = 0; i < 5; ++i) Dw.Meooe(src_e, out_grid_o);
  auto t2 = usecond();
  for (int i = 0; i < reps; ++i) Dw.Meooe(src_e, out_grid_o);
  auto t3 = usecond();
  double grid_ms_per = double(t3 - t2) * 1e-3 / reps;

  // Unpack out_buf (ODD half) into a Grid LatticeFermion on RBGrid4.
  LatticeFermion out_quda_o(&RBGrid4);
  out_quda_o.Checkerboard() = Odd;
  Quda::eo_buffer_half_to_fermion_rb(out_buf.data(), 1 /*ODD*/, out_quda_o);

  // Compare.
  RealD n_grid = norm2(out_grid_o);
  RealD n_quda = norm2(out_quda_o);
  ComplexD ip  = innerProduct(out_grid_o, out_quda_o);
  RealD cos_v  = (n_grid > 0 && n_quda > 0)
                     ? real(ip) / std::sqrt(n_grid * n_quda)
                     : 0.0;
  RealD factor = (n_quda > 0) ? real(ip) / n_quda : 0.0;

  std::cout << GridLogMessage << "===== Dslash benchmark (csw=0) =====" << std::endl;
  std::cout << GridLogMessage << "  |out_grid|² = " << n_grid << std::endl;
  std::cout << GridLogMessage << "  |out_quda|² = " << n_quda << std::endl;
  std::cout << GridLogMessage << "  cos(grid, quda)  = " << cos_v << std::endl;
  std::cout << GridLogMessage << "  factor (⟨g,q⟩/|q|²) = " << factor << std::endl;
  std::cout << GridLogMessage << "  Grid Meooe:    " << grid_ms_per << " ms/call" << std::endl;
  std::cout << GridLogMessage << "  QUDA dslashQuda: " << quda_ms_per << " ms/call" << std::endl;
  std::cout << GridLogMessage << "  speedup: " << grid_ms_per / quda_ms_per << "×" << std::endl;

  Quda::finalize();
  Grid_finalize();
  return 0;
}
