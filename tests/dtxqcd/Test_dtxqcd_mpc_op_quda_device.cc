// Test_dtxqcd_mpc_op_quda_device: 4⁴ bit-equivalence between
// DTXQCDMpcOpQUDA::M (host-mode pipeline, validated by
// Test_dtxqcd_mpc_op_quda) and DTXQCDMpcOpQUDA::M_device (M-wrap.5a:
// device-resident inner pipeline + Option-α host aux roundtrip).
//
// Gate: per-slot rel diff ≤ 1e-12 between M_host and M_device outputs
// across 10 random DTXQCDFermionDoubled inputs.
//
// Run:
//   ./Test_dtxqcd_mpc_op_quda_device --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/util/QudaInit.h>
#include <iomanip>
#include <iostream>

using namespace Grid;

namespace {
RealD rel_diff(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion diff(a.Grid());
  diff = a - b;
  RealD nb = norm2(b);
  if (nb == 0.0) return 0.0;
  return std::sqrt(norm2(diff) / nb);
}

void randomize_doubled(GridParallelRNG &pRNG, DTXQCDFermionDoubled &v) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.upper.f[a]);
    gaussian(pRNG, v.lower.f[a]);
  }
}
}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({501, 502, 503, 504});

  // Production-matched setup (csw, mass, λ=3 aux amplitude).
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD aux_scale = 1.0 / 3.0;

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d_field(&Grid);
  LatticeDtxqcdN     n_field(&Grid);
  LatticeDtxqcdS     s_field(&Grid);
  LatticeDtxqcdP     p_field(&Grid);

  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s_field);
  DtxqcdRealScalarGaussian(pRNG, p_field);
  DtxqcdHermitianCFGaussian(pRNG, d_field);
  DtxqcdHermitianCFGaussian(pRNG, n_field);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(sigma);
    DtxqcdRealSymmetricCFInPlace(pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, d_field);
    DtxqcdComplexSymmetricCFGaussian(pRNG, n_field);
  }
  sigma   = aux_scale * sigma;
  pi      = aux_scale * pi;
  s_field = aux_scale * s_field;
  p_field = aux_scale * p_field;
  d_field = aux_scale * d_field;
  n_field = aux_scale * n_field;

  // ---------- Wrapper instance ----------------------------------------------
  DTXQCDMpcOpQUDA Mop_quda(U, Grid, RBGrid, mass, csw,
                            sigma, pi, d_field, n_field, s_field, p_field);

  int V = Grid::Quda::local_volume(&Grid);
  size_t buf_bytes = 24 * V * sizeof(double);

  // ---------- 10-spinor M_host vs M_device comparison ------------------------
  const int Ntrials = 10;
  RealD max_rel[4] = {0.0, 0.0, 0.0, 0.0};
  int exitcode = 0;

  for (int t = 0; t < Ntrials; ++t) {
    DTXQCDFermionDoubled v(&Grid), out_host(&Grid), out_device_grid(&Grid);
    randomize_doubled(pRNG, v);

    // Host-mode reference (validated bit-equiv vs Grid in M-wrap.4-v2).
    Mop_quda.M(v, out_host);

    // Pack v -> host EO buffers, allocate device buffers, copy H2D.
    std::array<std::vector<double>, DtxqcdNf> h_in_u, h_in_l, h_out_u, h_out_l;
    double *d_in_u[DtxqcdNf], *d_in_l[DtxqcdNf];
    double *d_out_u[DtxqcdNf], *d_out_l[DtxqcdNf];
    double *d_scratch_l[DtxqcdNf];
    for (int a = 0; a < DtxqcdNf; ++a) {
      h_in_u[a].assign(24 * V, 0.0);
      h_in_l[a].assign(24 * V, 0.0);
      h_out_u[a].assign(24 * V, 0.0);
      h_out_l[a].assign(24 * V, 0.0);
      Grid::Quda::fermion_to_eo_buffer(v.upper.f[a], h_in_u[a].data());
      Grid::Quda::fermion_to_eo_buffer(v.lower.f[a], h_in_l[a].data());
      d_in_u[a]      = (double *)acceleratorAllocDevice(buf_bytes);
      d_in_l[a]      = (double *)acceleratorAllocDevice(buf_bytes);
      d_out_u[a]     = (double *)acceleratorAllocDevice(buf_bytes);
      d_out_l[a]     = (double *)acceleratorAllocDevice(buf_bytes);
      d_scratch_l[a] = (double *)acceleratorAllocDevice(buf_bytes);
      acceleratorCopyToDevice(h_in_u[a].data(), d_in_u[a], buf_bytes);
      acceleratorCopyToDevice(h_in_l[a].data(), d_in_l[a], buf_bytes);
    }

    Mop_quda.M_device(d_in_u, d_in_l, d_out_u, d_out_l, d_scratch_l,
                       /*dagger=*/false);

    // Copy D2H, unpack to Grid layout, free device buffers.
    for (int a = 0; a < DtxqcdNf; ++a) {
      acceleratorCopyFromDevice(d_out_u[a], h_out_u[a].data(), buf_bytes);
      acceleratorCopyFromDevice(d_out_l[a], h_out_l[a].data(), buf_bytes);
      Grid::Quda::eo_buffer_to_fermion(h_out_u[a].data(),
                                        out_device_grid.upper.f[a]);
      Grid::Quda::eo_buffer_to_fermion(h_out_l[a].data(),
                                        out_device_grid.lower.f[a]);
      acceleratorFreeDevice(d_in_u[a]);
      acceleratorFreeDevice(d_in_l[a]);
      acceleratorFreeDevice(d_out_u[a]);
      acceleratorFreeDevice(d_out_l[a]);
      acceleratorFreeDevice(d_scratch_l[a]);
    }

    RealD r_u0 = rel_diff(out_device_grid.upper.f[0], out_host.upper.f[0]);
    RealD r_u1 = rel_diff(out_device_grid.upper.f[1], out_host.upper.f[1]);
    RealD r_l0 = rel_diff(out_device_grid.lower.f[0], out_host.lower.f[0]);
    RealD r_l1 = rel_diff(out_device_grid.lower.f[1], out_host.lower.f[1]);
    max_rel[0] = std::max(max_rel[0], r_u0);
    max_rel[1] = std::max(max_rel[1], r_u1);
    max_rel[2] = std::max(max_rel[2], r_l0);
    max_rel[3] = std::max(max_rel[3], r_l1);

    std::cout << GridLogMessage << "trial " << t << "  rel: "
              << std::scientific << std::setprecision(3)
              << "u0=" << r_u0 << " u1=" << r_u1
              << " l0=" << r_l0 << " l1=" << r_l1 << std::endl;
  }

  std::cout << GridLogMessage << "----" << std::endl;
  std::cout << GridLogMessage << "M_device vs M_host MAX REL: "
            << "upper.f[0]=" << max_rel[0]
            << "  upper.f[1]=" << max_rel[1]
            << "  lower.f[0]=" << max_rel[2]
            << "  lower.f[1]=" << max_rel[3] << std::endl;

  const RealD tol = 1e-12;
  bool ok = (max_rel[0] < tol && max_rel[1] < tol && max_rel[2] < tol && max_rel[3] < tol);
  std::cout << GridLogMessage << "M_device GATE: " << (ok ? "PASS" : "FAIL")
            << "  (tol " << tol << ")" << std::endl;
  if (!ok) exitcode = 1;

  Grid::Quda::finalize();
  Grid_finalize();
  return exitcode;
}
