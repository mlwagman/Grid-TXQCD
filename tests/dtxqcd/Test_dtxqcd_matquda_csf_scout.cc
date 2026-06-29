// Style B perf scout — Path A (raw cudaMalloc + MatQuda input_location=CUDA)
// vs Path B (persistent QUDA-native ColorSpinorField + persistent Dirac +
// direct dirac->M).
//
// Hypothesis from Option β data: MatQuda's per-call alloc/repack/Dirac::create
// overhead dominates at 16³×48 mpi=1.1.1.4 (~23 ms/call observed).  Style B
// bypasses that by allocating native CSF + Dirac once and calling dirac->M
// directly.  Gate: Path B per-call ≤ 45 ms → proceed with full Style B
// refactor.  Path B > 60 ms → diagnosis wrong; stop.
//
// Run:
//   ./Test_dtxqcd_matquda_csf_scout --grid 16.16.16.48 --mpi 1.1.1.4

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/util/QudaInit.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Grid;

namespace {

// Wallclock helper.  cudaDeviceSynchronize is required for honest GPU timing
// (MatQuda / dirac->M return as soon as the kernels are queued).
double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

void sync_all() {
  // acceleratorCopySynchronise only syncs the COPY stream — not the compute
  // stream where MatQuda / dirac->M launch their kernels.  Force a full
  // device-wide sync for honest GPU timing.
  cudaDeviceSynchronize();
}

// L2 of (a - b) / L2(a) per element, both as flat double arrays.
double rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double n = 0.0, d = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double dd = a[i] - b[i];
    n += dd * dd;
    d += a[i] * a[i];
  }
  if (d == 0.0) return 0.0;
  return std::sqrt(n / d);
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{16, 16, 16, 48});
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
  pRNG.SeedFixedIntegers({701, 702, 703, 704});

  std::cout << GridLogMessage << "Style B perf scout"
            << "  latt=" << latt
            << "  mpi="  << mpi
            << "  vComplex::Nsimd=" << vComplex::Nsimd() << std::endl;

  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD aux_scale = 1.0 / 3.0;

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  // Zero aux — we only care about per-MatQuda cost, not the aux step.
  LatticeDtxqcdSigma sigma(&Grid);  sigma = Zero();
  LatticeDtxqcdPi    pi(&Grid);     pi    = Zero();
  LatticeDtxqcdD     d_field(&Grid);d_field = Zero();
  LatticeDtxqcdN     n_field(&Grid);n_field = Zero();
  LatticeDtxqcdS     s_field(&Grid);s_field = Zero();
  LatticeDtxqcdP     p_field(&Grid);p_field = Zero();
  (void)aux_scale;

  // Instantiate the production wrapper just to set up gauge+clover via the
  // exact same recipe (anti-periodic-t baked, QUDA QDP gauge order, clover
  // computed in-QUDA from gauge).
  DTXQCDMpcOpQUDA Mop_quda(U, Grid, RBGrid, mass, csw,
                            sigma, pi, d_field, n_field, s_field, p_field);

  int V = Grid::Quda::local_volume(&Grid);
  size_t buf_bytes = 24 * V * sizeof(double);
  std::cout << GridLogMessage << "V_local=" << V
            << "  buf_bytes_per_spinor=" << buf_bytes << std::endl;

  QudaInvertParam &inv_param = Mop_quda.InvertParam();

  // Random input spinor: pack one Grid LatticeFermion into a flat 24·V_local
  // DIRAC_ORDER+EVEN_ODD host buffer (matches production's Path A).
  LatticeFermion v(&Grid);
  gaussian(pRNG, v);

  std::vector<double> h_in(24 * V), h_out_A(24 * V), h_out_B(24 * V);
  Grid::Quda::fermion_to_eo_buffer(v, h_in.data());

  double *d_in  = (double *)acceleratorAllocDevice(buf_bytes);
  double *d_out = (double *)acceleratorAllocDevice(buf_bytes);
  acceleratorCopyToDevice(h_in.data(), d_in, buf_bytes);
  sync_all();

  const int Nwarm = 10;
  const int Ntime = 500;

  // ----------------------------------------------------------------------
  // Path A: raw cudaMalloc buffer + MatQuda(input_location=CUDA), exactly
  // as DTXQCDMpcOpQUDA::M_device does inside Stage B Option β.
  // ----------------------------------------------------------------------
  inv_param.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;
  inv_param.dagger          = QUDA_DAG_NO;

  // Warm up (kernel autotune cache, allocator).
  for (int k = 0; k < Nwarm; ++k) MatQuda(d_out, d_in, &inv_param);
  sync_all();

  double t_A0 = now_ms();
  for (int k = 0; k < Ntime; ++k) MatQuda(d_out, d_in, &inv_param);
  sync_all();
  double t_A1 = now_ms();
  double per_call_A_ms = (t_A1 - t_A0) / Ntime;

  // Save Path A result for bit-equiv check.
  acceleratorCopyFromDevice(d_out, h_out_A.data(), buf_bytes);

  std::cout << GridLogMessage << std::fixed << std::setprecision(3)
            << "Path A (raw cudaMalloc + MatQuda input_location=CUDA): "
            << per_call_A_ms << " ms / call  (" << Ntime << " calls / "
            << (t_A1 - t_A0) << " ms total)" << std::endl;

  // ----------------------------------------------------------------------
  // Path B: persistent QUDA-native ColorSpinorField + persistent Dirac +
  // direct dirac->M.
  // ----------------------------------------------------------------------
  inv_param.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;
  inv_param.dagger          = QUDA_DAG_NO;

  // Match what MatQuda does internally: build a cpuParam wrapping the raw
  // device buffer with the user's specified layout, then a cudaParam in
  // QUDA-native layout.  pc=false here because solution_type=QUDA_MAT_SOLUTION
  // (the DTXQCDMpcOpQUDA setup).
  bool pc = false;
  // gauge_param_.X is int[4]; ctor wants const lat_dim_t& (array<int, QUDA_MAX_DIM>).
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = Mop_quda.GaugeParam().X[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  quda::ColorSpinorParam cpuParam(d_in, inv_param, X_full,
                                  pc, QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField in_ref(cpuParam);  // wraps d_in, DIRAC_ORDER+EVEN_ODD

  quda::ColorSpinorParam cudaParam(cpuParam, inv_param,
                                   QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField in_native(cudaParam);   // FLOAT2/NATIVE layout
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorField out_native(cudaParam);

  // Populate in_native once: ref → native (one D→D format conversion).
  in_native.copy(in_ref);

  // Print field-order finding for the report (numeric — look up in
  // enum_quda.h: 1=NATIVE, 2=SPACE_SPIN_COLOR, 3=SPACE_COLOR_SPIN, ...).
  std::cout << GridLogMessage << "in_ref.FieldOrder()="
            << (int)in_ref.FieldOrder()
            << "  in_native.FieldOrder()=" << (int)in_native.FieldOrder()
            << "  (QUDA_NATIVE_FIELD_ORDER=" << (int)QUDA_NATIVE_FIELD_ORDER
            << ")" << std::endl;

  // Build persistent Dirac.
  quda::DiracParam diracParam;
  quda::setDiracParam(diracParam, &inv_param, pc);
  quda::Dirac *dirac = quda::Dirac::create(diracParam);

  // Warm up.
  for (int k = 0; k < Nwarm; ++k) dirac->M(out_native, in_native);
  sync_all();

  double t_B0 = now_ms();
  for (int k = 0; k < Ntime; ++k) dirac->M(out_native, in_native);
  sync_all();
  double t_B1 = now_ms();
  double per_call_B_ms = (t_B1 - t_B0) / Ntime;

  // Sanity check: insist on sync every iter — if the per-call cost shifts,
  // we know QUDA was batching launches without us syncing.
  double t_Bs0 = now_ms();
  for (int k = 0; k < Ntime; ++k) { dirac->M(out_native, in_native); sync_all(); }
  double t_Bs1 = now_ms();
  double per_call_B_sync_ms = (t_Bs1 - t_Bs0) / Ntime;
  std::cout << GridLogMessage << std::fixed << std::setprecision(3)
            << "Path B with PER-CALL sync (sanity): "
            << per_call_B_sync_ms << " ms / call  (" << Ntime << " calls / "
            << (t_Bs1 - t_Bs0) << " ms total)" << std::endl;

  // Pull Path B result into the raw buffer layout (DIRAC_ORDER+EVEN_ODD on
  // device) so we can compare bytewise to Path A.  This is a one-shot
  // post-loop cost, not part of per-call timing.
  quda::ColorSpinorParam outRefParam = cpuParam;
  outRefParam.v = d_out;
  quda::ColorSpinorField out_ref(outRefParam);
  out_ref.copy(out_native);
  sync_all();
  acceleratorCopyFromDevice(d_out, h_out_B.data(), buf_bytes);

  std::cout << GridLogMessage << std::fixed << std::setprecision(3)
            << "Path B (persistent CSF + persistent Dirac + dirac->M):  "
            << per_call_B_ms << " ms / call  (" << Ntime << " calls / "
            << (t_B1 - t_B0) << " ms total)" << std::endl;

  // ----------------------------------------------------------------------
  // Bit-equiv gate.
  // ----------------------------------------------------------------------
  // Note: MatQuda applies a kappa rescale (line 2429-2441 of interface_quda.cpp);
  // dirac->M does not.  With solution_type=MAT + mass_normalization=MASS, the
  // ax(0.5/kappa) is applied to out in MatQuda but not in our raw dirac->M.
  // We compensate by scaling h_out_B by 0.5/kappa to match.
  double scale = 0.5 / inv_param.kappa;
  for (size_t i = 0; i < h_out_B.size(); ++i) h_out_B[i] *= scale;
  double rel = rel_diff(h_out_A, h_out_B);
  std::cout << GridLogMessage
            << "Path A vs Path B (after kappa-rescale match): rel_diff="
            << std::scientific << std::setprecision(3) << rel
            << "  (gate ≤ 1e-12)" << std::endl;

  bool bit_ok = rel < 1e-12;
  std::cout << GridLogMessage << "BIT-EQUIV: " << (bit_ok ? "PASS" : "FAIL")
            << std::endl;

  // ----------------------------------------------------------------------
  // Verdict.
  // ----------------------------------------------------------------------
  std::cout << GridLogMessage << "----" << std::endl;
  std::cout << GridLogMessage << std::fixed << std::setprecision(3)
            << "SUMMARY  Path A=" << per_call_A_ms << " ms/call"
            << "  Path B=" << per_call_B_ms << " ms/call"
            << "  speedup=" << (per_call_A_ms / per_call_B_ms) << "×"
            << std::endl;

  std::string verdict;
  if (per_call_B_ms <= 45.0)
    verdict = "PROCEED (Path B ≤ 45 ms gate — Style B refactor justified)";
  else if (per_call_B_ms <= 60.0)
    verdict = "MARGINAL (Path B 45-60 ms — risky proceed)";
  else
    verdict = "STOP (Path B > 60 ms — diagnosis wrong; per-call ceiling lives elsewhere)";
  std::cout << GridLogMessage << "VERDICT: " << verdict << std::endl;

  delete dirac;
  acceleratorFreeDevice(d_in);
  acceleratorFreeDevice(d_out);

  Grid::Quda::finalize();
  Grid_finalize();
  return 0;
}
