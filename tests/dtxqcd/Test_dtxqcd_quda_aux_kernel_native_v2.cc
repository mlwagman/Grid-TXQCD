// Style B Path B Session 1 v2 — bit-equiv test for the aux contraction kernel
// running on QUDA NATIVE (FLOAT2, UKQCD basis) layout via FloatNOrder accessor.
//
// Reference: flat-24V `ApplyAuxKernel` from dtxqcd_quda_aux_kernel_device.h
// (already bit-equiv vs Grid-SIMD M-wrap.3 reference).
//
// Test path: pack inputs to flat-24V, csf.copy → native (UKQCD basis), run new
// `ApplyAuxKernelNative` (which applies toRel inline, runs aux math in DR,
// applies toNonRel/2 inline on save), then csf.copy native → host, unpack to
// Grid.  Compare with the reference output per-element.
//
// Gate: max rel diff ≤ 1e-14 across all 4 (transpose_aux, use_dn_conj) combos.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

using namespace Grid;

static RealD rel_diff_fermion(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion d(a.Grid()); d = a - b;
  RealD na = std::sqrt(norm2(a));
  RealD nd = std::sqrt(norm2(d));
  return (na > 0.0) ? nd / na : nd;
}

// Allocate one native ColorSpinorField from a device buffer, using
// inv_param-driven gamma_basis and a CUDA-native FLOAT2 layout.
static quda::ColorSpinorField *make_native_csf(double *d_buf, size_t buf_bytes,
                                               QudaInvertParam &inv_param,
                                               const int X_full_dims[4]) {
  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  QudaInvertParam inv = inv_param;
  inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv.output_location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorParam cpuParam(d_buf, inv, X_full, pc,
                                  QUDA_CUDA_FIELD_LOCATION);
  // Promote to cuda native:
  quda::ColorSpinorParam cudaParam(cpuParam, inv, QUDA_CUDA_FIELD_LOCATION);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  return new quda::ColorSpinorField(cudaParam);
}

// csf.copy from a host flat-24V SSC buffer (interpreted as DR basis) into a
// native CSF.  Returns the temp cpu CSF (caller owns) — keep alive until we're
// done with the csf.copy.
static void copy_flat_to_native(double *d_flat, size_t buf_bytes,
                                quda::ColorSpinorField &native_csf,
                                QudaInvertParam &inv_param,
                                const int X_full_dims[4]) {
  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  QudaInvertParam inv = inv_param;
  inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv.output_location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorParam cpuParam(d_flat, inv, X_full, pc,
                                  QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField cpu_csf(cpuParam);
  native_csf.copy(cpu_csf);
  cudaDeviceSynchronize();
}

static void copy_native_to_flat(quda::ColorSpinorField &native_csf,
                                double *d_flat, size_t buf_bytes,
                                QudaInvertParam &inv_param,
                                const int X_full_dims[4]) {
  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  QudaInvertParam inv = inv_param;
  inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv.output_location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorParam cpuParam(d_flat, inv, X_full, pc,
                                  QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField cpu_csf(cpuParam);
  cpu_csf.copy(native_csf);
  cudaDeviceSynchronize();
}

static int run_case(GridCartesian &grid, GridRedBlackCartesian &rb,
                    GridParallelRNG &pRNG, LatticeGaugeField &U,
                    bool transpose_aux, bool use_dn_conj) {
  std::cout << GridLogMessage << "=== case (transpose_aux="
            << (transpose_aux ? "true" : "false")
            << ", use_dn_conj="
            << (use_dn_conj ? "true" : "false") << ") ===" << std::endl;

  // ---------- Random aux + inputs ----------
  LatticeDtxqcdSigma sigma(&grid);
  LatticeDtxqcdPi    pi   (&grid);
  LatticeDtxqcdD     d    (&grid);
  LatticeDtxqcdN     n    (&grid);
  LatticeDtxqcdS     s    (&grid);
  LatticeDtxqcdP     p    (&grid);
  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdComplexSymmetricCFGaussian(pRNG, d);
  DtxqcdComplexSymmetricCFGaussian(pRNG, n);
  DtxqcdRealScalarGaussian(pRNG, s);
  DtxqcdRealScalarGaussian(pRNG, p);

  DTXQCDFermionDoubled in(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, in.upper.f[a]);
    gaussian(pRNG, in.lower.f[a]);
  }

  // ---------- Reference: flat-24V kernel ----------
  DTXQCDFermionDoubled out_ref(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_ref.upper.f[a] = Zero();
    out_ref.lower.f[a] = Zero();
  }
  DtxqcdQudaAuxKernel::ApplyFusedDtxqcdAuxKernel(
      sigma, pi, d, n, s, p, in, out_ref,
      transpose_aux, use_dn_conj, /*accumulate=*/true);

  int V = Quda::local_volume(&grid);
  std::size_t bytes_spinor = 24 * V * sizeof(double);

  // ---------- Native path setup ----------
  // We need inv_param + gauge_param.  Get them from a DTXQCDMpcOpQUDA wrapper.
  DTXQCDMpcOpQUDA Mop_quda(U, grid, rb, /*mass=*/-0.245, /*csw=*/1.249,
                            sigma, pi, d, n, s, p);
  QudaInvertParam &inv_param = Mop_quda.InvertParam();
  const int X_full_dims[4] = {
      Mop_quda.GaugeParam().X[0], Mop_quda.GaugeParam().X[1],
      Mop_quda.GaugeParam().X[2], Mop_quda.GaugeParam().X[3]};

  // Allocate device buffers for flat input + flat output, used for csf.copy
  // bridging.
  double *d_iu0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_iu1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_il0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_il1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_ou0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_ou1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_ol0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_ol1 = (double*)acceleratorAllocDevice(bytes_spinor);

  // Build 4 native input CSFs and 4 native output CSFs.
  quda::ColorSpinorField *iu0_csf = make_native_csf(d_iu0, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *iu1_csf = make_native_csf(d_iu1, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *il0_csf = make_native_csf(d_il0, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *il1_csf = make_native_csf(d_il1, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *ou0_csf = make_native_csf(d_ou0, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *ou1_csf = make_native_csf(d_ou1, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *ol0_csf = make_native_csf(d_ol0, bytes_spinor,
                                                    inv_param, X_full_dims);
  quda::ColorSpinorField *ol1_csf = make_native_csf(d_ol1, bytes_spinor,
                                                    inv_param, X_full_dims);

  // Pack 4 inputs to flat-24V and csf.copy → native.
  std::vector<double> host_buf(24 * V);
  Quda::fermion_to_eo_buffer(in.upper.f[0], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), d_iu0, bytes_spinor);
  copy_flat_to_native(d_iu0, bytes_spinor, *iu0_csf, inv_param, X_full_dims);

  Quda::fermion_to_eo_buffer(in.upper.f[1], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), d_iu1, bytes_spinor);
  copy_flat_to_native(d_iu1, bytes_spinor, *iu1_csf, inv_param, X_full_dims);

  Quda::fermion_to_eo_buffer(in.lower.f[0], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), d_il0, bytes_spinor);
  copy_flat_to_native(d_il0, bytes_spinor, *il0_csf, inv_param, X_full_dims);

  Quda::fermion_to_eo_buffer(in.lower.f[1], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), d_il1, bytes_spinor);
  copy_flat_to_native(d_il1, bytes_spinor, *il1_csf, inv_param, X_full_dims);

  // Zero outputs (native) — use quda::blas::zero.
  quda::blas::zero(*ou0_csf);
  quda::blas::zero(*ou1_csf);
  quda::blas::zero(*ol0_csf);
  quda::blas::zero(*ol1_csf);
  cudaDeviceSynchronize();

  // ---------- Build perm table (once per case is fine; could cache) ----------
  // Use ou0_csf as the probe target (it's about to be overwritten anyway).
  // Actually build a separate disposable native CSF for the probe so we don't
  // perturb the zero-initialised outputs.
  double *d_probe_scratch = (double*)acceleratorAllocDevice(bytes_spinor);
  quda::ColorSpinorField *probe_csf =
      make_native_csf(d_probe_scratch, bytes_spinor, inv_param, X_full_dims);
  std::vector<int> perm = DtxqcdQudaAuxKernelNativeV2::build_perm_table(
      *probe_csf, inv_param, X_full_dims, &grid);
  delete probe_csf;
  acceleratorFreeDevice(d_probe_scratch);

  // Upload perm to device.
  size_t perm_bytes = perm.size() * sizeof(int);
  int *perm_d = (int*)acceleratorAllocDevice(perm_bytes);
  acceleratorCopyToDevice(perm.data(), perm_d, perm_bytes);
  acceleratorCopySynchronise();

  // ---------- Pack aux into device cache (production EO order, same as flat) ----------
  DtxqcdQudaAuxKernelDevice::DeviceAuxCache aux_cache;
  DtxqcdQudaAuxKernelDevice::allocate_aux_cache(aux_cache, V);
  DtxqcdQudaAuxKernelDevice::pack_aux_to_device(sigma, pi, d, n, s, p,
                                                aux_cache);

  // ---------- Run native kernel ----------
  DtxqcdQudaAuxKernelNativeV2::ApplyAuxKernelNative(
      aux_cache, perm_d,
      *iu0_csf, *iu1_csf, *il0_csf, *il1_csf,
      *ou0_csf, *ou1_csf, *ol0_csf, *ol1_csf,
      transpose_aux, use_dn_conj);
  cudaDeviceSynchronize();

  // ---------- Pull outputs back: csf.copy native → host flat, unpack to Grid ----------
  DTXQCDFermionDoubled out_dev(&grid);
  copy_native_to_flat(*ou0_csf, d_ou0, bytes_spinor, inv_param, X_full_dims);
  acceleratorCopyFromDevice(d_ou0, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.upper.f[0]);

  copy_native_to_flat(*ou1_csf, d_ou1, bytes_spinor, inv_param, X_full_dims);
  acceleratorCopyFromDevice(d_ou1, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.upper.f[1]);

  copy_native_to_flat(*ol0_csf, d_ol0, bytes_spinor, inv_param, X_full_dims);
  acceleratorCopyFromDevice(d_ol0, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.lower.f[0]);

  copy_native_to_flat(*ol1_csf, d_ol1, bytes_spinor, inv_param, X_full_dims);
  acceleratorCopyFromDevice(d_ol1, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.lower.f[1]);

  // ---------- Compare ----------
  RealD max_rel = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    RealD ru = rel_diff_fermion(out_dev.upper.f[a], out_ref.upper.f[a]);
    RealD rl = rel_diff_fermion(out_dev.lower.f[a], out_ref.lower.f[a]);
    std::cout << GridLogMessage << "  upper.f[" << a << "] rel = " << ru
              << "   lower.f[" << a << "] rel = " << rl << std::endl;
    max_rel = std::max({max_rel, ru, rl});
  }
  std::cout << GridLogMessage << "  max_rel = " << max_rel
            << "   (gate ≤ 1e-14)" << std::endl;

  delete iu0_csf; delete iu1_csf; delete il0_csf; delete il1_csf;
  delete ou0_csf; delete ou1_csf; delete ol0_csf; delete ol1_csf;
  acceleratorFreeDevice(d_iu0); acceleratorFreeDevice(d_iu1);
  acceleratorFreeDevice(d_il0); acceleratorFreeDevice(d_il1);
  acceleratorFreeDevice(d_ou0); acceleratorFreeDevice(d_ou1);
  acceleratorFreeDevice(d_ol0); acceleratorFreeDevice(d_ol1);
  acceleratorFreeDevice(perm_d);
  DtxqcdQudaAuxKernelDevice::free_aux_cache(aux_cache);

  return (max_rel <= 1e-14) ? 0 : 1;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt({4, 4, 4, 4});
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--grid") {
      std::vector<int> dd;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) dd.push_back(std::stoi(tok));
      if (dd.size() == (size_t)Nd) latt = Coordinate(dd);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridParallelRNG       pRNG(&Grid_);
  pRNG.SeedFixedIntegers({5001, 5002, 5003, 5004});

  std::cout << GridLogMessage << "Lattice: ";
  for (int d_ = 0; d_ < Nd; ++d_) std::cout << latt[d_] << (d_ + 1 < Nd ? "×" : "");
  std::cout << std::endl;

  // Need a gauge field for DTXQCDMpcOpQUDA setup.
  LatticeGaugeField U(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, U);

  int rc = 0;
  rc |= run_case(Grid_, RBGrid, pRNG, U, /*transpose_aux=*/true,  /*use_dn_conj=*/true);
  rc |= run_case(Grid_, RBGrid, pRNG, U, /*transpose_aux=*/false, /*use_dn_conj=*/false);
  rc |= run_case(Grid_, RBGrid, pRNG, U, /*transpose_aux=*/true,  /*use_dn_conj=*/false);
  rc |= run_case(Grid_, RBGrid, pRNG, U, /*transpose_aux=*/false, /*use_dn_conj=*/true);

  std::cout << GridLogMessage
            << (rc == 0
                  ? "[ok] Style B Path B Session 1 v2 PASS — native aux kernel bit-equiv"
                  : "[FAIL] Style B Path B Session 1 v2 — exceeded 1e-14")
            << std::endl;

  Grid::Quda::finalize();
  Grid_finalize();
  return rc;
}
