// Style B Path B Session 2 — bit-equiv test for PreMatLowerKernel ported to
// QUDA NATIVE/FLOAT2 layout via FloatNOrder accessor.
//
// Reference: flat-24V `Quda::PreMatLowerKernel` from dtxqcd_quda_apply_C.h.
// Test path: pack input to flat-24V, csf.copy → native (UKQCD basis), run new
// `ApplyPreMatLowerNative` (which does UKQCD→DR via toRel, DR→UKQCD via
// toNonRelHalf), then csf.copy native → host, unpack to Grid.  Compare with
// the reference output per-element.
//
// Gate: max rel diff ≤ 1e-14.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_apply_C.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

using namespace Grid;

static RealD rel_diff_buf(const std::vector<double> &a,
                          const std::vector<double> &b) {
  double na2 = 0.0, nd2 = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    na2 += a[i] * a[i];
    double d = a[i] - b[i];
    nd2 += d * d;
  }
  double na = std::sqrt(na2);
  double nd = std::sqrt(nd2);
  return (na > 0.0) ? nd / na : nd;
}

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
  quda::ColorSpinorParam cudaParam(cpuParam, inv, QUDA_CUDA_FIELD_LOCATION);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  return new quda::ColorSpinorField(cudaParam);
}

static void copy_flat_to_native(double *d_flat, quda::ColorSpinorField &native,
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
  native.copy(cpu_csf);
  cudaDeviceSynchronize();
}

static void copy_native_to_flat(quda::ColorSpinorField &native, double *d_flat,
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
  cpu_csf.copy(native);
  cudaDeviceSynchronize();
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
  pRNG.SeedFixedIntegers({6001, 6002, 6003, 6004});

  std::cout << GridLogMessage << "Lattice: ";
  for (int d_ = 0; d_ < Nd; ++d_) std::cout << latt[d_] << (d_ + 1 < Nd ? "×" : "");
  std::cout << std::endl;

  // Gauge for DTXQCDMpcOpQUDA setup (just to extract inv_param + gauge_param).
  LatticeGaugeField U(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, U);

  // Random Grid input.
  LatticeFermion in_grid(&Grid_);
  gaussian(pRNG, in_grid);

  int V = Quda::local_volume(&Grid_);
  std::size_t bytes_spinor = 24 * V * sizeof(double);

  // ---------- Reference: flat-24V kernel ----------
  std::vector<double> h_in(24 * V);
  Quda::fermion_to_eo_buffer(in_grid, h_in.data());

  double *d_in = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_scratch = (double*)acceleratorAllocDevice(bytes_spinor);
  acceleratorCopyToDevice(h_in.data(), d_in, bytes_spinor);

  Quda::PreMatLowerKernel(d_scratch, d_in, V);
  cudaDeviceSynchronize();

  std::vector<double> h_ref(24 * V);
  acceleratorCopyFromDevice(d_scratch, h_ref.data(), bytes_spinor);

  // ---------- Native path ----------
  // Need inv_param via DTXQCDMpcOpQUDA wrapper.
  LatticeDtxqcdSigma sigma(&Grid_); LatticeDtxqcdPi pi(&Grid_);
  LatticeDtxqcdD     dd_(&Grid_);   LatticeDtxqcdN  n_ (&Grid_);
  LatticeDtxqcdS     s_(&Grid_);    LatticeDtxqcdP  p_ (&Grid_);
  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdComplexSymmetricCFGaussian(pRNG, dd_);
  DtxqcdComplexSymmetricCFGaussian(pRNG, n_);
  DtxqcdRealScalarGaussian(pRNG, s_);
  DtxqcdRealScalarGaussian(pRNG, p_);

  DTXQCDMpcOpQUDA Mop_quda(U, Grid_, RBGrid, -0.245, 1.249,
                            sigma, pi, dd_, n_, s_, p_);
  QudaInvertParam &inv_param = Mop_quda.InvertParam();
  const int X_full_dims[4] = {
      Mop_quda.GaugeParam().X[0], Mop_quda.GaugeParam().X[1],
      Mop_quda.GaugeParam().X[2], Mop_quda.GaugeParam().X[3]};

  double *d_in_native = (double*)acceleratorAllocDevice(bytes_spinor);
  double *d_out_native = (double*)acceleratorAllocDevice(bytes_spinor);
  acceleratorCopyToDevice(h_in.data(), d_in_native, bytes_spinor);

  quda::ColorSpinorField *in_csf = make_native_csf(
      d_in_native, bytes_spinor, inv_param, X_full_dims);
  quda::ColorSpinorField *out_csf = make_native_csf(
      d_out_native, bytes_spinor, inv_param, X_full_dims);

  // Pack input to native (csf.copy host flat → device native).
  copy_flat_to_native(d_in_native, *in_csf, inv_param, X_full_dims);
  quda::blas::zero(*out_csf);
  cudaDeviceSynchronize();

  // Run native kernel.
  DtxqcdQudaPreMatLowerNative::ApplyPreMatLowerNative(*in_csf, *out_csf);
  cudaDeviceSynchronize();

  // Pull native output back to flat.
  double *d_out_flat = (double*)acceleratorAllocDevice(bytes_spinor);
  copy_native_to_flat(*out_csf, d_out_flat, inv_param, X_full_dims);

  std::vector<double> h_native(24 * V);
  acceleratorCopyFromDevice(d_out_flat, h_native.data(), bytes_spinor);

  // ---------- Compare ----------
  RealD rel = rel_diff_buf(h_ref, h_native);
  std::cout << GridLogMessage << "PreMatLower native vs flat-24V rel = "
            << rel << "   (gate ≤ 1e-14)" << std::endl;

  delete in_csf;
  delete out_csf;
  acceleratorFreeDevice(d_in);
  acceleratorFreeDevice(d_scratch);
  acceleratorFreeDevice(d_in_native);
  acceleratorFreeDevice(d_out_native);
  acceleratorFreeDevice(d_out_flat);

  int rc = (rel <= 1e-14) ? 0 : 1;
  std::cout << GridLogMessage
            << (rc == 0
                  ? "[ok] Session 2 PreMatLower PASS"
                  : "[FAIL] Session 2 PreMatLower")
            << std::endl;

  Grid::Quda::finalize();
  Grid_finalize();
  return rc;
}
