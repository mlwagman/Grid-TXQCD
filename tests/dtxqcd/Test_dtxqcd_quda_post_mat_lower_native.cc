// Style B Path B Session 2 — bit-equiv test for PostMatLowerKernel ported to
// QUDA NATIVE/FLOAT2 layout via FloatNOrder accessor.  In-place op.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_apply_C.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_post_mat_lower_native.h>
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

static quda::ColorSpinorField *make_native_csf(double *d_buf, size_t,
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
  pRNG.SeedFixedIntegers({7001, 7002, 7003, 7004});

  std::cout << GridLogMessage << "Lattice: ";
  for (int d_ = 0; d_ < Nd; ++d_) std::cout << latt[d_] << (d_ + 1 < Nd ? "×" : "");
  std::cout << std::endl;

  LatticeGaugeField U(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeFermion in_grid(&Grid_);
  gaussian(pRNG, in_grid);

  int V = Quda::local_volume(&Grid_);
  std::size_t bytes_spinor = 24 * V * sizeof(double);

  // ---------- Reference: flat-24V in-place kernel ----------
  std::vector<double> h_in(24 * V);
  Quda::fermion_to_eo_buffer(in_grid, h_in.data());

  double *d_in = (double*)acceleratorAllocDevice(bytes_spinor);
  acceleratorCopyToDevice(h_in.data(), d_in, bytes_spinor);

  Quda::PostMatLowerKernel(d_in, V);
  cudaDeviceSynchronize();

  std::vector<double> h_ref(24 * V);
  acceleratorCopyFromDevice(d_in, h_ref.data(), bytes_spinor);

  // ---------- Native path ----------
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
  acceleratorCopyToDevice(h_in.data(), d_in_native, bytes_spinor);

  quda::ColorSpinorField *csf = make_native_csf(d_in_native, bytes_spinor,
                                                inv_param, X_full_dims);
  copy_flat_to_native(d_in_native, *csf, inv_param, X_full_dims);

  DtxqcdQudaPostMatLowerNative::ApplyPostMatLowerNative(*csf);
  cudaDeviceSynchronize();

  double *d_out_flat = (double*)acceleratorAllocDevice(bytes_spinor);
  copy_native_to_flat(*csf, d_out_flat, inv_param, X_full_dims);

  std::vector<double> h_native(24 * V);
  acceleratorCopyFromDevice(d_out_flat, h_native.data(), bytes_spinor);

  RealD rel = rel_diff_buf(h_ref, h_native);
  std::cout << GridLogMessage << "PostMatLower native vs flat-24V rel = "
            << rel << "   (gate ≤ 1e-14)" << std::endl;

  delete csf;
  acceleratorFreeDevice(d_in);
  acceleratorFreeDevice(d_in_native);
  acceleratorFreeDevice(d_out_flat);

  int rc = (rel <= 1e-14) ? 0 : 1;
  std::cout << GridLogMessage
            << (rc == 0
                  ? "[ok] Session 2 PostMatLower PASS"
                  : "[FAIL] Session 2 PostMatLower")
            << std::endl;

  Grid::Quda::finalize();
  Grid_finalize();
  return rc;
}
