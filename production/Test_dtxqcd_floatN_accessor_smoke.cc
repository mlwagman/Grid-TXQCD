// Style B Path B Session 1 — FloatNOrder accessor smoke test.
//
// Goal: prove that `quda::colorspinor::FloatNOrder<double, 4, 3>` is usable
// from a Grid `accelerator_for` body (i.e. without writing a separate .cu),
// and that load/save bit-exactly preserve data when wrapped around a no-op
// "scale by 2" kernel compared to QUDA's own `quda::blas::ax(2.0, csf)`.
//
// If this PASSES: architecture is proven; FloatNOrder works from Grid's nvcc
// compile path; we can port aux/gamma5/pre/post kernels using accelerator_for
// + accessor.load → ColorSpinor math → accessor.save.
//
// If this FAILS to compile: we need a separate .cu file and a dedicated nvcc
// invocation — significant build-system work for each new kernel.
//
// Run:
//   ./Test_dtxqcd_floatN_accessor_smoke --grid 4.4.4.4

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <blas_quda.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Grid;

namespace {

double rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double n = 0.0, d = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double dd = a[i] - b[i];
    n += dd * dd;
    d += a[i] * a[i];
  }
  if (d == 0.0) return std::sqrt(n);
  return std::sqrt(n / d);
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
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
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({901, 902, 903, 904});

  std::cout << GridLogMessage << "FloatN accessor smoke  latt=" << latt
            << "  mpi=" << mpi << std::endl;

  // Need a DTXQCDMpcOpQUDA wrapper just to set up inv_param + gauge_param.
  LatticeGaugeField U(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, U);
  LatticeDtxqcdSigma sigma(&Grid_);  sigma = Zero();
  LatticeDtxqcdPi    pi(&Grid_);     pi    = Zero();
  LatticeDtxqcdD     d_field(&Grid_);d_field = Zero();
  LatticeDtxqcdN     n_field(&Grid_);n_field = Zero();
  LatticeDtxqcdS     s_field(&Grid_);s_field = Zero();
  LatticeDtxqcdP     p_field(&Grid_);p_field = Zero();

  DTXQCDMpcOpQUDA Mop_quda(U, Grid_, RBGrid, /*mass=*/-0.245, /*csw=*/1.249,
                            sigma, pi, d_field, n_field, s_field, p_field);

  QudaInvertParam &inv_param = Mop_quda.InvertParam();

  int V = Quda::local_volume(&Grid_);
  size_t buf_bytes = 24 * V * sizeof(double);

  // Random LatticeFermion → pack to flat 24·V → upload to device.
  LatticeFermion v(&Grid_);
  gaussian(pRNG, v);

  std::vector<double> h_in(24 * V);
  Quda::fermion_to_eo_buffer(v, h_in.data());

  double *d_in  = (double *)acceleratorAllocDevice(buf_bytes);
  acceleratorCopyToDevice(h_in.data(), d_in, buf_bytes);
  acceleratorCopySynchronise();

  inv_param.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;

  // Build cpu CSF wrapping d_in (DIRAC_ORDER+EVEN_ODD), then a cuda CSF in
  // native layout.
  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = Mop_quda.GaugeParam().X[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  quda::ColorSpinorParam cpuParam(d_in, inv_param, X_full,
                                  pc, QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField in_ref(cpuParam);

  quda::ColorSpinorParam cudaParam(cpuParam, inv_param,
                                   QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField csf_blas(cudaParam);     // for blas::ax reference
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  quda::ColorSpinorField csf_kern(cudaParam);     // for our kernel

  // Populate both: in_ref (cpu SSC layout on d_in) → native (UKQCD basis +
  // QUDA x_cb reorder).
  csf_blas.copy(in_ref);
  csf_kern.copy(in_ref);
  cudaDeviceSynchronize();

  std::cout << GridLogMessage << "FieldOrder ref=" << (int)in_ref.FieldOrder()
            << " kern=" << (int)csf_kern.FieldOrder()
            << " (FLOAT2=" << (int)QUDA_FLOAT2_FIELD_ORDER << ")"
            << std::endl;

  // ----------------------------------------------------------------------
  // Reference: quda::blas::ax(2.0, csf_blas)
  // ----------------------------------------------------------------------
  quda::blas::ax(2.0, csf_blas);
  cudaDeviceSynchronize();

  // ----------------------------------------------------------------------
  // Our kernel: scale by 2.0 via FloatNOrder.load/save.
  // ----------------------------------------------------------------------
  //
  // FloatNOrder<double, Ns=4, Nc=3> with default spin_project=false.
  // load(complex<double> out[12], int x_cb, int parity) reads 12 complex.
  // save(complex<double> in[12], int x_cb, int parity) writes 12 complex.
  //
  // Note: load returns components in the *storage* (UKQCD) basis — no auto
  // basis-transform.  Since scaling is basis-invariant, that's fine here.
  // FloatNOrder<Float, Ns, Nc, N_> — N_=2 for double-prec FLOAT2 storage.
  using AccTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;
  AccTy accessor(csf_kern);

  int volumeCB = csf_kern.VolumeCB();
  int nParity = (csf_kern.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::cout << GridLogMessage << "Kernel range: nParity=" << nParity
            << " volumeCB=" << volumeCB << std::endl;

  // Inner loop iterates (parity, x_cb).  Use a flat index = parity*VCB + x_cb.
  size_t N = (size_t)nParity * (size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;
    quda::complex<double> spinor[12];
    accessor.load(spinor, x_cb, parity);
    for (int s = 0; s < 12; ++s) spinor[s] *= 2.0;
    accessor.save(spinor, x_cb, parity);
  });
  cudaDeviceSynchronize();

  // ----------------------------------------------------------------------
  // Compare: pull both back to host SSC and diff.
  // ----------------------------------------------------------------------
  std::vector<double> h_blas(24 * V), h_kern(24 * V);

  // Reuse cpuParam structure but point at a fresh device buffer for each
  // download.
  double *d_blas = (double *)acceleratorAllocDevice(buf_bytes);
  double *d_kern = (double *)acceleratorAllocDevice(buf_bytes);

  quda::ColorSpinorParam outParamB = cpuParam;
  outParamB.v = d_blas;
  quda::ColorSpinorField out_ref_blas(outParamB);
  out_ref_blas.copy(csf_blas);

  quda::ColorSpinorParam outParamK = cpuParam;
  outParamK.v = d_kern;
  quda::ColorSpinorField out_ref_kern(outParamK);
  out_ref_kern.copy(csf_kern);

  cudaDeviceSynchronize();

  acceleratorCopyFromDevice(d_blas, h_blas.data(), buf_bytes);
  acceleratorCopyFromDevice(d_kern, h_kern.data(), buf_bytes);
  acceleratorCopySynchronise();

  double rel = rel_diff(h_blas, h_kern);
  std::cout << GridLogMessage << std::scientific << std::setprecision(4)
            << "Echo kernel vs quda::blas::ax(2.0): rel_diff = " << rel
            << "   (gate ≤ 1e-15)" << std::endl;

  bool pass = rel < 1e-15;
  std::cout << GridLogMessage << "SMOKE: " << (pass ? "PASS" : "FAIL")
            << std::endl;

  acceleratorFreeDevice(d_in);
  acceleratorFreeDevice(d_blas);
  acceleratorFreeDevice(d_kern);

  Grid::Quda::finalize();
  Grid_finalize();
  return pass ? 0 : 1;
}
