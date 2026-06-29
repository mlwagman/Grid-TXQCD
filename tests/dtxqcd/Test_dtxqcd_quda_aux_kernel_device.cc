// Test_dtxqcd_quda_aux_kernel_device: M-wrap.5b.2 Option β device-aux unit test.
//
// Validates dtxqcd_quda_aux_kernel_device.h's `ApplyAuxKernel` (which operates
// on flat-EO device buffers) against the validated Grid-SIMD reference
// `ApplyFusedDtxqcdAuxKernel` from dtxqcd_quda_aux_kernel.h (M-wrap.3).
//
// All 4 (transpose_aux, use_dn_conj) flag combinations exercised; gate
// max rel diff ≤ 1e-14 across all 4 spinor slots.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>
#include <Grid/util/QudaFieldConvert.h>

using namespace Grid;

static RealD rel_diff_fermion(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion d(a.Grid()); d = a - b;
  RealD na = std::sqrt(norm2(a));
  RealD nd = std::sqrt(norm2(d));
  return (na > 0.0) ? nd / na : nd;
}

static int run_case(GridCartesian &grid, GridParallelRNG &pRNG,
                    bool transpose_aux, bool use_dn_conj) {
  std::cout << GridLogMessage << "=== case (transpose_aux="
            << (transpose_aux ? "true" : "false")
            << ", use_dn_conj="
            << (use_dn_conj ? "true" : "false") << ") ===" << std::endl;

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

  double *in_u0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *in_u1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *in_l0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *in_l1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *ou_u0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *ou_u1 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *ou_l0 = (double*)acceleratorAllocDevice(bytes_spinor);
  double *ou_l1 = (double*)acceleratorAllocDevice(bytes_spinor);

  std::vector<double> host_buf(24 * V);
  Quda::fermion_to_eo_buffer(in.upper.f[0], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), in_u0, bytes_spinor);
  Quda::fermion_to_eo_buffer(in.upper.f[1], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), in_u1, bytes_spinor);
  Quda::fermion_to_eo_buffer(in.lower.f[0], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), in_l0, bytes_spinor);
  Quda::fermion_to_eo_buffer(in.lower.f[1], host_buf.data());
  acceleratorCopyToDevice(host_buf.data(), in_l1, bytes_spinor);

  std::vector<double> zero_buf(24 * V, 0.0);
  acceleratorCopyToDevice(zero_buf.data(), ou_u0, bytes_spinor);
  acceleratorCopyToDevice(zero_buf.data(), ou_u1, bytes_spinor);
  acceleratorCopyToDevice(zero_buf.data(), ou_l0, bytes_spinor);
  acceleratorCopyToDevice(zero_buf.data(), ou_l1, bytes_spinor);

  DtxqcdQudaAuxKernelDevice::DeviceAuxCache aux_cache;
  DtxqcdQudaAuxKernelDevice::allocate_aux_cache(aux_cache, V);
  DtxqcdQudaAuxKernelDevice::pack_aux_to_device(sigma, pi, d, n, s, p,
                                                aux_cache);

  DtxqcdQudaAuxKernelDevice::ApplyAuxKernel(
      aux_cache,
      in_u0, in_u1, in_l0, in_l1,
      ou_u0, ou_u1, ou_l0, ou_l1,
      V, transpose_aux, use_dn_conj);

  DTXQCDFermionDoubled out_dev(&grid);
  acceleratorCopyFromDevice(ou_u0, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.upper.f[0]);
  acceleratorCopyFromDevice(ou_u1, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.upper.f[1]);
  acceleratorCopyFromDevice(ou_l0, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.lower.f[0]);
  acceleratorCopyFromDevice(ou_l1, host_buf.data(), bytes_spinor);
  Quda::eo_buffer_to_fermion(host_buf.data(), out_dev.lower.f[1]);

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

  DtxqcdQudaAuxKernelDevice::free_aux_cache(aux_cache);
  acceleratorFreeDevice(in_u0); acceleratorFreeDevice(in_u1);
  acceleratorFreeDevice(in_l0); acceleratorFreeDevice(in_l1);
  acceleratorFreeDevice(ou_u0); acceleratorFreeDevice(ou_u1);
  acceleratorFreeDevice(ou_l0); acceleratorFreeDevice(ou_l1);

  return (max_rel <= 1e-14) ? 0 : 1;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

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
  GridCartesian Grid_(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({4001, 4002, 4003, 4004});

  std::cout << GridLogMessage << "Lattice: ";
  for (int d_ = 0; d_ < Nd; ++d_) std::cout << latt[d_] << (d_ + 1 < Nd ? "×" : "");
  std::cout << std::endl;

  int rc = 0;
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/true,  /*use_dn_conj=*/true);
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/false, /*use_dn_conj=*/false);
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/true,  /*use_dn_conj=*/false);
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/false, /*use_dn_conj=*/true);

  std::cout << GridLogMessage
            << (rc == 0
                  ? "[ok] M-wrap.5b.2 Option β PASS — device aux kernel bit-exact"
                  : "[FAIL] M-wrap.5b.2 Option β — exceeded 1e-14")
            << std::endl;

  Grid_finalize();
  return rc;
}
