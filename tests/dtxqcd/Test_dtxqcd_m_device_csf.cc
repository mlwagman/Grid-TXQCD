// Style B Path B Session 3 — bit-equiv test for M_device_csf (Style C).
//
// M_device_csf operates on caller-owned native ColorSpinorFields (FLOAT2,
// halo-padded, UKQCD basis).  This proves the architecture: no per-call
// csf.copy inside the Mat path; all inner kernels (PreMat / Mat / PostMat /
// aux) run on the same native CSF storage.
//
// Reference: M_device (Style A / Option β) on raw flat-EO double buffers.
// Both paths receive the same Grid spinor input, packed to flat-24V EO
// layout.  Path A: M_device on raw buffers → unpack.  Path B: csf.copy →
// native, M_device_csf, csf.copy → flat, unpack.
//
// Gate: max rel diff ≤ 1e-12 (loose vs aux native's 1e-14; the M-vec
// composition includes more numerical operations).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>

using namespace Grid;

static RealD rel_diff_fermion(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion d(a.Grid()); d = a - b;
  RealD na = std::sqrt(norm2(a));
  RealD nd = std::sqrt(norm2(d));
  return (na > 0.0) ? nd / na : nd;
}

static quda::ColorSpinorField *make_native_csf(QudaInvertParam &inv_param,
                                               const int X_full_dims[4]) {
  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  QudaInvertParam inv = inv_param;
  inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv.output_location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorParam cpuParam(nullptr, inv, X_full, pc,
                                  QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorParam cudaParam(cpuParam, inv, QUDA_CUDA_FIELD_LOCATION);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  return new quda::ColorSpinorField(cudaParam);
}

static void copy_flat_to_native(double *d_flat,
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
                                double *d_flat,
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

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  // Default to 4⁴.
  Coordinate latt({4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dd;
      GridCmdOptionIntVector(argv[i + 1], dd);
      if (dd.size() == (size_t)Nd) latt = Coordinate(dd);
    }
  }

  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian grid(latt, simd_layout, mpi);
  GridRedBlackCartesian rb(&grid);
  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  // ------------------------------------------------------------------
  // Set up aux + gauge.
  // ------------------------------------------------------------------
  LatticeDtxqcdSigma sigma(&grid);
  LatticeDtxqcdPi    pi   (&grid);
  LatticeDtxqcdD     d    (&grid);
  LatticeDtxqcdN     n    (&grid);
  LatticeDtxqcdS     s    (&grid);
  LatticeDtxqcdP     p    (&grid);
  random(pRNG, sigma);
  random(pRNG, pi);
  random(pRNG, d);
  random(pRNG, n);
  random(pRNG, s);
  random(pRNG, p);

  LatticeGaugeField U(&grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  // ------------------------------------------------------------------
  // Random input doubled fermion.
  // ------------------------------------------------------------------
  DTXQCDFermionDoubled in(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    random(pRNG, in.upper.f[a]);
    random(pRNG, in.lower.f[a]);
  }

  // ------------------------------------------------------------------
  // Build the wrapper.
  // ------------------------------------------------------------------
  RealD mass = -0.245, csw = 1.249;
  DTXQCDMpcOpQUDA Mop(U, grid, rb, mass, csw,
                      sigma, pi, d, n, s, p,
                      /*anti_periodic_t=*/true);
  QudaInvertParam &inv_param = Mop.InvertParam();
  QudaGaugeParam  &gauge_param = Mop.GaugeParam();

  int V = Quda::local_volume(&grid);
  size_t bytes = 24 * V * sizeof(double);
  int X_full_dims[4];
  for (int d_ = 0; d_ < 4; ++d_) X_full_dims[d_] = gauge_param.X[d_];

  // ------------------------------------------------------------------
  // Pack input to flat-24V EO buffers (shared between Path A and Path B).
  // ------------------------------------------------------------------
  std::vector<double> host_buf(24 * V);
  std::array<double *, DtxqcdNf> d_in_u, d_in_l, d_in_u_B, d_in_l_B;
  for (int a = 0; a < DtxqcdNf; ++a) {
    d_in_u[a]   = (double *)acceleratorAllocDevice(bytes);
    d_in_l[a]   = (double *)acceleratorAllocDevice(bytes);
    d_in_u_B[a] = (double *)acceleratorAllocDevice(bytes);
    d_in_l_B[a] = (double *)acceleratorAllocDevice(bytes);
  }
  for (int a = 0; a < DtxqcdNf; ++a) {
    Quda::fermion_to_eo_buffer(in.upper.f[a], host_buf.data());
    acceleratorCopyToDevice(host_buf.data(), d_in_u[a], bytes);
    acceleratorCopyToDevice(host_buf.data(), d_in_u_B[a], bytes);
    Quda::fermion_to_eo_buffer(in.lower.f[a], host_buf.data());
    acceleratorCopyToDevice(host_buf.data(), d_in_l[a], bytes);
    acceleratorCopyToDevice(host_buf.data(), d_in_l_B[a], bytes);
  }

  // ------------------------------------------------------------------
  // PATH A: existing M_device on raw flat-EO buffers (reference).
  // ------------------------------------------------------------------
  std::array<double *, DtxqcdNf> d_out_u, d_out_l, d_scratch_l;
  for (int a = 0; a < DtxqcdNf; ++a) {
    d_out_u[a]      = (double *)acceleratorAllocDevice(bytes);
    d_out_l[a]      = (double *)acceleratorAllocDevice(bytes);
    d_scratch_l[a]  = (double *)acceleratorAllocDevice(bytes);
  }

  Mop.M_device(d_in_u.data(), d_in_l.data(), d_out_u.data(), d_out_l.data(),
               d_scratch_l.data(), /*dagger=*/false);

  DTXQCDFermionDoubled out_A(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    acceleratorCopyFromDevice(d_out_u[a], host_buf.data(), bytes);
    Quda::eo_buffer_to_fermion(host_buf.data(), out_A.upper.f[a]);
    acceleratorCopyFromDevice(d_out_l[a], host_buf.data(), bytes);
    Quda::eo_buffer_to_fermion(host_buf.data(), out_A.lower.f[a]);
  }

  // ------------------------------------------------------------------
  // PATH B: M_device_csf on native ColorSpinorFields.
  // ------------------------------------------------------------------
  // Allocate 8 native CSFs (4 in, 4 out) + 2 scratch_lower CSFs.
  std::array<quda::ColorSpinorField *, DtxqcdNf> csf_in_u, csf_in_l;
  std::array<quda::ColorSpinorField *, DtxqcdNf> csf_out_u, csf_out_l;
  std::array<quda::ColorSpinorField *, DtxqcdNf> csf_scratch_l;
  for (int a = 0; a < DtxqcdNf; ++a) {
    csf_in_u[a]       = make_native_csf(inv_param, X_full_dims);
    csf_in_l[a]       = make_native_csf(inv_param, X_full_dims);
    csf_out_u[a]      = make_native_csf(inv_param, X_full_dims);
    csf_out_l[a]      = make_native_csf(inv_param, X_full_dims);
    csf_scratch_l[a]  = make_native_csf(inv_param, X_full_dims);
  }

  // Populate CSFs from flat-EO inputs via csf.copy.
  for (int a = 0; a < DtxqcdNf; ++a) {
    copy_flat_to_native(d_in_u_B[a], *csf_in_u[a], inv_param, X_full_dims);
    copy_flat_to_native(d_in_l_B[a], *csf_in_l[a], inv_param, X_full_dims);
  }

  // Zero the output CSFs (the aux native kernel accumulates).
  for (int a = 0; a < DtxqcdNf; ++a) {
    quda::vector_ref<quda::ColorSpinorField> y_ref_u{*csf_out_u[a]};
    quda::vector_ref<quda::ColorSpinorField> y_ref_l{*csf_out_l[a]};
    quda::vector<quda::Complex> z{quda::Complex(0.0, 0.0)};
    quda::blas::ax(z, y_ref_u);
    quda::blas::ax(z, y_ref_l);
  }

  // Apply M_device_csf.
  Mop.M_device_csf(csf_in_u.data(), csf_in_l.data(),
                   csf_out_u.data(), csf_out_l.data(),
                   csf_scratch_l.data(), /*dagger=*/false);

  // Unpack: csf.copy native → flat → fermion.
  std::vector<double *> d_out_u_B(DtxqcdNf), d_out_l_B(DtxqcdNf);
  for (int a = 0; a < DtxqcdNf; ++a) {
    d_out_u_B[a] = (double *)acceleratorAllocDevice(bytes);
    d_out_l_B[a] = (double *)acceleratorAllocDevice(bytes);
  }
  for (int a = 0; a < DtxqcdNf; ++a) {
    copy_native_to_flat(*csf_out_u[a], d_out_u_B[a], inv_param, X_full_dims);
    copy_native_to_flat(*csf_out_l[a], d_out_l_B[a], inv_param, X_full_dims);
  }

  DTXQCDFermionDoubled out_B(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    acceleratorCopyFromDevice(d_out_u_B[a], host_buf.data(), bytes);
    Quda::eo_buffer_to_fermion(host_buf.data(), out_B.upper.f[a]);
    acceleratorCopyFromDevice(d_out_l_B[a], host_buf.data(), bytes);
    Quda::eo_buffer_to_fermion(host_buf.data(), out_B.lower.f[a]);
  }

  // ------------------------------------------------------------------
  // Compare per-flavor.
  // ------------------------------------------------------------------
  RealD max_rel = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    RealD ru = rel_diff_fermion(out_A.upper.f[a], out_B.upper.f[a]);
    RealD rl = rel_diff_fermion(out_A.lower.f[a], out_B.lower.f[a]);
    std::cout << GridLogMessage << "flavor " << a
              << " rel_diff upper=" << ru
              << "  lower=" << rl << std::endl;
    max_rel = std::max({max_rel, ru, rl});
  }

  // ------------------------------------------------------------------
  // 4⁴ correctness gate.
  // ------------------------------------------------------------------
  RealD gate = 1.0e-12;
  bool pass = (max_rel <= gate);
  std::cout << GridLogMessage
            << "[Test_dtxqcd_m_device_csf] max_rel_diff = " << max_rel
            << "   (gate " << gate << ")   "
            << (pass ? "PASS" : "FAIL") << std::endl;

  // ------------------------------------------------------------------
  // Cleanup.
  // ------------------------------------------------------------------
  for (int a = 0; a < DtxqcdNf; ++a) {
    delete csf_in_u[a]; delete csf_in_l[a];
    delete csf_out_u[a]; delete csf_out_l[a];
    delete csf_scratch_l[a];
    acceleratorFreeDevice(d_in_u[a]);
    acceleratorFreeDevice(d_in_l[a]);
    acceleratorFreeDevice(d_in_u_B[a]);
    acceleratorFreeDevice(d_in_l_B[a]);
    acceleratorFreeDevice(d_out_u[a]);
    acceleratorFreeDevice(d_out_l[a]);
    acceleratorFreeDevice(d_scratch_l[a]);
    acceleratorFreeDevice(d_out_u_B[a]);
    acceleratorFreeDevice(d_out_l_B[a]);
  }

  Grid::Quda::finalize();
  Grid_finalize();
  return pass ? 0 : 1;
}
