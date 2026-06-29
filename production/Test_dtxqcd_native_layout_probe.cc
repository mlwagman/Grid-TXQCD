// Native layout probe — empirically derive the QUDA NATIVE field-order index
// mapping for double-precision Wilson spinor on a CUDA-located ColorSpinorField.
//
// Hypothesis (from QUDA header read):
//   For Ns=4, Nc=3, Float=double, NATIVE resolves to FLOAT2:
//     N=2, M=12, Nrem=0
//     field[parity*offset + (volumeCB * complex_idx + x_cb) * N + slot]
//     complex_idx = spin*Nc + color  (spin outer, color inner)
//
// Two phases:
//   Phase A: write a known pattern into native via accelerator_for,
//            read back via csf.copy to SPACE_SPIN_COLOR.  Verify (parity, x_cb,
//            spin, color, ri) we wrote maps to expected (lex_site, spin, color, ri)
//            in SSC layout.
//   Phase B: prints offset, volumeCB, Bytes from the field itself.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>

#include <iostream>
#include <iomanip>

using namespace Grid;

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
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({701, 702, 703, 704});

  std::cout << GridLogMessage << "Probe latt=" << latt << std::endl;

  // Need a DTXQCDMpcOpQUDA wrapper just to set up inv_param + gauge_param
  // correctly (matches production's solution_type, gamma_basis, etc.).
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
  std::cout << GridLogMessage << "V_local=" << V
            << "  buf_bytes_per_spinor=" << buf_bytes << std::endl;

  // Allocate raw SPACE_SPIN_COLOR device buffer for the cpu-CSF wrapping.
  double *d_in  = (double *)acceleratorAllocDevice(buf_bytes);
  double *d_out = (double *)acceleratorAllocDevice(buf_bytes);

  inv_param.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;
  inv_param.dagger          = QUDA_DAG_NO;

  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = Mop_quda.GaugeParam().X[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  quda::ColorSpinorParam cpuParam(d_in, inv_param, X_full,
                                  pc, QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField in_ref(cpuParam);

  quda::ColorSpinorParam cudaParam(cpuParam, inv_param, QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField in_native(cudaParam);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  quda::ColorSpinorField out_native(cudaParam);

  // Report field metadata.
  std::cout << GridLogMessage << "in_ref.FieldOrder()    = " << (int)in_ref.FieldOrder() << "  (SSC=1)" << std::endl;
  std::cout << GridLogMessage << "in_native.FieldOrder() = " << (int)in_native.FieldOrder() << "  (NATIVE=0)" << std::endl;
  std::cout << GridLogMessage << "in_native.Volume()     = " << in_native.Volume() << std::endl;
  std::cout << GridLogMessage << "in_native.VolumeCB()   = " << in_native.VolumeCB() << std::endl;
  std::cout << GridLogMessage << "in_native.Bytes()      = " << in_native.Bytes() << std::endl;
  std::cout << GridLogMessage << "in_native.SiteSubset() = " << (int)in_native.SiteSubset() << "  (FULL=2 PARITY=1)" << std::endl;
  std::cout << GridLogMessage << "in_native.SiteOrder()  = " << (int)in_native.SiteOrder() << std::endl;
  std::cout << GridLogMessage << "in_ref.GammaBasis()    = " << (int)in_ref.GammaBasis() << "  (DR=0 UKQCD=1 CHIRAL=2)" << std::endl;
  std::cout << GridLogMessage << "in_native.GammaBasis() = " << (int)in_native.GammaBasis() << std::endl;
  std::cout << GridLogMessage << "inv_param.gamma_basis  = " << (int)inv_param.gamma_basis << std::endl;
  std::cout << GridLogMessage << "in_ref.Precision()     = " << (int)in_ref.Precision() << "  (DOUBLE=8)" << std::endl;
  std::cout << GridLogMessage << "in_native.Precision()  = " << (int)in_native.Precision() << std::endl;

  size_t native_bytes = in_native.Bytes();
  int volumeCB = in_native.VolumeCB();
  // Per the QUDA header: offset (in Float units) = a.Bytes() / (2 * sizeof(Float))
  size_t offset_doubles = native_bytes / (2 * sizeof(double));
  std::cout << GridLogMessage << "Derived offset (doubles per parity) = "
            << offset_doubles << "  expected ≈ 24*volumeCB = " << (24 * volumeCB)
            << std::endl;

  // Phase A: write a known pattern from SSC host into native via csf.copy,
  // then dump the raw native buffer and check the FLOAT2 layout formula.
  //
  // Pattern: h_in[lex_site * 24 + spin*6 + color*2 + ri] = code(lex_site, spin, color, ri)
  // where the code is uniquely decodable: 1e6 * lex_site + 1000 * (spin*Nc+color) + 10*ri.

  std::vector<double> h_in(24 * V);
  for (int lex_site = 0; lex_site < V; ++lex_site) {
    for (int spin = 0; spin < 4; ++spin) {
      for (int color = 0; color < 3; ++color) {
        for (int ri = 0; ri < 2; ++ri) {
          double code = 1e6 * lex_site
                      + 1000.0 * (spin * 3 + color)
                      + 10.0 * ri
                      + 1.0;
          h_in[lex_site * 24 + spin * 6 + color * 2 + ri] = code;
        }
      }
    }
  }
  acceleratorCopyToDevice(h_in.data(), d_in, buf_bytes);
  acceleratorCopySynchronise();

  // Convert SSC (d_in is wrapped as in_ref) → native (in_native).
  in_native.copy(in_ref);
  cudaDeviceSynchronize();

  // Dump the native raw buffer.
  std::vector<double> h_native(offset_doubles * 2, 0.0);   // 2 parities × offset doubles
  acceleratorCopyFromDevice(in_native.data<double*>(), h_native.data(),
                            h_native.size() * sizeof(double));
  acceleratorCopySynchronise();

  // Verify the formula by inverting: for each (parity, x_cb, complex_idx, ri),
  // compute native_idx and read h_native[native_idx].  Decode the value back to
  // (lex_site, spin, color, ri) via the pattern code formula and verify that
  // (lex_site -> parity, x_cb).
  //
  // Native index formula under test:
  //   native_idx = parity * offset + (volumeCB * complex_idx + x_cb) * 2 + slot
  // We need to figure out: what is the mapping (parity, x_cb) → lex_site?
  //
  // Phase A1: dump the first N entries to see what's there at parity=0 x_cb=0
  // for each complex_idx + slot, and find the lex_site that goes there.

  std::cout << GridLogMessage << "\n=== Phase A: native FLOAT2 layout verification ===" << std::endl;
  std::cout << GridLogMessage << "Probing parity=0 x_cb=0 across complex_idx=0..11 slot=0..1:" << std::endl;
  for (int complex_idx = 0; complex_idx < 12; ++complex_idx) {
    for (int slot = 0; slot < 2; ++slot) {
      size_t idx = (size_t)volumeCB * complex_idx * 2 + 0 * 2 + slot;
      double v = h_native[idx];
      int lex_site = (int)std::floor(v / 1e6);
      int code_complex = (int)std::floor((v - 1e6 * lex_site) / 1000.0);
      int code_slot = (int)std::floor((v - 1e6 * lex_site - 1000.0 * code_complex) / 10.0);
      std::cout << GridLogMessage
                << "  complex_idx=" << complex_idx << " slot=" << slot
                << " native_idx=" << idx
                << " val=" << std::fixed << std::setprecision(1) << v
                << " → lex_site=" << lex_site
                << " code_complex=" << code_complex
                << " code_slot=" << code_slot << std::endl;
    }
  }

  // Phase A2: dump parity=0 across x_cb=0..min(8,volumeCB-1) at complex_idx=0 slot=0
  // to see the x_cb → lex_site mapping for parity 0.
  std::cout << GridLogMessage << "\nParity=0 x_cb=0..7 at complex_idx=0 slot=0:" << std::endl;
  for (int x_cb = 0; x_cb < std::min(8, volumeCB); ++x_cb) {
    size_t idx = (size_t)volumeCB * 0 * 2 + x_cb * 2 + 0;
    double v = h_native[idx];
    int lex_site = (int)std::floor(v / 1e6);
    std::cout << GridLogMessage << "  x_cb=" << x_cb
              << " native_idx=" << idx << " val=" << v
              << " → lex_site=" << lex_site << std::endl;
  }
  std::cout << GridLogMessage << "Parity=1 x_cb=0..7 at complex_idx=0 slot=0:" << std::endl;
  for (int x_cb = 0; x_cb < std::min(8, volumeCB); ++x_cb) {
    size_t idx = offset_doubles + (size_t)volumeCB * 0 * 2 + x_cb * 2 + 0;
    double v = h_native[idx];
    int lex_site = (int)std::floor(v / 1e6);
    std::cout << GridLogMessage << "  x_cb=" << x_cb
              << " native_idx=" << idx << " val=" << v
              << " → lex_site=" << lex_site << std::endl;
  }

  // Phase B: compare expected fermion_to_eo_buffer ordering — at parity=0
  // x_cb=lex>>1 of even-parity lex sites.  This is the "production" EO order
  // used by the existing flat-24V kernel.
  std::cout << GridLogMessage << "\nExpected fermion_to_eo_buffer EO ordering (production convention):" << std::endl;
  int lc[4] = {latt[0], latt[1], latt[2], latt[3]};
  for (int parity = 0; parity < 2; ++parity) {
    std::cout << GridLogMessage << "  parity=" << parity << " first 6 lex sites:";
    int count = 0;
    for (int lex = 0; lex < V && count < 6; ++lex) {
      int x = lex % lc[0];
      int y = (lex / lc[0]) % lc[1];
      int z = (lex / (lc[0]*lc[1])) % lc[2];
      int t = lex / (lc[0]*lc[1]*lc[2]);
      int p = (x + y + z + t) & 1;
      if (p == parity) {
        std::cout << " lex=" << lex << "(x_cb=" << (lex/2) << ")";
        count++;
      }
    }
    std::cout << std::endl;
  }

  acceleratorFreeDevice(d_in);
  acceleratorFreeDevice(d_out);
  Grid::Quda::finalize();
  Grid_finalize();
  return 0;
}
