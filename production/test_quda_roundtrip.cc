// Phase 2 round-trip test: Grid ↔ QUDA host-buffer conversion.
//
// Verifies that fermion_to_lex_buffer / lex_buffer_to_fermion and
// gauge_to_lex_buffers / lex_buffers_to_gauge are inverse operations
// (norm2 of the difference must be exactly 0 for double-precision data —
// the conversion is just a memory reshuffle, no arithmetic).
//
// Also exercises lex_to_eo_permute / eo_to_lex_permute round-trip.
//
// Run:
//   mpirun -np 1 ./test_quda_roundtrip --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/util/QudaFieldConvert.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  int V = Quda::local_volume(&Grid4);
  std::cout << GridLogMessage << "Local volume = " << V << std::endl;

  // --------------------------------------------------------------------------
  // Fermion round-trip.
  // --------------------------------------------------------------------------
  LatticeFermion src(&Grid4), out(&Grid4);
  random(pRNG, src);
  out = Zero();

  std::vector<double> ferm_buf(24 * V);
  Quda::fermion_to_lex_buffer(src, ferm_buf.data());
  Quda::lex_buffer_to_fermion(ferm_buf.data(), out);

  LatticeFermion diff = src - out;
  RealD nf_src  = norm2(src);
  RealD nf_diff = norm2(diff);
  std::cout << GridLogMessage << "Fermion: norm2(src)  = " << nf_src  << std::endl;
  std::cout << GridLogMessage << "Fermion: norm2(diff) = " << nf_diff << std::endl;

  bool ferm_pass = (nf_diff == 0.0);
  std::cout << GridLogMessage << "Fermion round-trip: "
            << (ferm_pass ? "PASS" : "FAIL") << std::endl;

  // --------------------------------------------------------------------------
  // Gauge round-trip.
  // --------------------------------------------------------------------------
  LatticeGaugeField U_src(&Grid4), U_out(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U_src);
  U_out = Zero();

  std::vector<std::vector<double>> gauge_bufs(4, std::vector<double>(18 * V));
  double *bufs[4] = {gauge_bufs[0].data(), gauge_bufs[1].data(),
                     gauge_bufs[2].data(), gauge_bufs[3].data()};
  Quda::gauge_to_lex_buffers(U_src, bufs);
  Quda::lex_buffers_to_gauge(bufs, U_out);

  LatticeGaugeField U_diff = U_src - U_out;
  RealD ng_src  = norm2(U_src);
  RealD ng_diff = norm2(U_diff);
  std::cout << GridLogMessage << "Gauge: norm2(src)  = " << ng_src  << std::endl;
  std::cout << GridLogMessage << "Gauge: norm2(diff) = " << ng_diff << std::endl;

  bool gauge_pass = (ng_diff == 0.0);
  std::cout << GridLogMessage << "Gauge round-trip: "
            << (gauge_pass ? "PASS" : "FAIL") << std::endl;

  // --------------------------------------------------------------------------
  // Lex ↔ EO permutation round-trip on the fermion buffer.
  // --------------------------------------------------------------------------
  std::vector<double> ferm_eo(24 * V), ferm_back(24 * V);
  Coordinate lc = Grid4.LocalDimensions();
  Quda::lex_to_eo_permute(ferm_buf.data(), ferm_eo.data(),  V, 24, lc);
  Quda::eo_to_lex_permute(ferm_eo.data(),  ferm_back.data(), V, 24, lc);

  RealD eo_diff = 0.0;
  for (int i = 0; i < 24 * V; ++i) {
    double d = ferm_buf[i] - ferm_back[i];
    eo_diff += d * d;
  }
  std::cout << GridLogMessage << "EO permute: norm2(lex - eo→lex) = "
            << eo_diff << std::endl;

  bool eo_pass = (eo_diff == 0.0);
  std::cout << GridLogMessage << "EO permutation round-trip: "
            << (eo_pass ? "PASS" : "FAIL") << std::endl;

  // Sanity: EO buffer should NOT equal lex buffer (otherwise the permute
  // is a no-op and we have a bug).
  RealD lex_eo_diff = 0.0;
  for (int i = 0; i < 24 * V; ++i) {
    double d = ferm_buf[i] - ferm_eo[i];
    lex_eo_diff += d * d;
  }
  std::cout << GridLogMessage << "EO permute: norm2(lex - eo) = "
            << lex_eo_diff << " (should be > 0)" << std::endl;

  bool overall = ferm_pass && gauge_pass && eo_pass && (lex_eo_diff > 0);
  std::cout << GridLogMessage
            << "OVERALL Phase 2 round-trip: "
            << (overall ? "PASS" : "FAIL") << std::endl;

  Grid_finalize();
  return overall ? 0 : 1;
}
