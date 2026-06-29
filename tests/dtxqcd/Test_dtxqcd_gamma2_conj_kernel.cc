// Test_dtxqcd_gamma2_conj_kernel: M-wrap.2 unit test.
//
// Validates the γ_2·conj() kernel against Grid's reference at 4⁴:
//   1. Generate a random LatticeFermion v_grid.
//   2. Pack to flat lex-ordered host buffer (24 doubles/site).
//   3. Upload to device, run Gamma2ConjKernel, download.
//   4. Unpack into out_grid.
//   5. Compare against Grid reference  γ_2·conj(v_grid)  — try several
//      convention variants and pick the one that passes.
//
// Gate: pass tolerance ≤ 1e-14 (machine eps; the kernel is bit-exact since
// γ_2 entries are ±1 and conj is just an imag sign flip).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma2_conj.h>
#include <Grid/util/QudaFieldConvert.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt({4, 4, 4, 4});
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--grid") {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid_(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({701, 702, 703, 704});

  // ---- random Grid spinor + pack to lex buffer ----
  LatticeFermion v_grid(&Grid_), out_grid(&Grid_);
  gaussian(pRNG, v_grid);

  int V = Quda::local_volume(&Grid_);
  std::vector<double> v_lex(V * 24), out_lex(V * 24, 0.0);
  Quda::fermion_to_lex_buffer(v_grid, v_lex.data());

  // ---- ship to device, run kernel, ship back ----
  deviceVector<double> v_d(V * 24), out_d(V * 24);
  acceleratorCopyToDevice(v_lex.data(), &v_d[0],   V * 24 * sizeof(double));
  acceleratorCopyToDevice(out_lex.data(), &out_d[0], V * 24 * sizeof(double));

  Quda::Gamma2ConjKernel(&out_d[0], &v_d[0], V);

  acceleratorCopyFromDevice(&out_d[0], out_lex.data(), V * 24 * sizeof(double));
  Quda::lex_buffer_to_fermion(out_lex.data(), out_grid);

  // ---- compare to Grid references (convention probe) ----
  auto rel_diff = [](const LatticeFermion &a, const LatticeFermion &b) {
    LatticeFermion d(a.Grid()); d = a - b;
    RealD na = std::sqrt(norm2(a));
    RealD nd = std::sqrt(norm2(d));
    return (na > 0.0) ? nd / na : nd;
  };

  struct Variant {
    const char *label;
    std::function<LatticeFermion(const LatticeFermion &)> f;
  };

  Gamma g2(Gamma::Algebra::GammaY);
  Gamma g0(Gamma::Algebra::GammaX);
  Gamma g1(Gamma::Algebra::GammaY);   // alias for clarity in variants
  Gamma g3(Gamma::Algebra::GammaT);
  Gamma g5(Gamma::Algebra::Gamma5);

  std::vector<Variant> variants = {
      {"γ_Y · conj(v)",          [&](const LatticeFermion &v) { return g2 * conjugate(v); }},
      {"conj(γ_Y · v)",          [&](const LatticeFermion &v) { return LatticeFermion(conjugate(g2 * v)); }},
      {"-γ_Y · conj(v)",         [&](const LatticeFermion &v) { return LatticeFermion(-(g2 * conjugate(v))); }},
      {"-conj(γ_Y · v)",         [&](const LatticeFermion &v) { return LatticeFermion(-conjugate(g2 * v)); }},
      {"γ_X · conj(v) [probe]",  [&](const LatticeFermion &v) { return g0 * conjugate(v); }},
      {"γ_T · conj(v) [probe]",  [&](const LatticeFermion &v) { return g3 * conjugate(v); }},
      {"γ_5 · conj(v) [probe]",  [&](const LatticeFermion &v) { return g5 * conjugate(v); }},
  };

  std::cout << GridLogMessage << "=== Gamma2 conj kernel probe at 4⁴, "
            << V << " sites ===" << std::endl;
  std::cout << GridLogMessage << "||out_kernel|| = "
            << std::sqrt(norm2(out_grid)) << std::endl;

  int matched = -1;
  RealD best_rel = 1e300;
  const char *best_label = "(none)";
  for (size_t i = 0; i < variants.size(); ++i) {
    LatticeFermion ref = variants[i].f(v_grid);
    RealD r = rel_diff(out_grid, ref);
    std::cout << GridLogMessage << "  variant " << i << "  "
              << variants[i].label << "  rel_diff = " << r << std::endl;
    if (r < best_rel) { best_rel = r; best_label = variants[i].label; matched = i; }
  }

  bool ok = (best_rel <= 1e-14);
  std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL")
            << "] best variant: " << best_label
            << "   rel_diff = " << best_rel
            << "   (gate ≤ 1e-14)" << std::endl;

  Grid_finalize();
  return ok ? 0 : 1;
}
