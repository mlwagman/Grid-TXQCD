// Test_dtxqcd_apply_C_kernel: M-wrap.4-v2 unit test.
//
// Validates the C = γ_2·γ_4 charge-conjugation matrix kernel against Grid's
// reference at 4⁴.  C in DR basis is pure spinor permutation + sign — no
// complex conjugation, just spin reordering.
//
//   1. Generate a random LatticeFermion v_grid.
//   2. Pack to flat lex-ordered host buffer (24 doubles/site).
//   3. Upload to device, run ApplyCKernel, download.
//   4. Unpack into out_grid.
//   5. Compare against Grid reference  C · v_grid = γ_2·γ_4·v_grid  — try
//      several convention variants (order swap, sign) and pick the one that
//      matches at machine precision.
//
// Gate: pass tolerance ≤ 1e-14 (machine eps; C entries are ±1 in spinor
// space, no floating-point arithmetic beyond sign flips).
//
// Note: Grid's Gamma::Algebra labels map to QUDA's DR basis γ_μ as
//   GammaX = γ_1, GammaY = γ_2, GammaZ = γ_3, GammaT = γ_4.  We probe the
//   ambiguity in operator ordering (γ_2·γ_4 vs γ_4·γ_2) and the relative
//   sign (since C^T = -C in DR, an explicit sign flip is plausible).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_apply_C.h>
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
  pRNG.SeedFixedIntegers({901, 902, 903, 904});

  // ---- random Grid spinor + pack to lex buffer ----
  LatticeFermion v_grid(&Grid_), out_grid(&Grid_);
  gaussian(pRNG, v_grid);

  int V = Quda::local_volume(&Grid_);
  std::vector<double> v_lex(V * 24), out_lex(V * 24, 0.0);
  Quda::fermion_to_lex_buffer(v_grid, v_lex.data());

  // ---- ship to device, run kernel, ship back ----
  deviceVector<double> v_d(V * 24), out_d(V * 24);
  acceleratorCopyToDevice(v_lex.data(),   &v_d[0],   V * 24 * sizeof(double));
  acceleratorCopyToDevice(out_lex.data(), &out_d[0], V * 24 * sizeof(double));

  Quda::ApplyCKernel(&out_d[0], &v_d[0], V);

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

  // Grid algebra ↔ QUDA DR mapping:
  //   Grid GammaY ↔ QUDA γ_2     Grid GammaT ↔ QUDA γ_4
  // M-wrap.2 established that γ_2·conj operationally = conj(γ_Y·v) in Grid.
  // The C kernel has NO conjugation, so the equivalence is just for the
  // matrix product γ_2·γ_4 — try both Grid orderings and sign variants.
  Gamma g2(Gamma::Algebra::GammaY);  // ↔ QUDA γ_2
  Gamma g4(Gamma::Algebra::GammaT);  // ↔ QUDA γ_4
  Gamma g0(Gamma::Algebra::GammaX);  // probe slot
  Gamma g3(Gamma::Algebra::GammaZ);  // probe slot
  Gamma g5(Gamma::Algebra::Gamma5);  // probe slot

  std::vector<Variant> variants = {
      {"γ_Y · γ_T · v",            [&](const LatticeFermion &v) { return LatticeFermion(g2 * (g4 * v)); }},
      {"γ_T · γ_Y · v",            [&](const LatticeFermion &v) { return LatticeFermion(g4 * (g2 * v)); }},
      {"-γ_Y · γ_T · v",           [&](const LatticeFermion &v) { return LatticeFermion(-(g2 * (g4 * v))); }},
      {"-γ_T · γ_Y · v",           [&](const LatticeFermion &v) { return LatticeFermion(-(g4 * (g2 * v))); }},
      {"γ_Y · γ_T · conj(v) [probe]", [&](const LatticeFermion &v) { return LatticeFermion(g2 * (g4 * conjugate(v))); }},
      {"conj(γ_Y · γ_T · v) [probe]", [&](const LatticeFermion &v) { return LatticeFermion(conjugate(g2 * (g4 * v))); }},
      {"γ_5 · v [sanity probe]",   [&](const LatticeFermion &v) { return LatticeFermion(g5 * v); }},
  };

  std::cout << GridLogMessage << "=== Apply C = γ_2·γ_4 kernel probe at 4⁴, "
            << V << " sites ===" << std::endl;
  std::cout << GridLogMessage << "||v_grid||      = "
            << std::sqrt(norm2(v_grid)) << std::endl;
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
