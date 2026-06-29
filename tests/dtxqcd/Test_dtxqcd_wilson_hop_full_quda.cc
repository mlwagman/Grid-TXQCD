// Test_dtxqcd_wilson_hop_full_quda: equivalence check for
// USE_FULL_PF_QUDA_WILSON=1 path against the non-EO Grid backend (parent
// DTXQCDWilsonCloverRationalFullAction).
//
// Path A: parent DTXQCDWilsonCloverRationalFullAction
//   per-pole Wilson hop via Wu.DhopDeriv / Wl.DhopDeriv on the CPU.
//
// Path B: child DTXQCDWilsonCloverRationalFullActionQudaPrimitive with
//   USE_FULL_PF_QUDA_WILSON=1 and DTXQCD_TEST_QUDA_WILSON=1.
//   The child overrides AccumulateHoppingForceAllPoles to batch every
//   (flavor, pole) RHS through computeCloverWilsonForceWithSchurFields.
//
// What we compare: per-μ gauge force norm, cos, ratio (per-μ Ta- and raw).
// Aux components MUST match bit-for-bit (the override touches only the
// Wilson hop part of dSdU.U).
//
// W.1 convention hypothesis: off_scale_wilson = 1.0 (vs EO's 2.0 — no
// Schur (1±γ)/2 projector to compensate on non-EO mass-form halves).
//
// Run:
//   ./Test_dtxqcd_wilson_hop_full_quda --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullActionQudaPrimitive.h>
#include <cstdlib>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int exitcode = 0;
  auto check = [&](const char *name, RealD value, RealD target, RealD tol) {
    bool ok = (std::abs(value - target) < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << value << "  (target " << target
              << " tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Random U + aux ----------
  GridParallelRNG pRNG(&Grid);  pRNG.SeedFixedIntegers({701, 702, 703, 704});

  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian (pRNG, U.s);
  DtxqcdRealScalarGaussian (pRNG, U.p);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(U.sigma);
    DtxqcdRealSymmetricCFInPlace(U.pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
  }

  // FD-test mass: positive (mass=0.4), matching Test_dtxqcd_qudafull_force.
  // csw at production b6.1 value (1.249) so the Wilson hop code path is
  // exercised realistically (csw == 0 makes the QUDA path fall back).
  const RealD mass = 0.4;
  const RealD csw  = 1.24930970916466;
  const RealD kappa = 0.5 / (4.0 + mass);
  std::cout << GridLogMessage
            << "Setup: mass=" << mass << " csw=" << csw << " κ=" << kappa
            << std::endl;

  int rat_degree = 6;
  if (const char *e_d = std::getenv("DTXQCD_TEST_RAT_DEGREE")) {
    if (*e_d) rat_degree = std::atoi(e_d);
  }
  OneFlavourRationalParams rp(/*lo*/        1.0e-1,
                              /*hi*/        2.0e2,
                              /*MaxIter*/   20000,
                              /*tol*/       1.0e-8,
                              /*degree*/    rat_degree,
                              /*precision*/ 50,
                              /*BoundsCheckFreq*/ 0,
                              /*mdtol*/     1.0e-8);
  std::cout << GridLogMessage
            << "[FD test] RAT_DEGREE = " << rat_degree << std::endl;

  // Two action instances — parent (Grid) and child (QUDA hop).
  // Per the σ-piece test pattern: each refresh() consumes RNG, so each action
  // gets its OWN RNG seeded identically.
  GridParallelRNG pRNG_A(&Grid);  pRNG_A.SeedFixedIntegers({901, 902, 903, 904});
  GridParallelRNG pRNG_B(&Grid);  pRNG_B.SeedFixedIntegers({901, 902, 903, 904});
  GridSerialRNG sRNG_A;           sRNG_A.SeedFixedIntegers({611, 612, 613, 614});
  GridSerialRNG sRNG_B;           sRNG_B.SeedFixedIntegers({611, 612, 613, 614});

  DTXQCDWilsonCloverRationalFullAction action_A(
      Grid, RBGrid, mass, rp, csw);
  DTXQCDWilsonCloverRationalFullActionQudaPrimitive action_B(
      Grid, RBGrid, mass, rp, csw);

  action_A.refresh(U, sRNG_A, pRNG_A);
  action_B.refresh(U, sRNG_B, pRNG_B);

  DTXQCDField dSdU_A(&Grid), dSdU_B(&Grid);

  // ---- Path A: parent (Grid backend) — env gate OFF ----
  unsetenv("USE_FULL_PF_QUDA_WILSON");
  std::cout << GridLogMessage << "Path A: Grid Wilson hop (parent class)"
            << std::endl;
  action_A.deriv(U, dSdU_A);

  // ---- Path B: child with env gate ON ----
  setenv("USE_FULL_PF_QUDA_WILSON", "1", /*overwrite=*/1);
  std::cout << GridLogMessage
            << "Path B: USE_FULL_PF_QUDA_WILSON=1 (child class)"
            << std::endl;
  action_B.deriv(U, dSdU_B);

  // ---- Compare gauge dSdU.U per μ via Ta-projection and raw norm ----
  auto Ta_of = [&](const LatticeGaugeField &G) {
    LatticeGaugeField T(&Grid);
    for (int mu = 0; mu < Nd; ++mu)
      PokeIndex<LorentzIndex>(T, Ta(PeekIndex<LorentzIndex>(G, mu)), mu);
    return T;
  };
  LatticeGaugeField Ta_A = Ta_of(dSdU_A.U);
  LatticeGaugeField Ta_B = Ta_of(dSdU_B.U);

  std::cout << GridLogMessage
            << "===== Path A vs Path B equivalence =====" << std::endl;

  RealD n_A_total = norm2(Ta_A);
  RealD n_B_total = norm2(Ta_B);
  ComplexD ip_total = innerProduct(Ta_A, Ta_B);
  RealD cos_total = (n_A_total > 0 && n_B_total > 0)
                       ? real(ip_total) / std::sqrt(n_A_total * n_B_total) : 0.0;
  RealD ratio_total = (n_A_total > 0) ? std::sqrt(n_B_total / n_A_total) : 0.0;
  std::cout << GridLogMessage
            << "  Ta total: |A|²=" << n_A_total
            << " |B|²=" << n_B_total
            << " cos=" << cos_total
            << " |B|/|A|=" << ratio_total << std::endl;
  check("Ta total cos", cos_total, 1.0, 1e-4);
  check("Ta total |B|/|A|", ratio_total, 1.0, 1e-3);

  for (int mu = 0; mu < Nd; ++mu) {
    auto T_A_mu = PeekIndex<LorentzIndex>(Ta_A, mu);
    auto T_B_mu = PeekIndex<LorentzIndex>(Ta_B, mu);
    RealD nA = norm2(T_A_mu);
    RealD nB = norm2(T_B_mu);
    ComplexD ip = innerProduct(T_A_mu, T_B_mu);
    RealD cos_mu = (nA > 0 && nB > 0) ? real(ip) / std::sqrt(nA * nB) : 0.0;
    RealD ratio_mu = (nA > 0) ? std::sqrt(nB / nA) : 0.0;
    std::cout << GridLogMessage
              << "  μ=" << mu
              << "  |A|²=" << nA << "  |B|²=" << nB
              << "  cos=" << cos_mu << "  |B|/|A|=" << ratio_mu << std::endl;
    char namebuf[64];
    std::snprintf(namebuf, sizeof(namebuf), "μ=%d cos", mu);
    check(namebuf, cos_mu, 1.0, 1e-4);
    std::snprintf(namebuf, sizeof(namebuf), "μ=%d |B|/|A|", mu);
    check(namebuf, ratio_mu, 1.0, 1e-3);
  }

  // ---- Aux components should be IDENTICAL between A and B (the override
  // touches only the Wilson hop part of dSdU.U; aux + clover-σ go through
  // the unchanged AccumulateSiteForcesAll and the clover Cmunu chain). ----
  RealD aux_diff_sigma = norm2(LatticeDtxqcdSigma(dSdU_A.sigma - dSdU_B.sigma));
  RealD aux_diff_pi    = norm2(LatticeDtxqcdPi   (dSdU_A.pi    - dSdU_B.pi));
  RealD aux_diff_d     = norm2(LatticeDtxqcdD    (dSdU_A.d     - dSdU_B.d));
  RealD aux_diff_n     = norm2(LatticeDtxqcdN    (dSdU_A.n     - dSdU_B.n));
  RealD aux_diff_s     = norm2(LatticeDtxqcdS    (dSdU_A.s     - dSdU_B.s));
  RealD aux_diff_p     = norm2(LatticeDtxqcdP    (dSdU_A.p     - dSdU_B.p));
  std::cout << GridLogMessage
            << "Aux diff norms (expect ≈ 0):"
            << "  σ=" << aux_diff_sigma
            << "  π=" << aux_diff_pi
            << "  d=" << aux_diff_d
            << "  n=" << aux_diff_n
            << "  s=" << aux_diff_s
            << "  p=" << aux_diff_p << std::endl;
  check("aux σ diff", aux_diff_sigma, 0.0, 1e-12);
  check("aux π diff", aux_diff_pi,    0.0, 1e-12);
  check("aux d diff", aux_diff_d,     0.0, 1e-12);
  check("aux n diff", aux_diff_n,     0.0, 1e-12);
  check("aux s diff", aux_diff_s,     0.0, 1e-12);
  check("aux p diff", aux_diff_p,     0.0, 1e-12);

  std::cout << GridLogMessage
            << (exitcode == 0 ? "[QUDA-WILSON-W.1 PASS]"
                              : "[QUDA-WILSON-W.1 FAIL]")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
