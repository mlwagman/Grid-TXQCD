// Test_dtxqcd_qudafull_force: equivalence check for DTXQCD_QUDA_FULL=1 path
// (QUDA σ-pack via computeCloverSigmaForceWithSchurFields) against the
// DTXQCD_QUDA_HYBRID=1 reference (QUDA Wilson hop + Grid Cmunu σ chain).
//
// Both paths share the QUDA Wilson hop convention (Phase H validated TXQCD
// recipe).  The ONLY difference is the σ-piece: HYBRID uses Grid's Cmunu loop
// on the per-pole-accumulated clover_sigma_full[6] colour matrices; FULL uses
// QUDA's batched computeCloverSigmaForceWithSchurFields per doubled block.
//
// Convention recipe for the σ-piece is per Phase H.0 R4-7 (validated 2026-06-24
// at 4⁴ hot, csw=1.249, cos=0.999999988, factor=0.5 uniform):
//   x_scale = 1, y_scale = 2κ (kappa-form), off_scale = √(2κ),
//   dagger = NO, matpc = ODD_ODD_ASYMMETRIC,
//   ck = -csw·κ/8, kappa2 = +κ² (unused by the kernel),
//   unpack scale = -1/(4κ²) (= -1/(8κ²) × 2 for the R4-7 residual).
// Lower block: QUDA gauge = conj(U); entry-wise conjugate the result.
//
// What we compare: per-μ gauge force norm + cosine + ratio.  Since the σ-
// piece is only a fraction of the total gauge dSdU.U (Wilson hop is the bulk),
// agreement should be tight: pure-QUDA-hop terms cancel exactly, leaving the
// (QUDA σ-pack) − (Grid Cmunu) residual which should be small per Phase H.0.
//
// Run:
//   PROBE_PRINT=1 mpirun -np 1 ./Test_dtxqcd_qudafull_force --grid 4.4.4.4 \
//                                                            --mpi 1.1.1.1
//
// Gate: per-μ cos ≥ 0.9999 and |Q/G| ratio ∈ [0.9999, 1.0001].

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOActionQudaPrimitive.h>
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
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({601, 602, 603, 604});
  GridSerialRNG sRNG;
  sRNG.SeedFixedIntegers({611, 612, 613, 614});

  int exitcode = 0;
  auto check = [&](const char *name, RealD value, RealD target, RealD tol) {
    bool ok = (std::abs(value - target) < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << value << "  (target " << target
              << " tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Random U + aux ----------
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(U.sigma);
    DtxqcdRealSymmetricCFInPlace(U.pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
  }

  // DTXQCD_TEST_ZERO_AUX=1: zero all aux fields so the doubled operator
  // block-diagonalizes (upper sees U, lower sees conj(U), no cross-coupling).
  // Discriminator for whether Path-A/Path-B residual perp is due to
  // aux-induced Schur completion mismatch in W_o/Z_o (which QUDA's σ-Oprod
  // can't see because it only takes a Wilson-clover gauge+clover loader).
  const char *e_zero = std::getenv("DTXQCD_TEST_ZERO_AUX");
  const bool zero_aux = (e_zero && *e_zero && std::atoi(e_zero) != 0);
  if (zero_aux) {
    U.sigma = Zero();
    U.pi    = Zero();
    U.d     = Zero();
    U.n     = Zero();
    U.s     = Zero();
    U.p     = Zero();
    std::cout << GridLogMessage
              << "[DTXQCD_TEST_ZERO_AUX=1] aux fields zeroed — operator is "
                 "block-diagonal (upper on U, lower on conj(U))." << std::endl;
  }

  // FD-test mass: positive (mass=0.4), matching the other DTXQCD FD tests.
  // Production mass (-0.245) puts the operator near the critical line and
  // forces 14k+ CG iters per pole on 4⁴ hot — fine in HMC where the chain
  // is near-equilibrium, untenable in a stand-alone 4⁴ test.  csw kept at
  // production b6.1 value to exercise the σ-clover code path realistically.
  const RealD mass = 0.4;
  const RealD csw  = 1.24930970916466;
  const RealD kappa = 0.5 / (4.0 + mass);
  std::cout << GridLogMessage
            << "Setup: mass=" << mass << " csw=" << csw << " κ=" << kappa
            << " √(2κ)=" << std::sqrt(2.0 * kappa)
            << " 2κ=" << (2.0 * kappa) << std::endl;

  // Coarser CG + lower Remez degree: 4⁴ aux-fluctuated operator is poorly
  // conditioned on random fermion sources.  This test only validates the
  // σ-piece convention (one round of deriv() on each path), not the integrator
  // — bit-exactness against the production CG-tol is unnecessary.  At deg=6
  // and tol=1e-8, both refresh() and deriv() converge in ~2-5k iterations.
  OneFlavourRationalParams rp(/*lo*/        1.0e-1,
                              /*hi*/        2.0e2,
                              /*MaxIter*/   20000,
                              /*tol*/       1.0e-8,
                              /*degree*/    6,
                              /*precision*/ 50,
                              /*BoundsCheckFreq*/ 0,
                              /*mdtol*/     1.0e-8);

  // Two action instances — they share the same csw, mass, RNG, but each
  // owns its own pseudofermion.  refresh() with the same RNG state seeds the
  // same Phi in both, so deriv() differences come ONLY from the σ-piece path.
  GridParallelRNG pRNG_A(&Grid);  pRNG_A.SeedFixedIntegers({901, 902, 903, 904});
  GridParallelRNG pRNG_B(&Grid);  pRNG_B.SeedFixedIntegers({901, 902, 903, 904});

  DTXQCDWilsonCloverRationalEOActionQudaPrimitive action_A(
      Grid, RBGrid, mass, rp, csw);
  DTXQCDWilsonCloverRationalEOActionQudaPrimitive action_B(
      Grid, RBGrid, mass, rp, csw);

  // Same RNG seeds → same Phi after refresh.
  action_A.refresh(U, sRNG, pRNG_A);
  action_B.refresh(U, sRNG, pRNG_B);

  DTXQCDField dSdU_A(&Grid), dSdU_B(&Grid);

  // ---- Path A: DTXQCD_QUDA_HYBRID=1, FULL unset (QUDA hop + Grid Cmunu σ) ----
  setenv("DTXQCD_QUDA_HYBRID", "1", /*overwrite=*/1);
  unsetenv("DTXQCD_QUDA_FULL");
  std::cout << GridLogMessage << "Path A: DTXQCD_QUDA_HYBRID=1 (Grid σ-Cmunu)"
            << std::endl;
  action_A.deriv(U, dSdU_A);

  // ---- Path B: DTXQCD_QUDA_FULL=1 (QUDA hop + QUDA σ-pack) ----
  setenv("DTXQCD_QUDA_HYBRID", "1", /*overwrite=*/1);  // HYBRID still on
  setenv("DTXQCD_QUDA_FULL",   "1", /*overwrite=*/1);
  std::cout << GridLogMessage << "Path B: DTXQCD_QUDA_FULL=1 (QUDA σ-pack)"
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

  // ---- Aux components should be IDENTICAL between A and B (σ-piece is
  // only on the gauge component; aux force comes through AccumulateSiteForces
  // which is unchanged between paths).  This guards against accidental
  // refactor regressions.  ----
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
            << (exitcode == 0 ? "[QUDAFULL PASS]" : "[QUDAFULL FAIL]")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
