// QCD LogDet Clover EO force test.
// Validates QCDLogDetCloverEOAction against TXQCDLogDetCloverEOAction at aux=0.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG sRNG;
  pRNG.SeedFixedIntegers({11, 12, 13, 14});
  sRNG.SeedFixedIntegers({11, 12, 13, 14});

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  RealD mass = 0.3;
  RealD csw = 1.0;
  int exitcode = 0;

  auto check = [&](const char *name, RealD an, RealD fd) {
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    bool pass = rel < 1e-4;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  // Random Lie-algebra direction for FD tests
  std::array<LatticeColourMatrix, 4> Emu{
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid)};
  for (int mu = 0; mu < Nd; ++mu)
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

  // TXQCD setup (reference)
  TXQCDField Utx(&Grid);
  Utx.U = U;
  Utx.sigma = Zero(); Utx.pi = Zero();
  Utx.s = Zero(); Utx.p = Zero(); Utx.t = Zero();
  TXQCDLogDetCloverEOAction txaction(Grid, RBGrid, mass, csw);

  // QCD setup
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(U, Grid, RBGrid, mass, csw, csw);
  QCDLogDetCloverEOAction<WilsonImplR> qcdaction(Dw);

  // ============================================================
  // Test 1: S values match
  // ============================================================
  std::cout << GridLogMessage << "===== Test 1: S value comparison =====" << std::endl;

  RealD S_tx = txaction.S(Utx);
  RealD S_qcd = qcdaction.S(U);
  {
    RealD rel = std::abs(S_tx - S_qcd) / std::abs(S_tx);
    bool pass = rel < 1e-12;
    std::cout << GridLogMessage << "S_txqcd = " << S_tx << std::endl;
    std::cout << GridLogMessage << "S_qcd   = " << S_qcd << std::endl;
    std::cout << GridLogMessage << "rel     = " << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ============================================================
  // Test 2: TXQCD force FD test (baseline, should PASS)
  // ============================================================
  std::cout << GridLogMessage << "===== Test 2: TXQCD force (baseline) =====" << std::endl;

  TXQCDField dSdU_tx(&Grid);
  txaction.deriv(Utx, dSdU_tx);

  // TXQCD FD: an = -2 * Re Tr(E * F_A), fd = (S(+h)-S(-h))/(2h)
  RealD an_tx = 0;
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU_tx.U, mu);
    an_tx += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
  }
  an_tx *= -2.0;

  RealD h = 1e-4;
  {
    LatticeGaugeField Usaved = Utx.U;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      PokeIndex<LorentzIndex>(Utx.U, expMat(Emu[mu], h, 12) * Umu, mu);
    }
    RealD Sp = txaction.S(Utx);
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      PokeIndex<LorentzIndex>(Utx.U, expMat(Emu[mu], -h, 12) * Umu, mu);
    }
    RealD Sm = txaction.S(Utx);
    Utx.U = Usaved;
    check("TXQCD gauge", an_tx, (Sp - Sm) / (2.0 * h));
  }

  // ============================================================
  // Test 3: QCD force FD test (Grid pseudofermion convention)
  //   UdSdU = -dS/dU, so dSpred = -deltaS (ratio → -1)
  // ============================================================
  std::cout << GridLogMessage << "===== Test 3: QCD LogDet force =====" << std::endl;

  LatticeGaugeField dSdU_qcd(&Grid);
  qcdaction.deriv(U, dSdU_qcd);

  RealD dt = 1e-4;
  RealD dSpred = 0;
  {
    LatticeComplex dS_lc(&Grid);
    dS_lc = Zero();
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(dSdU_qcd, mu);
      fmu = Ta(fmu) * 2.0;
      dS_lc = dS_lc + trace(Emu[mu] * fmu) * dt;
    }
    dSpred = TensorRemove(sum(dS_lc)).real();

    LatticeGaugeField Uprime(U);
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(U, mu);
      PokeIndex<LorentzIndex>(Uprime, expMat(Emu[mu], dt, 12) * Umu, mu);
    }
    RealD deltaS = qcdaction.S(Uprime) - qcdaction.S(U);
    check("QCD LogDet force", -dSpred, deltaS);
  }

  // ============================================================
  // Test 4: QCD force matches +2*MeeDeriv basis sum
  //   (sign flipped to match Grid pseudofermion convention)
  // ============================================================
  std::cout << GridLogMessage << "===== Test 4: force vs MeeDeriv basis =====" << std::endl;
  {
    Dw.ImportGauge(U);

    LatticeGaugeField force_basis(&Grid);
    force_basis = Zero();

    LatticeFermion e_i(&RBGrid);
    LatticeFermion Minv_ei(&RBGrid);
    LatticeGaugeField tmp_force(&Grid);

    for (int s = 0; s < Ns; s++) {
      for (int c = 0; c < Nc; c++) {
        e_i = Zero();
        {
          autoView(e_v, e_i, CpuWrite);
          thread_foreach(ss, e_v, {
            e_v[ss]._internal._internal[s]._internal[c] = 1.0;
          });
        }
        e_i.Checkerboard() = Even;
        Dw.MooeeInv(e_i, Minv_ei);
        Dw.MeeDeriv(tmp_force, Minv_ei, e_i, DaggerNo);
        force_basis = force_basis + tmp_force;
      }
    }

    LatticeGaugeField expected = 2.0 * force_basis;
    LatticeGaugeField diff = dSdU_qcd - expected;
    RealD rdiff = std::sqrt(norm2(diff)) / std::sqrt(norm2(expected));
    bool pass = rdiff < 1e-12;
    std::cout << GridLogMessage << "||deriv - (+2*basis)|| / ||basis|| = " << rdiff
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;

    Dw.ImportGauge(U);
  }

  // ============================================================
  // Test 5: QCD force matches TXQCD force at aux=0
  //   Grid pseudofermion convention has extra sign relative to TXQCD:
  //   dSpred_grid = -dSpred_txqcd
  // ============================================================
  std::cout << GridLogMessage << "===== Test 5: QCD vs TXQCD force =====" << std::endl;
  {
    RealD an_qcd_grid = dSpred;
    std::cout << GridLogMessage << "dSpred (QCD Grid conv) = " << an_qcd_grid << std::endl;
    std::cout << GridLogMessage << "dSpred (TXQCD conv)    = " << an_tx * dt << std::endl;
    RealD rel = std::abs(an_qcd_grid + an_tx * dt) / std::abs(an_qcd_grid);
    bool pass = rel < 1e-10;
    std::cout << GridLogMessage << "rel = " << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
