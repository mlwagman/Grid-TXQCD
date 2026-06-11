// Test_dtxqcd_rational_full_force (v2): finite-difference force test for
// the non-EO 1/4-root Pfaffian pseudofermion action
// (DTXQCDWilsonCloverRationalFullAction) on the v2 aux roster.
//
// Same FD structure as v1 but the aux roster is now (sigma, pi, d, n,
// s, p) instead of (sigma, pi, t, d, n).  AuxInnerReal uses the
// matching transpose() convention on the four CF Hermitian fields and
// localInnerProduct on the singlet scalars.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_rational_full_force --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>

using namespace Grid;

static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(trace(A.sigma * adj(B.sigma)))).real();
  r += TensorRemove(sum(trace(A.pi    * adj(B.pi))))   .real();
  r += TensorRemove(sum(trace(A.d     * adj(B.d))))    .real();
  r += TensorRemove(sum(trace(A.n     * adj(B.n))))    .real();
  r += TensorRemove(sum(localInnerProduct(A.s, B.s))).real();
  r += TensorRemove(sum(localInnerProduct(A.p, B.p))).real();
  return r;
}

static void PerturbAux(const DTXQCDField &U, const DTXQCDField &Y, RealD scale,
                       DTXQCDField &out) {
  out.U     = U.U;
  out.sigma = U.sigma + scale * Y.sigma;
  out.pi    = U.pi    + scale * Y.pi;
  out.d     = U.d     + scale * Y.d;
  out.n     = U.n     + scale * Y.n;
  out.s     = U.s     + scale * Y.s;
  out.p     = U.p     + scale * Y.p;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--lat-2334") {
      latt = Coordinate(std::vector<int>{2, 2, 2, 4});
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({701, 702, 703, 704});
  GridSerialRNG sRNG;
  sRNG.SeedFixedIntegers({711, 712, 713, 714});

  int exitcode = 0;
  auto check = [&](const char *name, RealD num, RealD ana, RealD tol) {
    RealD rel = std::abs(num - ana)
              / std::max({std::abs(num), std::abs(ana), 1.0});
    bool ok = (rel < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " :  numeric = " << num << "  analytic = " << ana
              << "  rel = " << rel << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);

  DTXQCDField Y(&Grid);
  Y.U = Zero();
  DtxqcdHermitianCFGaussian(pRNG, Y.sigma);
  DtxqcdHermitianCFGaussian(pRNG, Y.pi);
  DtxqcdHermitianCFGaussian(pRNG, Y.d);
  DtxqcdHermitianCFGaussian(pRNG, Y.n);
  DtxqcdRealScalarGaussian(pRNG, Y.s);
  DtxqcdRealScalarGaussian(pRNG, Y.p);

  const RealD mass = 0.4;
  const RealD h    = 1e-4;

  OneFlavourRationalParams rp(/*lo*/        1.0e-1,
                              /*hi*/        2.0e2,
                              /*MaxIter*/   20000,
                              /*tol*/       1.0e-12,
                              /*degree*/    12,
                              /*precision*/ 50,
                              /*BoundsCheckFreq*/ 0,
                              /*mdtol*/     1.0e-12);

  // ---------- csw = 0 ----------
  DTXQCDWilsonCloverRationalFullAction action(Grid, RBGrid, mass, rp,
                                              /*csw=*/0.0);
  action.refresh(U, sRNG, pRNG);

  DTXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  auto check_piece = [&](const char *name, std::function<void(DTXQCDField&)> mask) {
    DTXQCDField Y_piece(&Grid);
    Y_piece = Zero();
    Y_piece.sigma = Y.sigma;  Y_piece.pi = Y.pi;
    Y_piece.d     = Y.d;      Y_piece.n  = Y.n;
    Y_piece.s     = Y.s;      Y_piece.p  = Y.p;
    Y_piece.U = Zero();
    mask(Y_piece);

    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbAux(U, Y_piece, +h, Up);
    PerturbAux(U, Y_piece, -h, Um);
    RealD num = (action.S(Up) - action.S(Um)) / (2.0 * h);
    RealD ana = AuxInnerReal(dSdU, Y_piece);
    check(name, num, ana, 1e-3);
  };

  check_piece("csw=0 sigma-only", [](DTXQCDField &P) {
    P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 pi-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 d-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 n-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 s-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.p = Zero();
  });
  check_piece("csw=0 p-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero();
  });
  check_piece("csw=0 all-aux", [](DTXQCDField &) {});

  auto gauge_fd_check = [&](DTXQCDWilsonCloverRationalFullAction &act,
                            const DTXQCDField &dS, const char *name) {
    std::array<LatticeColourMatrix, 4> Emu{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid)};
    for (int mu = 0; mu < Nd; ++mu)
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

    RealD ana = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dS.U, mu);
      ana += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    ana *= -2.0;

    LatticeGaugeField Usaved = U.U;

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu  = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], h, 12);
      PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
    }
    RealD Sp = act.S(U);
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu  = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], -h, 12);
      PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
    }
    RealD Sm = act.S(U);
    U.U = Usaved;

    RealD fd = (Sp - Sm) / (2.0 * h);
    check(name, fd, ana, 1e-3);

    RealD nU = std::sqrt(norm2(dS.U));
    std::cout << GridLogMessage << name << " |dSdU.U| = " << nU << std::endl;
    if (nU < 1e-10) {
      std::cout << GridLogError << "[FAIL] gauge force vanishes" << std::endl;
      exitcode = 1;
    }
  };

  gauge_fd_check(action, dSdU, "csw=0 gauge (hopping) FD vs analytic");

  // ---------- csw = 1.25 ----------
  DTXQCDWilsonCloverRationalFullAction action2(Grid, RBGrid, mass, rp,
                                               /*csw=*/1.25);
  pRNG.SeedFixedIntegers({701, 702, 703, 704});
  sRNG.SeedFixedIntegers({711, 712, 713, 714});
  action2.refresh(U, sRNG, pRNG);

  DTXQCDField dSdU2(&Grid);
  action2.deriv(U, dSdU2);
  {
    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbAux(U, Y, +h, Up);
    PerturbAux(U, Y, -h, Um);
    RealD num = (action2.S(Up) - action2.S(Um)) / (2.0 * h);
    RealD ana = AuxInnerReal(dSdU2, Y);
    check("csw=1.25 all-aux FD vs analytic", num, ana, 1e-4);
  }

  gauge_fd_check(action2, dSdU2, "csw=1.25 gauge (hopping+clover) FD vs analytic");

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
