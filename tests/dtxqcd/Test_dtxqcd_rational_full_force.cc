// Test_dtxqcd_rational_full_force: finite-difference force test for the
// non-EO 1/4-root Pfaffian pseudofermion action
// (DTXQCDWilsonCloverRationalFullAction).  Sibling of
// Test_dtxqcd_rational_aux_force; same FD structure, but the action
// now operates on the full doubled M (not Mpc), and there is no
// LogDet companion to add.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_rational_full_force --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>

using namespace Grid;

static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(localInnerProduct(A.sigma, B.sigma))).real();
  r += TensorRemove(sum(localInnerProduct(A.pi,    B.pi))).real();
  r += 0.5 * TensorRemove(sum(localInnerProduct(A.t, B.t))).real();
  r += TensorRemove(sum(trace(A.d * transpose(B.d)))).real();
  r += TensorRemove(sum(trace(A.n * transpose(B.n)))).real();
  return r;
}

static void PerturbAux(const DTXQCDField &U, const DTXQCDField &Y, RealD scale,
                       DTXQCDField &out) {
  out.U     = U.U;
  out.sigma = U.sigma + scale * Y.sigma;
  out.pi    = U.pi    + scale * Y.pi;
  out.t     = U.t     + scale * Y.t;
  out.d     = U.d     + scale * Y.d;
  out.n     = U.n     + scale * Y.n;
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
  DtxqcdRealGaussian(pRNG, U.sigma);
  DtxqcdRealGaussian(pRNG, U.pi);
  DtxqcdGaussianAntisymTensor(pRNG, U.t);
  DtxqcdHermitianGaussian(pRNG, U.d);
  DtxqcdHermitianGaussian(pRNG, U.n);

  DTXQCDField Y(&Grid);
  Y.U = Zero();
  DtxqcdRealGaussian(pRNG, Y.sigma);
  DtxqcdRealGaussian(pRNG, Y.pi);
  DtxqcdGaussianAntisymTensor(pRNG, Y.t);
  DtxqcdHermitianGaussian(pRNG, Y.d);
  DtxqcdHermitianGaussian(pRNG, Y.n);

  const RealD mass = 0.4;
  const RealD h    = 1e-4;

  // Rational bracket: full M^dag M spectrum is wider than Mpc^dag Mpc by
  // the M_ee factor (~ kDim24 * (mass+4)^2 floor).  Keep the lo end the
  // same as the EO test, but bump hi to 200 to cover the wider top.
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
    Y_piece.t = Y.t;          Y_piece.d  = Y.d;          Y_piece.n  = Y.n;
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
    P.pi = Zero(); P.t = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("csw=0 pi-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.t = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("csw=0 t-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("csw=0 d-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.t = Zero(); P.n = Zero();
  });
  check_piece("csw=0 n-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.t = Zero(); P.d = Zero();
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
