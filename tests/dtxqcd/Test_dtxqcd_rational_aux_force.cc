// Test_dtxqcd_rational_aux_force: finite-difference force test for the
// aux-field derivatives of DTXQCDWilsonCloverRationalEOAction (1/4-root Pfaffian
// pseudofermion on the Schur complement Mpc).
//
// For the analytic deriv() to match the FD (S(U + h Y) - S(U - h Y))/(2h), the
// pseudofermion Phi must be HELD FIXED while we vary U.  Phi_ is set once at
// the start by refresh() with a fixed RNG seed; S() and deriv() are called
// without an intervening refresh.
//
// Aux directions exercise the full aux force chain (sigma^A, pi^A,
// t^A_{mu,nu}, d, n) at csw = 0 (no clover) and csw = 1.25 (with clover, but
// still aux-only perturbation — gauge held fixed).  Gauge directions exercise
// the hopping (and clover) gauge force chain at csw = 0 (hopping only) and
// csw = 1.25 (hopping + clover).  All checks compare FD vs analytic deriv()
// after a single fixed-seed refresh().
//
// Run: ./tests/dtxqcd/Test_dtxqcd_rational_aux_force --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>

using namespace Grid;

// Same convention as Test_dtxqcd_logdet_aux_force: sigma/pi/t are real triplets
// in a complex container; d, n are Hermitian color matrices stored as full
// complex with the natural-Wirtinger convention.  See comments there.
static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(localInnerProduct(A.sigma, B.sigma))).real();
  r += TensorRemove(sum(localInnerProduct(A.pi,    B.pi))).real();
  // antisymmetric t double-counts (mu<nu and nu<mu), so divide by 2.
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
  // Default 4^4.  Pass --lat-2334 to fall back to the smaller 2^3 x 4
  // volume used during initial bring-up.  The SIMD-cached EO makes 4^4
  // tractable for FD validation now that Mooee / MooeeInv are O(48^2) per
  // site per call rather than O(48^3) (per-call LU was the v1 bottleneck).
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

  // ---------- Random U + aux + direction ----------
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

  // Rational params: lo/hi bracket the Mpc^dag Mpc spectrum.  At 4^4 with
  // mass=0.4, csw=0..1.25 and random aux, the spectrum extends from O(0.1)
  // (Wilson-Schur "mass" floor for this mass) to O(50).  Pushing lo too
  // small (1e-5) puts a multi-shift pole at sigma_min ~ lo, where the
  // shifted operator's condition number blows up and CG fails to converge
  // in 5000 iters.  lo=0.1 brackets the spectrum from below without
  // creating ill-conditioned shifted systems; MaxIter bumped to 20000 so
  // the lowest-shift solve has headroom.
  OneFlavourRationalParams rp(/*lo*/        1.0e-1,
                              /*hi*/        64.0,
                              /*MaxIter*/   20000,
                              /*tol*/       1.0e-12,
                              /*degree*/    12,
                              /*precision*/ 50,
                              /*BoundsCheckFreq*/ 0,
                              /*mdtol*/     1.0e-12);

  // ---------- csw = 0 ----------
  DTXQCDWilsonCloverRationalEOAction action(Grid, RBGrid, mass, rp, /*csw=*/0.0);
  action.refresh(U, sRNG, pRNG);             // sets Phi_ from frozen RNG seq.

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

  // ---------- csw = 0 gauge perturbation ----------
  // Perturb U_mu -> exp(h E_mu) U_mu and compare FD vs analytic dSdU.U.
  // The integrator-convention "Convention A" force gives
  //     dS/dh = -2 Re Tr(E_mu * F_mu) summed over Lorentz.
  // (Mirrors LogDet's gauge-FD test.)
  auto gauge_fd_check = [&](DTXQCDWilsonCloverRationalEOAction &act,
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
  DTXQCDWilsonCloverRationalEOAction action2(Grid, RBGrid, mass, rp, /*csw=*/1.25);
  // Reset RNGs so Phi_ in action2 starts from the same draw structure (the
  // exact value differs from action — refresh applies a different operator).
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
    check("csw=1.25 all-aux FD vs analytic", num, ana, 1e-5);
  }

  gauge_fd_check(action2, dSdU2, "csw=1.25 gauge (hopping+clover) FD vs analytic");

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
