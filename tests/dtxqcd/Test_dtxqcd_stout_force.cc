// Finite-difference GAUGE-force test for the stout-smeared DTXQCD EO
// production actions.
//
// Gates two pieces that were previously untested on the EO production path:
//   1. The stout smearer DTXQCDSmearedConfiguration (gauge component through
//      Grid's stout chain, aux fields pass-through) -- the smeared-gauge force
//      chain rule.
//   2. The clover gauge force for the doubled 48x48 fermion on the EO path
//      (DTXQCDWilsonCloverRationalEOAction + DTXQCDLogDetCloverEOAction).
//
// Mechanics (mirrors tests/txqcd/Test_txqcd_stout_force.cc):
//   perturb thin links  U.U[mu] -> expMat(E[mu], h) * U[mu]
//   Smearer.set_Field(U)  (re-smears the gauge component),  S(Smearer)
//   central difference  (S(+h) - S(-h)) / 2h
//   analytic gauge force from deriv(Smearer, dSdU) (which applies the stout
//   chain rule via U.smeared_force) read as the Convention-A inner product
//     an = -2 * sum_mu Re Tr(E[mu] * dSdU.U[mu]).
//
// The aux fields are NOT perturbed here (the aux force is covered by
// Test_dtxqcd_rational_aux_force / Test_dtxqcd_logdet_aux_force); the aux
// fields pass through the stout map unchanged so they only set the saddle.
//
// Coverage (each a gauge_test call, tol 1e-3):
//   csw=1.25, Nsmear=2: PF (rational EO), LogDet EO, Gauge adapter <- KEY case
//   csw=1.25, Nsmear=0: PF (rational EO), LogDet EO  (isolates EO clover force)
//   csw=0,    Nsmear=2: PF (rational EO)             (isolates stout chain rule)
//
// Run single-GPU:
//   srun ... ./Test_dtxqcd_stout_force --grid 4.4.4.4 --mpi 1.1.1.1 --shm 256

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG   sRNG;
  pRNG.SeedFixedIntegers({71, 72, 73, 74});
  sRNG.SeedFixedIntegers({81, 82, 83, 84});

  // ---- composite field: hot gauge + doubled-aux init (scaled to a sane
  //      saddle), matching the aux roster used by the EO production actions ----
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);
  // The HMC subspace projection (must match the manifold the force kernel was
  // derived for; see Test_dtxqcd_rational_full_force).
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(U.sigma);
    DtxqcdRealSymmetricCFInPlace(U.pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
  }
  const RealD aux_scale = 0.2;  // small aux -> sane saddle
  U.sigma = aux_scale * U.sigma;
  U.pi    = aux_scale * U.pi;
  U.d     = aux_scale * U.d;
  U.n     = aux_scale * U.n;
  U.s     = aux_scale * U.s;
  U.p     = aux_scale * U.p;

  const RealD mass = 0.3;
  const RealD rho  = 0.1;
  const RealD beta = 5.6;
  const RealD h    = 1e-4;

  OneFlavourRationalParams rp(/*lo*/        1.0e-1,
                              /*hi*/        2.0e2,
                              /*MaxIter*/   20000,
                              /*tol*/       1.0e-12,
                              /*degree*/    12,
                              /*precision*/ 50,
                              /*BoundsCheckFreq*/ 0,
                              /*mdtol*/     1.0e-12);

  int exitcode = 0;

  // Shared random gauge-algebra perturbation directions (one set, reused by
  // every gauge_test so the FD samples the same direction the analytic force
  // is projected onto).
  std::array<LatticeColourMatrix, 4> Emu{
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid)};
  for (int mu = 0; mu < Nd; ++mu)
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

  // Reusable gauge-force FD check on the EO production actions.  Nsmear==0 takes
  // the unsmeared path (S(U)/deriv(U,.)/refresh(U,..)); Nsmear>0 routes through
  // the stout smearer (S(Smearer)/deriv(Smearer,.)/refresh(Smearer,..)).
  auto gauge_test = [&](const char *name, Action<DTXQCDField> &action,
                        unsigned int Nsmear, bool do_refresh) {
    Smear_Stout<PeriodicGimplR> Stout(rho);
    DTXQCDSmearedConfiguration Smearer(&Grid, Nsmear, Stout);
    action.is_smeared = (Nsmear > 0);

    // refresh (heatbath) once on the saddle config, BEFORE deriv/S evals so the
    // pseudofermion is held fixed across the central difference.
    if (do_refresh) {
      if (Nsmear > 0) {
        Smearer.set_Field(U);
        action.refresh(Smearer, sRNG, pRNG);
      } else {
        action.refresh(U, sRNG, pRNG);
      }
    }

    // ---- analytic gauge force ----
    DTXQCDField dSdU(&Grid);
    if (Nsmear > 0) {
      Smearer.set_Field(U);
      action.deriv(Smearer, dSdU);  // includes the stout chain rule
    } else {
      action.deriv(U, dSdU);
    }

    RealD an = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;  // Convention A

    // ---- central-difference FD: perturb thin links, re-smear, eval S ----
    LatticeGaugeField Usaved = U.U;
    auto eval_S = [&](RealD eps) -> RealD {
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], eps, 12) * Umu, mu);
      }
      RealD s;
      if (Nsmear > 0) {
        Smearer.set_Field(U);
        s = action.S(Smearer);
      } else {
        s = action.S(U);
      }
      U.U = Usaved;
      return s;
    };
    RealD Sp = eval_S(+h);
    RealD Sm = eval_S(-h);
    RealD fd = (Sp - Sm) / (2.0 * h);

    RealD rel = std::abs(an - fd) / std::max({std::abs(an), std::abs(fd), 1.0});
    bool pass = (rel < 1e-3);

    RealD nU = std::sqrt(norm2(dSdU.U));
    if (nU < 1e-10) {
      std::cout << GridLogError << "[" << name
                << "] gauge force vanishes (|dSdU.U|=" << nU << ")" << std::endl;
      pass = false;
    }

    std::cout << GridLogMessage << "[" << name << "] AN=" << an << " FD=" << fd
              << " rel=" << rel << " |dSdU.U|=" << nU
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  // ======================================================================
  //  csw = 1.25, Nsmear = 2  -- the KEY case: stout + clover, chroma-matched
  //                             gauge force on the EO production actions.
  // ======================================================================
  std::cout << GridLogMessage
            << "===== csw=1.25, Nsmear=2 (stout + clover) =====" << std::endl;
  {
    DTXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rp, 1.25);
    gauge_test("PF rational EO  csw=1.25 Nsmear=2", PF, 2, /*do_refresh=*/true);
  }
  {
    DTXQCDLogDetCloverEOAction LD(Grid, RBGrid, mass, 1.25);
    gauge_test("LogDet EO       csw=1.25 Nsmear=2", LD, 2, /*do_refresh=*/false);
  }
  {
    DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GA(beta);
    gauge_test("Gauge adapter   csw=1.25 Nsmear=2", GA, 2, /*do_refresh=*/false);
  }

  // ======================================================================
  //  csw = 1.25, Nsmear = 0  -- isolate the EO clover gauge force (unsmeared).
  // ======================================================================
  std::cout << GridLogMessage
            << "===== csw=1.25, Nsmear=0 (clover, no stout) =====" << std::endl;
  {
    DTXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rp, 1.25);
    gauge_test("PF rational EO  csw=1.25 Nsmear=0", PF, 0, /*do_refresh=*/true);
  }
  {
    DTXQCDLogDetCloverEOAction LD(Grid, RBGrid, mass, 1.25);
    gauge_test("LogDet EO       csw=1.25 Nsmear=0", LD, 0, /*do_refresh=*/false);
  }

  // ======================================================================
  //  csw = 0, Nsmear = 2  -- isolate the stout chain rule (no clover).
  // ======================================================================
  std::cout << GridLogMessage
            << "===== csw=0, Nsmear=2 (stout, no clover) =====" << std::endl;
  {
    DTXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rp, 0.0);
    gauge_test("PF rational EO  csw=0    Nsmear=2", PF, 2, /*do_refresh=*/true);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME DTXQCD STOUT FORCE CHECKS FAILED"
                         : "ALL DTXQCD STOUT FORCE CHECKS PASSED")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
