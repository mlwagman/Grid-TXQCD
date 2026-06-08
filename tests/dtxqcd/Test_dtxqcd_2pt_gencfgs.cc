// Step 1 of the DTXQCD 2pt Fierz-equivalence suite: generate DTXQCD
// configurations on 4^3 x 8.  Mirror of Test_txqcd_2pt_gencfgs.cc with the
// action roster updated to the diquark-tensor variant.  The QCD reference
// ensemble (configs_2pt_qcd_nf2) is shared with the TXQCD test, so it is
// NOT regenerated here -- run Test_txqcd_2pt_gencfgs first (or after) to
// produce the QCD half.
//
// Action structure (matches TXQCD's Wilson Fierz test):
//   L1 (inner, dt fine):   DTXQCDWilsonCloverRationalEOAction (csw=0 = Wilson)
//                        + DTXQCDLogDetCloverEOAction (csw=0)
//                        + DTXQCDAuxiliaryFieldGaussianAction
//   L2 (outer, mult=4):    Wilson plaquette via DTXQCDGaugeActionAdapter
//
// Integrator: ForceGradient, MDsteps=10, trajL=0.5, eps=0.05.  This is the
// same setting the TXQCD Wilson Fierz test runs at and a reasonable
// starting point for DTXQCD -- the per-pole force magnitudes in the
// 1/4-root RHMC differ from TXQCD's 1/2-root only at the residue
// distribution level, and the doubled fermion's per-site Mooee block is
// what the cached EO operator already handles.  Tune from here if dH
// drifts.

#include "Test_dtxqcd_2pt_utils.h"
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>

using namespace DtxqcdTest2pt;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;

  // ==================== DTXQCD ====================
  if (dtxqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "DTXQCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating DTXQCD configs (" << total_traj
              << " trajectories)..." << std::endl;
    mkdir_p(dtxqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_dtxqcd_checkpoint();

    // Rational parameters.  TXQCD's Wilson Fierz test gets away with
    // lo=1e-4 because TXQCD's Mpc^dag Mpc has a clean spectral gap at
    // mass=0.3.  DTXQCD's doubled operator picks up lower-eigenvalue
    // modes from the d, n off-diagonal coupling, and the multi-shift CG
    // poles near sigma ~ lo blow up if lo is below the actual spectrum
    // bottom -- the FD validation at 4^4 confirmed lo=0.1 was the right
    // production floor (raising lo, not lowering it, was the fix).
    RealD cg_tol = 1e-8;
    OneFlavourRationalParams rat_params(
        /*lo=*/1e-1, /*hi=*/64.0,
        /*MaxIter=*/cg_max, /*tolerance=*/cg_tol,
        /*degree=*/12, /*precision=*/64,
        /*BoundsCheckFreq=*/100,
        /*mdtolerance=*/1e-6);

    DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda);
    // csw=0 => Wilson (no clover term).  The rational/LogDet code paths
    // skip the clover assembly when csw == 0.
    DTXQCDLogDetCloverEOAction                   LogDet(Grid, RBGrid, mass, 0.0);
    DTXQCDWilsonCloverRationalEOAction
        PF(Grid, RBGrid, mass, rat_params, 0.0);

    // Three-level hierarchy.  TXQCD's Wilson Fierz test bundles AuxGaussian
    // into L1 with PF and LogDet -- a smoke run at that structure showed
    // catastrophic feedback on DTXQCD: the 1/4-root RHMC and the doubled
    // LogDet dump force into the aux slots, that pumps aux momentum, that
    // grows the aux fields, which in turn blows up AuxGaussian's
    // lambda^2 * aux force (observed Force_max ~ 4e4, Fdt_max ~ 1400 by
    // mid-trajectory on cold start).  Separating AuxGaussian onto its own
    // outer level with a x4 multiplier gives it its own time scale and
    // lets the gauge sub-integrator (also x4) carry the fast-varying
    // contributions.  Same structure gen_dtxqcd_cfgs uses.
    typedef Representations<EmptyRep<DTXQCDField>> Reps;
    ActionLevel<DTXQCDField, Reps> L1(1);
    L1.push_back(&PF);
    L1.push_back(&LogDet);
    ActionLevel<DTXQCDField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    ActionLevel<DTXQCDField, Reps> L3(4);
    L3.push_back(&AuxAction);
    ActionSet<DTXQCDField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);
    Aset.push_back(L3);

    IntegratorParameters MD;
    MD.name    = "ForceGradient";
    MD.MDsteps = 10;
    MD.trajL   = 0.5;

    DTXQCDField U(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming DTXQCD from checkpoint at traj "
                << latest << std::endl;
      LoadDtxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);
    }

    int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - no_metrop - start_traj;
    HMCp.NoMetropolisUntil   = no_metrop;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<DTXQCDCompositeImpl> Smear;
    typedef ForceGradient<DTXQCDCompositeImpl,
                          NoSmearing<DTXQCDCompositeImpl>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(U);

    CheckpointerParameters CPp;
    CPp.config_prefix = dtxqcd_cfg_dir() + "/ckpoint_lat";
    CPp.rng_prefix    = dtxqcd_cfg_dir() + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    DTXQCDCheckpointer ckpt(CPp);

    DtxqcdDiagnostics diag(dtxqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"PseudoFermion", &PF},
        {"LogDet",        &LogDet},
        {"AuxGaussian",   &AuxAction},
        {"Gauge",         &GaugeAction}
    }, Grid, RBGrid, pRNG, mass, n_vev_noise);

    std::vector<HmcObservable<DTXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD (Nf=2 Wilson) — shared with TXQCD test ====================
  // The QCD reference ensemble is generated by Test_txqcd_2pt_gencfgs in
  // configs_2pt_qcd_nf2/.  Re-generate it here if it's missing; otherwise
  // skip so a single sequential run of {txqcd_gencfgs, dtxqcd_gencfgs}
  // doesn't produce two copies.
  if (TxqcdTest2pt::qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 configs (" << total_traj
              << " trajectories)..." << std::endl;
    mkdir_p(TxqcdTest2pt::qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD from checkpoint at traj "
                << latest << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      SU<Nc>::ColdConfiguration(Umu);
    }

    WilsonFermionD FermOp(Umu, Grid, RBGrid, mass);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
    Nf2.is_smeared = false;

    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&Nf2);
    ActionLevel<LatticeGaugeField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name    = "ForceGradient";
    MD.MDsteps = 10;
    MD.trajL   = 0.5;

    int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - no_metrop - start_traj;
    HMCp.NoMetropolisUntil   = no_metrop;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef ForceGradient<PeriodicGimplR,
                          NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    TxqcdTest2pt::QcdCheckpointer ckpt;
    ckpt.cfg_prefix    = TxqcdTest2pt::qcd_cfg_dir() + "/ckpoint_lat";
    ckpt.rng_prefix    = TxqcdTest2pt::qcd_cfg_dir() + "/ckpoint_rng";
    ckpt.save_interval = meas_skip;

    TxqcdTest2pt::QcdDiagnostics diag(
        TxqcdTest2pt::qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
            {"Nf2",   &Nf2},
            {"Gauge", &GaugeAction}
        }, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "DTXQCD 2pt config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
