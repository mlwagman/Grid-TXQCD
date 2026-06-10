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
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
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

    // Rational bracket tuned to Test_dtxqcd_spectrum measurements on
    // 4^3 x 8, mass = 0.3, csw = 0, weak gauge + thermal aux at
    // AUX_FLUCT_LAMBDA = 10 (see DTXQCDCompositeImpl::FillAuxFields):
    //
    //   lambda_min(Mpc^dag Mpc) ~ 0.11
    //   lambda_max(Mpc^dag Mpc) ~ 37
    //
    // bracket lo = 0.05 (well below lambda_min) and hi = 80 (1.5x
    // measured lambda_max + headroom for HMC drift) keeps the shifted
    // multi-shift CG well-conditioned.  AUX_FLUCT_LAMBDA = 10 is exported
    // below before the cold-start init.
    RealD cg_tol = 1e-8;
    // RAT_LO / RAT_HI / RAT_DEGREE env knobs let us tighten the rational
    // bracket to reduce 4th-order ForceGradient remainder.  Defaults match
    // the original EO-tuned values (broad bracket, high degree).
    RealD rat_lo     = 0.05;
    RealD rat_hi     = 80.0;
    int   rat_degree = 12;
    if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
    if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
    if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
    std::cout << GridLogMessage << "DTXQCD rational: lo=" << rat_lo
              << " hi=" << rat_hi << " degree=" << rat_degree << std::endl;
    OneFlavourRationalParams rat_params(
        /*lo=*/rat_lo, /*hi=*/rat_hi,
        /*MaxIter=*/cg_max, /*tolerance=*/cg_tol,
        /*degree=*/rat_degree, /*precision=*/64,
        /*BoundsCheckFreq=*/100,
        /*mdtolerance=*/1e-6);

    RealD lambda_run = lambda;
    if (const char *l = std::getenv("LAMBDA"); l && *l) {
      lambda_run = std::atof(l);
    }
    std::cout << GridLogMessage << "DTXQCD lambda = " << lambda_run << std::endl;
    RealD mass_run = mass;
    if (const char *m = std::getenv("MASS"); m && *m) {
      mass_run = std::atof(m);
    }
    std::cout << GridLogMessage << "DTXQCD mass = " << mass_run << std::endl;
    DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);
    // csw=0 => Wilson (no clover term).  The rational/LogDet code paths
    // skip the clover assembly when csw == 0.
    // USE_FULL_PF=1 swaps the EO Schur 1/4-root + LogDet pair for a single
    // non-EO 1/4-root pseudofermion on the full doubled M^dag M.  The
    // 2026-06-09 PSD diagnostic (Test_dtxqcd_psd_check) showed the EO Schur
    // Mpc explodes by 10^3-10^5 above aux_std ~ 0.5 while the full M stays
    // mild; the production HMC breakdowns at lambda=3 are EO-only.  Non-EO
    // bypasses the cliff at the cost of slower per-CG multishift.
    bool use_full_pf = false;
    if (const char *u = std::getenv("USE_FULL_PF"); u && *u) {
      use_full_pf = (std::atoi(u) != 0);
    }
    std::cout << GridLogMessage
              << "DTXQCD pseudofermion = "
              << (use_full_pf ? "FULL (non-EO)" : "EO Schur")
              << std::endl;

    DTXQCDLogDetCloverEOAction                   LogDet(Grid, RBGrid, mass_run, 0.0);
    DTXQCDWilsonCloverRationalEOAction
        PF(Grid, RBGrid, mass_run, rat_params, 0.0);
    DTXQCDWilsonCloverRationalFullAction
        PF_full(Grid, RBGrid, mass_run, rat_params, 0.0);

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
    if (use_full_pf) {
      L1.push_back(&PF_full);     // single rational on full M^dag M;
                                  // LogDet folded in (no companion).
    } else {
      L1.push_back(&PF);
      L1.push_back(&LogDet);
    }
    ActionLevel<DTXQCDField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    int aux_mult = 4;
    if (const char *m = std::getenv("AUX_MULT"); m && *m) {
      aux_mult = std::atoi(m);
    }
    std::cout << GridLogMessage << "DTXQCD aux multiplier = " << aux_mult << std::endl;
    ActionLevel<DTXQCDField, Reps> L3(aux_mult);
    L3.push_back(&AuxAction);
    ActionSet<DTXQCDField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);
    Aset.push_back(L3);

    IntegratorParameters MD;
    MD.name    = "ForceGradient";
    // MDSTEPS env: scaling experiment for the LogDet instability.  At
    // MDsteps = 10 + trajL = 0.1 (eps = 0.01), LogDet Fdt_max bounces
    // between sub-1 (stable) and 3000+ (CG failed, force trash) within
    // the first 3 MD substeps because per-site M_ee_48 sites cross
    // near-singular eigenvalues.  Halving eps via doubled MDsteps tells
    // us whether the instability is purely a step-size issue (stays
    // bounded longer with finer eps) or a per-trajectory cliff that
    // refinement can't reach (LogDet blows up at the same point in
    // configuration-space regardless of step size).
    MD.MDsteps = 10;
    if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) {
      MD.MDsteps = std::atoi(ms);
    }
    MD.trajL   = 0.1;
    if (const char *tl = std::getenv("TRAJL"); tl && *tl) {
      MD.trajL = std::atof(tl);
    }
    std::cout << GridLogMessage << "Integrator: MDsteps=" << MD.MDsteps
              << " trajL=" << MD.trajL
              << " eps=" << (MD.trajL / MD.MDsteps) << std::endl;

    DTXQCDField U(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming DTXQCD from checkpoint at traj "
                << latest << std::endl;
      LoadDtxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      // Thermal aux init at the spectrum-safe fluctuation width.  Without
      // AUX_FLUCT_LAMBDA the env-knob defaults to lambda = 3 -> outlier
      // aux sites near the Pfaffian sign boundary; FLUCT = 10 keeps the
      // doubled M well-conditioned on the cold gauge while HMC evolves
      // the aux toward the physical 1/lambda width.
      if (std::getenv("AUX_FLUCT_LAMBDA") == nullptr) setenv("AUX_FLUCT_LAMBDA", "10.0", 0);
      DTXQCDCompositeImpl::ThermalAuxConfiguration(pRNG, U, lambda_run, /*wf=*/0.1);
      if (const char *z = std::getenv("ZERO_DN_INIT"); z && std::atoi(z) != 0) {
        std::cout << GridLogMessage << "Zeroing d, n diquark fields at init" << std::endl;
        U.d = Zero();
        U.n = Zero();
      }
      if (const char *z = std::getenv("ZERO_ALL_AUX"); z && std::atoi(z) != 0) {
        std::cout << GridLogMessage << "Zeroing ALL aux fields at init (cold)" << std::endl;
        U.sigma = Zero();
        U.pi    = Zero();
        U.t     = Zero();
        U.d     = Zero();
        U.n     = Zero();
      }
    }

    int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
    if (const char *nm = std::getenv("NO_METROP"); nm && *nm) {
      no_metrop = std::atoi(nm);
    }
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
    if (const char *nm = std::getenv("NO_METROP"); nm && *nm) {
      no_metrop = std::atoi(nm);
    }
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
