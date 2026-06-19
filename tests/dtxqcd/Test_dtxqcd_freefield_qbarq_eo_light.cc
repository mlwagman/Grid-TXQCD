// Test_dtxqcd_freefield_qbarq_eo_light:
//
// Light-mass DTXQCD free-field Fierz test using the EO action stack
// (production-matching): RationalEO on M_pc + LogDetCloverEO on M_ee +
// AuxGaussian.  Sibling: Test_dtxqcd_freefield_qbarq_light uses
// RationalFullAction alone (no LogDet); both must agree on Σ_DTX/Σ_W and
// ⟨s⟩ saddle if the Pfaffian factor accounting is consistent across
// stacks.
//
// Default corner: m=0.1 (κ ≈ 0.122) and λ=10.  This is the
// BC-sensitive regime where the silent PBC default in DTXQCDMeooeDoubled
// produced a 7%-style gap before the APBC fix; the test acts as a
// regression guard for that.
//
// PASS criterion: |Σ_DTXQCD / Σ_W − 1| < PASS_TOL (default 0.02).
//
// Env knobs: MASS, LAMBDA, CSW, MDSTEPS, TRAJL, N_THERM, N_PROD
//            RAT_LO, RAT_HI, RAT_DEGREE, MD_CG_TOL, N_NOISE
//            GAUGE_INIT (cold|tepid[:AMP]|hot|nersc:PATH)
//            PASS_TOL (default 0.02)
//            CFG_DIR (default free_dtxqcd_light)

#include "Test_dtxqcd_2pt_utils.h"
#include "Test_dtxqcd_fierz_check_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Light-mass defaults — exercises the APBC convention strongly.
  RealD lambda_run = 10.0;
  RealD mass_run   = 0.1;
  RealD csw_run    = 0.0;
  int mdsteps      = 20;
  RealD trajL      = 1.0;
  int n_therm_run  = 30;
  int n_prod_run   = 40;
  int meas_skip_run = 4;
  std::string cfg_dir = "free_dtxqcd_eo_light";

  if (const char *v = std::getenv("LAMBDA");    v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");      v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW");       v && *v) csw_run    = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");   v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");     v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");   v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");    v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");   v && *v) cfg_dir = v;

  RealD pass_tol = 0.02;
  if (const char *v = std::getenv("PASS_TOL"); v && *v) pass_tol = std::atof(v);

  int n_noise = 64;
  if (const char *v = std::getenv("N_NOISE"); v && *v) n_noise = std::atoi(v);
  RealD meas_cg_tol = 1e-10;
  if (const char *v = std::getenv("MEAS_CG_TOL"); v && *v) meas_cg_tol = std::atof(v);

  RealD md_cg_tol = 1e-6;
  if (const char *v = std::getenv("MD_CG_TOL"); v && *v) md_cg_tol = std::atof(v);

  // Bracket sized for M48 spectrum at m=0.1: eigenvalues ~ (m+4)² .. ~70
  RealD rat_lo = 0.005, rat_hi = 80.0;
  int rat_deg = 12;
  if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo  = std::atof(v);
  if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi  = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_deg = std::atoi(v);

  std::cout << GridLogMessage
            << "DTXQCD FREE-FIELD light test:"
            << " mass=" << mass_run << " (κ=" << 1.0/(2.0*(4.0+mass_run)) << ")"
            << " lambda=" << lambda_run << " csw=" << csw_run
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " N_THERM=" << n_therm_run << " N_PROD=" << n_prod_run
            << " RAT[" << rat_lo << "," << rat_hi << "]^" << rat_deg
            << " PASS_TOL=" << pass_tol << std::endl;

  std::vector<int> latt_dims{4, 4, 4, 8};
  Coordinate latt(latt_dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);

  int total_traj = n_therm_run + n_prod_run;
  TxqcdTest2pt::mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid_);
  int rng_off = 0;
  if (const char *v = std::getenv("RNG_SEED_OFFSET"); v && *v) rng_off = std::atoi(v);
  sRNG.SeedFixedIntegers({1 + rng_off, 2 + rng_off, 3 + rng_off, 4 + rng_off, 5 + rng_off});
  pRNG.SeedFixedIntegers({6 + rng_off, 7 + rng_off, 8 + rng_off, 9 + rng_off, 10 + rng_off});

  RealD beta_dummy = 6.0;
  DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_dummy);
  DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);

  OneFlavourRationalParams rat_params(rat_lo, rat_hi, /*MaxIter=*/30000,
                                       /*tol=*/1e-10,
                                       rat_deg, 64, /*BCFreq=*/100,
                                       /*mdtol=*/md_cg_tol);

  // EO action stack mirroring production: RationalEO covers |det M_pc|^1
  // (Nf=2 doubled = 2 Pfaffians), LogDet covers |det M_ee|; together =
  // |det M48|^1.  Non-EO sibling: Test_dtxqcd_freefield_qbarq_light uses
  // RationalFullAction on M48 alone.
  DTXQCDWilsonCloverRationalEOAction
      PF_eo(Grid_, RBGrid, mass_run, rat_params, csw_run);
  DTXQCDLogDetCloverEOAction LogDet(Grid_, RBGrid, mass_run, csw_run);

  setenv("DTXQCD_FREEZE_GAUGE", "1", 1);
  setenv("USE_FULL_PF", "0", 1);
  std::cout << GridLogMessage
            << "DTXQCD_FREEZE_GAUGE=1, USE_FULL_PF=0 (EO stack: RationalEO + LogDet)"
            << std::endl;

  typedef Representations<EmptyRep<DTXQCDField>> Reps;
  ActionLevel<DTXQCDField, Reps> L1(1);
  L1.push_back(&PF_eo);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  ActionSet<DTXQCDField, Reps> Aset;
  Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  DTXQCDField U(&Grid_);
  DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  DtxqcdInitFrozenGauge(pRNG, U);

  // Seed aux at the bare-Σ saddle = 3/(4+m) (matches existing DTXQCD smoke).
  // AUX_INIT_AUTO=1 runs the production self-consistent bisection from
  // gencfgs to seed at the true saddle Σ* satisfying Σ = Σ_DTXQCD(Σ).
  // Necessary at small λ where the bare-Σ formula overshoots.
  RealD Sigma_init = 3.0 / (4.0 + mass_run);
  if (const char *si = std::getenv("AUX_INIT"); si && *si) {
    Sigma_init = std::atof(si);
  }
  if (const char *sa = std::getenv("AUX_INIT_AUTO"); sa && std::atoi(sa) != 0) {
    int max_iter = 15;
    if (const char *m = std::getenv("AUX_INIT_MAX_ITER"); m && *m) max_iter = std::atoi(m);
    RealD aux_tol = 1e-2;
    if (const char *t = std::getenv("AUX_INIT_TOL"); t && *t) aux_tol = std::atof(t);
    Sigma_init = DtxqcdSelfConsistentAuxInit(pRNG, Grid_, RBGrid, U,
                                              lambda_run, mass_run, csw_run,
                                              Sigma_init, max_iter, aux_tol);
  } else {
    pRNG.SeedFixedIntegers({6 + rng_off, 7 + rng_off, 8 + rng_off, 9 + rng_off, 10 + rng_off});
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, Sigma_init);
    std::cout << GridLogMessage
              << "Aux seeded at bare saddle Σ=" << Sigma_init << std::endl;
  }

  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  HMCp.Trajectories        = total_traj - n_therm_run;
  HMCp.NoMetropolisUntil   = n_therm_run;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  NoSmearing<DTXQCDCompositeImpl> Smear;
  typedef ForceGradient<DTXQCDCompositeImpl,
                        NoSmearing<DTXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid_, MD, Aset, Smear);
  Smear.set_Field(U);

  // FIERZ_AVG_N_NOISE=K  → install averaging observer that does a small
  // Fierz check (K noise samples) on every prod-window trajectory and
  // accumulates aux trace VEVs.  Final ratio is the cfg-averaged
  // ⟨Σ_DTX⟩ / ⟨Σ_W⟩, with the cfg-fluctuation noise folded into the
  // standard error.  Default (knob absent) keeps the single-cfg final
  // check path.
  std::unique_ptr<DtxqcdFierzAveragingObserver> avg_obs;
  std::unique_ptr<DTXQCDCheckpointer> ckpt;
  std::vector<HmcObservable<DTXQCDField> *> Obs;
  // SAVE_TRACE=PATH writes gauge+aux every MEAS_SKIP trajs so any future
  // precision measurement (different N_NOISE, different ratio
  // definition, etc.) can be re-run post-hoc.  Off by default.
  if (const char *trace_path = std::getenv("SAVE_TRACE");
      trace_path && *trace_path) {
    int meas_skip = 5;
    if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip = std::atoi(v);
    TxqcdTest2pt::mkdir_p(trace_path);
    CheckpointerParameters CPp;
    CPp.config_prefix = std::string(trace_path) + "/ckpoint_lat";
    CPp.rng_prefix    = std::string(trace_path) + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    ckpt.reset(new DTXQCDCheckpointer(CPp));
    Obs.push_back(ckpt.get());
    std::cout << GridLogMessage << "SAVE_TRACE=" << trace_path
              << " (cfg every " << meas_skip << " trajs)" << std::endl;
  }
  int fierz_avg_n_noise = 16;
  if (const char *v = std::getenv("FIERZ_AVG_N_NOISE"); v && *v)
    fierz_avg_n_noise = std::atoi(v);
  if (fierz_avg_n_noise > 0) {
    avg_obs.reset(new DtxqcdFierzAveragingObserver(Grid_, RBGrid, mass_run,
                                                    csw_run, lambda_run,
                                                    n_therm_run,
                                                    fierz_avg_n_noise,
                                                    /*cg_tol=*/1e-8));
    Obs.push_back(avg_obs.get());
    std::cout << GridLogMessage
              << "FIERZ_AVG_N_NOISE=" << fierz_avg_n_noise
              << "  (in-line averaging from traj " << n_therm_run
              << " onward)" << std::endl;
  }
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  DtxqcdFierzCheckResult result;
  if (avg_obs) {
    result = avg_obs->finalize(pass_tol,
                                "Test_dtxqcd_freefield_qbarq_eo_light");
  } else {
    result = DtxqcdFierzCheck(U, Grid_, RBGrid, mass_run, csw_run,
                               n_noise, meas_cg_tol, pass_tol,
                               "Test_dtxqcd_freefield_qbarq_eo_light");
  }

  Grid_finalize();
  return result.pass ? 0 : 1;
}
