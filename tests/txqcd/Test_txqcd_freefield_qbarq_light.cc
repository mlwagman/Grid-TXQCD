// Test_txqcd_freefield_qbarq_light:
//
// Self-contained free-field Fierz unit test at light Wilson mass.
//
// Combines the HMC sampler from Test_txqcd_freefield_qbarq (frozen U=I,
// SADDLE_INIT aux, Nf=2 rational PF) with a post-HMC Σ_TX vs Σ_W
// comparison (same convention as Test_txqcd_trminv_compare) on the
// final equilibrated configuration, and returns PASS/FAIL exit.
//
// "Light" defaults are sized for m=0.1 — the κ=0.122 regime where the
// APBC vs PBC boundary-condition convention dominates the propagator and
// any mismatched-BC bug shows up as a 7% Σ_TX/Σ_W gap.  The test thus
// doubles as a regression guard for the BC default in TXQCDWilsonOp /
// TXQCDWilsonCloverFermionEO (see project-bc-mismatch-root-cause).
//
// PASS criterion: |Σ_TX / Σ_W − 1| < PASS_TOL (default 0.02).
//
// Env knobs (overrides):
//   MASS, LAMBDA, MDSTEPS, TRAJL, N_THERM, N_PROD
//   RAT_LO, RAT_HI, RAT_DEGREE, MEAS_CG_TOL, MD_CG_TOL, N_NOISE
//   FIERZ_AVG_N_NOISE  — install averaging observer (default off)
//   SAVE_TRACE=PATH    — write cfgs every MEAS_SKIP trajs (default off)
//   CSW       — clover coefficient (default 0; nonzero switches the
//                HMC to TXQCDWilsonCloverRationalEOAction and the Σ_W
//                reference to WilsonCloverFermion at the same csw)
//   GAUGE_INIT— cold | tepid[:AMP] | hot | nersc:PATH (see helper)
//   PASS_TOL  — relative PASS threshold (default 0.02)
//
// Typical run: ~10–15 min at the default (m=0.1, λ=10).  Quick-mode
// usage `MASS=1000 LAMBDA=1` reproduces the heavy-mass sanity in ~30 s.

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include "Test_txqcd_fierz_check_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptCloverOptlam;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_vev_noise;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Light-mass defaults — m=0.1 (κ=0.122) is the hardest finite-aux corner
  // and exercises the APBC convention strongly.
  RealD lambda_run = 10.0;
  RealD mass_run   = 0.1;
  RealD csw_run    = 0.0;
  int mdsteps      = 20;
  RealD trajL      = 1.0;
  int n_therm_run  = 30;
  int n_prod_run   = 40;

  if (const char *v = std::getenv("LAMBDA");    v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");      v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW");       v && *v) csw_run    = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");   v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");     v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");   v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");    v && *v) n_prod_run  = std::atoi(v);

  RealD pass_tol = 0.02;
  if (const char *v = std::getenv("PASS_TOL"); v && *v) pass_tol = std::atof(v);

  int n_noise = 64;
  if (const char *v = std::getenv("N_NOISE"); v && *v) n_noise = std::atoi(v);
  // Measurement-side CG tolerance for the Σ_TX, Σ_W trace estimators.
  // Accepts MEAS_CG_TOL (standardized name) or legacy CG_TOL.
  RealD cg_tol_meas = 1e-10;
  if (const char *v = std::getenv("MEAS_CG_TOL"); v && *v) cg_tol_meas = std::atof(v);
  else if (const char *v = std::getenv("CG_TOL"); v && *v) cg_tol_meas = std::atof(v);

  // RHMC bracket — default sized for m=0.1.
  RealD rat_lo = 0.005, rat_hi = 80.0;
  int rat_deg = 12;
  if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo  = std::atof(v);
  if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi  = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_deg = std::atoi(v);

  std::cout << GridLogMessage
            << "TXQCD FREE-FIELD light test:"
            << " mass=" << mass_run << " (κ=" << 1.0/(2.0*(4.0+mass_run)) << ")"
            << " lambda=" << lambda_run << " csw=" << csw_run
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " N_THERM=" << n_therm_run << " N_PROD=" << n_prod_run
            << " RAT[" << rat_lo << "," << rat_hi << "]^" << rat_deg
            << " PASS_TOL=" << pass_tol << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);

  int total_traj = n_therm_run + n_prod_run;

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid_);
  int seed_off = 0;
  if (const char *v = std::getenv("RNG_SEED"); v && *v) seed_off = std::atoi(v);
  sRNG.SeedFixedIntegers({1 + seed_off, 2 + seed_off, 3 + seed_off, 4 + seed_off, 5 + seed_off});
  pRNG.SeedFixedIntegers({6 + seed_off, 7 + seed_off, 8 + seed_off, 9 + seed_off, 10 + seed_off});

  // Rational params: outer tolerance 1e-10 for refresh/S, MD-solve tol
  // 1e-6 default (force is symplectic-corrected, so a looser tol is fine).
  // MD_CG_TOL knob lets a probe trade accuracy for wall time.
  RealD md_cg_tol = 1e-6;
  if (const char *v = std::getenv("MD_CG_TOL"); v && *v) md_cg_tol = std::atof(v);
  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, 1e-10,
                                       rat_deg, 64, 100, md_cg_tol);

  // Non-EO action stack: full-volume RHMC pseudofermion alone covers the
  // |det M_TX|^{Nf/2} weight; no LogDet needed.  csw≠0 is not supported
  // here — use the EO sibling (Test_txqcd_freefield_qbarq_eo_light) that
  // pairs RationalEO with LogDet.
  if (csw_run != 0.0) {
    std::cout << GridLogMessage
              << "[Test_txqcd_freefield_qbarq_light] non-EO stack requires csw=0;"
                 " got csw=" << csw_run << " — use the _eo sibling instead."
              << std::endl;
    Grid_finalize();
    return 1;
  }
  AuxiliaryFieldGaussianAction AuxAction(lambda_run);
  TXQCDWilsonRationalPseudoFermionAction PF_wilson(Grid_, RBGrid,
                                                    mass_run, rat_params);
  Action<TXQCDField> *PF = (Action<TXQCDField> *)&PF_wilson;
  std::cout << GridLogMessage << "PF: " << PF->action_name()
            << " (non-EO stack: PF only)" << std::endl;

  setenv("TXQCD_FREEZE_GAUGE", "1", 1);
  std::cout << GridLogMessage
            << "TXQCD_FREEZE_GAUGE=1 → gauge momentum forced to zero" << std::endl;

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(PF);
  L1.push_back(&AuxAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  TXQCDField U(&Grid_);
  TXQCDCompositeImpl::ColdConfiguration(pRNG, U);  // zeros aux + U=I
  TxqcdInitFrozenGauge(pRNG, U);                    // overlays U per GAUGE_INIT
  // Seed aux at the predicted free-field saddle Σ = 3/(4+m).  At weak
  // perturbed U this is still a good initial guess.
  RealD sigma_init = 3.0 / (4.0 + mass_run);
  TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, sigma_init);
  std::cout << GridLogMessage
            << "aux seeded at saddle Σ=" << sigma_init << std::endl;

  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  HMCp.Trajectories        = total_traj - n_therm_run;
  HMCp.NoMetropolisUntil   = n_therm_run;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef ForceGradient<TXQCDCompositeImpl,
                        NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid_, MD, Aset, Smear);
  Smear.set_Field(U);

  // Observers:
  //   SAVE_TRACE=PATH      — write cfgs every MEAS_SKIP trajs (off by default)
  //   FIERZ_AVG_N_NOISE=K  — in-line Fierz check with K noise/traj over the
  //                          prod window, averages folded into final ratio
  //                          (off by default; final check uses single-cfg path)
  std::unique_ptr<TXQCDCheckpointer> ckpt;
  std::unique_ptr<TxqcdFierzAveragingObserver> avg_obs;
  std::vector<HmcObservable<TXQCDField> *> Obs;
  if (const char *trace_path = std::getenv("SAVE_TRACE");
      trace_path && *trace_path) {
    int meas_skip = 10;
    if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip = std::atoi(v);
    mkdir_p(trace_path);
    CheckpointerParameters CPp;
    CPp.config_prefix = std::string(trace_path) + "/ckpoint_lat";
    CPp.rng_prefix    = std::string(trace_path) + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    ckpt.reset(new TXQCDCheckpointer(CPp));
    Obs.push_back(ckpt.get());
    std::cout << GridLogMessage << "SAVE_TRACE=" << trace_path
              << "  (cfg every " << meas_skip << " trajs)" << std::endl;
  }
  int fierz_avg_n_noise = 16;
  if (const char *v = std::getenv("FIERZ_AVG_N_NOISE"); v && *v)
    fierz_avg_n_noise = std::atoi(v);
  if (fierz_avg_n_noise > 0) {
    avg_obs.reset(new TxqcdFierzAveragingObserver(Grid_, RBGrid, mass_run,
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

  // ===== Post-HMC Fierz check =====
  TxqcdFierzCheckResult result;
  if (avg_obs) {
    result = avg_obs->finalize(pass_tol,
                                "Test_txqcd_freefield_qbarq_light");
  } else {
    result = TxqcdFierzCheck(U, Grid_, RBGrid, mass_run, csw_run,
                              n_noise, cg_tol_meas, pass_tol,
                              "Test_txqcd_freefield_qbarq_light");
  }

  Grid_finalize();
  return result.pass ? 0 : 1;
}
