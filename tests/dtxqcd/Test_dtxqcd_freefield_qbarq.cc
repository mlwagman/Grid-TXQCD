// Test_dtxqcd_freefield_qbarq:
//
// Free-field unit test for ⟨q̄q⟩ + aux saddles in DTXQCD.
// Mirror of TXQCD's Test_txqcd_freefield_qbarq.
//
// Setup: U=I (DTXQCD_FREEZE_GAUGE=1 keeps it that way), m=1000 → kappa≈0,
//        csw=0.  Hopping → 0, M48 ≈ block-diag (4+m) per site + X(aux).
//
// Predicted free-field Σ_M48 at U=I, m large (trminv_compare convention,
// /V with σ²=2 Gaussian noise):
//   Σ_M48 = Tr[M48^{-1}]/(V·N_F·something)·2 ≈ 2·V·48/(4+m)/V/(2·N_F)
//         = 48/((4+m)·N_F)·2/2N_F
// Actually simplest: Σ_M48 in trminv_block_compare normalization should be
// 2*Tr[M48^{-1}]/V/(2·N_F) = 2*(V·48)/(4+m)/V/(2·N_F) = 48/((4+m)·N_F) = 24/(4+m) for N_F=2.
// So Σ_M48 = Σ_W = 24/(4+m) at U=I, m large — Fierz target.
//
// Env knobs match TXQCD analog (LAMBDA, MASS, MDSTEPS, TRAJL, N_PROD, N_THERM, MEAS_SKIP, CFG_DIR).

#include "Test_dtxqcd_2pt_utils.h"
#include "Test_dtxqcd_fierz_check_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Quick smoke defaults — m=1000 (κ ≈ 5e-4, BC irrelevant) so the
  // entire run completes in well under 1 min.
  RealD lambda_run = 1.0;
  RealD mass_run   = 1000.0;
  RealD csw_run    = 0.0;
  int mdsteps      = 4;
  RealD trajL      = 1.0;
  int n_therm_run  = 10;
  int n_prod_run   = 20;
  int meas_skip_run = 5;
  std::string cfg_dir = "free_dtxqcd";

  if (const char *v = std::getenv("LAMBDA");    v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");      v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW");       v && *v) csw_run    = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");   v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");     v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");   v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");    v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");   v && *v) cfg_dir = v;

  setenv("DTXQCD_FREEZE_GAUGE", "1", 1);
  setenv("USE_FULL_PF", "1", 1);  // non-EO full PF, simpler at U=I

  std::cout << GridLogMessage
            << "DTXQCD FREE-FIELD test: lambda=" << lambda_run
            << " mass=" << mass_run << " csw=" << csw_run
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " cfg_dir=" << cfg_dir << std::endl;
  std::cout << GridLogMessage
            << "  predicted Σ = 24/(4+m) = " << (24.0/(4.0+mass_run)) << std::endl;

  std::vector<int> latt_dims{4,4,4,8};
  Coordinate latt(latt_dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm_run + n_prod_run;
  TxqcdTest2pt::mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  RealD beta_dummy = 6.0;  // unused: gauge force frozen
  DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_dummy);
  DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);

  // Rational bracket sized for M48 spectrum at m=1000: eigenvalues ~ (m+4)² = 1e6.
  RealD rat_lo     = 1e4;
  RealD rat_hi     = 1e7;
  int   rat_degree = 8;
  if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
  if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
  OneFlavourRationalParams rat_params(rat_lo, rat_hi, /*MaxIter=*/10000, /*tol=*/1e-8,
                                      rat_degree, 64, /*BCFreq=*/100, /*mdtol=*/1e-6);
  std::cout << GridLogMessage
            << "  rational bracket: lo=" << rat_lo
            << " hi=" << rat_hi << " degree=" << rat_degree << std::endl;

  DTXQCDWilsonCloverRationalFullAction
      PF_full(Grid, RBGrid, mass_run, rat_params, csw_run);
  DTXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_run, csw_run);

  typedef Representations<EmptyRep<DTXQCDField>> Reps;
  ActionLevel<DTXQCDField, Reps> L1(1);
  L1.push_back(&PF_full);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  ActionSet<DTXQCDField, Reps> Aset;
  Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  DTXQCDField U(&Grid);
  DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  DtxqcdInitFrozenGauge(pRNG, U);

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
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  // Optional observers (off by default):
  //   SAVE_TRACE=PATH      — write cfgs every MEAS_SKIP trajs
  //   FIERZ_AVG_N_NOISE=K  — in-line averaging Fierz measurement
  std::unique_ptr<DTXQCDCheckpointer> ckpt;
  std::unique_ptr<DtxqcdFierzAveragingObserver> avg_obs;
  std::vector<HmcObservable<DTXQCDField> *> Obs;
  if (const char *trace_path = std::getenv("SAVE_TRACE");
      trace_path && *trace_path) {
    int meas_skip = 10;
    if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip = std::atoi(v);
    TxqcdTest2pt::mkdir_p(trace_path);
    CheckpointerParameters CPp;
    CPp.config_prefix = std::string(trace_path) + "/ckpoint_lat";
    CPp.rng_prefix    = std::string(trace_path) + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    ckpt.reset(new DTXQCDCheckpointer(CPp));
    Obs.push_back(ckpt.get());
  }
  int fierz_avg_n_noise = 0;
  if (const char *v = std::getenv("FIERZ_AVG_N_NOISE"); v && *v)
    fierz_avg_n_noise = std::atoi(v);
  if (fierz_avg_n_noise > 0) {
    avg_obs.reset(new DtxqcdFierzAveragingObserver(Grid, RBGrid, mass_run,
                                                    csw_run, n_therm_run,
                                                    fierz_avg_n_noise,
                                                    /*cg_tol=*/1e-8));
    Obs.push_back(avg_obs.get());
  }
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  // ===== Post-HMC Fierz check =====
  RealD pass_tol = 0.02;
  if (const char *v = std::getenv("PASS_TOL"); v && *v) pass_tol = std::atof(v);
  int n_noise = 64;
  if (const char *v = std::getenv("N_NOISE"); v && *v) n_noise = std::atoi(v);
  RealD meas_cg_tol = 1e-10;
  if (const char *v = std::getenv("MEAS_CG_TOL"); v && *v) meas_cg_tol = std::atof(v);

  DtxqcdFierzCheckResult result;
  if (avg_obs) {
    result = avg_obs->finalize(pass_tol, "Test_dtxqcd_freefield_qbarq");
  } else {
    result = DtxqcdFierzCheck(U, Grid, RBGrid, mass_run, csw_run,
                               n_noise, meas_cg_tol, pass_tol,
                               "Test_dtxqcd_freefield_qbarq");
  }

  Grid_finalize();
  return result.pass ? 0 : 1;
}
