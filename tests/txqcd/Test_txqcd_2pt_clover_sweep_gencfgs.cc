// Sweep variant of Test_txqcd_2pt_clover_optlam_gencfgs.
//
// Env knobs (all optional, useful for parallel HMC sweeps over λ):
//   CFG_DIR   — output dir (default "configs_2pt_txqcd_csw1_sweep")
//   LAMBDA    — auxiliary-field coupling (default 5.33)
//   MASS      — Wilson mass (default 0.3)
//   CSW       — clover coefficient (default 1.0)
//   MDSTEPS   — integrator MD steps (default 10)
//   TRAJL     — trajectory length (default 0.5)
//   N_PROD    — production trajectories (default 500)
//   N_THERM   — thermalization trajectories (default 100)
//   MEAS_SKIP — measurement / cfg save interval (default 10)
//
// Only generates the TXQCD configs (no plain QCD side — we measure plain
// Wilson Σ_W as a Fierz reference directly on the TXQCD gauge field with
// Test_txqcd_trminv_compare).

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptCloverOptlam;
using TxqcdTest2pt::beta;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_vev_noise;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ---- Knobs from env ----
  RealD lambda_run = 5.33;
  RealD mass_run   = 0.3;
  RealD csw_run    = 1.0;
  int mdsteps      = 10;
  RealD trajL      = 0.5;
  int n_therm_run  = 100;
  int n_prod_run   = 500;
  int meas_skip_run = 10;
  std::string cfg_dir = "configs_2pt_txqcd_csw1_sweep";

  if (const char *v = std::getenv("LAMBDA"); v && *v)    lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS"); v && *v)      mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW"); v && *v)       csw_run    = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS"); v && *v)   mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL"); v && *v)     trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM"); v && *v)   n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD"); v && *v)    n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR"); v && *v)   cfg_dir = v;

  std::cout << GridLogMessage
            << "TXQCD sweep: lambda=" << lambda_run << " mass=" << mass_run
            << " csw=" << csw_run << " MDsteps=" << mdsteps
            << " trajL=" << trajL << " cfg_dir=" << cfg_dir
            << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm_run + n_prod_run;

  // ==================== TXQCD ====================
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  // Find latest checkpoint in cfg_dir
  int latest = -1;
  for (int t = meas_skip_run; t <= total_traj; t += meas_skip_run) {
    std::string f = cfg_dir + "/ckpoint_lat." + std::to_string(t);
    struct stat st;
    if (stat(f.c_str(), &st) == 0) latest = t;
  }
  int start_traj = 0;

  RealD cg_tol = 1e-8;
  // Use a slightly broader rational bracket to cover small-λ aux growth.
  OneFlavourRationalParams rat_params(1e-4, 100.0, cg_max, cg_tol, 12, 64,
                                      100, 1e-6, 1e-4);

  GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
  AuxiliaryFieldGaussianAction           AuxAction(lambda_run);
  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass_run, rat_params, csw_run);
  TXQCDLogDetCloverEOAction         LogDet(Grid, RBGrid, mass_run, csw_run);

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  ActionLevel<TXQCDField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming TXQCD from " << cfg_dir
              << " at traj " << latest << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", latest);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  }

  int no_metrop = (start_traj < n_therm_run) ? (n_therm_run - start_traj) : 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = total_traj - no_metrop - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef ForceGradient<TXQCDCompositeImpl,
                        NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip_run;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  TxqcdTest2ptClover::TxqcdDiagnostics diag(cfg_dir + "/hmc_diagnostics", meas_skip_run, {
      {"PseudoFermion", &PF},
      {"LogDet",        &LogDet},
      {"AuxGaussian",   &AuxAction},
      {"Gauge",         &GaugeAction}
  }, Grid, RBGrid, pRNG, mass_run, csw_run, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}
