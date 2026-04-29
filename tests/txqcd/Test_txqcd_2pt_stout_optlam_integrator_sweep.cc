// Integrator parameter sweep for stout optlam TXQCD.
// Diagnoses aux-field autocorrelation under different HMC settings.
//
// Environment variables:
//   TRAJ_L     — trajectory length (default 0.5)
//   MD_STEPS   — number of MD steps (default 10)
//   AUX_LEVEL  — timescale multiplier for aux action (default 0)
//                0 = aux on fermion level, N>0 = aux on own level with multiplier N
//   N_TRAJ     — total trajectories to run (default 200)
//   SUFFIX     — appended to output directory name (default "")
//
// Output: configs_2pt_stout_intsweep_<params>/hmc_diagnostics.*.h5

#include "Test_txqcd_2pt_stout_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>

using namespace TxqcdTest2ptStoutOptlam;

constexpr RealD Sigma_l = 2.69;

static double getenv_dbl(const char *name, double def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atof(v) : def;
}
static int getenv_int(const char *name, int def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}
static std::string getenv_str(const char *name, const char *def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : std::string(def);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  double trajL       = getenv_dbl("TRAJ_L", 0.5);
  int    mdsteps     = getenv_int("MD_STEPS", 10);
  int    aux_level   = getenv_int("AUX_LEVEL", 0);
  int    n_traj      = getenv_int("N_TRAJ", 200);
  std::string suffix = getenv_str("SUFFIX", "");

  char dirname[256];
  std::snprintf(dirname, sizeof(dirname),
    "configs_2pt_stout_intsweep_tl%.2f_md%d_al%d%s",
    trajL, mdsteps, aux_level,
    suffix.empty() ? "" : ("_" + suffix).c_str());
  std::string cfg_dir(dirname);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  std::cout << GridLogMessage << "=== Integrator Sweep ===" << std::endl;
  std::cout << GridLogMessage << "  trajL      = " << trajL << std::endl;
  std::cout << GridLogMessage << "  MDsteps    = " << mdsteps << std::endl;
  std::cout << GridLogMessage << "  dt         = " << trajL/mdsteps << std::endl;
  std::cout << GridLogMessage << "  aux_level  = " << aux_level << std::endl;
  std::cout << GridLogMessage << "  N_traj     = " << n_traj << std::endl;
  std::cout << GridLogMessage << "  output     = " << cfg_dir << std::endl;

  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = TxqcdTest2pt::latest_txqcd_checkpoint(cfg_dir);

  RealD cg_tol = 1e-8;
  OneFlavourRationalParams rat_params(1e-4, 64.0, cg_max, cg_tol, 12, 64,
                                      100, 1e-6, 1e-4);

  GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
  GaugeAction.is_smeared = true;

  AuxiliaryFieldGaussianAction AuxAction(lambda);

  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rat_params, csw);
  PF.is_smeared = true;

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass, csw);
  LogDet.is_smeared = true;

  typedef Representations<EmptyRep<TXQCDField>> Reps;

  // ActionLevel has a reference member, so levels must outlive the ActionSet.
  // Declare all possible levels at function scope.
  ActionLevel<TXQCDField, Reps> L1(1);
  ActionLevel<TXQCDField, Reps> L2(4);
  ActionLevel<TXQCDField, Reps> Laux_sep(aux_level > 0 ? aux_level : 1);

  ActionSet<TXQCDField, Reps> Aset;

  if (aux_level > 0) {
    // 3-level: fermion (1x) | aux (aux_level x) | gauge (4x)
    L1.push_back(&PF);
    L1.push_back(&LogDet);
    Laux_sep.push_back(&AuxAction);
    L2.push_back(&GaugeAction);
    Aset.push_back(L1);
    Aset.push_back(Laux_sep);
    Aset.push_back(L2);
  } else {
    // 2-level: fermion+aux (1x) | gauge (4x)
    L1.push_back(&PF);
    L1.push_back(&LogDet);
    L1.push_back(&AuxAction);
    L2.push_back(&GaugeAction);
    Aset.push_back(L1);
    Aset.push_back(L2);
  }

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL = trajL;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj "
              << latest << std::endl;
    TxqcdTest2pt::LoadTxqcdConfig(U, sRNG, pRNG, latest, cfg_dir);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({51, 52, 53, 54, 55});
    pRNG.SeedFixedIntegers({56, 57, 58, 59, 60});
    TXQCDCompositeImpl::ThermalAuxConfiguration(pRNG, U, lambda, 0.1, Sigma_l);
  }

  int n_therm_sweep = 20;
  int no_metrop = (start_traj < n_therm_sweep) ? (n_therm_sweep - start_traj) : 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = n_traj - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  TXQCDSmearedConfiguration Smear(&Grid, stout_nsmear, Stout);

  typedef ForceGradient<TXQCDCompositeImpl,
                        TXQCDSmearedConfiguration, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  TxqcdSmearedDiagnostics<TXQCDSmearedConfiguration> diag(
      cfg_dir + "/hmc_diagnostics", meas_skip, {
      {"PseudoFermion", &PF}, {"LogDet", &LogDet},
      {"AuxGaussian", &AuxAction}, {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage << "Integrator sweep run complete: " << cfg_dir << std::endl;
  Grid_finalize();
  return 0;
}
