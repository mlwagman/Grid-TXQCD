// Full (non-EO) TXQCD sweep gencfgs.  Uses TXQCDWilsonPseudoFermionAction
// (S = phi^dag (M^dag M)^{-1} phi on the FULL M_W + Delta operator, NO
// EO preconditioning, NO clover).  Matches DTXQCD's USE_FULL_PF=1 path
// for clean comparison at small λ where EO Schur preconditioning can
// break down on either side.
//
// csw=0, mass=0.3 by default, 4³×8.
//
// Env knobs (all optional):
//   CFG_DIR   — output dir (required)
//   LAMBDA    — auxiliary-field coupling (required)
//   MASS      — Wilson mass (default 0.3)
//   MDSTEPS   — integrator MD steps (default 10)
//   TRAJL     — trajectory length (default 0.5)
//   N_PROD    — production trajectories (default 500)
//   N_THERM   — thermalization trajectories (default 100)
//   MEAS_SKIP — measurement / cfg save interval (default 10)

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

using namespace TxqcdTest2ptCloverOptlam;
using TxqcdTest2pt::beta;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_vev_noise;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  RealD lambda_run = 3.0;
  RealD mass_run   = 0.3;
  int mdsteps      = 10;
  RealD trajL      = 0.5;
  int n_therm_run  = 100;
  int n_prod_run   = 500;
  int meas_skip_run = 10;
  std::string cfg_dir = "configs_2pt_txqcd_full_sweep";

  if (const char *v = std::getenv("LAMBDA"); v && *v)    lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS"); v && *v)      mass_run   = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS"); v && *v)   mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL"); v && *v)     trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM"); v && *v)   n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD"); v && *v)    n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR"); v && *v)   cfg_dir = v;

  std::cout << GridLogMessage
            << "TXQCD FULL sweep (non-EO, csw=0): lambda=" << lambda_run
            << " mass=" << mass_run << " MDsteps=" << mdsteps
            << " trajL=" << trajL << " cfg_dir=" << cfg_dir
            << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm_run + n_prod_run;
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int latest = -1;
  for (int t = meas_skip_run; t <= total_traj; t += meas_skip_run) {
    std::string f = cfg_dir + "/ckpoint_lat." + std::to_string(t);
    struct stat st;
    if (stat(f.c_str(), &st) == 0) latest = t;
  }
  int start_traj = 0;

  // RHMC bracket per env knobs (matched to DTXQCD pattern).
  RealD rat_lo     = 0.05;
  RealD rat_hi     = 80.0;
  int   rat_degree = 12;
  if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
  if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
  std::cout << GridLogMessage
            << "TXQCD rational: lo=" << rat_lo << " hi=" << rat_hi
            << " degree=" << rat_degree << std::endl;
  RealD cg_tol = 1e-8;
  OneFlavourRationalParams rat_params(
      rat_lo, rat_hi, cg_max, cg_tol, rat_degree, 64,
      /*BoundsCheckFreq=*/100, /*mdtolerance=*/1e-6);

  GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
  AuxiliaryFieldGaussianAction           AuxAction(lambda_run);
  // FULL (non-EO) TXQCD RATIONAL pseudofermion: S = phi^dag (M^dag M)^{-1/2} phi.
  // M_TX = D_W (csw=0) + Delta(σ, π, s, p, t).  Internal Nf_tx=2 flavor block
  // means one PF carries (det M_W)^2 = Nf=2 Wilson at aux=0.  No clover, no LogDet.
  TXQCDWilsonRationalPseudoFermionAction PF(Grid, RBGrid, mass_run, rat_params);

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
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
    std::cout << GridLogMessage << "Resuming TXQCD FULL from " << cfg_dir
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

  // csw=0 for measurement compute_trminv (plain Wilson reference).
  TxqcdTest2ptClover::TxqcdDiagnostics diag(
      cfg_dir + "/hmc_diagnostics", meas_skip_run, {
      {"PseudoFermion", &PF},
      {"AuxGaussian",   &AuxAction},
      {"Gauge",         &GaugeAction}
  }, Grid, RBGrid, pRNG, mass_run, /*csw=*/0.0, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}
