// Stout optlam gencfgs with ThermalAuxConfiguration initialization.
// Uses known Sigma_l from measured VEVs to start aux fields near equilibrium.
// Only generates TXQCD configs (QCD configs are shared with the base stout run).

#include "Test_txqcd_2pt_stout_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptStoutOptlam;

// Sigma_l = lambda^2 / Nf * <Tr sigma>/V, measured from stout ensemble
constexpr RealD Sigma_l = 2.69;

inline std::string therminit_cfg_dir() {
  return "configs_2pt_txqcd_stout_optlam_therminit";
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;

  std::string cfg_dir = therminit_cfg_dir();

  if (TxqcdTest2pt::txqcd_configs_exist(cfg_dir)) {
    std::cout << GridLogMessage
              << "TXQCD stout therminit configs already exist, skipping."
              << std::endl;
    Grid_finalize();
    return 0;
  }

  std::cout << GridLogMessage
            << "Generating TXQCD stout optlam configs with ThermalAuxInit"
            << " (Sigma_l=" << Sigma_l << ", lambda=" << lambda
            << ", csw=" << csw << ", rho=" << stout_rho
            << ", Nsmear=" << stout_nsmear << ")..." << std::endl;
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

  AuxiliaryFieldGaussianAction           AuxAction(lambda);

  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rat_params, csw);
  PF.is_smeared = true;

  TXQCDLogDetCloverEOAction         LogDet(Grid, RBGrid, mass, csw);
  LogDet.is_smeared = true;

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
  MD.MDsteps = 10;
  MD.trajL = 0.5;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj "
              << latest << std::endl;
    TxqcdTest2pt::LoadTxqcdConfig(U, sRNG, pRNG, latest, cfg_dir);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({31, 32, 33, 34, 35});
    pRNG.SeedFixedIntegers({36, 37, 38, 39, 40});
    // Key change: ThermalAuxConfiguration with known Sigma_l
    TXQCDCompositeImpl::ThermalAuxConfiguration(pRNG, U, lambda, 0.1, Sigma_l);
    std::cout << GridLogMessage
              << "Initialized aux fields with ThermalAuxConfiguration:"
              << " Sigma_l=" << Sigma_l
              << ", <sigma_diag>=" << Sigma_l / (lambda * lambda)
              << std::endl;
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
      {"PseudoFermion", &PF},
      {"LogDet", &LogDet},
      {"AuxGaussian", &AuxAction},
      {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage << "Stout optlam therminit generation complete."
            << std::endl;
  Grid_finalize();
  return 0;
}
