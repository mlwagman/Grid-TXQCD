#include "params.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;
  mkdir_p(txqcd_cfg_dir());

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(txqcd_cfg_dir() + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(txqcd_cfg_dir() + "/ckpoint_lat_aux." + std::to_string(t)) &&
        file_exists(txqcd_cfg_dir() + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  OneFlavourRationalParams rat_params(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                      100, 1e-6, 1e-4);

  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  GaugeActionAdapter<SymanzikR> GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  AuxiliaryFieldGaussianAction AuxAction(lambda);

  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass_light, rat_params, csw);
  PF.is_smeared = true;

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_light, csw);
  LogDet.is_smeared = true;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  txqcd_cfg_dir() + "/ckpoint_lat",
                                  txqcd_cfg_dir() + "/ckpoint_rng", latest);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
  }

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF StrangeFermOp(U.U, Grid, RBGrid, mass_strange, csw, csw);
  OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                       100, 1e-6, 1e-4);
  OneFlavourRationalPseudoFermionAction<WilsonImplR> StrangePF(StrangeFermOp,
                                                                strange_rat);
  QCDActionAdapter StrangeAdapter(StrangePF);
  StrangeAdapter.is_smeared = true;

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  L1.push_back(&StrangeAdapter);
  ActionLevel<TXQCDField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = 10;
  MD.trajL = 0.5;

  int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = total_traj - no_metrop - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  TXQCDSmearedConfiguration Smear(&Grid, stout_nsmear_inv, Stout);

  typedef ForceGradient<TXQCDCompositeImpl,
                        TXQCDSmearedConfiguration, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = txqcd_cfg_dir() + "/ckpoint_lat";
  CPp.rng_prefix    = txqcd_cfg_dir() + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage << "TXQCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
