// Test_txqcd_freefield_nf2_qbarq:
// Same as Test_txqcd_freefield_qbarq but uses TXQCDWilsonRationalPseudoFermionAction
// = single rational PF instance = Nf=2 effective at aux=0
// (vs the non-rational version which gives |det M_TX|² = Nf=4 effective).
//
// Goal: test whether the Fierz cancellation in TXQCD construction is
// Nf-specific — if rational/Nf=2 gives ratio = 1 and non-rational/Nf=4
// gives 0.98, the action's Fierz identity targets Nf=2.

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptCloverOptlam;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_vev_noise;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  RealD lambda_run = 2.0;
  RealD mass_run   = 10.0;
  int mdsteps      = 4;
  RealD trajL      = 1.0;
  int n_therm_run  = 60;
  int n_prod_run   = 500;
  int meas_skip_run = 10;
  std::string cfg_dir = "free_txqcd_nf2";

  if (const char *v = std::getenv("LAMBDA");   v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");     v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");  v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");    v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");  v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");   v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");  v && *v) cfg_dir = v;

  setenv("TXQCD_FREEZE_GAUGE", "1", 1);

  // Rational bracket scaled to m: M^dag M at U=I, m=10 has eigenvalues ~(m+4)²=196.
  // Default to broad bracket; user can override.
  RealD rat_lo = 50.0, rat_hi = 500.0;
  int rat_deg = 8;
  if (const char *v = std::getenv("RAT_LO"); v && *v) rat_lo = std::atof(v);
  if (const char *v = std::getenv("RAT_HI"); v && *v) rat_hi = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_deg = std::atoi(v);

  std::cout << GridLogMessage
            << "TXQCD FREE-FIELD Nf=2: lambda=" << lambda_run
            << " mass=" << mass_run << " MDsteps=" << mdsteps
            << " trajL=" << trajL << " RAT_LO=" << rat_lo
            << " RAT_HI=" << rat_hi << " deg=" << rat_deg
            << " cfg_dir=" << cfg_dir << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm_run + n_prod_run;
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, 1e-10,
                                       rat_deg, 64, 100, 1e-6);

  AuxiliaryFieldGaussianAction           AuxAction(lambda_run);
  TXQCDWilsonRationalPseudoFermionAction PF(Grid, RBGrid, mass_run, rat_params);

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
  L1.push_back(&AuxAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  TXQCDField U(&Grid);
  TXQCDCompositeImpl::ColdConfiguration(pRNG, U);

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
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip_run;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}
