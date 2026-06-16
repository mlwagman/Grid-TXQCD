// Test_dtxqcd_freefield_qbarq_light:
//
// Light-mass DTXQCD free-field Fierz unit test.  Mirror of
// Test_txqcd_freefield_qbarq_light — runs DTXQCD HMC at frozen U with
// APBC time, measures Σ_DTXQCD / Σ_W on the equilibrated state, and
// returns PASS/FAIL.
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
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
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
  std::string cfg_dir = "free_dtxqcd_light";

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
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  RealD beta_dummy = 6.0;
  DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_dummy);
  DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);

  OneFlavourRationalParams rat_params(rat_lo, rat_hi, /*MaxIter=*/30000,
                                       /*tol=*/1e-10,
                                       rat_deg, 64, /*BCFreq=*/100,
                                       /*mdtol=*/md_cg_tol);

  DTXQCDWilsonCloverRationalFullAction
      PF_full(Grid_, RBGrid, mass_run, rat_params, csw_run);
  DTXQCDLogDetCloverEOAction LogDet(Grid_, RBGrid, mass_run, csw_run);

  setenv("DTXQCD_FREEZE_GAUGE", "1", 1);
  setenv("USE_FULL_PF", "1", 1);
  std::cout << GridLogMessage
            << "DTXQCD_FREEZE_GAUGE=1, USE_FULL_PF=1" << std::endl;

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

  DTXQCDField U(&Grid_);
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
  IntT MDyn(&Grid_, MD, Aset, Smear);
  Smear.set_Field(U);

  std::vector<HmcObservable<DTXQCDField> *> Obs = {};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  auto result = DtxqcdFierzCheck(U, Grid_, RBGrid, mass_run, csw_run,
                                  n_noise, meas_cg_tol, pass_tol,
                                  "Test_dtxqcd_freefield_qbarq_light");

  Grid_finalize();
  return result.pass ? 0 : 1;
}
