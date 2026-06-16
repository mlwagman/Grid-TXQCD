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
//   RAT_LO, RAT_HI, RAT_DEGREE, CG_TOL, N_NOISE
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
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
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
  RealD cg_tol_meas = 1e-10;
  if (const char *v = std::getenv("CG_TOL"); v && *v) cg_tol_meas = std::atof(v);

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
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, 1e-10,
                                       rat_deg, 64, 100, 1e-6);

  // PF selection by csw: at csw=0 keep the lighter non-EO Wilson PF
  // (proven fast smoke).  At csw!=0 switch to the clover-aware EO action
  // so the sampling weight matches the WilsonClover reference operator.
  AuxiliaryFieldGaussianAction AuxAction(lambda_run);
  TXQCDWilsonRationalPseudoFermionAction PF_wilson(Grid_, RBGrid,
                                                    mass_run, rat_params);
  TXQCDWilsonCloverRationalEOAction      PF_clover(Grid_, RBGrid,
                                                    mass_run, rat_params,
                                                    csw_run);
  Action<TXQCDField> *PF = (csw_run == 0.0)
                            ? (Action<TXQCDField> *)&PF_wilson
                            : (Action<TXQCDField> *)&PF_clover;
  std::cout << GridLogMessage << "PF: " << PF->action_name() << std::endl;

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

  // No checkpointer — the Fierz check runs on the in-memory final state.
  std::vector<HmcObservable<TXQCDField> *> Obs = {};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  // ===== Post-HMC: Σ_TX vs Σ_W on the equilibrated state =====
  auto result = TxqcdFierzCheck(U, Grid_, RBGrid, mass_run, csw_run,
                                 n_noise, cg_tol_meas, pass_tol,
                                 "Test_txqcd_freefield_qbarq_light");

  Grid_finalize();
  return result.pass ? 0 : 1;
}
