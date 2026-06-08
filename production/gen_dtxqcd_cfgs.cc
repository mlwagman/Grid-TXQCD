// DTXQCD production HMC driver — Nf=2 Wilson-clover with diquark-tensor
// auxiliary fields, sibling of gen_txqcd_cfgs but with the DTXQCD action
// roster (Pauli-triplet sigma/pi/t + Hermitian color d/n) and 1/4-root
// Pfaffian RHMC pseudofermion.
//
// Action layers:
//   Level 1 (inner): DTXQCDWilsonCloverRationalEOAction + DTXQCDLogDetCloverEOAction
//   Level 2 (gauge multiplier): plaquette gauge action via DTXQCDGaugeActionAdapter
//   Level 3 (aux multiplier):   DTXQCDAuxiliaryFieldGaussianAction
//
// Initial scope is deliberately minimal (no stout smearing, no QUDA, no
// diagnostic observer beyond the checkpointer) -- enough for a smoke run
// confirming dH finite and acceptance plausible.  The aux-correlator
// observers, eigenvalue diagnostics, and stout chain from gen_txqcd_cfgs_2plus1
// can graft on once the bare HMC trajectory is stable.
//
// Env overrides:
//   LATT=L.L.L.T     lattice size (default from params.h)
//   MASS_LIGHT_DTXQCD, LAMBDA_DTXQCD, CSW, BETA     physics
//   MDSTEPS, TRAJL, INTEGRATOR     integrator
//   RHMC_LO, RHMC_HI, RHMC_DEG, CG_TOL, CG_MAX     rational solver
//   N_SKIP           checkpoint save interval (default 10)
//   N_THERM, N_PROD overrides via params.h constexprs (recompile to change)

#include "params.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>

using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  // ---- geometry ----
  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  // ---- physics parameters (env-overridable via params.h knobs) ----
  const RealD mass  = mass_light_dtxqcd;
  const RealD csw_  = csw;
  const RealD beta_ = beta;
  const RealD lam   = lambda_dtxqcd;
  const RealD tol   = TXQCDProduction::detail::env_real("CG_TOL", 1e-10);
  const int   cgmax = TXQCDProduction::detail::env_int ("CG_MAX", 10000);
  std::cout << GridLogMessage
            << "DTXQCD physics: mass=" << mass << " csw=" << csw_
            << " beta=" << beta_ << " lambda=" << lam
            << " cg_tol=" << tol << " cg_max=" << cgmax << std::endl;

  // ---- rational params for the RHMC pseudofermion ----
  OneFlavourRationalParams rp(
      /*lo=*/         TXQCDProduction::detail::env_real("RHMC_LO",   1.0e-1),
      /*hi=*/         TXQCDProduction::detail::env_real("RHMC_HI",   64.0),
      /*MaxIter=*/    cgmax,
      /*tolerance=*/  tol,
      /*degree=*/     TXQCDProduction::detail::env_int ("RHMC_DEG",  12),
      /*precision=*/  50,
      /*BoundsCheckFreq=*/ 0,
      /*mdtolerance=*/ tol);
  std::cout << GridLogMessage
            << "RHMC rational: lo=" << rp.lo << " hi=" << rp.hi
            << " degree=" << rp.degree << " tol=" << rp.tolerance << std::endl;

  // ---- composite field (gauge + aux), cold initialisation ----
  DTXQCDField U(&Grid);
  DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);  // U=I, aux=0

  // ---- actions ----
  DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_);
  DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lam);
  DTXQCDLogDetCloverEOAction                   LogDet(Grid, RBGrid, mass, csw_);
  DTXQCDWilsonCloverRationalEOAction
      PFAction(Grid, RBGrid, mass, rp, csw_);

  // ---- action levels.  Level 1 (innermost) bundles the two fermion-bilinear
  //      monomials that need the finest dt; gauge gets a 2x multiplier and
  //      aux a 4x multiplier mirroring the TXQCD production hierarchy.
  typedef Representations<EmptyRep<DTXQCDField>> Reps;
  ActionLevel<DTXQCDField, Reps> L1(1);
  L1.push_back(&PFAction);
  L1.push_back(&LogDet);
  ActionLevel<DTXQCDField, Reps> L2(TXQCDProduction::detail::env_int("GAUGE_MULT", 2));
  L2.push_back(&GaugeAction);
  ActionLevel<DTXQCDField, Reps> L3(TXQCDProduction::detail::env_int("AUX_MULT",  4));
  L3.push_back(&AuxAction);
  ActionSet<DTXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);
  Aset.push_back(L3);

  // ---- integrator ----
  IntegratorParameters MD;
  MD.name    = "ForceGradient";
  if (const char *env = std::getenv("INTEGRATOR"); env && *env) MD.name = env;
  MD.MDsteps = 20;
  if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) MD.MDsteps = std::atoi(ms);
  MD.trajL   = std::sqrt(2.0);
  if (const char *tl = std::getenv("TRAJL"); tl && *tl) MD.trajL = std::atof(tl);
  std::cout << GridLogMessage
            << "INTEGRATOR=" << MD.name << " MDsteps=" << MD.MDsteps
            << " trajL=" << MD.trajL << std::endl;

  // ---- HMC parameters ----
  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  // Trajectories default to n_therm + n_prod; override via TRAJ env var for
  // smoke runs (TRAJ=3) or scans.
  HMCp.Trajectories        = TXQCDProduction::detail::env_int("TRAJ",
                                                              n_therm + n_prod);
  HMCp.NoMetropolisUntil   = TXQCDProduction::detail::env_int("NO_METROP", 0);
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  // ---- checkpointing ----
  const std::string cfg_dir = dtxqcd_cfg_dir();
  mkdir_p(cfg_dir);
  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  DTXQCDCheckpointer ckpt(CPp);

  // ---- run.  v1: gauge-only smearer (no stout); add DTXQCD smearer later
  //      once stout-chained DTXQCDSmearedConfiguration is ported.
  NoSmearing<DTXQCDCompositeImpl> Smear;
  Smear.set_Field(U);

  std::vector<HmcObservable<DTXQCDField> *> Obs = {&ckpt};

  typedef ForceGradient<DTXQCDCompositeImpl,
                        NoSmearing<DTXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage
            << "DTXQCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
