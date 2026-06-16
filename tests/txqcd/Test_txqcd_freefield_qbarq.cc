// Test_txqcd_freefield_qbarq:
//
// Free-field unit test for ⟨q̄q⟩ + aux saddles in TXQCD.
//
// Setup:
//   U = Identity (gauge frozen at U_μ(x) = I)
//   m = large (default 100) → kappa = 1/(2(4+m)) ~ 0 → hopping ≈ 0
//   csw = 0
//
// In this limit the TXQCD Dirac operator is purely diagonal in space-time:
//   M(x) = (4+m) · I_24 + X(aux(x))
// where X is the local aux insertion (σ + s + (π+p)γ5 + t·σ_μν).
//
// All sites are independent ⇒ HMC samples a per-site Gaussian × logdet
// measure.  No gauge dynamics (no gauge force at U=1, kappa=0).
//
// Analytic free-field predictions (aux=0 limit, λ → ∞):
//   ⟨q̄q⟩ = Tr[M⁻¹]/(V · D_block) = 1/(4+m) per spin-color mode
//   Σ (our convention, /(2V·Nf)) = (Ns·Nc)/(2·(4+m)) = 6/(4+m) for Nc=3
//   ⟨s⟩, ⟨σ⟩ → 0
//
// At finite λ, the saddle ⟨s⟩ ≠ 0; checking it matches saddle equation
// from the local action  S_site = (λ²/2)Tr(aux²) + (λ²/4)(s²+p²) - log det(M_site).
//
// Env knobs:
//   LAMBDA    — aux coupling (required)
//   MASS      — Wilson mass (default 100 — effectively kappa=0)
//   MDSTEPS   — integrator MD steps (default 4)
//   TRAJL     — trajectory length (default 1.0)
//   N_PROD    — production trajectories (default 200)
//   N_THERM   — thermalization trajectories (default 50)

#include "Test_txqcd_2pt_clover_optlam_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptCloverOptlam;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_vev_noise;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  RealD lambda_run = 2.0;
  RealD mass_run   = 100.0;          // effectively kappa=0
  RealD csw_run    = 0.0;
  int mdsteps      = 4;
  RealD trajL      = 1.0;
  int n_therm_run  = 50;
  int n_prod_run   = 200;
  int meas_skip_run = 10;
  bool dh_only     = false;          // if true, no Metropolis (pure dH probe)
  std::string cfg_dir = "free_txqcd";

  if (const char *v = std::getenv("LAMBDA");   v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");     v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");  v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");    v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");  v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");   v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");  v && *v) cfg_dir = v;
  if (const char *v = std::getenv("DH_ONLY");  v && *v) dh_only = (std::atoi(v) != 0);

  // Analytic free-field prediction in our /(2V) convention.
  const RealD sigma_free = (Ns * Nc) / (2.0 * (4.0 + mass_run));   // = 6/(4+m) for Nc=3

  std::cout << GridLogMessage
            << "TXQCD FREE-FIELD test: lambda=" << lambda_run
            << " mass=" << mass_run << " (kappa~" << 1.0/(2.0*(4.0+mass_run)) << ")"
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " cfg_dir=" << cfg_dir << std::endl;
  std::cout << GridLogMessage
            << "  predicted free-field Σ = (Ns·Nc)/(2(4+m)) = " << sigma_free
            << "  (≡ 4Nc/(4+m) / 2 in user convention)" << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm_run + n_prod_run;
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  int rng_seed = 0;
  if (const char *v = std::getenv("RNG_SEED"); v && *v) rng_seed = std::atoi(v);
  sRNG.SeedFixedIntegers({1 + rng_seed, 2 + rng_seed, 3 + rng_seed,
                          4 + rng_seed, 5 + rng_seed});
  pRNG.SeedFixedIntegers({6 + rng_seed, 7 + rng_seed, 8 + rng_seed,
                          9 + rng_seed, 10 + rng_seed});
  std::cout << GridLogMessage << "  RNG_SEED offset = " << rng_seed << std::endl;

  // RATIONAL Nf=2 PF: S = phi^dag (M^dag M)^{-1/2} phi → weight |det M_TX|.
  // At aux=0 with internal Nf=2 flavor block: |det M_TX| = (det D_W)² = Nf=2 vanilla Wilson.
  // This matches TXQCD's Fierz construction which is Nf=2-specific.
  //
  // Using the non-rational PF (|det M_TX|² = Nf=4 effective) produces a ~2%
  // Σ_TX/Σ_W bias at finite m because the Fierz identity doesn't cancel the
  // extra Nf=4 four-quark combinations.  See project_txqcd_fierz_is_nf2_specific.
  //
  // Rational bracket: M^dag M at U=I has eigenvalues ~ (m+4)².  Defaults
  // sized for m=1000 (M²~1e6); override with RAT_LO/RAT_HI/RAT_DEGREE for
  // smaller m runs.
  RealD rat_lo = 1e4, rat_hi = 2e6;
  int rat_deg  = 8;
  if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo  = std::atof(v);
  if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi  = std::atof(v);
  if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_deg = std::atoi(v);
  RealD pf_cg_tol = 1e-10;
  if (const char *v = std::getenv("CG_TOL"); v && *v) pf_cg_tol = std::atof(v);
  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, pf_cg_tol,
                                       rat_deg, 64, 100, 1e-6);
  std::cout << GridLogMessage
            << "  RHMC bracket: lo=" << rat_lo << " hi=" << rat_hi
            << " degree=" << rat_deg << std::endl;

  AuxiliaryFieldGaussianAction           AuxAction(lambda_run);
  TXQCDWilsonRationalPseudoFermionAction PF(Grid, RBGrid, mass_run, rat_params);

  // FREEZE gauge by zeroing gauge momentum (TXQCD_FREEZE_GAUGE knob, read by
  // TXQCDCompositeImpl::generate_momenta).  No gauge action needed — U just
  // doesn't move because its momentum stays zero.  Confirms below by
  // logging plaq each traj.
  setenv("TXQCD_FREEZE_GAUGE", "1", 1);
  std::cout << GridLogMessage << "  TXQCD_FREEZE_GAUGE=1 → gauge momentum forced to zero"
            << std::endl;

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
  // Initialize U = I and aux fields.
  TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  // SIGMA_INIT env knob: seed aux at the predicted saddle using
  // FillAuxFields(Sigma).  Sigma=0 keeps aux=0 (cold start).
  // SADDLE_INIT=1 computes the free-field Sigma automatically:
  //   <q̄q>_free / 2 = Ns·Nc/(4·(4+m)) = 3/(4+m)   (compute_trminv "/4V" convention)
  // and uses it as the seed (saddle <s> ≈ Nf·Σ/(√2·Nc·λ²)).
  RealD sigma_init = 0.0;
  if (const char *v = std::getenv("SIGMA_INIT"); v && *v) {
    sigma_init = std::atof(v);
  } else if (const char *v = std::getenv("SADDLE_INIT");
             v && std::atoi(v) != 0) {
    sigma_init = 3.0 / (4.0 + mass_run);
  }
  if (sigma_init != 0.0) {
    TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, sigma_init);
    std::cout << GridLogMessage
              << "  aux seeded at saddle: Sigma=" << sigma_init
              << "  → <s>_init ~ Nf·Σ/(√2·Nc·λ²) = "
              << (TxqcdNf * sigma_init / (std::sqrt(2.0) * Nc * lambda_run * lambda_run))
              << std::endl;
  }
  // verify U = I (plaq should be 1)
  RealD init_plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage << "  U cold start: plaq = " << init_plaq
            << " (expect 1)" << std::endl;

  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  HMCp.Trajectories        = total_traj - (dh_only ? total_traj : n_therm_run);
  HMCp.NoMetropolisUntil   = dh_only ? total_traj : n_therm_run;
  HMCp.MetropolisTest      = !dh_only;
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

  TxqcdTest2ptClover::TxqcdDiagnostics diag(
      cfg_dir + "/hmc_diagnostics", meas_skip_run, {
      {"PseudoFermion", &PF},
      {"AuxGaussian",   &AuxAction}
  }, Grid, RBGrid, pRNG, mass_run, csw_run, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}
