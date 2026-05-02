// Pure Wilson Nf=2 TXQCD vs QCD gauge generation on an 8^3 x 24 lattice
// with lighter quark mass.  Same action structure as Test_txqcd_2pt_gencfgs
// but with the wilson8x24 utils overriding lattice/beta/mass and exposing a
// runtime LAMBDA env var (with cfg-dir auto-suffix) so optlam vs reference
// runs can share one binary.

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetEOAction.h>

using namespace TxqcdTest2ptWilson8x24;

// Env-var overrides for short pipeline tests (N_THERM/N_PROD).
static int env_int(const char *name, int def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}

// WHICH={qcd,txqcd,both} -- gen one or both ensembles.  Useful so we can
// generate QCD first, measure Sigma_l on it to set lambda_opt, then run
// TXQCD with WHICH=txqcd LAMBDA=<opt>.
static std::string which_run() {
  const char *v = std::getenv("WHICH");
  std::string w = (v && *v) ? std::string(v) : std::string("both");
  return w;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int n_therm_use = env_int("N_THERM", n_therm);
  int n_prod_use  = env_int("N_PROD",  n_prod);
  int total_traj  = n_therm_use + n_prod_use;

  RealD lam = lambda_runtime();
  std::string WHICH = which_run();
  bool do_qcd   = (WHICH == "qcd"   || WHICH == "both");
  bool do_txqcd = (WHICH == "txqcd" || WHICH == "both");
  std::cout << GridLogMessage << "WHICH=" << WHICH
            << " (qcd=" << do_qcd << ", txqcd=" << do_txqcd << ")"
            << std::endl;

  // ==================== TXQCD ====================
  if (do_txqcd && txqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "TXQCD wilson8x24 configs already exist, skipping." << std::endl;
  } else if (do_txqcd) {
    std::cout << GridLogMessage
              << "Generating TXQCD wilson8x24 configs (8^3x24, beta=" << beta
              << ", m_l=" << mass_runtime() << ", csw=" << csw << ", lambda=" << lam
              << ", " << total_traj << " traj)..." << std::endl;
    mkdir_p(txqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_txqcd_checkpoint();

    RealD cg_tol = 1e-8;
    OneFlavourRationalParams rat_params(1e-4, 64.0, cg_max, cg_tol, 12, 64,
                                        100, 1e-6, 1e-4);

    GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    AuxiliaryFieldGaussianAction           AuxAction(lam);
    TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass_runtime(), rat_params);
    TXQCDLogDetEOAction         LogDet(Grid, RBGrid, mass_runtime());

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
      std::cout << GridLogMessage << "Resuming TXQCD from checkpoint at traj "
                << latest << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
    }

    int no_metrop = (start_traj < n_therm_use) ? (n_therm_use - start_traj) : 0;
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
    CPp.config_prefix = txqcd_cfg_dir() + "/ckpoint_lat";
    CPp.rng_prefix    = txqcd_cfg_dir() + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    TXQCDCheckpointer ckpt(CPp);

    TxqcdDiagnostics diag(txqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"PseudoFermion", &PF},
        {"LogDet", &LogDet},
        {"AuxGaussian", &AuxAction},
        {"Gauge", &GaugeAction}
    }, Grid, RBGrid, pRNG, mass_runtime(), csw, n_vev_noise);

    std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD ====================
  if (do_qcd && qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD wilson8x24 configs already exist, skipping." << std::endl;
  } else if (do_qcd) {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 wilson8x24 configs (" << total_traj
              << " traj)..." << std::endl;
    mkdir_p(qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD from checkpoint at traj "
                << latest << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      SU<Nc>::ColdConfiguration(Umu);
    }

    WilsonFermionD FermOp(Umu, Grid, RBGrid, mass_runtime());
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
    Nf2.is_smeared = false;

    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&Nf2);
    ActionLevel<LatticeGaugeField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name = "ForceGradient";
    MD.MDsteps = 10;
    MD.trajL = 0.5;

    int no_metrop = (start_traj < n_therm_use) ? (n_therm_use - start_traj) : 0;
    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - no_metrop - start_traj;
    HMCp.NoMetropolisUntil   = no_metrop;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef ForceGradient<PeriodicGimplR,
                          NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    QcdCheckpointer ckpt;
    ckpt.cfg_prefix    = qcd_cfg_dir() + "/ckpoint_lat";
    ckpt.rng_prefix    = qcd_cfg_dir() + "/ckpoint_rng";
    ckpt.save_interval = meas_skip;

    QcdDiagnostics diag(qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"Nf2", &Nf2},
        {"Gauge", &GaugeAction}
    }, Grid, RBGrid, pRNG, mass_runtime(), csw, n_vev_noise);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage
            << "Wilson 8x24 config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
