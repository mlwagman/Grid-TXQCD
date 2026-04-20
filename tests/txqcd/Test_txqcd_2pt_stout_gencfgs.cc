// Step 1 (Stout): Generate TXQCD and QCD gauge configurations with stout smearing.
// Same HMC settings as clover version, plus stout smearing of gauge links.
//
// TXQCD: rational PF + LogDet (Wilson-Clover) + Gaussian aux + Wilson gauge,
//        all evaluated on stout-smeared gauge links.
// QCD:   TwoFlavour PF (Wilson-Clover) + Wilson gauge, stout-smeared.

#include "Test_txqcd_2pt_stout_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptStout;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;

  // ==================== TXQCD ====================
  if (txqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "TXQCD stout configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating TXQCD stout configs (" << total_traj
              << " trajectories, csw=" << csw << ", rho=" << stout_rho
              << ", Nsmear=" << stout_nsmear << ")..." << std::endl;
    mkdir_p(txqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_txqcd_checkpoint();

    RealD cg_tol = 1e-8;
    OneFlavourRationalParams rat_params(1e-4, 64.0, cg_max, cg_tol, 12, 64,
                                        100, 1e-6, 1e-4);

    GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    GaugeAction.is_smeared = true;

    AuxiliaryFieldGaussianAction           AuxAction(lambda);
    // AuxAction.is_smeared stays false — aux fields are not smeared

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
      std::cout << GridLogMessage << "Resuming TXQCD stout from checkpoint at traj "
                << latest << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      // Tepid start: Grid's stout smearing has a 0/0 singularity on cold configs
      TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
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
    CPp.config_prefix = txqcd_cfg_dir() + "/ckpoint_lat";
    CPp.rng_prefix    = txqcd_cfg_dir() + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    TXQCDCheckpointer ckpt(CPp);

    TxqcdSmearedDiagnostics<TXQCDSmearedConfiguration> diag(
        txqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"PseudoFermion", &PF},
        {"LogDet", &LogDet},
        {"AuxGaussian", &AuxAction},
        {"Gauge", &GaugeAction}
    }, Smear, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD ====================
  if (qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD stout configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 stout configs (" << total_traj
              << " trajectories, csw=" << csw << ", rho=" << stout_rho
              << ", Nsmear=" << stout_nsmear << ")..." << std::endl;
    mkdir_p(qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD stout from checkpoint at traj "
                << latest << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      SU<Nc>::TepidConfiguration(pRNG, Umu);
    }

    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF FermOp(Umu, Grid, RBGrid, mass, csw, csw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
    Nf2.is_smeared = true;

    WilsonGaugeActionR GaugeAction(beta);
    GaugeAction.is_smeared = true;

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
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear, Stout);

    typedef ForceGradient<PeriodicGimplR,
                          SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    QcdCheckpointer ckpt;
    ckpt.cfg_prefix    = qcd_cfg_dir() + "/ckpoint_lat";
    ckpt.rng_prefix    = qcd_cfg_dir() + "/ckpoint_rng";
    ckpt.save_interval = meas_skip;

    QcdSmearedDiagnostics<SmearedConfiguration<PeriodicGimplR>> diag(
        qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"Nf2", &Nf2},
        {"Gauge", &GaugeAction}
    }, Smear, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "Stout config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
