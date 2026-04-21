// Step 1 (Clover): Generate TXQCD and QCD gauge configurations with csw != 0.
// Same HMC settings as the Wilson version; only the fermion action changes.
//
// TXQCD: rational PF + LogDet (Wilson-Clover) + Gaussian aux + Wilson gauge.
// QCD:   TwoFlavour PF (Wilson-Clover) + Wilson gauge.

#include "Test_txqcd_2pt_clover_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>

using namespace TxqcdTest2ptClover;

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
              << "TXQCD clover configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating TXQCD clover configs (" << total_traj
              << " trajectories, csw=" << csw << ")..." << std::endl;
    mkdir_p(txqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_txqcd_checkpoint();

    RealD cg_tol = 1e-8;
    OneFlavourRationalParams rat_params(1e-4, 64.0, cg_max, cg_tol, 12, 64,
                                        100, 1e-6, 1e-4);

    GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
    AuxiliaryFieldGaussianAction           AuxAction(lambda);
    TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rat_params, csw);
    TXQCDLogDetCloverEOAction         LogDet(Grid, RBGrid, mass, csw);

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
      std::cout << GridLogMessage << "Resuming TXQCD clover from checkpoint at traj "
                << latest << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
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
    }, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD ====================
  if (qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD clover configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 clover configs (" << total_traj
              << " trajectories, csw=" << csw << ")..." << std::endl;
    mkdir_p(qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD clover from checkpoint at traj "
                << latest << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      SU<Nc>::ColdConfiguration(Umu);
    }

    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF FermOp(Umu, Grid, RBGrid, mass, csw, csw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    QCDLogDetCloverEOAction<WilsonImplR> LogDetAction(FermOp);
    TwoFlavourSchurCloverAction<WilsonImplR> SchurPF(FermOp, CG, CG);

    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&LogDetAction);
    L1.push_back(&SchurPF);
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
        {"LogDet", &LogDetAction},
        {"SchurPF", &SchurPF},
        {"Gauge", &GaugeAction}
    }, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "Clover config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
