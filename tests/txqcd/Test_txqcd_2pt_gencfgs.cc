// Step 1: Generate TXQCD and QCD gauge configurations for the 2pt test suite.
// Skips generation if configs already exist at all measurement trajectories.
//
// TXQCD: one rational PF (|det M_TX|^1, Nf_tx=2 -> Nf=2 Wilson at aux=0)
//        + Gaussian aux action + Wilson gauge action.
// QCD:   one TwoFlavour PF (|det M_W|^2 = Nf=2 Wilson) + Wilson gauge action.

#include "Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetEOAction.h>

using namespace TxqcdTest2pt;

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
              << "TXQCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating TXQCD configs (" << total_traj << " trajectories)..."
              << std::endl;
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
    TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass, rat_params);
    TXQCDLogDetEOAction         LogDet(Grid, RBGrid, mass);

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
      TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
    }

    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - start_traj;
    HMCp.NoMetropolisUntil   = n_therm;
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

    std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD ====================
  if (qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 configs (" << total_traj
              << " trajectories)..." << std::endl;
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

    WilsonFermionD FermOp(Umu, Grid, RBGrid, mass);
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

    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - start_traj;
    HMCp.NoMetropolisUntil   = n_therm;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef ForceGradient<PeriodicGimplR,
                          NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    // QCD checkpointer using NerscIO
    struct QcdCkpt : public HmcObservable<LatticeGaugeField> {
      std::string cfg_prefix, rng_prefix;
      int save_interval;
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &sR,
                              GridParallelRNG &pR) override {
        if (t % save_interval != 0) return;
        NerscIO::writeRNGState(sR, pR,
                               rng_prefix + "." + std::to_string(t));
        NerscIO::writeConfiguration<GaugeStats>(
            U, cfg_prefix + "." + std::to_string(t), 0, 1);
      }
    };
    QcdCkpt ckpt;
    ckpt.cfg_prefix    = qcd_cfg_dir() + "/ckpoint_lat";
    ckpt.rng_prefix    = qcd_cfg_dir() + "/ckpoint_rng";
    ckpt.save_interval = meas_skip;

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "Config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
