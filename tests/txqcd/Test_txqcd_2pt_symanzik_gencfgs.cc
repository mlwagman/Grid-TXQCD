// Step 1 (Symanzik): Generate Nf=2+1 TXQCD and QCD gauge configurations with
// stout smearing and tadpole-improved Lüscher-Weisz (Symanzik) gauge action.
//
// Light quarks (u,d): TXQCD RHMC + LogDet (or QCD TwoFlavour PF)
// Strange quark (s):  standard QCD OneFlavourRational RHMC via QCDActionAdapter

#include "Test_txqcd_2pt_symanzik_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptSymanzik;

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
              << "TXQCD symanzik configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating TXQCD symanzik configs (" << total_traj
              << " trajectories, csw=" << csw << ", rho=" << stout_rho
              << ", Nsmear=" << stout_nsmear << ")..." << std::endl;
    mkdir_p(txqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_txqcd_checkpoint();

    RealD cg_tol = 1e-8;
    OneFlavourRationalParams rat_params(1e-4, 200.0, cg_max, cg_tol, 12, 64,
                                        100, 1e-6, 1e-4);

    typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
    GaugeActionAdapter<SymanzikR> GaugeAction(beta, u0);
    GaugeAction.is_smeared = true;

    AuxiliaryFieldGaussianAction           AuxAction(lambda);

    TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rat_params, csw);
    PF.is_smeared = true;

    TXQCDLogDetCloverEOAction         LogDet(Grid, RBGrid, mass, csw);
    LogDet.is_smeared = true;

    TXQCDField U(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming TXQCD symanzik from checkpoint at traj "
                << latest << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
    }

    // Strange quark: standard QCD one-flavor RHMC, wrapped for TXQCD HMC
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF StrangeFermOp(U.U, Grid, RBGrid, mass_s, csw, csw);
    OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, cg_tol, 12, 64,
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

    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - start_traj;
    HMCp.NoMetropolisUntil   = n_therm;
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

    TxqcdDiagnostics diag(txqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"PseudoFermion", &PF},
        {"LogDet", &LogDet},
        {"AuxGaussian", &AuxAction},
        {"StrangeQuark", &StrangeAdapter},
        {"Gauge", &GaugeAction}
    });

    std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD ====================
  if (qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD symanzik configs already exist, skipping generation."
              << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2+1 symanzik configs (" << total_traj
              << " trajectories, csw=" << csw << ", rho=" << stout_rho
              << ", Nsmear=" << stout_nsmear << ")..." << std::endl;
    mkdir_p(qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD symanzik from checkpoint at traj "
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

    // Strange quark: one-flavor RHMC
    WCF StrangeFermOp(Umu, Grid, RBGrid, mass_s, csw, csw);
    OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, 1e-8, 12, 64,
                                         100, 1e-6, 1e-4);
    OneFlavourRationalPseudoFermionAction<WilsonImplR> StrangePF(StrangeFermOp,
                                                                  strange_rat);
    StrangePF.is_smeared = true;

    SymanzikGaugeAction<PeriodicGimplR> GaugeAction(beta, u0);
    GaugeAction.is_smeared = true;

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&Nf2);
    L1.push_back(&StrangePF);
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

    QcdDiagnostics diag(qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"Nf2", &Nf2},
        {"StrangeQuark", &StrangePF},
        {"Gauge", &GaugeAction}
    });

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "Symanzik config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
