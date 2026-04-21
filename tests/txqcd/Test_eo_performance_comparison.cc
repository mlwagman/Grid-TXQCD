// 3-way HMC performance comparison:
//   1. QCD non-EO  (TwoFlavourPseudoFermion on full volume)
//   2. QCD EO      (LogDet + TwoFlavourSchurClover on half volume)
//   3. TXQCD EO    (TXQCDLogDetClover + TXQCDWilsonCloverRationalEO + AuxGaussian)
//
// All use the same lattice, mass, csw, beta, integrator, and trajectory length.
// Identical RNG seeds so the gauge evolution starts from the same point
// (though TXQCD has extra auxiliary fields, so trajectories diverge immediately).

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavour.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt({4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  constexpr RealD mass   = 0.3;
  constexpr RealD csw    = 1.0;
  constexpr RealD beta   = 5.6;
  constexpr RealD lambda = 3.0;
  constexpr int   n_traj = 5;

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

  std::cout << GridLogMessage << "============================================" << std::endl;
  std::cout << GridLogMessage << "  EO Preconditioning Performance Comparison" << std::endl;
  std::cout << GridLogMessage << "  Lattice: 4^3 x 8, mass=" << mass
            << ", csw=" << csw << ", beta=" << beta << std::endl;
  std::cout << GridLogMessage << "  Trajectories: " << n_traj
            << ", LeapFrog, MDsteps=8, trajL=0.5" << std::endl;
  std::cout << GridLogMessage << "============================================" << std::endl;

  // ==================== 1. QCD Non-EO ====================
  RealD time_qcd_noneo;
  {
    std::cout << GridLogMessage << std::endl;
    std::cout << GridLogMessage
              << "===== 1. QCD Non-EO (TwoFlavourPseudoFermion) =====" << std::endl;

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

    LatticeGaugeField Umu(&Grid);
    SU<Nc>::ColdConfiguration(Umu);

    WCF FermOp(Umu, Grid, RBGrid, mass, csw, csw);
    ConjugateGradient<LatticeFermion> CG(1e-8, 10000);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
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
    MD.name    = "LeapFrog";
    MD.MDsteps = 8;
    MD.trajL   = 0.5;

    HMCparameters HMCp;
    HMCp.StartTrajectory     = 0;
    HMCp.Trajectories        = n_traj;
    HMCp.NoMetropolisUntil   = 0;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef LeapFrog<PeriodicGimplR, NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);

    RealD t0 = usecond();
    HMC.evolve();
    time_qcd_noneo = (usecond() - t0) / 1e6;

    std::cout << GridLogMessage << "QCD Non-EO wall time: " << time_qcd_noneo
              << " s (" << time_qcd_noneo / n_traj << " s/traj)" << std::endl;
  }

  // ==================== 2. QCD EO ====================
  RealD time_qcd_eo;
  {
    std::cout << GridLogMessage << std::endl;
    std::cout << GridLogMessage
              << "===== 2. QCD EO (LogDet + TwoFlavourSchurClover) =====" << std::endl;

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

    LatticeGaugeField Umu(&Grid);
    SU<Nc>::ColdConfiguration(Umu);

    WCF FermOp(Umu, Grid, RBGrid, mass, csw, csw);
    ConjugateGradient<LatticeFermion> CG(1e-8, 10000);
    QCDLogDetCloverEOAction<WilsonImplR> LogDet(FermOp);
    TwoFlavourSchurCloverAction<WilsonImplR> SchurPF(FermOp, CG, CG);
    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&LogDet);
    L1.push_back(&SchurPF);
    ActionLevel<LatticeGaugeField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name    = "LeapFrog";
    MD.MDsteps = 8;
    MD.trajL   = 0.5;

    HMCparameters HMCp;
    HMCp.StartTrajectory     = 0;
    HMCp.Trajectories        = n_traj;
    HMCp.NoMetropolisUntil   = 0;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef LeapFrog<PeriodicGimplR, NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);

    RealD t0 = usecond();
    HMC.evolve();
    time_qcd_eo = (usecond() - t0) / 1e6;

    std::cout << GridLogMessage << "QCD EO wall time: " << time_qcd_eo
              << " s (" << time_qcd_eo / n_traj << " s/traj)" << std::endl;
  }

  // ==================== 3. TXQCD EO ====================
  RealD time_txqcd;
  {
    std::cout << GridLogMessage << std::endl;
    std::cout << GridLogMessage
              << "===== 3. TXQCD EO (RHMC + LogDet + AuxGaussian) =====" << std::endl;

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

    OneFlavourRationalParams rat_params(1e-4, 64.0, 10000, 1e-8, 12, 64,
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
    MD.name    = "LeapFrog";
    MD.MDsteps = 8;
    MD.trajL   = 0.5;

    TXQCDField U(&Grid);
    TXQCDCompositeImpl::ColdConfiguration(pRNG, U);

    HMCparameters HMCp;
    HMCp.StartTrajectory     = 0;
    HMCp.Trajectories        = n_traj;
    HMCp.NoMetropolisUntil   = 0;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<TXQCDCompositeImpl> Smear;
    typedef LeapFrog<TXQCDCompositeImpl,
                     NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(U);

    std::vector<HmcObservable<TXQCDField> *> Obs;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);

    RealD t0 = usecond();
    HMC.evolve();
    time_txqcd = (usecond() - t0) / 1e6;

    std::cout << GridLogMessage << "TXQCD EO wall time: " << time_txqcd
              << " s (" << time_txqcd / n_traj << " s/traj)" << std::endl;
  }

  // ==================== Summary ====================
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "============================================" << std::endl;
  std::cout << GridLogMessage << "  PERFORMANCE SUMMARY (" << n_traj << " trajectories)" << std::endl;
  std::cout << GridLogMessage << "============================================" << std::endl;
  std::cout << GridLogMessage << "QCD Non-EO:  " << time_qcd_noneo << " s  ("
            << time_qcd_noneo / n_traj << " s/traj)" << std::endl;
  std::cout << GridLogMessage << "QCD EO:      " << time_qcd_eo << " s  ("
            << time_qcd_eo / n_traj << " s/traj)" << std::endl;
  std::cout << GridLogMessage << "TXQCD EO:    " << time_txqcd << " s  ("
            << time_txqcd / n_traj << " s/traj)" << std::endl;
  std::cout << GridLogMessage << "Speedup (QCD Non-EO / QCD EO):   "
            << time_qcd_noneo / time_qcd_eo << "x" << std::endl;
  std::cout << GridLogMessage << "Overhead (TXQCD EO / QCD EO):    "
            << time_txqcd / time_qcd_eo << "x" << std::endl;
  std::cout << GridLogMessage << "Overhead (TXQCD EO / QCD Non-EO): "
            << time_txqcd / time_qcd_noneo << "x" << std::endl;
  std::cout << GridLogMessage << "============================================" << std::endl;

  Grid_finalize();
  return 0;
}
