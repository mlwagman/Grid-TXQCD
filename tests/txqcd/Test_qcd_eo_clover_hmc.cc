// Quick HMC comparison: non-EO (TwoFlavourPseudoFermion) vs EO (LogDet + SchurPF).
// Runs a few trajectories of each and reports CG iteration counts and deltaH.

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavour.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt({4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  constexpr RealD mass = 0.3;
  constexpr RealD csw  = 1.0;
  constexpr RealD beta = 5.6;
  constexpr int   n_traj = 5;

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

  // ==================== Non-EO HMC ====================
  {
    std::cout << GridLogMessage
              << "===== Non-EO HMC (TwoFlavourPseudoFermion) =====" << std::endl;

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
    HMC.evolve();

    std::cout << GridLogMessage << "Non-EO HMC complete." << std::endl;
  }

  // ==================== EO HMC ====================
  {
    std::cout << GridLogMessage
              << "===== EO HMC (LogDet + SchurPF) =====" << std::endl;

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
    HMC.evolve();

    std::cout << GridLogMessage << "EO HMC complete." << std::endl;
  }

  Grid_finalize();
  return 0;
}
