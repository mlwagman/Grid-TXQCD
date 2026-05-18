// TXQCD HMC driver, Möbius DWF prototype.
//
// Composite field = (gauge U) + (sigma, pi, s, p, t).  Two MD levels:
//   Level 1 (inner, dt fine):  TXQCDMobiusPseudoFermionAction + AuxGaussian.
//   Level 2 (outer, multiplier): GaugeActionAdapter<WilsonGaugeAction>.
//
// Mirrors HMC/TXQCD_Wilson_small.cc; the only changes are the 5D grids, the
// Möbius operator (with M5, b, c), and the Möbius PF action.

#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusPseudoFermionAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  const int Ls = 8;
  Coordinate latt4(std::vector<int>{4, 4, 4, 4});
  Coordinate simd  = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi   = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridSerialRNG    sRNG;            sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  GridParallelRNG  pRNG4(UGrid);    pRNG4.SeedFixedIntegers({6, 7, 8, 9, 10});

  // ---- physics parameters ----
  RealD beta   = 5.6;
  RealD lambda = 3.0;
  RealD mass   = 0.05;            // light Möbius m_l
  RealD M5     = 1.8;             // standard
  RealD b      = 1.5;             // Möbius b
  RealD c      = 0.5;             // Möbius c   (b+c = 2)
  RealD cg_tol = 1e-8;
  int   cg_max = 10000;

  // ---- actions ----
  GaugeActionAdapter<WilsonGaugeActionR>  GaugeAction(beta);
  AuxiliaryFieldGaussianAction             AuxAction(lambda);
  TXQCDMobiusPseudoFermionAction           PFAction(*FGrid, *FrbGrid,
                                                    *UGrid, *UrbGrid,
                                                    mass, M5, b, c,
                                                    cg_tol, cg_max);

  typedef Representations<EmptyRep<TXQCDField>> TxqcdReps;
  ActionLevel<TXQCDField, TxqcdReps> Level1(1);
  Level1.push_back(&PFAction);
  Level1.push_back(&AuxAction);
  ActionLevel<TXQCDField, TxqcdReps> Level2(4);
  Level2.push_back(&GaugeAction);

  ActionSet<TXQCDField, TxqcdReps> Aset;
  Aset.push_back(Level1);
  Aset.push_back(Level2);

  IntegratorParameters MD;
  MD.name    = "LeapFrog";
  MD.MDsteps = 80;
  MD.trajL   = 0.5;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory    = 0;
  HMCparams.Trajectories       = 10;
  HMCparams.NoMetropolisUntil  = 0;
  HMCparams.MetropolisTest     = true;
  HMCparams.PerformRandomShift = false;
  HMCparams.StartingType       = "ColdStart";
  HMCparams.MD                 = MD;

  NoSmearing<TXQCDCompositeImpl> Smearer;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, TxqcdReps> IntegratorT;
  IntegratorT MDynamics(UGrid, MD, Aset, Smearer);

  TXQCDField U(UGrid);
  TXQCDCompositeImpl::ColdConfiguration(pRNG4, U);
  Smearer.set_Field(U);

  std::vector<HmcObservable<TXQCDField> *> Observables;
  HybridMonteCarlo<IntegratorT> HMC(HMCparams, MDynamics, sRNG, pRNG4,
                                    Observables, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}

#endif
