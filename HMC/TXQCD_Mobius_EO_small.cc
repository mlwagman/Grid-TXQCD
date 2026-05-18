// TXQCD HMC driver, EO-preconditioned Möbius DWF prototype.
//
// Composite field = (gauge U) + (sigma, pi, s, p, t).  Two MD levels:
//   Level 1 (inner): two TXQCDMobiusRationalEOAction (|det M|^2 = det M†M,
//                     i.e. Nf=2-equivalent to one full PF) + AuxGaussian.
//   Level 2 (outer):  GaugeActionAdapter<WilsonGaugeAction>.
//
// Mirrors HMC/TXQCD_Mobius_small.cc but swaps the full (M†M)^{-1} pseudo-
// fermion action for the EO/Schur rational (M†M)^{-1/2} action — CG runs on
// the half-volume odd sublattice via TXQCDMobiusSchurOp, which is the EO
// speedup.  Same lattice / Möbius params so the two scouts are directly
// comparable on dH, acceptance, and trajectory time.

#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusRationalEOAction.h>

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

  // ---- physics parameters (match TXQCD_Mobius_small.cc) ----
  RealD beta   = 5.6;
  RealD lambda = 3.0;
  RealD mass   = 0.05;
  RealD M5     = 1.8;
  RealD b      = 1.5;
  RealD c      = 0.5;            // b+c = 2

  // Rational params: (M†M)^{-1/2}, degree 12 ⇒ Remez err ~1e-9.
  OneFlavourRationalParams rat_params(/*lo=*/1e-4, /*hi=*/64.0,
                                      /*maxit=*/20000, /*tol=*/1e-10,
                                      /*degree=*/12, /*precision=*/64,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1e-7,
                                      /*BoundsCheckTol=*/1e-4);

  // ---- actions ----
  GaugeActionAdapter<WilsonGaugeActionR>  GaugeAction(beta);
  AuxiliaryFieldGaussianAction             AuxAction(lambda);
  // Two PFs on the shared TXQCDField: |det M_TX|^2 (Nf=2-equivalent).
  TXQCDMobiusRationalEOAction PF1(*FGrid, *FrbGrid, *UGrid, *UrbGrid,
                                  mass, M5, b, c, rat_params);
  TXQCDMobiusRationalEOAction PF2(*FGrid, *FrbGrid, *UGrid, *UrbGrid,
                                  mass, M5, b, c, rat_params);

  typedef Representations<EmptyRep<TXQCDField>> TxqcdReps;
  ActionLevel<TXQCDField, TxqcdReps> Level1(1);
  Level1.push_back(&PF1);
  Level1.push_back(&PF2);
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
