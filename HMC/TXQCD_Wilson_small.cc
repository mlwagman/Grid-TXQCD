// TXQCD HMC driver, Wilson-fermion prototype (Phase 4d).
//
// Composite Field = (gauge U) + (sigma, pi, s, p, t). Integrator templated on
// TXQCDCompositeImpl, two MD levels:
//   Level 1 (inner, dt fine):  TXQCDWilsonPseudoFermionAction + AuxGaussian.
//   Level 2 (outer, multiplier): GaugeActionAdapter<WilsonGaugeAction>.
//
// Bypasses HMCResourceManager (which assumes a gauge-only Field) and wires the
// HybridMonteCarlo + Integrator + RNGs by hand.

#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  // ---- geometry ----
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG    sRNG;
  GridParallelRNG  pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  // ---- physics parameters ----
  RealD beta   = 5.6;     // Wilson gauge coupling
  RealD lambda = 3.0;     // aux Gaussian width: per-DOF variance = 1/lambda^2 ~ 0.11
  RealD mass   = 0.3;     // Wilson quark mass (well inside positive M^dag M regime)
  RealD cg_tol = 1e-8;
  int   cg_max = 10000;

  // ---- actions on the composite field ----
  GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
  AuxiliaryFieldGaussianAction            AuxAction(lambda);
  TXQCDWilsonPseudoFermionAction          PFAction(Grid, RBGrid, mass, cg_tol, cg_max);

  // RepresentationsPolicy must accept TXQCDField (the default NoHirep is
  // FundamentalRep<Nc> and tries to call update_representation on a Lattice
  // gauge field). EmptyRep<TXQCDField> is a no-op.
  typedef Representations<EmptyRep<TXQCDField>> TxqcdReps;

  // ---- action set: Level 1 inner, Level 2 outer (multiplier=4) ----
  ActionLevel<TXQCDField, TxqcdReps> Level1(1);
  Level1.push_back(&PFAction);
  Level1.push_back(&AuxAction);
  ActionLevel<TXQCDField, TxqcdReps> Level2(4);
  Level2.push_back(&GaugeAction);

  ActionSet<TXQCDField, TxqcdReps> Aset;
  Aset.push_back(Level1);
  Aset.push_back(Level2);

  // ---- integrator + HMC ----
  IntegratorParameters MD;
  MD.name    = "LeapFrog";
  MD.MDsteps = 80;
  MD.trajL   = 0.5;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory   = 0;
  HMCparams.Trajectories      = 10;
  HMCparams.NoMetropolisUntil = 0;
  HMCparams.MetropolisTest    = true;
  HMCparams.PerformRandomShift = false;
  HMCparams.StartingType      = "ColdStart";
  HMCparams.MD                = MD;

  NoSmearing<TXQCDCompositeImpl> Smearer;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, TxqcdReps> IntegratorT;
  IntegratorT MDynamics(&Grid, MD, Aset, Smearer);

  // ---- initial composite field ----
  TXQCDField U(&Grid);
  TXQCDCompositeImpl::ColdConfiguration(pRNG, U);  // U=I, aux=0
  Smearer.set_Field(U);

  std::vector<HmcObservable<TXQCDField> *> Observables;  // empty: smoke test only
  HybridMonteCarlo<IntegratorT> HMC(HMCparams, MDynamics, sRNG, pRNG,
                                    Observables, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}

#endif
