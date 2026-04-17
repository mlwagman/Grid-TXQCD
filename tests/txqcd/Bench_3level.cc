// Benchmark: 3-level integrator with separate aux-field timescale.
//
// 2-level (current):  L0(fermion+logdet+aux, ×1) | L1(gauge, ×N)
// 3-level (new):      L0(fermion+logdet, ×1) | L1(gauge, ×G) | L2(aux, ×A)
//
// Aux on the innermost (fastest) level because its force is essentially free.

#include "Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetEOAction.h>

using namespace TxqcdTest2pt;

typedef Representations<EmptyRep<TXQCDField>> Reps;

static const int bench_traj = 5;

static HMCparameters make_hmcp() {
  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  HMCp.Trajectories        = bench_traj;
  HMCp.NoMetropolisUntil   = bench_traj;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  return HMCp;
}

// 2-level: ferm+aux(×1) | gauge(×G)
void bench_2level(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                  OneFlavourRationalParams rat_params,
                  const std::string &label, int md_steps, int gauge_mult) {
  std::cout << GridLogMessage << "===== " << label << " =====" << std::endl;

  GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
  AuxiliaryFieldGaussianAction           AA(lambda);
  TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass, rat_params);
  TXQCDLogDetEOAction         LD(Grid, RBGrid, mass);

  ActionLevel<TXQCDField, Reps> L0(1);
  L0.push_back(&PF); L0.push_back(&LD); L0.push_back(&AA);
  ActionLevel<TXQCDField, Reps> L1(gauge_mult);
  L1.push_back(&GA);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L0); Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "ForceGradient"; MD.MDsteps = md_steps; MD.trajL = 0.5;
  HMCparameters HMCp = make_hmcp(); HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef ForceGradient<TXQCDCompositeImpl,
                        NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1,2,3,4,5}); pRNG.SeedFixedIntegers({6,7,8,9,10});
  TXQCDField U(&Grid); TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smear.set_Field(U);

  std::vector<HmcObservable<TXQCDField>*> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

// 3-level: ferm(×1) | gauge(×G) | aux(×A)
void bench_3level(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                  OneFlavourRationalParams rat_params,
                  const std::string &label, int md_steps,
                  int gauge_mult, int aux_mult) {
  std::cout << GridLogMessage << "===== " << label << " =====" << std::endl;

  GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
  AuxiliaryFieldGaussianAction           AA(lambda);
  TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass, rat_params);
  TXQCDLogDetEOAction         LD(Grid, RBGrid, mass);

  ActionLevel<TXQCDField, Reps> L0(1);
  L0.push_back(&PF); L0.push_back(&LD);
  ActionLevel<TXQCDField, Reps> L1(gauge_mult);
  L1.push_back(&GA);
  ActionLevel<TXQCDField, Reps> L2(aux_mult);
  L2.push_back(&AA);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L0); Aset.push_back(L1); Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient"; MD.MDsteps = md_steps; MD.trajL = 0.5;
  HMCparameters HMCp = make_hmcp(); HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef ForceGradient<TXQCDCompositeImpl,
                        NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1,2,3,4,5}); pRNG.SeedFixedIntegers({6,7,8,9,10});
  TXQCDField U(&Grid); TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smear.set_Field(U);

  std::vector<HmcObservable<TXQCDField>*> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  RealD cg_tol = 1e-8;
  OneFlavourRationalParams rat_params(1e-4, 64.0, cg_max, cg_tol, 12, 64,
                                      100, 1e-6, 1e-4);

  // Section 1: 2-level reference
  bench_2level(Grid, RBGrid, rat_params,
               "1. 2-lvl FG-10: ferm+aux(x1) | gauge(x4)", 10, 4);

  // Section 2-6: 3-level with aux on innermost
  bench_3level(Grid, RBGrid, rat_params,
               "2. 3-lvl FG-10: ferm(x1) | gauge(x2) | aux(x2)", 10, 2, 2);
  bench_3level(Grid, RBGrid, rat_params,
               "3. 3-lvl FG-10: ferm(x1) | gauge(x1) | aux(x4)", 10, 1, 4);
  bench_3level(Grid, RBGrid, rat_params,
               "4. 3-lvl FG-10: ferm(x1) | gauge(x4) | aux(x2)", 10, 4, 2);
  bench_3level(Grid, RBGrid, rat_params,
               "5. 3-lvl FG-10: ferm(x1) | gauge(x2) | aux(x4)", 10, 2, 4);
  bench_3level(Grid, RBGrid, rat_params,
               "6. 3-lvl FG-10: ferm(x1) | gauge(x4) | aux(x4)", 10, 4, 4);

  // Section 7-8: 2-level references
  bench_2level(Grid, RBGrid, rat_params,
               "7. 2-lvl FG-10: ferm+aux(x1) | gauge(x8)", 10, 8);
  bench_2level(Grid, RBGrid, rat_params,
               "8. 2-lvl FG-20: ferm+aux(x1) | gauge(x4) [ref]", 20, 4);

  std::cout << GridLogMessage << "Benchmark complete." << std::endl;
  Grid_finalize();
  return 0;
}
