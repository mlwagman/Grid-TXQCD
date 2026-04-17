// Benchmark: TXQCD HMC performance across integrators and preconditioning.
//
// Sections:
//   1. Full-grid LeapFrog 80 steps (baseline)
//   2. EO LeapFrog 80 steps
//   3. EO MinimumNorm2 40 steps
//   4. EO ForceGradient 20 steps, L2×4
//   5. EO ForceGradient 15 steps, L2×4
//   6. EO ForceGradient 10 steps, L2×4
//   7. EO ForceGradient 20 steps, L2×8
//   8. EO ForceGradient 10 steps, L2×8

#include "Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>
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

// Each section is a standalone function so all objects share the same lifetime.

void section_1(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
               OneFlavourRationalParams rat_params) {
  std::cout << GridLogMessage
            << "===== 1. FULL-GRID LeapFrog 80, L2x4 =====" << std::endl;

  GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
  AuxiliaryFieldGaussianAction           AA(lambda);
  TXQCDWilsonRationalPseudoFermionAction PF(Grid, RBGrid, mass, rat_params);

  ActionLevel<TXQCDField, Reps> L0(1);
  L0.push_back(&PF); L0.push_back(&AA);
  ActionLevel<TXQCDField, Reps> L1(4);
  L1.push_back(&GA);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L0); Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "LeapFrog"; MD.MDsteps = 80; MD.trajL = 0.5;
  HMCparameters HMCp = make_hmcp(); HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1,2,3,4,5}); pRNG.SeedFixedIntegers({6,7,8,9,10});
  TXQCDField U(&Grid); TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smear.set_Field(U);

  std::vector<HmcObservable<TXQCDField>*> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

void section_2(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
               OneFlavourRationalParams rat_params) {
  std::cout << GridLogMessage
            << "===== 2. EO LeapFrog 80, L2x4 =====" << std::endl;

  GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
  AuxiliaryFieldGaussianAction           AA(lambda);
  TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass, rat_params);
  TXQCDLogDetEOAction         LD(Grid, RBGrid, mass);

  ActionLevel<TXQCDField, Reps> L0(1);
  L0.push_back(&PF); L0.push_back(&LD); L0.push_back(&AA);
  ActionLevel<TXQCDField, Reps> L1(4);
  L1.push_back(&GA);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L0); Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "LeapFrog"; MD.MDsteps = 80; MD.trajL = 0.5;
  HMCparameters HMCp = make_hmcp(); HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1,2,3,4,5}); pRNG.SeedFixedIntegers({6,7,8,9,10});
  TXQCDField U(&Grid); TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smear.set_Field(U);

  std::vector<HmcObservable<TXQCDField>*> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

void section_3(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
               OneFlavourRationalParams rat_params) {
  std::cout << GridLogMessage
            << "===== 3. EO MN2 40, L2x4 =====" << std::endl;

  GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
  AuxiliaryFieldGaussianAction           AA(lambda);
  TXQCDWilsonRationalEOAction PF(Grid, RBGrid, mass, rat_params);
  TXQCDLogDetEOAction         LD(Grid, RBGrid, mass);

  ActionLevel<TXQCDField, Reps> L0(1);
  L0.push_back(&PF); L0.push_back(&LD); L0.push_back(&AA);
  ActionLevel<TXQCDField, Reps> L1(4);
  L1.push_back(&GA);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L0); Aset.push_back(L1);

  IntegratorParameters MD;
  MD.name = "MinimumNorm2"; MD.MDsteps = 40; MD.trajL = 0.5;
  HMCparameters HMCp = make_hmcp(); HMCp.MD = MD;

  NoSmearing<TXQCDCompositeImpl> Smear;
  typedef MinimumNorm2<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1,2,3,4,5}); pRNG.SeedFixedIntegers({6,7,8,9,10});
  TXQCDField U(&Grid); TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smear.set_Field(U);

  std::vector<HmcObservable<TXQCDField>*> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

// EO ForceGradient helper
void section_fg(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
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
  typedef ForceGradient<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, Reps> IntT;
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

  section_1(Grid, RBGrid, rat_params);
  section_2(Grid, RBGrid, rat_params);
  section_3(Grid, RBGrid, rat_params);
  section_fg(Grid, RBGrid, rat_params, "4. EO FG 20, L2x4", 20, 4);
  section_fg(Grid, RBGrid, rat_params, "5. EO FG 15, L2x4", 15, 4);
  section_fg(Grid, RBGrid, rat_params, "6. EO FG 10, L2x4", 10, 4);
  section_fg(Grid, RBGrid, rat_params, "7. EO FG 20, L2x8", 20, 8);
  section_fg(Grid, RBGrid, rat_params, "8. EO FG 10, L2x8", 10, 8);

  std::cout << GridLogMessage << "Benchmark complete." << std::endl;
  Grid_finalize();
  return 0;
}
