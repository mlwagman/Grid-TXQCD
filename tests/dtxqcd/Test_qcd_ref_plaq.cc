// Test_qcd_ref_plaq:
//
// Pure QCD reference HMC for plaquette comparison vs Test_dtxqcd_fierz_full_qcd.
// Wilson gauge + Nf=2 Wilson fermion RHMC at the same physical params, no aux.

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  RealD mass_run  = 0.3;
  RealD beta_run  = 6.0;
  int mdsteps     = 20;
  RealD trajL     = std::sqrt(2.0);
  int n_therm     = 100;
  int n_prod      = 500;

  if (const char *v = std::getenv("MASS");    v && *v) mass_run = std::atof(v);
  if (const char *v = std::getenv("BETA");    v && *v) beta_run = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS"); v && *v) mdsteps  = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");   v && *v) trajL    = std::atof(v);
  if (const char *v = std::getenv("N_THERM"); v && *v) n_therm  = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");  v && *v) n_prod   = std::atoi(v);

  Coordinate latt = GridDefaultLatt();
  if (latt.size() == 0) latt = Coordinate(std::vector<int>{4,4,4,4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  std::cout << GridLogMessage
            << "QCD REF HMC: mass=" << mass_run << " beta=" << beta_run
            << " lattice=" << latt[0] << "." << latt[1] << "." << latt[2]
            << "." << latt[3] << "  N_THERM=" << n_therm
            << " N_PROD=" << n_prod << " MDSTEPS=" << mdsteps
            << " trajL=" << trajL << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({101, 102, 103, 104, 105});
  pRNG.SeedFixedIntegers({201, 202, 203, 204, 205});

  // APBC time, matches DTXQCD default
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;

  WilsonGaugeActionR GaugeAction(beta_run);

  // Nf=2 RHMC pseudofermion on plain Wilson Dw using OneFlavour rational^2.
  // For Nf=2: weight ∝ |det Dw|² = (Dw^†Dw)^1.  Single-flavour rational with
  // power=-1/2 gives |det Dw|; we need two of them for Nf=2.
  LatticeGaugeField U(&Grid);
  SU<Nc>::ColdConfiguration(U);
  WilsonFermion<WilsonImplR> Dw(U, Grid, RBGrid, mass_run, ip);

  OneFlavourRationalParams rp(/*lo=*/0.05, /*hi=*/200.0, /*MaxIter=*/10000,
                              /*tol=*/1e-8, /*degree=*/12, /*precision=*/64,
                              /*BoundsCheckFreq=*/100, /*mdtol=*/1e-6);
  OneFlavourRationalPseudoFermionAction<WilsonImplR> PF1(Dw, rp);
  OneFlavourRationalPseudoFermionAction<WilsonImplR> PF2(Dw, rp);

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  L1.push_back(&PF1);
  L1.push_back(&PF2);
  ActionLevel<LatticeGaugeField, Reps> L2(2);
  L2.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name    = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  HMCparameters HMCp;
  HMCp.StartTrajectory    = 0;
  HMCp.Trajectories       = n_prod;
  HMCp.NoMetropolisUntil  = n_therm;
  HMCp.MetropolisTest     = true;
  HMCp.PerformRandomShift = false;
  HMCp.StartingType       = "ColdStart";
  HMCp.MD = MD;

  NoSmearing<PeriodicGimplR> Smear;
  Smear.set_Field(U);
  typedef ForceGradient<PeriodicGimplR, NoSmearing<PeriodicGimplR>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);

  std::vector<HmcObservable<LatticeGaugeField> *> Obs;
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  RealD plaq_final = WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
  std::cout << GridLogMessage
            << "[Test_qcd_ref_plaq] final plaq = " << plaq_final << std::endl;

  Grid_finalize();
  return 0;
}
