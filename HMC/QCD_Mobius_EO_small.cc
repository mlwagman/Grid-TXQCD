// QCD-equivalent HMC driver for the TXQCD Möbius DWF Fierz comparison.
// Mirrors HMC/TXQCD_Mobius_EO_small.cc with the TXQCD aux fields removed
// (det M_QCD)^2 instead of (det M_TX)^2 — same Wilson gauge action, same
// Möbius fermion parameters, same MD integrator settings, so trajectories
// from the two drivers form a matched pair for Fierz-identity validation.
//
// At "large" λ (above the EO-Schur tensor-condensation cliff documented in
// the appendix), the EO solver path works for both drivers and a 50-cfg
// Fierz row can be produced with EO-speedup turned on.
//
// Envs: LAMBDA (not used by this QCD driver — set by the matching TXQCD
// driver only), NTRAJ (default 10), NTHERM (default 0), STARTTRAJ (default
// 0), CKPT_PREFIX (default "ckpoint_QCDMobiusEO_lat").

#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourEvenOddRational.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  const char *e;

  const int Ls = 8;
  Coordinate latt4(std::vector<int>{4, 4, 4, 4});
  if ((e = std::getenv("LATT4")) && *e) {
    latt4.resize(0);
    std::stringstream ss(e); std::string tok;
    while (std::getline(ss, tok, '.')) latt4.push_back(std::atoi(tok.c_str()));
  }
  Coordinate simd  = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi   = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridSerialRNG    sRNG;            sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  GridParallelRNG  pRNG4(UGrid);    pRNG4.SeedFixedIntegers({6, 7, 8, 9, 10});

  // ---- physics parameters (match TXQCD_Mobius_EO_small.cc verbatim) ----
  RealD beta   = 5.6;
  RealD mass   = 0.05;
  RealD M5     = 1.8;
  RealD b      = 1.5;
  RealD c      = 0.5;
  std::cout << GridLogMessage << "[scout-qcd-eo] beta=" << beta
            << " m=" << mass << " M5=" << M5 << " b=" << b << " c=" << c
            << " Ls=" << Ls << std::endl;

  // Rational params: same as TXQCD scout for matched HMC comparability.
  OneFlavourRationalParams rat_params(/*lo=*/1e-4, /*hi=*/64.0,
                                      /*maxit=*/20000, /*tol=*/1e-10,
                                      /*degree=*/12, /*precision=*/64,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1e-7,
                                      /*BoundsCheckTol=*/1e-4);

  // ---- gauge action ----
  WilsonGaugeActionR GaugeAction(beta);

  // ---- fermion operator + EO rational PFs (Nf=2-equivalent: two 1-flavor PFs) ----
  LatticeGaugeField Umu(UGrid);
  SU<Nc>::ColdConfiguration(Umu);
  typedef WilsonImplR Impl;
  typedef MobiusFermionD FermionAction;
  FermionAction FermOp(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                       mass, M5, b, c);
  OneFlavourEvenOddRationalPseudoFermionAction<Impl> PF1(FermOp, rat_params);
  OneFlavourEvenOddRationalPseudoFermionAction<Impl> PF2(FermOp, rat_params);

  // ---- action levels: PFs inner (faster), gauge outer (more steps) ----
  typedef Representations<EmptyRep<LatticeGaugeField>> QcdReps;
  ActionLevel<LatticeGaugeField, QcdReps> Level1(1);
  Level1.push_back(&PF1);
  Level1.push_back(&PF2);
  ActionLevel<LatticeGaugeField, QcdReps> Level2(4);
  Level2.push_back(&GaugeAction);

  ActionSet<LatticeGaugeField, QcdReps> Aset;
  Aset.push_back(Level1);
  Aset.push_back(Level2);

  // ---- MD parameters (verbatim match) ----
  IntegratorParameters MD;
  MD.name    = "LeapFrog";
  MD.MDsteps = 80;
  MD.trajL   = 0.5;

  // ---- HMC parameters (envs for trajectory control + checkpointing) ----
  int ntraj  = (e = std::getenv("NTRAJ"))      ? std::atoi(e) : 10;
  int ntherm = (e = std::getenv("NTHERM"))     ? std::atoi(e) : 0;
  int start  = (e = std::getenv("STARTTRAJ"))  ? std::atoi(e) : 0;
  std::string ckpt_prefix =
    (e = std::getenv("CKPT_PREFIX")) ? e : "ckpoint_QCDMobiusEO_lat";
  std::cout << GridLogMessage << "[scout-qcd-eo] ntraj=" << ntraj
            << " ntherm=" << ntherm << " start=" << start
            << " ckpt_prefix=" << ckpt_prefix << std::endl;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory    = start;
  HMCparams.Trajectories       = ntraj;
  HMCparams.NoMetropolisUntil  = ntherm;
  HMCparams.MetropolisTest     = true;
  HMCparams.PerformRandomShift = false;
  HMCparams.StartingType       = (start == 0) ? "ColdStart" : "CheckpointStart";
  HMCparams.MD                 = MD;

  // ---- integrator + checkpointer ----
  typedef LeapFrog<PeriodicGimplR, NoSmearing<PeriodicGimplR>, QcdReps> IntegratorT;
  NoSmearing<PeriodicGimplR> Smearer;
  IntegratorT MDynamics(UGrid, MD, Aset, Smearer);

  CheckpointerParameters CPparams;
  CPparams.config_prefix = ckpt_prefix;
  CPparams.rng_prefix    = ckpt_prefix + "_rng";
  CPparams.saveInterval  = 1;
  CPparams.format        = "IEEE64BIG";
  NerscHmcCheckpointer<PeriodicGimplR> CheckPoint(CPparams);
  std::vector<HmcObservable<LatticeGaugeField> *> Observables;
  Observables.push_back(&CheckPoint);

  Smearer.set_Field(Umu);
  HybridMonteCarlo<IntegratorT> HMC(HMCparams, MDynamics, sRNG, pRNG4,
                                    Observables, Umu);
  HMC.evolve();

  Grid_finalize();
  return 0;
}

#endif
