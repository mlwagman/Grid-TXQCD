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
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  const char *e;
  const int Ls = 8;
  // LATT4="Lx.Ly.Lz.Lt" (default 4.4.4.4)
  std::vector<int> dims = {4, 4, 4, 4};
  if ((e = std::getenv("LATT4")) && *e) {
    dims.clear();
    std::stringstream ss(e); std::string tok;
    while (std::getline(ss, tok, '.')) dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt4(dims);
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
  const char *lam_env = std::getenv("LAMBDA");
  RealD lambda = (lam_env && *lam_env) ? std::atof(lam_env) : 3.0;
  std::cout << GridLogMessage << "[scout-non-eo] lambda=" << lambda << std::endl;
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

  // ---- HMC params (envs) ----
  int ntraj  = (e = std::getenv("NTRAJ"))     ? std::atoi(e) : 10;
  int ntherm = (e = std::getenv("NTHERM"))    ? std::atoi(e) : 0;
  int start  = (e = std::getenv("STARTTRAJ")) ? std::atoi(e) : 0;
  int save_every = (e = std::getenv("MEAS_SKIP")) ? std::atoi(e) : 0;
  std::string ckpt_prefix =
    (e = std::getenv("CKPT_PREFIX")) ? e : "";
  std::cout << GridLogMessage << "[scout-non-eo] ntraj=" << ntraj
            << " ntherm=" << ntherm << " start=" << start
            << " meas_skip=" << save_every
            << " ckpt_prefix=" << (ckpt_prefix.empty() ? "<disabled>" : ckpt_prefix)
            << std::endl;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory    = start;
  HMCparams.Trajectories       = ntraj;
  HMCparams.NoMetropolisUntil  = ntherm;
  HMCparams.MetropolisTest     = true;
  HMCparams.PerformRandomShift = false;
  HMCparams.StartingType       = (start == 0) ? "ColdStart" : "CheckpointStart";
  HMCparams.MD                 = MD;

  NoSmearing<TXQCDCompositeImpl> Smearer;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, TxqcdReps> IntegratorT;
  IntegratorT MDynamics(UGrid, MD, Aset, Smearer);

  TXQCDField U(UGrid);
  TXQCDCompositeImpl::ColdConfiguration(pRNG4, U);
  Smearer.set_Field(U);

  std::vector<HmcObservable<TXQCDField> *> Observables;
  std::unique_ptr<TXQCDCheckpointer> CheckPoint;
  if (!ckpt_prefix.empty() && save_every > 0) {
    CheckpointerParameters CPp;
    CPp.config_prefix = ckpt_prefix;
    CPp.rng_prefix    = ckpt_prefix + "_rng";
    CPp.saveInterval  = save_every;
    CPp.format        = "IEEE64BIG";
    CheckPoint = std::make_unique<TXQCDCheckpointer>(CPp);
    Observables.push_back(CheckPoint.get());
  }
  HybridMonteCarlo<IntegratorT> HMC(HMCparams, MDynamics, sRNG, pRNG4,
                                    Observables, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}

#endif
