#include "params.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;
  mkdir_p(qcd_cfg_dir());

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(qcd_cfg_dir() + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(qcd_cfg_dir() + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  LatticeGaugeField Umu(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    std::string cf = qcd_cfg_dir() + "/ckpoint_lat." + std::to_string(latest);
    std::string rf = qcd_cfg_dir() + "/ckpoint_rng." + std::to_string(latest);
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rf);
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cf);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
    pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
    SU<Nc>::TepidConfiguration(pRNG, Umu);
  }

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF FermOp(Umu, Grid, RBGrid, mass_light, csw, csw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, cg_max);
  TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
  Nf2.is_smeared = true;

  WCF StrangeFermOp(Umu, Grid, RBGrid, mass_strange, csw, csw);
  OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                       100, 1e-6, 1e-4);
  OneFlavourRationalPseudoFermionAction<WilsonImplR> StrangePF(StrangeFermOp,
                                                                strange_rat);
  StrangePF.is_smeared = true;

  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  SymanzikR GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  L1.push_back(&Nf2);
  L1.push_back(&StrangePF);
  ActionLevel<LatticeGaugeField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = 10;
  MD.trajL = 0.5;

  int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = total_traj - no_metrop - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear_inv, Stout);

  typedef ForceGradient<PeriodicGimplR,
                        SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(Umu);

  struct Ckpt : public HmcObservable<LatticeGaugeField> {
    std::string cfg_prefix, rng_prefix;
    int interval;
    void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &sR,
                            GridParallelRNG &pR) override {
      if (t % interval != 0) return;
      typedef GaugeStatistics<PeriodicGimplR> GS;
      NerscIO::writeRNGState(sR, pR, rng_prefix + "." + std::to_string(t));
      NerscIO::writeConfiguration<GS>(U, cfg_prefix + "." + std::to_string(t), 0, 1);
    }
  };
  Ckpt ckpt;
  ckpt.cfg_prefix = qcd_cfg_dir() + "/ckpoint_lat";
  ckpt.rng_prefix = qcd_cfg_dir() + "/ckpoint_rng";
  ckpt.interval   = meas_skip;

  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
  HMC.evolve();

  std::cout << GridLogMessage << "QCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
