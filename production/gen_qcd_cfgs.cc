#include "params.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>

using namespace TXQCDProduction;

struct QcdDiag : public HmcObservable<LatticeGaugeField> {
  struct ActionRef { std::string name; Action<LatticeGaugeField> *action; };

  std::string prefix_;
  int interval_;
  std::vector<ActionRef> actions_;
  SmearedConfiguration<PeriodicGimplR> &smear_;
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG &prng_;

  std::vector<int>    traj_;
  std::vector<RealD>  plaq_, vev_trminv_;
  std::vector<std::vector<RealD>> force_avg_, force_max_, fdt_avg_, fdt_max_;

  QcdDiag(const std::string &prefix, int interval,
          std::vector<ActionRef> actions,
          SmearedConfiguration<PeriodicGimplR> &smear,
          GridCartesian &grid, GridRedBlackCartesian &rbgrid,
          GridParallelRNG &prng)
      : prefix_(prefix), interval_(interval), actions_(std::move(actions)),
        smear_(smear), grid_(grid), rbgrid_(rbgrid), prng_(prng) {}

  void TrajectoryComplete(int traj, LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    traj_.push_back(traj);
    plaq_.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(U));

    int na = (int)actions_.size();
    std::vector<RealD> fa(na), fm(na), fdta(na), fdtm(na);
    for (int i = 0; i < na; ++i) {
      fa[i]   = actions_[i].action->deriv_norm_average();
      fm[i]   = actions_[i].action->deriv_max_average();
      fdta[i] = actions_[i].action->Fdt_norm_average();
      fdtm[i] = actions_[i].action->Fdt_max_average();
    }
    force_avg_.push_back(fa); force_max_.push_back(fm);
    fdt_avg_.push_back(fdta); fdt_max_.push_back(fdtm);

    smear_.set_Field(U);
    LatticeGaugeField Usm = smear_.get_SmearedU();
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    RealD V = (RealD)grid_.gSites();
    RealD acc = 0.0;
    for (int h = 0; h < n_vev_noise; ++h) {
      LatticeFermion eta(&grid_), b(&grid_), x(&grid_);
      gaussian(prng_, eta);
      Dw.Mdag(eta, b);
      x = Zero();
      CG(HermOp, b, x);
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    vev_trminv_.push_back(acc / n_vev_noise);

    if (traj % interval_ == 0) {
      std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
      Hdf5Writer wr(fname);
      write(wr, "traj", traj_);
      write(wr, "plaq", plaq_);
      write(wr, "vev_trminv", vev_trminv_);
      write(wr, "force_avg", force_avg_);
      write(wr, "force_max", force_max_);
      write(wr, "fdt_avg", fdt_avg_);
      write(wr, "fdt_max", fdt_max_);
      std::vector<std::string> names;
      for (auto &a : actions_) names.push_back(a.name);
      write(wr, "action_names", names);
      traj_.clear(); plaq_.clear(); vev_trminv_.clear();
      force_avg_.clear(); force_max_.clear();
      fdt_avg_.clear(); fdt_max_.clear();
      std::cout << GridLogMessage << "Diagnostics written to " << fname << std::endl;
    }
  }
};

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

  // Light quarks (Nf=2): EO-preconditioned LogDet + Schur
  WCF FermOp(Umu, Grid, RBGrid, mass_light, csw, csw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, cg_max);
  QCDLogDetCloverEOAction<WilsonImplR> LightLogDet(FermOp, 2);
  LightLogDet.is_smeared = true;
  TwoFlavourSchurCloverAction<WilsonImplR> LightSchurPF(FermOp, CG, CG);
  LightSchurPF.is_smeared = true;

  // Strange quark (Nf=1): EO-preconditioned LogDet + Schur RHMC
  WCF StrangeFermOp(Umu, Grid, RBGrid, mass_strange, csw, csw);
  OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  StrangeLogDet.is_smeared = true;
  OneFlavourSchurCloverRationalAction<WilsonImplR> StrangeSchurPF(
      StrangeFermOp, strange_rat);
  StrangeSchurPF.is_smeared = true;

  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  SymanzikR GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  L1.push_back(&LightLogDet);
  L1.push_back(&LightSchurPF);
  L1.push_back(&StrangeLogDet);
  L1.push_back(&StrangeSchurPF);
  ActionLevel<LatticeGaugeField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = 10;
  MD.trajL = sqrt(2.0);

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

  QcdDiag diag(qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
      {"LightLogDet", &LightLogDet},
      {"LightSchurPF", &LightSchurPF},
      {"StrangeLogDet", &StrangeLogDet},
      {"StrangeSchurPF", &StrangeSchurPF},
      {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG);

  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
  HMC.evolve();

  std::cout << GridLogMessage << "QCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
