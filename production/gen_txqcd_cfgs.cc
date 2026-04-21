#include "params.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>

using namespace TXQCDProduction;

struct TxqcdDiag : public HmcObservable<TXQCDField> {
  struct ActionRef { std::string name; Action<TXQCDField> *action; };

  std::string prefix_;
  int interval_;
  std::vector<ActionRef> actions_;
  TXQCDSmearedConfiguration &smear_;
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG &prng_;

  std::vector<int>    traj_;
  std::vector<RealD>  plaq_, vev_sigma_, vev_s_, vev_trminv_;
  std::vector<std::vector<RealD>> force_avg_, force_max_, fdt_avg_, fdt_max_;

  TxqcdDiag(const std::string &prefix, int interval,
            std::vector<ActionRef> actions,
            TXQCDSmearedConfiguration &smear,
            GridCartesian &grid, GridRedBlackCartesian &rbgrid,
            GridParallelRNG &prng)
      : prefix_(prefix), interval_(interval), actions_(std::move(actions)),
        smear_(smear), grid_(grid), rbgrid_(rbgrid), prng_(prng) {}

  void TrajectoryComplete(int traj, TXQCDField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    traj_.push_back(traj);
    plaq_.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U));

    RealD V = (RealD)U.Grid()->gSites();
    vev_sigma_.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
    vev_s_.push_back(TensorRemove(sum(trace(U.s))).real() / V);

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
    LatticeGaugeField Usm = smear_.get_SmearedU().U;
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
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
      write(wr, "vev_sigma", vev_sigma_);
      write(wr, "vev_s", vev_s_);
      write(wr, "vev_trminv", vev_trminv_);
      write(wr, "force_avg", force_avg_);
      write(wr, "force_max", force_max_);
      write(wr, "fdt_avg", fdt_avg_);
      write(wr, "fdt_max", fdt_max_);
      std::vector<std::string> names;
      for (auto &a : actions_) names.push_back(a.name);
      write(wr, "action_names", names);
      traj_.clear(); plaq_.clear();
      vev_sigma_.clear(); vev_s_.clear(); vev_trminv_.clear();
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
  mkdir_p(txqcd_cfg_dir());

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(txqcd_cfg_dir() + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(txqcd_cfg_dir() + "/ckpoint_lat_aux." + std::to_string(t)) &&
        file_exists(txqcd_cfg_dir() + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  OneFlavourRationalParams rat_params(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                      100, 1e-6, 1e-4);

  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  GaugeActionAdapter<SymanzikR> GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  AuxiliaryFieldGaussianAction AuxAction(lambda);

  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass_light, rat_params, csw);
  PF.is_smeared = true;

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_light, csw);
  LogDet.is_smeared = true;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  txqcd_cfg_dir() + "/ckpoint_lat",
                                  txqcd_cfg_dir() + "/ckpoint_rng", latest);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
  }

  // Strange quark (Nf=1): EO-preconditioned LogDet + Schur RHMC, wrapped for TXQCD HMC
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF StrangeFermOp(U.U, Grid, RBGrid, mass_strange, csw, csw);
  OneFlavourRationalParams strange_rat(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  QCDActionAdapter StrangeLogDetAdapter(StrangeLogDet);
  StrangeLogDetAdapter.is_smeared = true;
  OneFlavourSchurCloverRationalAction<WilsonImplR> StrangeSchurPF(
      StrangeFermOp, strange_rat);
  QCDActionAdapter StrangeSchurAdapter(StrangeSchurPF);
  StrangeSchurAdapter.is_smeared = true;

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  L1.push_back(&StrangeLogDetAdapter);
  L1.push_back(&StrangeSchurAdapter);
  ActionLevel<TXQCDField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<TXQCDField, Reps> Aset;
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
  TXQCDSmearedConfiguration Smear(&Grid, stout_nsmear_inv, Stout);

  typedef ForceGradient<TXQCDCompositeImpl,
                        TXQCDSmearedConfiguration, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = txqcd_cfg_dir() + "/ckpoint_lat";
  CPp.rng_prefix    = txqcd_cfg_dir() + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  TxqcdDiag diag(txqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
      {"PseudoFermion", &PF},
      {"LogDet", &LogDet},
      {"AuxGaussian", &AuxAction},
      {"StrangeLogDet", &StrangeLogDetAdapter},
      {"StrangeSchurPF", &StrangeSchurAdapter},
      {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage << "TXQCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
