#include "params.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverHasenbuschAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>

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
    RealD pl = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
    plaq_.push_back(pl);

    RealD V = (RealD)U.Grid()->gSites();
    RealD vs = TensorRemove(sum(trace(U.sigma))).real() / V;
    RealD vc = TensorRemove(sum(trace(U.s))).real() / V;
    vev_sigma_.push_back(vs);
    vev_s_.push_back(vc);
    std::cout << GridLogMessage << "[TxqcdDiag] traj=" << traj << " plaq=" << pl
              << " vev_sigma=" << vs << " vev_s=" << vc << std::endl;

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
    // Antiperiodic time BC to match chroma <boundary>1 1 1 -1</boundary>.
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
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

  // LAMBDA env var override for per-stream lambda scan (e.g. parallel chains
  // at lambda=1.5, 2.5, 2.75, 3.0, ...).  Each lambda is its own independent
  // HMC chain with its own cfg dir; RNG seeds are derived from lambda so the
  // chains decorrelate.
  RealD lambda_runtime = lambda;
  if (const char *s = std::getenv("LAMBDA"); s && *s) lambda_runtime = std::atof(s);
  std::ostringstream tag_ss;
  tag_ss << std::fixed << std::setprecision(4) << lambda_runtime;
  std::string cfg_dir = "cfgs/txqcd_lam" + tag_ss.str();
  // Optional suffix so Hasenbusch / tuning variants don't clobber the main
  // lambda-scan streams.  e.g. SUFFIX=_hasen0p10 → cfgs/txqcd_lam6.6000_hasen0p10.
  if (const char *sfx = std::getenv("SUFFIX"); sfx && *sfx) cfg_dir += sfx;
  int seed_offset = (int)std::round(lambda_runtime * 1000);
  std::cout << GridLogMessage << "LAMBDA=" << lambda_runtime
            << " cfg_dir=" << cfg_dir
            << " seed_offset=" << seed_offset << std::endl;

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int total_traj = n_therm + n_prod;
  if (const char *nt = std::getenv("N_TRAJ"); nt && *nt) total_traj = std::atoi(nt);
  std::cout << GridLogMessage << "total_traj=" << total_traj << std::endl;
  mkdir_p(cfg_dir);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(cfg_dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(cfg_dir + "/ckpoint_lat_aux." + std::to_string(t)) &&
        file_exists(cfg_dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  // 10 poles on the rational (chroma ref uses 10-12), MD tol 1e-6.
  OneFlavourRationalParams rat_params(1e-4, 200.0, cg_max, cg_tol, 10, 64,
                                      100, 1e-6, 1e-4);

  // Grid's SymanzikGaugeAction uses RBC convention — NOT chroma's.
  // Chroma's LW_TREE_GAUGEACT literally sets c0=β, c1=−β/(20·u0²).
  // Match chroma's XML-β convention for reference-ensemble compatibility.
  typedef PlaqPlusRectangleAction<PeriodicGimplR> PlaqRectR;
  GaugeActionAdapter<PlaqRectR> GaugeAction(beta, -beta / (20.0 * u0 * u0));
  // Chroma's LW_TREE_GAUGEACT operates on the THIN (unsmeared) gauge links;
  // stout only wraps the fermion via STOUT_FERM_STATE.  Set is_smeared=false.
  GaugeAction.is_smeared = false;

  AuxiliaryFieldGaussianAction AuxAction(lambda_runtime);

  // Hasenbusch mass preconditioning (HASEN_DM env var).  When HASEN_DM > 0,
  // split |det M_light| into |det M_heavy| * |det M_light / M_heavy| with
  // mass_heavy = mass_light + HASEN_DM.  Heavy mass → better-conditioned CG,
  // ratio force suppressed by ~Δm.  HASEN_DM=0 (default) keeps the single
  // rational at mass_light.  Typical tuning: start with Δm ≈ 0.05-0.10 for
  // Wilson-Clover with mass_light = -0.245.
  double hasen_dm = 0.0;
  if (const char *hd = std::getenv("HASEN_DM"); hd && *hd) hasen_dm = std::atof(hd);
  const RealD mass_heavy = mass_light + hasen_dm;
  std::cout << GridLogMessage << "HASEN_DM=" << hasen_dm
            << "  mass_light=" << mass_light
            << "  mass_heavy=" << mass_heavy << std::endl;

  // Full rational at mass_light (used when HASEN_DM == 0).
  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass_light, rat_params, csw);
  PF.is_smeared = true;

  // Hasenbusch split pair (used when HASEN_DM > 0).
  TXQCDWilsonCloverRationalEOAction PF_heavy(Grid, RBGrid, mass_heavy,
                                              rat_params, csw);
  PF_heavy.is_smeared = true;
  TXQCDWilsonCloverHasenbuschAction PF_ratio(Grid, RBGrid,
                                              mass_light, mass_heavy,
                                              rat_params, csw);
  PF_ratio.is_smeared = true;

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_light, csw);
  LogDet.is_smeared = true;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", latest);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({1 + seed_offset, 2 + seed_offset, 3 + seed_offset,
                            4 + seed_offset, 5 + seed_offset});
    pRNG.SeedFixedIntegers({6 + seed_offset, 7 + seed_offset, 8 + seed_offset,
                            9 + seed_offset, 10 + seed_offset});
    // START_TYPE controls the initial composite field.  Default "thermal":
    // weak-field gauge (scale=WEAK_FIELD_SCALE, default 0.1 = chroma's WEAK_FIELD)
    // + aux drawn from the Gaussian action's equilibrium (variance 1/lambda^2).
    std::string start_type = "thermal";
    if (const char *st = std::getenv("START_TYPE"); st && *st) start_type = st;
    double wf_scale = 0.1;
    if (const char *ws = std::getenv("WEAK_FIELD_SCALE"); ws && *ws) wf_scale = std::atof(ws);
    // Aux-field equilibrium offset is parameterized by Σ ≡ vev_trminv/2 =
    // Tr[M⁻¹]/(4V) on the stout-smeared weak-field gauge (per the TXQCD aux
    // init convention memory).  Equilibrium relations:
    //   <σ_aa>  = Σ / λ²
    //   <s_ii>  = (N_f/(√2·N_c)) · Σ / λ²
    //
    // Three input modes (in priority order):
    //   AUX_INIT=value  → set Σ explicitly to this number
    //   AUX_SIGMA_L=v   → legacy env var; Σ = −AUX_SIGMA_L/2  (since old
    //                     code used Σ_l_old = −2·vev_trminv)
    //   AUX_INIT_AUTO=1 (default if neither is set) → measure vev_trminv on
    //                     the just-generated weak-field gauge with stout
    //                     smearing + antiperiodic time BC, set Σ = vev_trminv/2.
    RealD Sigma = 0.0;
    bool auto_measure = false;
    if (const char *si = std::getenv("AUX_INIT"); si && *si) {
      Sigma = std::atof(si);
    } else if (const char *sl = std::getenv("AUX_SIGMA_L"); sl && *sl) {
      Sigma = -std::atof(sl) / 2.0;
    } else {
      auto_measure = true;
    }
    std::cout << GridLogMessage << "START_TYPE=" << start_type
              << "  WEAK_FIELD_SCALE=" << wf_scale
              << (auto_measure ? "  AUX_INIT_AUTO=1"
                               : "  AUX_INIT=" + std::to_string(Sigma))
              << std::endl;
    if (start_type == "hot") {
      TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    } else if (start_type == "cold") {
      TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
    } else if (start_type == "tepid") {
      TXQCDCompositeImpl::TepidConfiguration(pRNG, U);
    } else {
      // Step 1: weak-field gauge.
      TXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, wf_scale);
      // Step 2: optionally auto-measure Σ on the just-generated gauge with
      // production stout-smearing and antiperiodic time BC, matching the
      // M_ee operator that the TXQCD HMC will use.
      if (auto_measure) {
        Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
        SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv,
                                                        Stout);
        SmearMeas.set_Field(U.U);
        LatticeGaugeField Usm = SmearMeas.get_SmearedU();
        WilsonImplParams impl_p;
        impl_p.boundary_phases.resize(Nd, 1.0);
        impl_p.boundary_phases[Nd - 1] = -1.0;
        typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>>
            MeasFermOp;
        MeasFermOp Dw(Usm, Grid, RBGrid, mass_light, csw, csw,
                      WilsonAnisotropyCoefficients(), impl_p);
        MdagMLinearOperator<MeasFermOp, LatticeFermion> HermOp(Dw);
        ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
        RealD V = (RealD)Grid.gSites();
        RealD acc = 0.0;
        // Use a SEPARATE pRNG for the noise vectors so the main pRNG state
        // (which feeds aux-field generation in step 3) remains the same as
        // it would be in a non-AUX_INIT_AUTO run with the same seed.
        GridParallelRNG noisePRNG(&Grid);
        noisePRNG.SeedFixedIntegers(
            {seed_offset + 11, seed_offset + 12, seed_offset + 13,
             seed_offset + 14, seed_offset + 15});
        for (int h = 0; h < n_vev_noise; ++h) {
          LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
          gaussian(noisePRNG, eta);
          Dw.Mdag(eta, b);
          x = Zero();
          CG(HermOp, b, x);
          acc += innerProduct(eta, x).real() / (2.0 * V);
        }
        RealD vev_trminv = acc / n_vev_noise;
        Sigma = vev_trminv / 2.0;
        std::cout << GridLogMessage
                  << "[AUX_INIT_AUTO] vev_trminv=" << vev_trminv
                  << "  Σ=" << Sigma
                  << "  → <σ_aa>=" << Sigma / (lambda_runtime * lambda_runtime)
                  << "  <s_ii>="
                  << (TxqcdNf * Sigma) /
                         (std::sqrt(2.0) * Nc * lambda_runtime * lambda_runtime)
                  << std::endl;
      }
      // Step 3: fill aux fields (σ, π, s, p, t) using the measured Σ.
      TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime, Sigma);
    }
  }

  // Strange quark (Nf=1): EO-preconditioned LogDet + Schur RHMC, wrapped for TXQCD HMC.
  // Uses the mixed-precision rational action (matches gen_qcd_cfgs.cc): MD force
  // runs ConjugateGradientMultiShiftMixedPrec with reliable updates, refresh and
  // S keep full-DP multishift CG.  Roughly 2× faster than full-DP on the strange
  // force eval, which dominates the per-traj cost outside the TXQCD light deriv.
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>> WCF_f;
  // Antiperiodic time BC to match chroma <boundary>1 1 1 -1</boundary>.
  WilsonImplParams strange_impl_p;
  strange_impl_p.boundary_phases.resize(Nd, 1.0);
  strange_impl_p.boundary_phases[Nd - 1] = -1.0;
  WilsonImplParams strange_impl_pF;
  strange_impl_pF.boundary_phases.resize(Nd, 1.0);
  strange_impl_pF.boundary_phases[Nd - 1] = -1.0;

  // Single-precision sibling grids + gauge field for the MP CG.
  GridCartesian        StrangeGridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian StrangeRBGridF(&StrangeGridF);
  LatticeGaugeFieldF StrangeUmuF(&StrangeGridF);
  {
    LatticeColourMatrix  U_d(&Grid);
    LatticeColourMatrixF U_f(&StrangeGridF);
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(U.U, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(StrangeUmuF, U_f, mu);
    }
  }
  WCF_f StrangeFermOpF(StrangeUmuF, StrangeGridF, StrangeRBGridF, mass_strange,
                       csw, csw, WilsonAnisotropyCoefficients(), strange_impl_pF);

  WCF StrangeFermOp(U.U, Grid, RBGrid, mass_strange, csw, csw,
                    WilsonAnisotropyCoefficients(), strange_impl_p);
  // Chroma-matched bounds for the rat_3strange monomial: lo=1e-4, hi=32,
  // force degree=13.  See gen_qcd_cfgs.cc note for details.
  OneFlavourRationalParams strange_rat(1e-4, 32.0, cg_max, cg_tol, 13, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  QCDActionAdapter StrangeLogDetAdapter(StrangeLogDet);
  StrangeLogDetAdapter.is_smeared = true;
  // MP rational: deriv() uses ConjugateGradientMultiShiftMixedPrec.
  OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF> StrangeSchurPF(
      StrangeFermOp, StrangeFermOpF, &StrangeRBGridF, strange_rat, 50);
  QCDActionAdapter StrangeSchurAdapter(StrangeSchurPF);
  StrangeSchurAdapter.is_smeared = true;

  // Nested levels:
  //   L1 (outer, MDsteps):  fermion actions (expensive CG, large coarse dt).
  //   L2 (x GAUGE_MULT):    gauge action (cheap, Fdt ~ 0.1 already fine at x4).
  //   L3 (x AUX_MULT):      aux Gaussian action.  Oscillator freq = lambda,
  //                         restoring force ~ lambda^2 * aux.  Force eval is
  //                         a few flops per site so the substep count is
  //                         essentially free computationally, and finer dt
  //                         here costs very little.  Default 16 gives
  //                         MDsteps*4*16 = 448 aux substeps at MDsteps=7.
  int gauge_mult = 4;
  int aux_mult   = 16;
  if (const char *gm = std::getenv("GAUGE_MULT"); gm && *gm) gauge_mult = std::atoi(gm);
  if (const char *am = std::getenv("AUX_MULT"); am && *am)   aux_mult   = std::atoi(am);
  std::cout << GridLogMessage << "GAUGE_MULT=" << gauge_mult
            << "  AUX_MULT=" << aux_mult << std::endl;
  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  if (hasen_dm > 0.0) {
    L1.push_back(&PF_heavy);
    L1.push_back(&PF_ratio);
  } else {
    L1.push_back(&PF);
  }
  L1.push_back(&LogDet);
  L1.push_back(&StrangeLogDetAdapter);
  L1.push_back(&StrangeSchurAdapter);
  ActionLevel<TXQCDField, Reps> L2(gauge_mult);
  L2.push_back(&GaugeAction);
  ActionLevel<TXQCDField, Reps> L3(aux_mult);
  L3.push_back(&AuxAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);
  Aset.push_back(L3);

  // INTEGRATOR env var: "MinimumNorm2" (default) or "ForceGradient".
  // ForceGradient is what the passing small-lattice Fierz test uses; finer
  // effective step size for given MDsteps thanks to 4th-order-with-gradient
  // structure.
  std::string integrator_name = "MinimumNorm2";
  if (const char *env = std::getenv("INTEGRATOR"); env && *env)
    integrator_name = env;

  IntegratorParameters MD;
  MD.name = integrator_name;
  MD.MDsteps = 7;
  if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) MD.MDsteps = std::atoi(ms);
  MD.trajL = sqrt(2.0);
  if (const char *tl = std::getenv("TRAJL"); tl && *tl) MD.trajL = std::atof(tl);
  std::cout << GridLogMessage << "INTEGRATOR=" << integrator_name
            << "  MDsteps=" << MD.MDsteps
            << "  trajL=" << MD.trajL << std::endl;

  // NoMetropolisUntil: default 10 - start_traj ≥ 0; override via NO_METROP env
  // var (0 = chroma-style Metropolis-from-trajectory-0).
  int no_metrop = std::max(0, 10 - start_traj);
  if (const char *nm = std::getenv("NO_METROP"); nm && *nm) no_metrop = std::atoi(nm);
  std::cout << GridLogMessage << "NoMetropolisUntil=" << no_metrop << std::endl;
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
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  std::vector<TxqcdDiag::ActionRef> diag_actions;
  if (hasen_dm > 0.0) {
    diag_actions.push_back({"PseudoFermionHeavy", &PF_heavy});
    diag_actions.push_back({"PseudoFermionRatio", &PF_ratio});
  } else {
    diag_actions.push_back({"PseudoFermion", &PF});
  }
  diag_actions.push_back({"LogDet", &LogDet});
  diag_actions.push_back({"AuxGaussian", &AuxAction});
  diag_actions.push_back({"StrangeLogDet", &StrangeLogDetAdapter});
  diag_actions.push_back({"StrangeSchurPF", &StrangeSchurAdapter});
  diag_actions.push_back({"Gauge", &GaugeAction});
  TxqcdDiag diag(cfg_dir + "/hmc_diagnostics", meas_skip, diag_actions,
                 Smear, Grid, RBGrid, pRNG);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};

  // Branch on integrator choice.  HybridMonteCarlo is templated on the
  // integrator type, so both code paths are compiled and we pick at runtime.
  if (integrator_name == "ForceGradient") {
    typedef ForceGradient<TXQCDCompositeImpl,
                          TXQCDSmearedConfiguration, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  } else {
    typedef MinimumNorm2<TXQCDCompositeImpl,
                         TXQCDSmearedConfiguration, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "TXQCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
