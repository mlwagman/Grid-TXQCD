#include "params.h"
#include <cstring>
#include <cstdio>
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CompactWilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>
#include <Grid/algorithms/iterative/ConjugateGradientMixedPrec.h>
#include <Grid/algorithms/iterative/ConjugateGradientMultiShiftMixedPrec.h>

using namespace TXQCDProduction;

// Mixed-precision CG wrapper that satisfies the OperatorFunction<FieldD>
// interface expected by TwoFlavourSchurCloverAction as its derivative solver.
// Assumes the single-precision Schur operator has already been updated (by
// TwoFlavourSchurCloverActionMP::deriv() below) to match the current gauge.
namespace Grid {
template <class FieldD, class FieldF, class SchurOpD, class SchurOpF>
class MixedPrecCGWrapper : public OperatorFunction<FieldD> {
 public:
  using OperatorFunction<FieldD>::operator();

  MixedPrecCGWrapper(RealD tol, int max_inner, int max_outer,
                     GridBase *rbgrid_f,
                     SchurOpD &schur_d, SchurOpF &schur_f)
      : tol_(tol), max_inner_(max_inner), max_outer_(max_outer),
        rbgrid_f_(rbgrid_f), schur_d_(schur_d), schur_f_(schur_f) {}

  void operator()(LinearOperatorBase<FieldD> &LinOp_unused,
                  const FieldD &src, FieldD &sol) override {
    MixedPrecisionConjugateGradient<FieldD, FieldF> MPCG(
        tol_, max_inner_, max_outer_, rbgrid_f_, schur_f_, schur_d_);
    MPCG(src, sol);
  }

 private:
  RealD tol_;
  int max_inner_, max_outer_;
  GridBase *rbgrid_f_;
  SchurOpD &schur_d_;
  SchurOpF &schur_f_;
};

// Subclass of TwoFlavourSchurCloverAction that keeps the SP operator in sync
// with the raw gauge field before each deriv().  Passes the CG work to the
// mixed-precision wrapper installed as the DerivativeSolver.
template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class TwoFlavourSchurCloverActionMP
    : public TwoFlavourSchurCloverAction<ImplD, FermOpD_> {
 public:
  typedef TwoFlavourSchurCloverAction<ImplD, FermOpD_> Base;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  TwoFlavourSchurCloverActionMP(typename Base::FermionOperator &opD,
                                FermOpF &opF,
                                OperatorFunction<typename Base::FermionField> &DS,
                                OperatorFunction<typename Base::FermionField> &AS)
      : Base(opD, DS, AS), opF_(opF) {}

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    // Refresh SP operator from the raw gauge field U (per-mu precisionChange).
    typename ImplF::GaugeField UmuF(opF_.GaugeGrid());
    typename ImplD::GaugeLinkField U_d(U.Grid());
    typename ImplF::GaugeLinkField U_f(opF_.GaugeGrid());
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(U, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
    opF_.ImportGauge(UmuF);
    // Delegate to base; base's DerivativeSolver is MixedPrecCGWrapper which
    // will invoke MPCG against the now-synced SP Schur operator.
    Base::deriv(U, dSdU);
  }

 private:
  FermOpF &opF_;
};
}  // namespace Grid

// OneFlavourSchurCloverRationalActionMP — mixed-precision rational action
// shared with gen_txqcd_cfgs.cc.
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>

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
    RealD pl = WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
    plaq_.push_back(pl);
    std::cout << GridLogMessage << "[QcdDiag] traj=" << traj << " plaq=" << pl
              << std::endl;

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
    // Antiperiodic BC in time (direction Nd-1=3) to match chroma's
    // <boundary>1 1 1 -1</boundary>.  Grid's default is all-periodic.
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
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

  // Multi-stream support: if the STREAM_ID env var is set (0, 1, 2, ...),
  // write to a stream-indexed cfg dir and offset RNG seeds so each stream is
  // an independent Markov chain.  STREAM_ID unset => default behaviour
  // (cfgs/qcd, original seed) for backwards compatibility.
  int stream_id = -1;
  if (const char *sid = std::getenv("STREAM_ID"); sid && *sid) stream_id = std::atoi(sid);
  std::string cfg_dir = (stream_id < 0)
      ? qcd_cfg_dir()
      : (qcd_cfg_dir() + "_s" + std::to_string(stream_id));
  // Optional suffix so action-variant / tuning variants don't clobber the
  // main stream dirs (e.g. SUFFIX=_lwfix → cfgs/qcd_s0_lwfix).
  if (const char *sfx = std::getenv("SUFFIX"); sfx && *sfx) cfg_dir += sfx;
  std::cout << GridLogMessage << "STREAM_ID=" << stream_id
            << " cfg_dir=" << cfg_dir << std::endl;

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
        file_exists(cfg_dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }

  LatticeGaugeField Umu(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    std::string cf = cfg_dir + "/ckpoint_lat." + std::to_string(latest);
    std::string rf = cfg_dir + "/ckpoint_rng." + std::to_string(latest);
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rf);
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cf);
    start_traj = latest;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    // Import an external thermalized config (chroma LIME or NERSC) as the
    // starting gauge field — skips tepid-start thermalization entirely.
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    // Auto-detect NERSC vs LIME by magic bytes (same logic as compute_plaq).
    FILE *f = std::fopen(ic, "rb");
    char magic[16] = {0};
    if (f) { std::fread(magic, 1, sizeof(magic), f); std::fclose(f); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(Umu, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(Umu, header);
      IR.close();
    }
    // Still need RNG seeds — use stream-id offset as default.
    int sid = std::max(0, stream_id);
    sRNG.SeedFixedIntegers({11 + 100*sid, 12 + 100*sid, 13 + 100*sid,
                            14 + 100*sid, 15 + 100*sid});
    pRNG.SeedFixedIntegers({16 + 100*sid, 17 + 100*sid, 18 + 100*sid,
                            19 + 100*sid, 20 + 100*sid});
  } else {
    // Offset RNG seeds by 100 × stream_id so streams are independent.
    int sid = std::max(0, stream_id);
    sRNG.SeedFixedIntegers({11 + 100*sid, 12 + 100*sid, 13 + 100*sid,
                            14 + 100*sid, 15 + 100*sid});
    pRNG.SeedFixedIntegers({16 + 100*sid, 17 + 100*sid, 18 + 100*sid,
                            19 + 100*sid, 20 + 100*sid});
    std::string start_type = "tepid";
    if (const char *st = std::getenv("START_TYPE"); st && *st) start_type = st;
    // WEAK_FIELD_SCALE: U_mu = exp(i * scale * Σ_a c_a t_a) per link via
    // LieRandomize.  Grid's default TepidConfiguration uses scale=0.01 which
    // gives plaq ≈ 0.99996; chroma's WEAK_FIELD uses ≈0.1 giving plaq ≈ 0.997.
    // Default raised to 0.1 to match chroma's convention.
    double wf_scale = 0.1;
    if (const char *ws = std::getenv("WEAK_FIELD_SCALE"); ws && *ws) wf_scale = std::atof(ws);
    std::cout << GridLogMessage << "START_TYPE=" << start_type
              << "  WEAK_FIELD_SCALE=" << wf_scale << std::endl;
    if (start_type == "hot") {
      SU<Nc>::HotConfiguration(pRNG, Umu);
    } else if (start_type == "cold") {
      SU<Nc>::ColdConfiguration(pRNG, Umu);
    } else {
      // Scaled tepid / weak-field: LieRandomize per link with caller-chosen
      // scale, bypassing Grid's hard-coded 0.01.
      LatticeColourMatrix Ulink(Umu.Grid());
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::LieRandomize(pRNG, Ulink, wf_scale);
        PokeIndex<LorentzIndex>(Umu, Ulink, mu);
      }
    }
  }

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

  // Single-precision grid + gauge field + operator for mixed-precision CG.
  GridCartesian        GridF(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian RBGridF(&GridF);
  LatticeGaugeFieldF UmuF(&GridF);
  {
    LatticeColourMatrix  U_d(&Grid);
    LatticeColourMatrixF U_f(&GridF);
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(Umu, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
  }
  typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>> WCF_f;
  // Antiperiodic BC in time to match chroma's <boundary>1 1 1 -1</boundary>.
  // Grid default is all-periodic → wrong det(M) → equilibrium plaq shift.
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time
  WilsonImplParams impl_pF;
  impl_pF.boundary_phases.resize(Nd, 1.0);
  impl_pF.boundary_phases[Nd - 1] = -1.0;
  WCF_f FermOpF(UmuF, GridF, RBGridF, mass_light, csw, csw,
                WilsonAnisotropyCoefficients(), impl_pF);

  // Light quarks (Nf=2): EO-preconditioned LogDet + Schur.
  // Action solver (accept/reject): tight DP CG at cg_tol=1e-8.
  // Derivative solver (MD force): mixed-precision CG (SP inner + DP correction),
  // outer tolerance 1e-6 (mdtol bias cancels on Metropolis accept/reject).
  WCF FermOp(Umu, Grid, RBGrid, mass_light, csw, csw,
             WilsonAnisotropyCoefficients(), impl_p);
  RealD cg_action_tol = cg_tol;
  if (const char *t = std::getenv("CG_TOL"); t && *t) cg_action_tol = std::atof(t);
  std::cout << GridLogMessage << "CG_TOL=" << cg_action_tol << std::endl;
  ConjugateGradient<LatticeFermion> CG_action(cg_action_tol, cg_max);
  SchurDifferentiableOperator<WilsonImplR> SchurOpD(FermOp);
  SchurDifferentiableOperator<WilsonImplF> SchurOpF(FermOpF);
  // CG_MD_TOL env var: MD force CG tolerance.  Default 1e-6 (chroma-style),
  // but force-FD test showed 0.2% force-action mismatch in TwoFlavourSchurMP
  // at this tol → tightening here lets us probe whether the residual
  // mismatch is just sloppy CG or something structural.
  RealD cg_md_tol = 1e-6;
  if (const char *t = std::getenv("CG_MD_TOL"); t && *t) cg_md_tol = std::atof(t);
  std::cout << GridLogMessage << "CG_MD_TOL=" << cg_md_tol << std::endl;
  MixedPrecCGWrapper<LatticeFermion, LatticeFermionF,
                     SchurDifferentiableOperator<WilsonImplR>,
                     SchurDifferentiableOperator<WilsonImplF>>
      CG_md(cg_md_tol, cg_max, 50, &RBGridF, SchurOpD, SchurOpF);
  QCDLogDetCloverEOAction<WilsonImplR> LightLogDet(FermOp, 2);
  LightLogDet.is_smeared = true;
  // SOLVER_DEBUG=1 forces DP CG_action also for the derivative — used to
  // disentangle MP-CG bug from structural deriv vs S inconsistency.
  bool solver_debug = false;
  if (const char *t = std::getenv("SOLVER_DEBUG"); t && std::atoi(t)) solver_debug = true;
  TwoFlavourSchurCloverActionMP<WilsonImplR, WilsonImplF>
      LightSchurPF(FermOp, FermOpF,
                   solver_debug ? (OperatorFunction<LatticeFermion>&)CG_action : (OperatorFunction<LatticeFermion>&)CG_md,
                   CG_action);
  LightSchurPF.is_smeared = true;

  // Strange quark (Nf=1): EO-preconditioned LogDet + Schur RHMC.
  // MD force uses mixed-precision multishift CG (SP inner + DP reliable).
  WCF StrangeFermOp(Umu, Grid, RBGrid, mass_strange, csw, csw,
                    WilsonAnisotropyCoefficients(), impl_p);
  WCF_f StrangeFermOpF(UmuF, GridF, RBGridF, mass_strange, csw, csw,
                       WilsonAnisotropyCoefficients(), impl_pF);
  // Chroma-matched bounds for the rat_3strange monomial on this ensemble:
  // <lowerMin>0.0001</lowerMin> <upperMax>32</upperMax>, force <degree>13</degree>.
  // Was (1e-4, 200, 10) — hi=200 wasted Remez fit on a region the spectrum
  // doesn't reach (real top is ~24-30); degree=10 was less accurate than
  // chroma's 13.  Force eval cost grows ~30% from extra poles; trade is
  // fewer cleanup steps + tighter bound on |dH|.
  OneFlavourRationalParams strange_rat(1e-4, 32.0, cg_max, cg_tol, 13, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  StrangeLogDet.is_smeared = true;
  OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>
      StrangeSchurPF(StrangeFermOp, StrangeFermOpF, &RBGridF, strange_rat, 50);
  StrangeSchurPF.is_smeared = true;

  // Grid's SymanzikGaugeAction(β,u0) uses the RBC/Iwasaki convention
  // (c_plaq = β·(1−8c1) ≈ 1.96β, c_rect = −β/(12·u0²)) — NOT the Lüscher-
  // Weisz tree-level action chroma's LW_TREE_GAUGEACT uses.  Chroma's source
  // literally sets  c0 = β  (the 5/3 is absorbed into β_XML) and
  //                 c1 = −c0/(20·u0²) = −β/(20·u0²).
  // So to match the chroma reference ensemble cl3_16_48_b6p1 at XML β=6.1
  // we construct PlaqPlusRectangleAction directly with chroma's coefficients.
  typedef PlaqPlusRectangleAction<PeriodicGimplR> PlaqRectR;
  PlaqRectR GaugeAction(beta, -beta / (20.0 * u0 * u0));
  // Chroma's LW_TREE_GAUGEACT operates on the THIN (unsmeared) gauge links;
  // stout only wraps the fermion via STOUT_FERM_STATE.  Setting
  // is_smeared=false here makes the gauge action see thin U directly —
  // matching chroma.  is_smeared=true was the cause of the Δplaq=0.12
  // equilibrium shift vs chroma's 0.5135.
  GaugeAction.is_smeared = false;

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  // Outer (fermion) level: 7 steps/trajectory.  Inner (gauge) level: 4 gauge
  // sub-steps per fermion step -> 28 gauge force evaluations per trajectory.
  // Mirrors chroma's Min-Norm 7 / 7x4 convention on this ensemble.
  // GAUGE_INNER_MULT env var lets us probe whether multi-rate is the source
  // of FG instability at MDs=7 — set to 1 to put gauge force at outer level.
  int gauge_inner_mult = 4;
  if (const char *m = std::getenv("GAUGE_INNER_MULT"); m && *m) gauge_inner_mult = std::atoi(m);
  std::cout << GridLogMessage << "GAUGE_INNER_MULT=" << gauge_inner_mult << std::endl;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  ActionLevel<LatticeGaugeField, Reps> L2(std::max(1, gauge_inner_mult));
  L1.push_back(&LightLogDet);
  L1.push_back(&LightSchurPF);
  L1.push_back(&StrangeLogDet);
  L1.push_back(&StrangeSchurPF);
  ActionSet<LatticeGaugeField, Reps> Aset;
  if (gauge_inner_mult <= 1) {
    L1.push_back(&GaugeAction);
    Aset.push_back(L1);
  } else {
    L2.push_back(&GaugeAction);
    Aset.push_back(L1);
    Aset.push_back(L2);
  }

  // INTEGRATOR env var: "MinimumNorm2" (default) or "ForceGradient".  Added
  // to allow direct comparison against TXQCD runs using the same integrator.
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
  // var (0 = chroma-style Metropolis-from-trajectory-0).  With thermal aux /
  // weak-field gauge the initial configuration is close enough to equilibrium
  // that skipping Metropolis for the first few trajs risks settling into a
  // pure-MD equilibrium that differs from the Metropolis one.
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
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear_inv, Stout);
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
  ckpt.cfg_prefix = cfg_dir + "/ckpoint_lat";
  ckpt.rng_prefix = cfg_dir + "/ckpoint_rng";
  ckpt.interval   = meas_skip;

  QcdDiag diag(cfg_dir + "/hmc_diagnostics", meas_skip, {
      {"LightLogDet", &LightLogDet},
      {"LightSchurPF", &LightSchurPF},
      {"StrangeLogDet", &StrangeLogDet},
      {"StrangeSchurPF", &StrangeSchurPF},
      {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG);

  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};

  // TEST_FORCE_FD: skip HMC, run force-FD consistency check on each action
  // through the full Smearer wiring (so we test the smeared_force chain rule
  // for is_smeared=true actions).  Compares dS_actual = S(U_+ε) − S(U_−ε)
  // against analytic dS_pred = −2ε Σ Re tr(p · U·∂S/∂U).  Ratio → 1 means
  // deriv() is consistent with S() at this U.  Diverging ratio at small ε →
  // analytic-vs-numeric force inconsistency.
  if (const char *t = std::getenv("TEST_FORCE_FD"); t && std::atoi(t)) {
    std::vector<std::pair<std::string, Action<LatticeGaugeField>*>> actions = {
        {"PlaqRect",      &GaugeAction},
        {"LightLogDet",   &LightLogDet},
        {"LightSchurPF",  &LightSchurPF},
        {"StrangeLogDet", &StrangeLogDet},
        {"StrangeSchurPF", &StrangeSchurPF}};

    // FD_NO_SMEAR=1 disables stout smearing on all fermion actions for the
    // test — isolates whether bug is in plain Schur+clover deriv() or in the
    // SmearedConfiguration chain-rule plumbing.
    if (const char *fns = std::getenv("FD_NO_SMEAR"); fns && std::atoi(fns)) {
      LightLogDet.is_smeared = false;
      LightSchurPF.is_smeared = false;
      StrangeLogDet.is_smeared = false;
      StrangeSchurPF.is_smeared = false;
      std::cout << GridLogMessage << "[FD] FD_NO_SMEAR=1 — fermion is_smeared=false" << std::endl;
    }

    // Refresh pseudofermions against the loaded U (smeared, where applicable).
    Smear.set_Field(Umu);
    for (auto &p : actions) p.second->refresh(Smear, sRNG, pRNG);

    LatticeGaugeField mom(&Grid);
    {
      LatticeColourMatrix mommu(&Grid);
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
        PokeIndex<LorentzIndex>(mom, mommu, mu);
      }
    }

    LatticeGaugeField Umu_save(&Grid); Umu_save = Umu;
    LatticeGaugeField UdSdU(&Grid);
    for (auto &[name, act] : actions) {
      // Restore U + smear chain to the reference cfg.
      Umu = Umu_save;
      Smear.set_Field(Umu);
      // Compute analytic derivative through Smearer (respects is_smeared +
      // applies smeared_force chain rule when needed).
      act->deriv(Smear, UdSdU);
      // Apply the same projection the integrator does after deriv().  Only the
      // Lie-algebra (Ta) part of dSdU is physically meaningful — anything else
      // is dropped before the momentum update.
      UdSdU = PeriodicGimplR::projectForce(UdSdU);
      ComplexD dSpred(0.0, 0.0);
      for (int mu = 0; mu < Nd; ++mu) {
        auto UdSdUmu = PeekIndex<LorentzIndex>(UdSdU, mu);
        auto pmu     = PeekIndex<LorentzIndex>(mom, mu);
        LatticeComplex dS_mu = -2.0 * trace(pmu * UdSdUmu);
        dSpred += TensorRemove(sum(dS_mu));
      }
      RealD S0 = act->S(Smear);
      std::cout << GridLogMessage << "[FD][" << name << "] S(U)=" << std::setprecision(15) << S0
                << "  dS_pred=" << dSpred << std::endl;
      for (RealD eps : {1e-2, 1e-3, 1e-4, 1e-5}) {
        // U_± = (1 ± ε p) U with the smear chain refreshed via set_Field.
        LatticeGaugeField Up(&Grid), Um(&Grid);
        {
          autoView(Up_v, Up, CpuWrite);
          autoView(Um_v, Um, CpuWrite);
          autoView(U_v,  Umu_save, CpuRead);
          autoView(p_v,  mom, CpuRead);
          thread_foreach(i, p_v, {
            for (int mu = 0; mu < Nd; ++mu) {
              Up_v[i](mu) = U_v[i](mu) + p_v[i](mu) * U_v[i](mu) * eps;
              Um_v[i](mu) = U_v[i](mu) - p_v[i](mu) * U_v[i](mu) * eps;
            }
          });
        }
        Umu = Up; Smear.set_Field(Umu);
        RealD Sp = act->S(Smear);
        Umu = Um; Smear.set_Field(Umu);
        RealD Sm = act->S(Smear);
        RealD dS_actual = Sp - Sm;
        ComplexD ratio = dS_actual / (2.0 * eps * dSpred);
        std::cout << GridLogMessage << "[FD][" << name << "]"
                  << " ε=" << std::scientific << std::setprecision(2) << eps
                  << " dS_act/2ε=" << std::setprecision(8) << dS_actual / (2.0 * eps)
                  << " dS_pred=" << dSpred << " ratio=" << ratio << std::endl;
      }
    }
    Grid_finalize();
    return 0;
  }

  // Branch on integrator.  Both types compiled, chosen at runtime.
  if (integrator_name == "ForceGradient") {
    typedef ForceGradient<PeriodicGimplR,
                          SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  } else {
    typedef MinimumNorm2<PeriodicGimplR,
                         SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "QCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
