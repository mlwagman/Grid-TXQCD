#include "params.h"
#include "eig_diag.h"
#include "aux_correlator.h"
#include <cstdio>
#include <cstring>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOActionQudaPrimitive.h>
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
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaForceRationalActionMP.h>
#endif

using namespace TXQCDProduction;

// Boolean env-var helper: treat unset, "", "0", "false", "no" as false; any
// other value as true. Replaces the older `getenv(...) != nullptr` pattern
// where `export X=0` confusingly still evaluated as TRUE (since the variable
// existed). Diagnosed 2026-05-25 during the b6.5 2-node Meooe pathology
// hunt: `export QUDA_FORCE=0` left QUDA enabled and triggered the 145× slow
// Wilson Meooe on 2-node; `unset QUDA_FORCE` correctly disabled it. See
// production/overnight_2node_diag/FINDINGS.md.
static inline bool env_enabled(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !*v) return false;
  if (v[0] == '0' && v[1] == '\0') return false;
  if (std::strcmp(v, "false") == 0 || std::strcmp(v, "False") == 0 ||
      std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "no") == 0 ||
      std::strcmp(v, "No") == 0    || std::strcmp(v, "NO") == 0) return false;
  return true;
}

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
  // Per-traj eigenvalue diagnostic (γ5·M signed Rayleigh quotients on Ritz
  // vectors of the Chebyshev-filtered M†M, sorted by smallest |λ|).  Enabled
  // by EIG_DIAG=1 env var; empty inner vectors when disabled.
  std::vector<std::vector<RealD>> eig_M2_, eig_g5M_;
  // Sign-problem order parameters (basis-independent): smallest |γ5·M|
  // and #modes with |γ5·M|<zero_eps.  These are the sharp det-sign
  // diagnostics — immune to ± near-degenerate-pair relabeling.
  std::vector<RealD> eig_minabs_, eig_nnear_;

  // Per-traj aux wall-wall correlators.  Computed every traj (collective
  // sliceSum on every rank — see aux_correlator.h); accumulated here and
  // flushed to h5 inside the rank-0 write block.  Outer dim = traj index
  // since last write; inner dim already flat per channel (T, Nf²·T, or
  // NtPairs·T).  Mirrors the schema of meas_aux_txqcd's per-cfg h5 except
  // with the trajectory axis prepended.
  std::vector<std::vector<ComplexD>> aux_C_sigma_, aux_C_pi_iso_,
                                     aux_C_pi_ab_, aux_C_s_, aux_C_p_, aux_C_t_;
  std::vector<std::vector<ComplexD>> aux_wall_sigma_, aux_wall_pi_tr_,
                                     aux_wall_pi_ab_, aux_wall_s_,
                                     aux_wall_p_, aux_wall_t_;

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

    // Per-component σ + s breakdown (same as gen_txqcd_cfgs.cc Nf=3 driver).
    // See that file for the analytical/diagnostic motivation (within-cfg vs
    // between-cfg cancellation of <σ_ab>=0 for a≠b).
    {
      std::ostringstream sd;
      sd << "[TxqcdDiag] traj=" << traj << " sigma_diag";
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeComplex sab(U.Grid());
        sab = PeekIndex<2>(U.sigma, a, a);
        ComplexD v = TensorRemove(sum(sab)) / V;
        sd << "[" << a << "]=" << v.real();
      }
      if (TxqcdNf >= 2) {
        sd << " sigma_offdiag(re,im)";
        for (int a = 0; a < TxqcdNf; ++a) {
          for (int b = a + 1; b < TxqcdNf; ++b) {
            LatticeComplex sab(U.Grid());
            sab = PeekIndex<2>(U.sigma, a, b);
            ComplexD v = TensorRemove(sum(sab)) / V;
            sd << "[" << a << b << "]=(" << v.real() << "," << v.imag() << ")";
          }
        }
      }
      std::cout << GridLogMessage << sd.str() << std::endl;

      // Six-quantity stats per aux component — see gen_txqcd_cfgs.cc for full doc.
      struct CStats {
        RealD mean_re, mean_im, mean_modsq, var_re, var_im, var_modsq;
      };
      auto cstats = [V](const LatticeComplex &sab) -> CStats {
        ComplexD m = TensorRemove(sum(sab)) / V;
        RealD m_abs_sq = norm2(sab) / V;
        RealD re_sq_re = TensorRemove(sum(sab * sab)).real() / V;
        LatticeComplex modsq(sab.Grid());
        modsq = sab * conjugate(sab);
        RealD m_modsq_sq = TensorRemove(sum(modsq * modsq)).real() / V;
        CStats s;
        s.mean_re    = m.real();
        s.mean_im    = m.imag();
        s.mean_modsq = m_abs_sq;
        s.var_re     = std::max(RealD(0.5) * (m_abs_sq + re_sq_re)
                                  - m.real() * m.real(), RealD(0.0));
        s.var_im     = std::max(RealD(0.5) * (m_abs_sq - re_sq_re)
                                  - m.imag() * m.imag(), RealD(0.0));
        s.var_modsq  = std::max(m_modsq_sq - m_abs_sq * m_abs_sq, RealD(0.0));
        return s;
      };
      auto dump_mat = [&cstats, traj](const std::string &name,
                                       auto &field, int N) {
        static const char *keys[6] = {
          "mean_re", "var_re", "mean_im", "var_im", "mean_modsq", "var_modsq"};
        std::ostringstream out[6];
        for (int k = 0; k < 6; ++k)
          out[k] << "[TxqcdDiag] traj=" << traj << " "
                 << name << "_" << keys[k] << " diag";
        for (int a = 0; a < N; ++a) {
          LatticeComplex sab(field.Grid());
          sab = PeekIndex<2>(field, a, a);
          CStats s = cstats(sab);
          out[0] << "[" << a << "]=" << s.mean_re;
          out[1] << "[" << a << "]=" << s.var_re;
          out[2] << "[" << a << "]=" << s.mean_im;
          out[3] << "[" << a << "]=" << s.var_im;
          out[4] << "[" << a << "]=" << s.mean_modsq;
          out[5] << "[" << a << "]=" << s.var_modsq;
        }
        if (N >= 2) {
          for (int k = 0; k < 6; ++k) out[k] << " offdiag";
          for (int a = 0; a < N; ++a) {
            for (int b = a + 1; b < N; ++b) {
              LatticeComplex sab(field.Grid());
              sab = PeekIndex<2>(field, a, b);
              CStats s = cstats(sab);
              out[0] << "[" << a << b << "]=" << s.mean_re;
              out[1] << "[" << a << b << "]=" << s.var_re;
              out[2] << "[" << a << b << "]=" << s.mean_im;
              out[3] << "[" << a << b << "]=" << s.var_im;
              out[4] << "[" << a << b << "]=" << s.mean_modsq;
              out[5] << "[" << a << b << "]=" << s.var_modsq;
            }
          }
        }
        for (int k = 0; k < 6; ++k)
          std::cout << GridLogMessage << out[k].str() << std::endl;
      };
      dump_mat("sigma", U.sigma, TxqcdNf);
      dump_mat("pi",    U.pi,    TxqcdNf);
      dump_mat("s_color", U.s,   Nc);
      dump_mat("p_color", U.p,   Nc);
      {
        LatticeSFieldC t01(U.t.Grid());
        t01 = PeekIndex<1>(U.t, 0, 1);
        dump_mat("t01", t01, Nc);
      }
    }
    {
      std::ostringstream sd;
      sd << "[TxqcdDiag] traj=" << traj << " s_diag";
      for (int i = 0; i < Nc; ++i) {
        LatticeComplex sii(U.Grid());
        sii = PeekIndex<2>(U.s, i, i);
        ComplexD v = TensorRemove(sum(sii)) / V;
        sd << "[" << i << "]=" << v.real();
      }
      sd << " s_offdiag(re,im)";
      for (int i = 0; i < Nc; ++i) {
        for (int j = i + 1; j < Nc; ++j) {
          LatticeComplex sij(U.Grid());
          sij = PeekIndex<2>(U.s, i, j);
          ComplexD v = TensorRemove(sum(sij)) / V;
          sd << "[" << i << j << "]=(" << v.real() << "," << v.imag() << ")";
        }
      }
      std::cout << GridLogMessage << sd.str() << std::endl;
    }

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

    // ---- γ5·M signed eigenvalue diagnostic (opt-in via EIG_DIAG=1) --------
    // Uses the SAME TXQCD operator the HMC integrator sees (with current
    // dynamical aux fields), so any zero-crossing of det(M) shows up as a
    // sign flip in the smallest eigenvalue across consecutive trajectories.
    std::vector<RealD> eig_M2, eig_g5M;
    if (eig_diag_enabled()) {
      EigDiagParams ep = eig_diag_params_from_env();
      std::array<RealD, TxqcdNf> mass_arr;
      for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
      if (TxqcdNf >= 3) mass_arr[TxqcdNf - 1] = mass_strange;
      TXQCDWilsonCloverOp Mop_eig(Usm, grid_, rbgrid_, mass_arr,
                                   U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);
      RunEigDiagTxqcd(Mop_eig, &grid_, prng_, ep, eig_M2, eig_g5M);
      std::cout << GridLogMessage << "[TxqcdDiag] traj=" << traj
                << " γ5·M signed lowest |·|:";
      for (auto e : eig_g5M) std::cout << " " << e;
      std::cout << std::endl;
    }
    eig_M2_.push_back(eig_M2);
    eig_g5M_.push_back(eig_g5M);
    {
      RealD eig_ma; int eig_nn;
      eig_order_params(eig_g5M, eig_diag_params_from_env().zero_eps,
                       eig_ma, eig_nn);
      eig_minabs_.push_back(eig_ma);
      eig_nnear_.push_back((RealD)eig_nn);
    }

    // Per-traj aux wall-wall correlators (collective: every rank).  Cost ≪1%
    // of a trajectory.  Accumulated; flushed at meas_skip intervals below.
    {
      AuxWallCorrelators awc = ComputeAuxWallCorrelators(U);
      aux_C_sigma_.push_back(std::move(awc.C_sigma));
      aux_C_pi_iso_.push_back(std::move(awc.C_pi_iso));
      aux_C_pi_ab_.push_back(std::move(awc.C_pi_ab_flat));
      aux_C_s_.push_back(std::move(awc.C_s));
      aux_C_p_.push_back(std::move(awc.C_p));
      aux_C_t_.push_back(std::move(awc.C_t_flat));
      aux_wall_sigma_.push_back(std::move(awc.wall_sigma));
      aux_wall_pi_tr_.push_back(std::move(awc.wall_pi_tr));
      aux_wall_pi_ab_.push_back(std::move(awc.wall_pi_ab_flat));
      aux_wall_s_.push_back(std::move(awc.wall_s));
      aux_wall_p_.push_back(std::move(awc.wall_p));
      aux_wall_t_.push_back(std::move(awc.wall_t_flat));
    }

    if (traj % interval_ == 0) {
      // Hdf5Writer is SERIAL — all MPI ranks racing to open the same file
      // throws H5::FileIException under file-locking.  Crashed b6.5 1283314
      // at traj 10 with 4 ranks.  Guard with rank 0 only; clear arrays on
      // all ranks afterwards so memory doesn't grow without bound.
      if (grid_.IsBoss()) {
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
        write(wr, "eig_M2", eig_M2_);
        write(wr, "eig_g5M", eig_g5M_);
        write(wr, "eig_min_abs_g5M", eig_minabs_);
        write(wr, "eig_n_near_zero", eig_nnear_);
        // Aux wall-wall correlators accumulated since last write
        // (one row per trajectory; columns = T, Nf²·T, or NtPairs·T).
        // Schema matches meas_aux_txqcd's per-cfg h5 with an outer traj axis.
        write(wr, "aux_C_sigma", aux_C_sigma_);
        write(wr, "aux_C_pi_iso", aux_C_pi_iso_);
        write(wr, "aux_C_pi_ab", aux_C_pi_ab_);
        write(wr, "aux_C_s", aux_C_s_);
        write(wr, "aux_C_p", aux_C_p_);
        write(wr, "aux_C_t", aux_C_t_);
        write(wr, "aux_wall_sigma", aux_wall_sigma_);
        write(wr, "aux_wall_pi_tr", aux_wall_pi_tr_);
        write(wr, "aux_wall_pi_ab", aux_wall_pi_ab_);
        write(wr, "aux_wall_s", aux_wall_s_);
        write(wr, "aux_wall_p", aux_wall_p_);
        write(wr, "aux_wall_t", aux_wall_t_);
        std::vector<std::string> names;
        for (auto &a : actions_) names.push_back(a.name);
        write(wr, "action_names", names);
        std::cout << GridLogMessage << "Diagnostics written to " << fname << std::endl;
      }
      traj_.clear(); plaq_.clear();
      vev_sigma_.clear(); vev_s_.clear(); vev_trminv_.clear();
      force_avg_.clear(); force_max_.clear();
      fdt_avg_.clear(); fdt_max_.clear();
      eig_M2_.clear(); eig_g5M_.clear();
      eig_minabs_.clear(); eig_nnear_.clear();
      aux_C_sigma_.clear(); aux_C_pi_iso_.clear(); aux_C_pi_ab_.clear();
      aux_C_s_.clear(); aux_C_p_.clear(); aux_C_t_.clear();
      aux_wall_sigma_.clear(); aux_wall_pi_tr_.clear();
      aux_wall_pi_ab_.clear(); aux_wall_s_.clear();
      aux_wall_p_.clear(); aux_wall_t_.clear();
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
  // Env overrides for wiring-test smokes that need fast CGs (RAT_LO=0.01
  // RAT_DEGREE=8 RAT_TOL=1e-5 → multishift converges in ≪100 iters on 4⁴).
  // Production leaves these unset and uses the defaults below.
  RealD rat_lo = 1e-4, rat_hi = 100.0, rat_tol = cg_tol;
  int rat_degree = 20;
  if (const char *e = std::getenv("RAT_LO");     e && *e) rat_lo     = std::atof(e);
  if (const char *e = std::getenv("RAT_HI");     e && *e) rat_hi     = std::atof(e);
  if (const char *e = std::getenv("RAT_TOL");    e && *e) rat_tol    = std::atof(e);
  if (const char *e = std::getenv("RAT_DEGREE"); e && *e) rat_degree = std::atoi(e);
  OneFlavourRationalParams rat_params(rat_lo, rat_hi, cg_max, rat_tol,
                                      rat_degree, 64, 100, 1e-6, 1e-4);

  // Grid's SymanzikGaugeAction uses RBC convention — NOT chroma's.
  // Chroma's LW_TREE_GAUGEACT literally sets c0=β, c1=−β/(20·u0²).
  // Match chroma's XML-β convention for reference-ensemble compatibility.
  typedef PlaqPlusRectangleAction<PeriodicGimplR> PlaqRectR;
  GaugeActionAdapter<PlaqRectR> GaugeAction(beta, -beta / (20.0 * u0 * u0));
  // Chroma's LW_TREE_GAUGEACT operates on the THIN (unsmeared) gauge links;
  // stout only wraps the fermion via STOUT_FERM_STATE.  Set is_smeared=false.
  GaugeAction.is_smeared = false;

  AuxiliaryFieldGaussianAction AuxAction(lambda_runtime);

  // Optional kinetic-term action for all 5 aux fields (σ, π, s, p, t):
  //   S_kin = (Z_σ/2) Σ_{x,μ} Tr[(σ(x+μ̂)-σ(x))²]   + same for π, s, p
  //         +  Z_t    Σ_{x,μ} Tr[(t(x+μ̂)-t(x))²]   (note: 1·Z_t, mirrors quadratic 2·λ²)
  // Decouples mean from variance: Z>0 damps high-momentum fluctuations of
  // that field while preserving its VEV (zero for π/p/t, Σ/λ² for σ, etc.).
  // Coefficients exactly mirror AuxGaussianAction with λ² → Z to preserve
  // Fierz at finite a.  Env knobs: SIGMA_KINETIC_Z, PI_KINETIC_Z,
  // S_KINETIC_Z, P_KINETIC_Z, T_KINETIC_Z (default 0 → no kinetic term).
  RealD Z_sigma_kin = 0.0, Z_pi_kin = 0.0, Z_s_kin = 0.0,
        Z_p_kin = 0.0, Z_t_kin = 0.0;
  if (const char *e = std::getenv("SIGMA_KINETIC_Z"); e && *e) Z_sigma_kin = std::atof(e);
  if (const char *e = std::getenv("PI_KINETIC_Z");    e && *e) Z_pi_kin    = std::atof(e);
  if (const char *e = std::getenv("S_KINETIC_Z");     e && *e) Z_s_kin     = std::atof(e);
  if (const char *e = std::getenv("P_KINETIC_Z");     e && *e) Z_p_kin     = std::atof(e);
  if (const char *e = std::getenv("T_KINETIC_Z");     e && *e) Z_t_kin     = std::atof(e);
  bool use_aux_kinetic = (Z_sigma_kin != 0.0) || (Z_pi_kin != 0.0) ||
                         (Z_s_kin != 0.0) || (Z_p_kin != 0.0) || (Z_t_kin != 0.0);
  AuxiliaryFieldKineticAction AuxKinAction(Z_sigma_kin, Z_pi_kin, Z_s_kin,
                                            Z_p_kin, Z_t_kin);
  std::cout << GridLogMessage << "[AuxKineticAction] SIGMA_KINETIC_Z=" << Z_sigma_kin
            << " PI_KINETIC_Z=" << Z_pi_kin
            << " S_KINETIC_Z=" << Z_s_kin
            << " P_KINETIC_Z=" << Z_p_kin
            << " T_KINETIC_Z=" << Z_t_kin
            << (use_aux_kinetic ? " (ACTIVE)" : " (inactive)") << std::endl;

  // Fierz-preserving σ_eff = σ - (Z/λ²)·Lap(σ) inside the Dirac operator.
  // When the aux kinetic term is active, vanilla TXQCD's σ-mediated 4-fermion
  // coupling becomes nonlocal 1/(λ²+Z·k̂²) and can no longer be Fierz-reduced
  // to a local mass shift.  TXQCD_FIERZ_LAP=1 enables a wrapper around the
  // TXQCD fermion actions that replaces σ with σ_eff inside the Dirac op and
  // applies the matching Lap chain rule on the force.  See
  // [[fierz-lap-shift]] memory and fierz_lap_shift.tex for the derivation.
  AuxFierzShift fierz_shift = AuxFierzShift::FromEnv(lambda_runtime);
  bool use_fierz_lap = fierz_shift.active();
  std::cout << GridLogMessage << fierz_shift.LogParameters()
            << (use_fierz_lap ? " (ACTIVE)" : " (inactive)") << std::endl;

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
  // Phase C: QUDA σ-piece hybrid via TXQCD_QUDA_HYBRID env (=1 enables).
  bool tx_quda_hybrid = env_enabled("TXQCD_QUDA_HYBRID");
  std::unique_ptr<TXQCDWilsonCloverRationalEOAction> PF_grid_holder;
  std::unique_ptr<TXQCDWilsonCloverRationalEOActionQudaPrimitive> PF_quda_holder;
  Action<TXQCDField> *PF = nullptr;
  if (tx_quda_hybrid) {
    PF_quda_holder = std::make_unique<TXQCDWilsonCloverRationalEOActionQudaPrimitive>(
        Grid, RBGrid, mass_light, rat_params, csw);
    PF_quda_holder->is_smeared = true;
    PF = PF_quda_holder.get();
    std::cout << GridLogMessage << "[TXQCD light Nf=2] using QUDA σ-piece hybrid" << std::endl;
  } else {
    PF_grid_holder = std::make_unique<TXQCDWilsonCloverRationalEOAction>(
        Grid, RBGrid, mass_light, rat_params, csw);
    PF_grid_holder->is_smeared = true;
    PF = PF_grid_holder.get();
  }

  // Hasenbusch split pair (used when HASEN_DM > 0).
  TXQCDWilsonCloverRationalEOAction PF_heavy(Grid, RBGrid, mass_heavy,
                                              rat_params, csw);
  PF_heavy.is_smeared = true;
  TXQCDWilsonCloverHasenbuschAction PF_ratio(Grid, RBGrid,
                                              mass_light, mass_heavy,
                                              rat_params, csw);
  PF_ratio.is_smeared = true;

  // ────────────────────────────────────────────────────────────────────────
  // N-level Hasenbusch ladder (HASEN_LADDER env var, comma-separated masses
  // light→heavy).  Overrides HASEN_DM when set.
  //
  // Example: HASEN_LADDER="-0.2416,-0.20,-0.10,0.05"  → 4-level chain:
  //   |det M(-0.2416)|
  //     = |det M(0.05)|                        (rational at heaviest)
  //       × |det M(-0.10) / M(0.05)|           (ratio)
  //       × |det M(-0.20) / M(-0.10)|          (ratio)
  //       × |det M(-0.2416) / M(-0.20)|        (ratio)
  //
  // Each link uses the existing TXQCDWilsonCloverHasenbuschAction class
  // (mathematically |det M_a / M_b| at masses a < b).  The chain composes
  // because the ratios telescope: M(m_1)/M(m_2) · M(m_2)/M(m_3) … M(m_{N-1})/M(m_N)
  // = M(m_1)/M(m_N), and we multiply back by |det M(m_N)| via the rational.
  //
  // For N=2 (HASEN_LADDER="m_light,m_heavy"), produces the same actions as
  // HASEN_DM = m_heavy - m_light.  Bit-exact recovery is a validation gate.
  // ────────────────────────────────────────────────────────────────────────
  std::vector<RealD> ladder_masses;
  if (const char *hl = std::getenv("HASEN_LADDER"); hl && *hl) {
    std::string s(hl);
    size_t pos = 0, comma;
    while ((comma = s.find(',', pos)) != std::string::npos) {
      ladder_masses.push_back(std::atof(s.substr(pos, comma - pos).c_str()));
      pos = comma + 1;
    }
    if (pos < s.size())
      ladder_masses.push_back(std::atof(s.substr(pos).c_str()));
  }

  // Build the ladder (only when HASEN_LADDER is set and has ≥2 levels).
  // ladder_rational holds the heaviest rational; ladder_ratios holds N-1
  // Hasenbusch ratio actions in order (m_1,m_2), (m_2,m_3), …, (m_{N-1},m_N).
  std::unique_ptr<TXQCDWilsonCloverRationalEOAction> ladder_rational;
  std::vector<std::unique_ptr<TXQCDWilsonCloverHasenbuschAction>> ladder_ratios;
  bool use_ladder = (ladder_masses.size() >= 2);
  if (use_ladder) {
    // Validate strictly increasing (light→heavy) mass list.
    for (size_t i = 1; i < ladder_masses.size(); ++i) {
      if (!(ladder_masses[i] > ladder_masses[i - 1])) {
        std::cerr << "HASEN_LADDER masses must be strictly increasing (light→heavy)."
                  << " Got: ";
        for (auto m : ladder_masses) std::cerr << m << " ";
        std::cerr << std::endl;
        std::exit(1);
      }
    }
    // Sanity: lightest must match mass_light (the physical light quark mass
    // we're sampling).  Otherwise the user is silently changing the action.
    if (std::abs(ladder_masses.front() - mass_light) > 1e-12) {
      std::cerr << "HASEN_LADDER first mass " << ladder_masses.front()
                << " must equal mass_light " << mass_light << std::endl;
      std::exit(1);
    }
    std::cout << GridLogMessage << "HASEN_LADDER (N=" << ladder_masses.size() << "):";
    for (auto m : ladder_masses) std::cout << " " << m;
    std::cout << std::endl;
    const RealD m_top = ladder_masses.back();
    ladder_rational = std::make_unique<TXQCDWilsonCloverRationalEOAction>(
        Grid, RBGrid, m_top, rat_params, csw);
    ladder_rational->is_smeared = true;
    for (size_t i = 0; i + 1 < ladder_masses.size(); ++i) {
      ladder_ratios.emplace_back(
          std::make_unique<TXQCDWilsonCloverHasenbuschAction>(
              Grid, RBGrid,
              /*mass_light=*/ladder_masses[i],
              /*mass_heavy=*/ladder_masses[i + 1],
              rat_params, csw));
      ladder_ratios.back()->is_smeared = true;
    }
  }

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_light, csw);
  LogDet.is_smeared = true;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj " << latest << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", latest);
    start_traj = latest;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    // Import an external thermalized gauge config (chroma LIME or NERSC).
    // Aux fields auto-initialized via Σ measured on the imported (smeared)
    // gauge, matching the AUX_INIT_AUTO path of the fresh-start branch.
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(U.U, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(U.U, header);
      IR.close();
    }
    sRNG.SeedFixedIntegers({1 + seed_offset, 2 + seed_offset, 3 + seed_offset,
                            4 + seed_offset, 5 + seed_offset});
    pRNG.SeedFixedIntegers({6 + seed_offset, 7 + seed_offset, 8 + seed_offset,
                            9 + seed_offset, 10 + seed_offset});
    // Σ priority: AUX_INIT env var override, else auto-measure on stout-smeared
    // gauge.  For chroma-equilibrium starts the static auto-measure under-shoots
    // by ~2× because of σ-back-reaction; AUX_INIT=3.5 (or the iterative path,
    // see AUX_INIT_ITERATIONS below) gets closer to true equilibrium.
    RealD Sigma = 0.0;
    if (const char *si = std::getenv("AUX_INIT"); si && *si) {
      Sigma = std::atof(si);
      std::cout << GridLogMessage << "[IMPORT_CFG+AUX_INIT] Σ=" << Sigma
                << " (manual override)" << std::endl;
    } else {
      Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
      SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv, Stout);
      SmearMeas.set_Field(U.U);
      LatticeGaugeField Usm = SmearMeas.get_SmearedU();
      WilsonImplParams impl_p_meas;
      impl_p_meas.boundary_phases.resize(Nd, 1.0);
      impl_p_meas.boundary_phases[Nd - 1] = -1.0;
      typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> MFO;
      MFO Dw(Usm, Grid, RBGrid, mass_light, csw, csw,
             WilsonAnisotropyCoefficients(), impl_p_meas);
      MdagMLinearOperator<MFO, LatticeFermion> HermOp(Dw);
      ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
      RealD V = (RealD)Grid.gSites(); RealD acc = 0.0;
      GridParallelRNG noisePRNG(&Grid);
      noisePRNG.SeedFixedIntegers(
          {seed_offset + 11, seed_offset + 12, seed_offset + 13,
           seed_offset + 14, seed_offset + 15});
      for (int h = 0; h < n_vev_noise; ++h) {
        LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
        gaussian(noisePRNG, eta); Dw.Mdag(eta, b); x = Zero();
        CG(HermOp, b, x);
        acc += innerProduct(eta, x).real() / (2.0 * V);
      }
      Sigma = (acc / n_vev_noise);  // Σ = vev_trminv (no /2 — see header)
      std::cout << GridLogMessage << "[IMPORT_CFG+AUX_AUTO] Σ=" << Sigma
                << "  → <σ_aa>=" << Sigma / (lambda_runtime * lambda_runtime)
                << std::endl;
    }
    TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime, Sigma);
    // If a kinetic action is active (any *_KINETIC_Z env > 0), apply the
    // FFT filter so the initial aux distribution matches the Fierz-correct
    // joint quadratic+kinetic equilibrium at the chosen (λ, Z) point.  No
    // extra thermalization needed.
    TXQCDKineticFilter::ApplyFromEnv(U, lambda_runtime, Sigma);

    // Optional: self-consistent iteration with TXQCD operator on top of the
    // initial Σ guess.  Each iteration refills aux from the previous Σ, then
    // measures Σ again using the FULL TXQCD M⁻¹ (which feels the σ
    // back-reaction).  Converges in 2-3 iterations on chroma cfg.
    if (const char *ni = std::getenv("AUX_INIT_ITERATIONS"); ni && *ni) {
      int n_iter = std::atoi(ni);
      Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
      SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv, Stout);
      SmearMeas.set_Field(U.U);
      LatticeGaugeField Usm = SmearMeas.get_SmearedU();
      WilsonImplParams impl_p_meas;
      impl_p_meas.boundary_phases.resize(Nd, 1.0);
      impl_p_meas.boundary_phases[Nd - 1] = -1.0;
      RealD V = (RealD)Grid.gSites();
      GridParallelRNG noisePRNG(&Grid);
      noisePRNG.SeedFixedIntegers(
          {seed_offset + 21, seed_offset + 22, seed_offset + 23,
           seed_offset + 24, seed_offset + 25});
      for (int it = 0; it < n_iter; ++it) {
        // Build TXQCD operator with current aux fields.
        std::array<RealD, TxqcdNf> mass_arr;
        for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
        TXQCDWilsonCloverOp Mop(Usm, Grid, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p_meas);
        // Stochastic Tr[M_TXQCD⁻¹] / (4 N_f V) = per-flavor Σ for degenerate flavors.
        RealD acc = 0.0;
        for (int h = 0; h < n_vev_noise; ++h) {
          TXQCDFermionNf eta(&Grid), b(&Grid), x(&Grid);
          for (int a = 0; a < TxqcdNf; ++a) gaussian(noisePRNG, eta.f[a]);
          Mop.Mdag(eta, b);
          x = Zero();
          // Inline single-shift CG on M_TXQCD†M_TXQCD.
          TXQCDFermionNf r(&Grid), p(&Grid), Mp(&Grid), MdMp(&Grid);
          r = b; p = r;
          RealD rsq = norm2(r);
          RealD bsq = std::max(norm2(b), 1e-30);
          RealD tol2 = 1e-16 * bsq;
          for (int cg_it = 0; cg_it < cg_max; ++cg_it) {
            Mop.M(p, Mp); Mop.Mdag(Mp, MdMp);
            ComplexD pAp = innerProduct(p, MdMp);
            ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
            for (int a = 0; a < TxqcdNf; ++a) x.f[a] = x.f[a] + alpha * p.f[a];
            for (int a = 0; a < TxqcdNf; ++a) r.f[a] = r.f[a] - alpha * MdMp.f[a];
            RealD rsq_new = norm2(r);
            if (rsq_new < tol2) break;
            RealD beta_cg = rsq_new / rsq;
            for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
            rsq = rsq_new;
          }
          acc += innerProduct(eta, x).real() / (2.0 * (RealD)TxqcdNf * V);
        }
        Sigma = (acc / n_vev_noise);  // Σ = vev_trminv (no /2 — see header)
        std::cout << GridLogMessage << "[AUX_ITER " << (it+1) << "/" << n_iter
                  << "] Σ=" << Sigma
                  << "  → <σ_aa>=" << Sigma / (lambda_runtime * lambda_runtime)
                  << std::endl;
        TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime, Sigma);
      }
      // Re-apply kinetic filter on the final iteration's aux draw.
      TXQCDKineticFilter::ApplyFromEnv(U, lambda_runtime, Sigma);
    }
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
    // Aux-field equilibrium offset is Σ = vev_trminv = Tr[M⁻¹]/(2V) on the
    // SINGLE-flavor QCD Wilson-clover op, stout-smeared gauge.  Confirmed on
    // dynamical streams to 0.2% (2026-05-17) — the old "Σ≡vev_trminv/2 =
    // Tr/(4V)" convention was WRONG (spurious /2; harmless in HMC since the
    // dynamics relaxes σ to the true saddle, but it under-initialised σ).
    //   <σ_aa>  = Σ / λ²            (per-flavor diagonal component)
    //   <s_ii>  = (N_f/(√2·N_c)) · Σ / λ²
    //   vev_sigma diag = N_f·<σ_aa>  (the trivial N_f flavor trace)
    //
    // Three input modes (in priority order):
    //   AUX_INIT=value  → set Σ explicitly to this number
    //   AUX_SIGMA_L=v   → legacy env var; Σ = −AUX_SIGMA_L  (AUX_SIGMA_L is
    //                     calibrated as −vev_trminv = −Σ; meaning unchanged)
    //   AUX_INIT_AUTO=1 (default if neither is set) → measure vev_trminv on
    //                     the gauge with stout smearing + antiperiodic time
    //                     BC, set Σ = vev_trminv.
    RealD Sigma = 0.0;
    bool auto_measure = false;
    if (const char *si = std::getenv("AUX_INIT"); si && *si) {
      Sigma = std::atof(si);
    } else if (const char *sl = std::getenv("AUX_SIGMA_L"); sl && *sl) {
      Sigma = -std::atof(sl);  // AUX_SIGMA_L = −Σ = −vev_trminv (no /2)
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
        Sigma = vev_trminv;  // Σ = vev_trminv (corrected 2026-05-17, no /2)
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
      // Apply kinetic FFT filter so weak-field start lands at the
      // Fierz-correct (λ, Z) equilibrium distribution from the get-go.
      TXQCDKineticFilter::ApplyFromEnv(U, lambda_runtime, Sigma);
    }
  }

  // ---- AUX_KICK: optionally perturb the loaded aux fields right after the cfg
  // load.  Use to attempt basin-flip recovery of stuck λ chains.  Three modes:
  //   AUX_KICK=zero               σ=π=s=p=t=0   (test: is BASIN_HI gauge-locked?)
  //   AUX_KICK=scale:<f>          σ→f·σ, π→f·π, … (partial kick)
  //   AUX_KICK=redraw[:<Σ>]       fresh Gaussian draw with given Σ
  //                               (or auto-measure on current gauge if no Σ)
  // Gauge field U.U is untouched.  Diagnostics: print <σ>, <s> before/after.
  if (const char *k = std::getenv("AUX_KICK"); k && *k) {
    auto report = [&](const char *tag) {
      RealD V = (RealD)Grid.gSites();
      RealD vs = TensorRemove(sum(trace(U.sigma))).real() / V;
      RealD vc = TensorRemove(sum(trace(U.s))).real() / V;
      RealD vp = TensorRemove(sum(trace(U.pi))).real() / V;
      RealD vps = TensorRemove(sum(trace(U.p))).real() / V;
      RealD vt = TensorRemove(sum(trace(U.t))).real() / V;
      RealD pl = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
      std::cout << GridLogMessage << "[AUX_KICK " << tag << "] plaq=" << pl
                << " <σ>=" << vs << " <π>=" << vp << " <s>=" << vc
                << " <p>=" << vps << " <t>=" << vt << std::endl;
    };
    std::string mode(k);
    report("before");
    if (mode == "zero") {
      U.sigma = Zero(); U.pi = Zero();
      U.s = Zero(); U.p = Zero(); U.t = Zero();
      std::cout << GridLogMessage << "[AUX_KICK] zeroed all aux fields" << std::endl;
    } else if (mode.rfind("scale:", 0) == 0) {
      RealD f = std::atof(mode.c_str() + 6);
      U.sigma = f * U.sigma; U.pi = f * U.pi;
      U.s = f * U.s; U.p = f * U.p; U.t = f * U.t;
      std::cout << GridLogMessage << "[AUX_KICK] scaled all aux fields by f="
                << f << std::endl;
    } else if (mode.rfind("redraw", 0) == 0) {
      RealD Sigma_kick = 0.0;
      if (mode.length() > 7 && mode[6] == ':') {
        Sigma_kick = std::atof(mode.c_str() + 7);
        std::cout << GridLogMessage << "[AUX_KICK] redraw with Σ="
                  << Sigma_kick << " (manual)" << std::endl;
      } else {
        // Auto-measure Σ on current (stout-smeared) gauge with antiperiodic BC.
        Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
        SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv,
                                                        Stout);
        SmearMeas.set_Field(U.U);
        LatticeGaugeField Usm = SmearMeas.get_SmearedU();
        WilsonImplParams impl_p_kick;
        impl_p_kick.boundary_phases.resize(Nd, 1.0);
        impl_p_kick.boundary_phases[Nd - 1] = -1.0;
        typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>>
            MeasFermOpKick;
        MeasFermOpKick Dw(Usm, Grid, RBGrid, mass_light, csw, csw,
                          WilsonAnisotropyCoefficients(), impl_p_kick);
        MdagMLinearOperator<MeasFermOpKick, LatticeFermion> HermOp(Dw);
        ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
        RealD V = (RealD)Grid.gSites();
        GridParallelRNG noisePRNG(&Grid);
        noisePRNG.SeedFixedIntegers({9001, 9002, 9003, 9004, 9005});
        RealD acc = 0.0;
        for (int h = 0; h < n_vev_noise; ++h) {
          LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
          gaussian(noisePRNG, eta);
          Dw.Mdag(eta, b);
          x = Zero();
          CG(HermOp, b, x);
          acc += innerProduct(eta, x).real() / (2.0 * V);
        }
        Sigma_kick = (acc / n_vev_noise);  // Σ = vev_trminv (no /2)
        std::cout << GridLogMessage << "[AUX_KICK] redraw with auto-measured Σ="
                  << Sigma_kick << std::endl;
      }
      TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime, Sigma_kick);
    } else {
      std::cout << GridLogMessage << "[AUX_KICK] unknown mode '" << mode
                << "', ignoring" << std::endl;
    }
    report("after ");
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
  OneFlavourRationalParams strange_rat(1e-4, 100.0, cg_max, cg_tol, 20, 64,
                                       100, 1e-6, 1e-4);
  QCDLogDetCloverEOAction<WilsonImplR> StrangeLogDet(StrangeFermOp, 1);
  QCDActionAdapter StrangeLogDetAdapter(StrangeLogDet);
  StrangeLogDetAdapter.is_smeared = true;
  // MP rational: deriv() uses ConjugateGradientMultiShiftMixedPrec.
  // QUDA_FORCE=1 (Phase D) swaps in the QUDA-force kernel variant — same
  // multishift outputs, but deriv() calls computeCloverForceQuda (Wilson+σ
  // fused) with PyQUDA's dagger=YES convention. Validated cos=1.0 vs Path A.
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeSchurBase_holder;
  Action<LatticeGaugeField> *StrangeSchurInner = nullptr;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeSchurQuda_holder;
  if (env_enabled("QUDA_FORCE")) {
    QudaCloverParams qp;
    qp.mass = mass_strange;
    qp.csw  = csw;
    qp.anti_periodic_t = true;
    qp.tol = cg_tol;
    qp.max_iter = cg_max;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    StrangeSchurQuda_holder = std::make_unique<
        OneFlavourSchurCloverQudaForceRationalActionMP<WilsonImplR, WilsonImplF>>(
        StrangeFermOp, StrangeFermOpF, &StrangeRBGridF, strange_rat, qp, 50);
    StrangeSchurInner = StrangeSchurQuda_holder.get();
    std::cout << GridLogMessage
              << "[TXQCD strange Nf=1] QUDA_FORCE active — full QUDA force kernel"
              << std::endl;
  } else
#endif
  {
    StrangeSchurBase_holder = std::make_unique<
        OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>(
        StrangeFermOp, StrangeFermOpF, &StrangeRBGridF, strange_rat, 50);
    StrangeSchurInner = StrangeSchurBase_holder.get();
  }
  QCDActionAdapter StrangeSchurAdapter(*StrangeSchurInner);
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

  // Stable storage for FierzShiftedAction wrappers — must outlive L1 since the
  // ActionLevel holds raw pointers.  Built ONCE per underlying TXQCD action so
  // L1 and diag_actions share the same instance and report consistent
  // diagnostics.  Inactive shift returns the underlying pointer unchanged.
  std::vector<std::unique_ptr<FierzShiftedAction>> fierz_wrappers;
  auto wrap = [&](Action<TXQCDField> *a) -> Action<TXQCDField> * {
    if (!use_fierz_lap) return a;
    fierz_wrappers.emplace_back(std::make_unique<FierzShiftedAction>(*a, fierz_shift));
    return fierz_wrappers.back().get();
  };

  // Pre-wrap every TXQCD fermion action exactly once.  QCDActionAdapter actions
  // (strange) are NOT wrapped because they're pure QCD and don't depend on σ.
  Action<TXQCDField> *PF_w        = wrap(PF);
  Action<TXQCDField> *PF_heavy_w  = wrap(&PF_heavy);
  Action<TXQCDField> *PF_ratio_w  = wrap(&PF_ratio);
  Action<TXQCDField> *LogDet_w    = wrap(&LogDet);
  Action<TXQCDField> *ladder_top_w = nullptr;
  std::vector<Action<TXQCDField> *> ladder_ratio_w;
  if (use_ladder) {
    ladder_top_w = wrap(ladder_rational.get());
    for (auto &r : ladder_ratios) ladder_ratio_w.push_back(wrap(r.get()));
  }

  if (use_ladder) {
    // Heaviest rational first, then ratios in order (light side to heavy side).
    L1.push_back(ladder_top_w);
    for (auto *r : ladder_ratio_w) L1.push_back(r);
  } else if (hasen_dm > 0.0) {
    L1.push_back(PF_heavy_w);
    L1.push_back(PF_ratio_w);
  } else {
    L1.push_back(PF_w);
  }
  L1.push_back(LogDet_w);
  L1.push_back(&StrangeLogDetAdapter);   // QCD — no σ dependence
  L1.push_back(&StrangeSchurAdapter);    // QCD — no σ dependence
  ActionLevel<TXQCDField, Reps> L2(gauge_mult);
  L2.push_back(&GaugeAction);
  ActionLevel<TXQCDField, Reps> L3(aux_mult);
  L3.push_back(&AuxAction);
  if (use_aux_kinetic) L3.push_back(&AuxKinAction);
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
  // Use the wrapped pointers so per-action force/Fdt diagnostics match what
  // the integrator actually evaluates.
  if (use_ladder) {
    diag_actions.push_back({"PseudoFermionLadder_top", ladder_top_w});
    for (size_t i = 0; i < ladder_ratio_w.size(); ++i) {
      diag_actions.push_back(
          {"PseudoFermionLadder_ratio" + std::to_string(i), ladder_ratio_w[i]});
    }
  } else if (hasen_dm > 0.0) {
    diag_actions.push_back({"PseudoFermionHeavy", PF_heavy_w});
    diag_actions.push_back({"PseudoFermionRatio", PF_ratio_w});
  } else {
    diag_actions.push_back({"PseudoFermion", PF_w});
  }
  diag_actions.push_back({"LogDet", LogDet_w});
  diag_actions.push_back({"AuxGaussian", &AuxAction});
  if (use_aux_kinetic) diag_actions.push_back({"AuxKinetic", &AuxKinAction});
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
