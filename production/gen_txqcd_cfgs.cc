#include "params.h"
#include "eig_diag.h"
#include <cstdio>
#include <cstring>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOActionQudaPrimitive.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalAction.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalActionQudaPrimitive.h>
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

// Boolean env-var helper: treat unset, "", "0", "false", "no" as false; any
// other value as true.  Mirrors gen_txqcd_cfgs_2plus1.cc so `export FOO=0`
// reliably disables (vs the older getenv()!=nullptr pattern which would treat
// "0" as enabled).
static inline bool env_enabled(const char *name) {
  const char *v = std::getenv(name);
  if (!v || !*v) return false;
  if (v[0] == '0' && v[1] == '\0') return false;
  if (std::strcmp(v, "false") == 0 || std::strcmp(v, "False") == 0 ||
      std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "no") == 0 ||
      std::strcmp(v, "No") == 0    || std::strcmp(v, "NO") == 0) return false;
  return true;
}

// This driver is now compiled at TXQCD_Nf=3 with diag mass {m_l, m_l, m_s}.
// The previous Nf=2 (light) + Nf=1 (strange QCD-wrap) structure had two
// problems: (a) the kaon (light-strange) propagator only had the σ-coupling
// on the light leg → asymmetric decay relative to chroma's Nf=3 ensemble;
// (b) the strange leg used Schur PF whose M_pc⁻¹ amplification dominated
// near-zero modes off-equilibrium → metastable plaq~0.534 basin trap.
// Nf=3 with three independent rational PFs (one per flavor) at masses
// {m_l, m_l, m_s} resolves both: all three flavors share the σ background
// and rational PFs are bounded-force.
static_assert(TxqcdNf == 3,
              "production/gen_txqcd_cfgs.cc requires TXQCD_Nf=3 "
              "(per-flavor diag mass {m_l, m_l, m_s}).");

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
  // Per-traj γ5·M signed eigenvalues (EIG_DIAG=1).
  std::vector<std::vector<RealD>> eig_M2_, eig_g5M_;
  // Sign-problem order parameters (basis-independent): smallest |γ5·M|
  // and #modes with |γ5·M|<zero_eps.  These are the sharp det-sign
  // diagnostics — immune to ± near-degenerate-pair relabeling.
  std::vector<RealD> eig_minabs_, eig_nnear_;

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

    // Per-component σ + s breakdown — distinguishes flavor-singlet trace(σ)
    // from non-singlet σ_ab off-diagonals.  Useful for diagnosing whether the
    // period-2 oscillation observed in cold-gauge Nf=3 thermalization lives in
    // the singlet (trace) mode or in non-singlet (off-diagonal flavor) modes.
    //
    // Also prints volume std-dev of each component to distinguish within-cfg
    // cancellation (large std-dev with σ_ab(x) sign-flipping in different
    // spatial regions) from between-cfg cancellation (coherent σ_ab on one
    // cfg with sign that varies across cfgs).  For the analytical
    // ⟨σ_ab⟩=0 (a≠b) to be recovered from a single-cfg volume average that's
    // ≠0, between-cfg cancellation is required; for a single-cfg volume avg
    // ≈0, within-cfg cancellation is happening.
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

      // Six-quantity stats per Hermitian-matrix aux component (σ, π, s, p, t_{01}):
      //   mean_re = <Re σ>;          var_re = <(Re σ)²> - <Re σ>²
      //   mean_im = <Im σ>;          var_im = <(Im σ)²> - <Im σ>²
      //   mean_modsq = <|σ|²>;       var_modsq = <|σ|⁴> - <|σ|²>²
      // All sum-based (no sqrt).  Identities used:
      //   <(Re σ)²> = (<|σ|²> + Re<σ²>) / 2
      //   <(Im σ)²> = (<|σ|²> - Re<σ²>) / 2
      //   <|σ|⁴>    = sum_x (σ_x σ_x*)²
      // var_re vs var_im distinguishes flavor-singlet-rotation from amplitude
      // dynamics; var_modsq probes the amplitude of |σ|² fluctuations.  For an
      // iid Gaussian Re/Im with mean 0, var_re=var_im=½<|σ|²> and
      // var_modsq=<|σ|²>².
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
        // t_{01}: Lorentz (μ=0, ν=1) slice of the antisymmetric Lorentz +
        // Hermitian-color tensor.  PeekIndex<1>(U.t, 0, 1) reduces the Lorentz
        // iMatrix to scalar, leaving a TxqcdSiteColorMatrix-shaped tensor, so we
        // can reuse the matrix dumper.  Other Lorentz slices behave the same way
        // by antisymmetry — one representative is sufficient.
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

    if (traj % interval_ == 0) {
      // See gen_txqcd_cfgs_2plus1.cc: Hdf5Writer is SERIAL; all ranks racing
      // throws H5::FileIException. Guard with rank 0; clear on all ranks.
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

  // Chroma-matched rational bounds for the cl3_16_48_b6p1_m0p2450 ensemble:
  // lowerMin=1e-4, upperMax=32, degree=15.  The previous hi=200, degree=10
  // values were vestigial from the Nf=2 light + Nf=1 strange QCD-wrap setup
  // where the strange-wrap operator had wider effective spectrum.  With
  // Nf=3 diag mass and mass_strange = mass_light = -0.245 the operator is
  // the same as chroma's pure Wilson-Clover; the chroma bounds give a
  // ~5-order-of-magnitude better Remez approximation (~1e-9 vs ~1e-4) →
  // smaller rational dH contribution + faster multishift CG (poles closer
  // to spectrum density).
  OneFlavourRationalParams rat_params(1e-4, 100.0, cg_max, cg_tol, 20, 64,
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

  // Optional kinetic-term action for all 5 aux fields — mirrors the 2+1
  // driver.  See gen_txqcd_cfgs_2plus1.cc and AuxKineticAction.h for the
  // Fierz-preserving coefficient convention.
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

  // Fierz-preserving σ_eff = σ - (Z/λ²)·Lap(σ) inside the Dirac op.  See
  // [[fierz-lap-shift]] memory and fierz_lap_shift.tex.  Enabled via
  // TXQCD_FIERZ_LAP=1; reuses the *_KINETIC_Z env vars for the Z values.
  AuxFierzShift fierz_shift = AuxFierzShift::FromEnv(lambda_runtime);
  bool use_fierz_lap = fierz_shift.active();
  std::cout << GridLogMessage << fierz_shift.LogParameters()
            << (use_fierz_lap ? " (ACTIVE)" : " (inactive)") << std::endl;

  // Nf=3 diag mass: {m_l, m_l, m_s}.  All three flavors share aux fields.
  // The TXQCDWilsonCloverRationalEOAction's per-flavor-mass overload takes
  // a std::array<RealD, TxqcdNf> directly; pass {mass_light, mass_light,
  // mass_strange} (mass_strange = mass_light by default in params.h, but
  // the array form allows non-degenerate setups).
  const std::array<RealD, TxqcdNf> mass_arr = {mass_light, mass_light, mass_strange};
  std::cout << GridLogMessage << "Nf=3 diag mass = {" << mass_arr[0] << ", "
            << mass_arr[1] << ", " << mass_arr[2] << "}" << std::endl;

  // Hasenbusch (HASEN_DM>0) is not currently per-flavor compatible — disable
  // for Nf=3.  Rational HMC bounds the force on its own; the M_pc⁻¹
  // amplification that motivated Hasenbusch in Schur-PF setups doesn't apply.
  if (const char *hd = std::getenv("HASEN_DM"); hd && std::atof(hd) > 0) {
    std::cerr << "ERROR: HASEN_DM>0 not supported in Nf=3 build.  "
              << "Use rational HMC alone.\n";
    std::exit(1);
  }

  // ONE rational pseudofermion per flavor (3 total) with the diag mass array.
  // Each carries its own pseudofermion field; refresh independently.
  // TXQCD_QUDA_HYBRID=1 swaps in the QUDA σ-piece hybrid action (Phase H).
  // Masses must be degenerate for the hybrid path (single κ inside QUDA) —
  // safe here because params.h defaults mass_strange = mass_light, and all
  // production b6.1/b6.5 setups override both to the same value.
  bool tx_quda_hybrid = env_enabled("TXQCD_QUDA_HYBRID");
  bool tx_full_pf     = env_enabled("USE_FULL_PF");
  if (tx_quda_hybrid) {
    bool degenerate = true;
    for (int a = 1; a < TxqcdNf; ++a)
      if (std::abs(mass_arr[a] - mass_arr[0]) > 1e-12) degenerate = false;
    if (!degenerate) {
      std::cerr << "ERROR: TXQCD_QUDA_HYBRID requires degenerate masses; got {";
      for (int a = 0; a < TxqcdNf; ++a)
        std::cerr << mass_arr[a] << (a + 1 < TxqcdNf ? ", " : "");
      std::cerr << "}\n";
      std::exit(1);
    }
  }
  std::unique_ptr<TXQCDWilsonCloverRationalEOAction>                PF_eo_grid;
  std::unique_ptr<TXQCDWilsonCloverRationalEOActionQudaPrimitive>   PF_eo_quda;
  std::unique_ptr<TXQCDWilsonCloverRationalAction>                  PF_full_grid;
  std::unique_ptr<TXQCDWilsonCloverRationalActionQudaPrimitive>     PF_full_quda;
  Action<TXQCDField> *PF = nullptr;
  if (tx_full_pf && tx_quda_hybrid) {
    PF_full_quda = std::make_unique<TXQCDWilsonCloverRationalActionQudaPrimitive>(
        Grid, RBGrid, mass_arr, rat_params, csw);
    PF_full_quda->is_smeared = true;
    PF = PF_full_quda.get();
    std::cout << GridLogMessage
              << "[TXQCD Nf=" << TxqcdNf << "] using non-EO + QUDA σ hybrid" << std::endl;
  } else if (tx_full_pf) {
    PF_full_grid = std::make_unique<TXQCDWilsonCloverRationalAction>(
        Grid, RBGrid, mass_arr, rat_params, csw);
    PF_full_grid->is_smeared = true;
    PF = PF_full_grid.get();
    std::cout << GridLogMessage
              << "[TXQCD Nf=" << TxqcdNf << "] using non-EO Grid rational" << std::endl;
  } else if (tx_quda_hybrid) {
    PF_eo_quda = std::make_unique<TXQCDWilsonCloverRationalEOActionQudaPrimitive>(
        Grid, RBGrid, mass_arr, rat_params, csw);
    PF_eo_quda->is_smeared = true;
    PF = PF_eo_quda.get();
    std::cout << GridLogMessage
              << "[TXQCD Nf=" << TxqcdNf << "] using QUDA σ-piece hybrid action"
              << std::endl;
  } else {
    PF_eo_grid = std::make_unique<TXQCDWilsonCloverRationalEOAction>(
        Grid, RBGrid, mass_arr, rat_params, csw);
    PF_eo_grid->is_smeared = true;
    PF = PF_eo_grid.get();
  }

  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_arr, csw);
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
    // Aux-field initialization Σ priority:
    //   (a) AUX_INIT env var → use that Σ directly (no auto-measure).
    //   (b) Otherwise auto-measure Σ on the imported (stout-smeared) gauge.
    // For chroma-equilibrium starts, the static auto-measure under-shoots the
    // true equilibrium Σ by ~2× because it doesn't account for σ-back-reaction
    // on ⟨ψ̄ψ⟩.  Use AUX_INIT=3.5 (or AUX_INIT_ITERATIONS=N for a self-consistent
    // loop) to get closer to equilibrium at trajectory 0.
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
    TXQCDKineticFilter::ApplyFromEnv(U, lambda_runtime, Sigma);

    // Self-consistent iteration: for chroma-cfg starts the bare auto-measure
    // under-shoots the equilibrium aux mean by ~2× (σ-back-reaction on ⟨ψ̄ψ⟩
    // is missing).  Each iteration measures Tr[M_TXQCD⁻¹] / (4·N_f·V) on the
    // CURRENT aux fields and refills.  Converges in 2-3 iterations on chroma.
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
        // Nf=3 mass array: light pair + strange.
        std::array<RealD, TxqcdNf> mass_arr;
        for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
        if (TxqcdNf >= 3) mass_arr[TxqcdNf - 1] = mass_strange;
        TXQCDWilsonCloverOp Mop(Usm, Grid, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p_meas);
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
        Sigma = (acc / n_vev_noise);  // Σ = vev_trminv (TXQCD-op self-consist)
        std::cout << GridLogMessage << "[AUX_ITER " << (it+1) << "/" << n_iter
                  << "] Σ=" << Sigma
                  << "  → <σ_aa>=" << Sigma / (lambda_runtime * lambda_runtime)
                  << std::endl;
        TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime, Sigma);
      }
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
    // Tr/(4V)" was WRONG (spurious /2; harmless in HMC, under-init'd σ).
    //   <σ_aa>  = Σ / λ²            (per-flavor diagonal component)
    //   <s_ii>  = (N_f/(√2·N_c)) · Σ / λ²
    //   vev_sigma diag = N_f·<σ_aa>  (trivial N_f flavor trace)
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
      TXQCDKineticFilter::ApplyFromEnv(U, lambda_runtime, Sigma);
    }
  }

  // The strange-as-separate-Nf=1-QCD-wrap section was removed when this driver
  // was converted to Nf=3 with diag mass {m_l, m_l, m_s}: the strange flavor
  // is now handled by the TXQCD operator's third flavor slot, sharing the
  // same aux-field background as the light flavors.

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

  // Fierz Lap shift wrapping — see gen_txqcd_cfgs_2plus1.cc for the same
  // pattern.  Inactive shift returns the underlying pointer (bit-exact
  // pass-through); active shift allocates one wrapper per action and stores
  // it in fierz_wrappers (must outlive L1 + diag_actions).
  std::vector<std::unique_ptr<FierzShiftedAction>> fierz_wrappers;
  auto wrap = [&](Action<TXQCDField> *a) -> Action<TXQCDField> * {
    if (!use_fierz_lap) return a;
    fierz_wrappers.emplace_back(std::make_unique<FierzShiftedAction>(*a, fierz_shift));
    return fierz_wrappers.back().get();
  };
  Action<TXQCDField> *PF_w     = wrap(PF);
  Action<TXQCDField> *LogDet_w = wrap(&LogDet);

  L1.push_back(PF_w);
  L1.push_back(LogDet_w);
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
  // Use wrapped pointers so diagnostics match what the integrator evaluates.
  diag_actions.push_back({"PseudoFermion", PF_w});
  diag_actions.push_back({"LogDet", LogDet_w});
  diag_actions.push_back({"AuxGaussian", &AuxAction});
  if (use_aux_kinetic) diag_actions.push_back({"AuxKinetic", &AuxKinAction});
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
