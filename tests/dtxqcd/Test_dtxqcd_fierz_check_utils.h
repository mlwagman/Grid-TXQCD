// Shared Fierz-check helpers for the DTXQCD free-field unit tests.
//
//   Σ_DTXQCD = Tr[M48^{-1}] / (V · 2·N_F)   (per-quark, /(2V) Hutchinson convention)
//   Σ_W       = Tr[D_W^{-1}] / V             (plain Wilson at the same U)
//
// All operators use APBC time (chroma physics convention, matching
// DTXQCDMeooeDoubled::DefaultImplParams() and TXQCD's analogous default).
//
// Mirror of tests/txqcd/Test_txqcd_fierz_check_utils.h — same structure
// adapted to the doubled DTXQCD operator (M48 = block-diag M_upper +
// M_lower, both 24-dim per site → 48 per site for both blocks).

#pragma once

#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/utils/GaugeGroup.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/parallelIO/NerscIO.h>

NAMESPACE_BEGIN(Grid);

// Initialize the frozen gauge field. Aux fields are assumed already
// initialized by the caller (DTXQCDCompositeImpl::ColdConfiguration +
// any seed sequence).
//
// GAUGE_INIT env knob:
//   "cold"        — U_μ(x) = I (default)
//   "tepid"       — Grid TepidConfiguration (small Lie-random amp=0.01)
//   "tepid:AMP"   — explicit amplitude
//   "hot"         — full SU(3) random
//   "nersc:PATH"  — read NERSC-format config
inline RealD DtxqcdInitFrozenGauge(GridParallelRNG &pRNG, DTXQCDField &U) {
  const char *env = std::getenv("GAUGE_INIT");
  std::string mode = (env && *env) ? env : "cold";
  if (mode == "cold") {
    U.U = 1.0;
  } else if (mode == "hot") {
    SU<Nc>::HotConfiguration(pRNG, U.U);
  } else if (mode.rfind("tepid", 0) == 0) {
    RealD amp = 0.01;
    if (mode.size() > 5 && mode[5] == ':') amp = std::atof(mode.c_str() + 6);
    typedef LatticeColourMatrix LCM;
    LCM Umu(U.U.Grid());
    for (int mu = 0; mu < Nd; ++mu) {
      SU<Nc>::LieRandomize(pRNG, Umu, amp);
      PokeIndex<LorentzIndex>(U.U, Umu, mu);
    }
    std::cout << GridLogMessage << "DtxqcdInitFrozenGauge: tepid amp=" << amp << std::endl;
  } else if (mode.rfind("nersc:", 0) == 0) {
    std::string path = mode.substr(6);
    FieldMetaData header;
    NerscIO::readConfiguration(U.U, header, path);
    std::cout << GridLogMessage
              << "DtxqcdInitFrozenGauge: loaded NERSC " << path << std::endl;
  } else {
    std::cerr << "DtxqcdInitFrozenGauge: unknown GAUGE_INIT='" << mode
              << "'.  Falling back to cold." << std::endl;
    U.U = 1.0;
  }
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage
            << "DtxqcdInitFrozenGauge: mode=" << mode << " plaq=" << plaq
            << " (1.0 = cold, < 1 = perturbed)" << std::endl;
  return plaq;
}

inline WilsonImplR::ImplParams DtxqcdFierzApbcImplParams() {
  WilsonImplR::ImplParams p;
  p.boundary_phases.resize(Nd, 1.0);
  p.boundary_phases[Nd - 1] = -1.0;
  return p;
}

// Σ_W reference: Tr[D_W^{-1}] / V on plain Wilson (csw=0) or WilsonClover
// (csw!=0) at the given U, APBC time.  Per-quark trace.
template <class Op>
inline RealD DtxqcdFierzWilsonLikeTrminv_(Op &Dw, GridCartesian &Grid_,
                                           GridParallelRNG &prng,
                                           int n_noise, RealD cg_tol) {
  MdagMLinearOperator<Op, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, 30000);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid_), b(&Grid_), x(&Grid_);
    gaussian(prng, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / n_noise;
}

inline RealD DtxqcdFierzPlainWilsonTrminv(LatticeGaugeField &U,
                                           GridCartesian &Grid_,
                                           GridRedBlackCartesian &RBGrid,
                                           GridParallelRNG &prng,
                                           RealD mass, RealD csw,
                                           int n_noise, RealD cg_tol) {
  auto impl_p = DtxqcdFierzApbcImplParams();
  if (csw == 0.0) {
    WilsonFermionD Dw(U, Grid_, RBGrid, mass, impl_p);
    return DtxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  } else {
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(U, Grid_, RBGrid, mass, csw, csw,
            WilsonAnisotropyCoefficients(), impl_p);
    return DtxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  }
}

// Σ_DTXQCD: stochastic Tr[M48^{-1}] / (V · 2·N_F) on the doubled DTXQCD
// operator (covers both upper and lower blocks).  Per-quark normalization
// → matches DtxqcdFierzPlainWilsonTrminv at aux=0.
inline RealD DtxqcdFierzOpTrminv(DTXQCDField &U,
                                  GridCartesian &Grid_,
                                  GridRedBlackCartesian &RBGrid,
                                  GridParallelRNG &prng,
                                  RealD mass, RealD csw,
                                  int n_noise, RealD cg_tol) {
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const int cg_max_iter = 30000;
  for (int h = 0; h < n_noise; ++h) {
    DTXQCDFermionDoubled eta(&Grid_), b(&Grid_), x(&Grid_);
    DTXQCDFermionDoubled r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(prng, eta.upper.f[a]);
      gaussian(prng, eta.lower.f[a]);
    }
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < DtxqcdNf; ++a) {
      r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
      p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max_iter; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  // Both upper + lower blocks → 2·Nf flavors total.
  return acc / (n_noise * 2.0 * (RealD)DtxqcdNf);
}

// Self-consistent aux seed via bisection.  Solves
//   g(Σ) = Σ - Σ_DTXQCD(Σ) = 0
// where Σ_DTXQCD(Σ) is the per-quark trace of M_DTXQCD^{-1} with aux
// filled at ⟨s⟩ = 2·Nf·Σ/λ².  Bisection bracket: [0, Σ_bare].  Mirrors
// the AUX_INIT_AUTO logic in Test_dtxqcd_2pt_gencfgs.cc.
//
// Returns the converged Σ.  The caller is responsible for the final
// FillAuxFields call (which this routine also performs as a side effect).
inline RealD DtxqcdSelfConsistentAuxInit(GridParallelRNG &pRNG,
                                          GridCartesian &Grid_,
                                          GridRedBlackCartesian &RBGrid,
                                          DTXQCDField &U, RealD lambda,
                                          RealD mass, RealD csw,
                                          RealD Sigma_bare, int aux_max_iter,
                                          RealD aux_tol) {
  auto measure_sigma_dtxqcd = [&](RealD Sigma_at) -> RealD {
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_at);
    GridParallelRNG noisePRNG(&Grid_);
    noisePRNG.SeedFixedIntegers({400, 410, 420, 430, 440});
    const int n_noise_iter = 4;
    return DtxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG, mass, csw,
                                n_noise_iter, /*cg_tol=*/1e-8);
  };
  RealD Sigma_lo = 0.0;
  RealD Sigma_hi = Sigma_bare;
  RealD sigma_dtxqcd_lo = measure_sigma_dtxqcd(Sigma_lo);
  RealD g_lo = Sigma_lo - sigma_dtxqcd_lo;
  RealD sigma_dtxqcd_hi = measure_sigma_dtxqcd(Sigma_hi);
  RealD g_hi = Sigma_hi - sigma_dtxqcd_hi;
  std::cout << GridLogMessage
            << "[AUX_INIT bracket] g(0)=" << g_lo
            << "  g(" << Sigma_hi << ")=" << g_hi << std::endl;
  RealD Sigma_star = Sigma_bare;
  if (g_lo * g_hi > 0.0) {
    std::cout << GridLogMessage
              << "[AUX_INIT_AUTO] bracket has same sign — fall back to Σ_bare/2"
              << std::endl;
    Sigma_star = Sigma_hi / 2.0;
  } else {
    for (int it = 0; it < aux_max_iter; ++it) {
      RealD Sigma_mid = 0.5 * (Sigma_lo + Sigma_hi);
      RealD sigma_dtxqcd_mid = measure_sigma_dtxqcd(Sigma_mid);
      RealD g_mid = Sigma_mid - sigma_dtxqcd_mid;
      std::cout << GridLogMessage
                << "[AUX_INIT iter " << it << "] Σ_mid = " << Sigma_mid
                << "  Σ_DTXQCD = " << sigma_dtxqcd_mid
                << "  g = " << g_mid << std::endl;
      if (g_mid * g_lo < 0.0) {
        Sigma_hi = Sigma_mid;
        g_hi = g_mid;
      } else {
        Sigma_lo = Sigma_mid;
        g_lo = g_mid;
      }
      if ((Sigma_hi - Sigma_lo) /
              std::max(0.5 * (Sigma_hi + Sigma_lo), 1e-30) < aux_tol) {
        break;
      }
    }
    Sigma_star = 0.5 * (Sigma_lo + Sigma_hi);
  }
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_star);
  std::cout << GridLogMessage
            << "[AUX_INIT_AUTO converged] Σ* = " << Sigma_star
            << "  → ⟨s⟩* = 2·Nf·Σ*/λ² = "
            << (2.0 * DtxqcdNf * Sigma_star / (lambda * lambda)) << std::endl;
  return Sigma_star;
}

struct DtxqcdFierzCheckResult {
  RealD sigma_dtx;
  RealD sigma_w;
  RealD ratio;
  RealD dev;
  bool pass;
};

// In-line averaging observer.  Runs a small Fierz check on every HMC
// trajectory once burn-in is past, accumulating per-traj samples of
// Σ_DTXQCD, Σ_W, and the aux trace VEVs.  Finalize() returns the
// ensemble-averaged ratio with cfg-fluctuation noise folded in.
class DtxqcdFierzAveragingObserver : public HmcObservable<DTXQCDField> {
 public:
  DtxqcdFierzAveragingObserver(GridCartesian &grid,
                                GridRedBlackCartesian &rbgrid,
                                RealD mass, RealD csw, RealD lambda,
                                int n_skip, int n_noise_per_traj,
                                RealD cg_tol = 1e-8)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
        lambda_(lambda), cg_tol_(cg_tol), n_skip_(n_skip),
        n_noise_per_traj_(n_noise_per_traj) {}

  void TrajectoryComplete(int traj, DTXQCDField &U,
                          GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    if (traj < n_skip_) return;
    // Inter-measurement stride to reduce autocorrelation in the per-traj
    // sample list (default 5 — safer than measuring every traj since
    // HMC autocorr at light mass is several trajs).  Override with
    // FIERZ_AVG_MEAS_STRIDE=K (K=1 disables striding).
    static const int meas_stride = []() {
      if (const char *v = std::getenv("FIERZ_AVG_MEAS_STRIDE"); v && *v) {
        return std::atoi(v);
      }
      return 5;
    }();
    if (meas_stride > 1 && ((traj - n_skip_) % meas_stride) != 0) return;
    RealD V = (RealD)grid_.gSites();
    auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
    auto tr_pi    = TensorRemove(sum(trace(U.pi)));
    auto tr_s     = TensorRemove(sum(trace(U.s)));
    auto tr_p     = TensorRemove(sum(trace(U.p)));
    auto tr_d     = TensorRemove(sum(trace(U.d)));
    auto tr_n     = TensorRemove(sum(trace(U.n)));
    tr_sigma_.push_back(tr_sigma.real() / V);
    tr_pi_.push_back(tr_pi.real() / V);
    tr_s_.push_back(tr_s.real() / V);
    tr_p_.push_back(tr_p.real() / V);
    tr_d_.push_back(tr_d.real() / V);
    tr_n_.push_back(tr_n.real() / V);
    n_sigma_sq_.push_back(norm2(U.sigma) / V);
    n_s_sq_.push_back(norm2(U.s) / V);
    n_d_sq_.push_back(norm2(U.d) / V);

    GridParallelRNG noisePRNG(&grid_);
    noisePRNG.SeedFixedIntegers({1000 + 7 * traj, 1100 + 7 * traj,
                                  1200 + 7 * traj, 1300 + 7 * traj,
                                  1400 + 7 * traj});
    RealD sigma_dtx = DtxqcdFierzOpTrminv(U, grid_, rbgrid_, noisePRNG,
                                           mass_, csw_, n_noise_per_traj_,
                                           cg_tol_);
    noisePRNG.SeedFixedIntegers({2000 + 7 * traj, 2100 + 7 * traj,
                                  2200 + 7 * traj, 2300 + 7 * traj,
                                  2400 + 7 * traj});
    // Use 4× the noise count for the plain-Wilson reference so its
    // stochastic error is sub-dominant to the DTXQCD aux-side error
    // when forming the ratio.
    RealD sigma_w = DtxqcdFierzPlainWilsonTrminv(U.U, grid_, rbgrid_,
                                                  noisePRNG, mass_, csw_,
                                                  4 * n_noise_per_traj_, cg_tol_);
    sigma_dtx_.push_back(sigma_dtx);
    sigma_w_.push_back(sigma_w);
    ratio_.push_back(sigma_dtx / sigma_w);
    std::cout << GridLogMessage
              << "[FierzAvg traj " << traj << "] Σ_DTX=" << sigma_dtx
              << " Σ_W=" << sigma_w << " ratio=" << (sigma_dtx / sigma_w)
              << "  ⟨Tr s⟩=" << (tr_s.real() / V)
              << "  ⟨Tr σ⟩=" << (tr_sigma.real() / V) << std::endl;
  }

  DtxqcdFierzCheckResult finalize(RealD pass_tol,
                                   const std::string &test_name) {
    int N = (int)sigma_dtx_.size();
    if (N == 0) {
      DtxqcdFierzCheckResult r{0, 0, 0, 0, false};
      std::cout << GridLogMessage
                << "[" << test_name << " AVG] No samples — FAIL"
                << std::endl;
      return r;
    }
    auto mean = [](const std::vector<RealD> &v) {
      RealD s = 0.0;
      for (auto x : v) s += x;
      return s / v.size();
    };
    auto stdev = [](const std::vector<RealD> &v, RealD m) {
      if (v.size() < 2) return 0.0;
      RealD s = 0.0;
      for (auto x : v) s += (x - m) * (x - m);
      return std::sqrt(s / (v.size() - 1));
    };
    RealD mean_sigma_dtx = mean(sigma_dtx_);
    RealD mean_sigma_w   = mean(sigma_w_);
    RealD ratio = mean_sigma_dtx / mean_sigma_w;
    RealD ratio_std = stdev(ratio_, mean(ratio_));
    RealD ratio_se = ratio_std / std::sqrt((RealD)N);
    RealD dev = std::fabs(ratio - 1.0);
    // PASS requires BOTH: |dev| < absolute tol AND dev < 3·SE (statistical).
    bool pass_abs   = dev < pass_tol;
    bool pass_3sig  = dev < 3.0 * ratio_se;
    bool pass       = pass_abs && pass_3sig;

    // Saddle predictions for aux VEVs (free-field, isotropic):
    //   Under DN_COMPLEX_SYMMETRIC: σ is real-symmetric (NOT traceless), so
    //   it shares the singlet load with s:
    //     ⟨Tr σ⟩ = ⟨s⟩ = Nf · Σ_W / λ²
    //   Default Hermitian-traceless convention:
    //     ⟨Tr σ⟩ = 0,   ⟨s⟩ = 2 · Nf · Σ_W / λ²   (singlet absorbs σ trace)
    RealD lam2 = lambda_ * lambda_;
    RealD pred_tr_sigma, pred_tr_s;
    if (DtxqcdDnComplexSymmetric()) {
      pred_tr_sigma = (RealD)DtxqcdNf * mean_sigma_w / lam2;
      pred_tr_s     = (RealD)DtxqcdNf * mean_sigma_w / lam2;
    } else {
      pred_tr_sigma = 0.0;
      pred_tr_s     = 2.0 * (RealD)DtxqcdNf * mean_sigma_w / lam2;
    }
    RealD mean_tr_sigma = mean(tr_sigma_);
    RealD mean_tr_s     = mean(tr_s_);
    RealD se_tr_sigma = stdev(tr_sigma_, mean_tr_sigma) / std::sqrt((RealD)N);
    RealD se_tr_s     = stdev(tr_s_, mean_tr_s)         / std::sqrt((RealD)N);
    RealD dev_tr_sigma = std::fabs(mean_tr_sigma - pred_tr_sigma);
    RealD dev_tr_s     = std::fabs(mean_tr_s     - pred_tr_s);
    // VEV saddle gate: abs |dev| < 5·pass_tol·|pred| AND dev < 3·SE.
    // The abs floor is intentionally looser than the Σ_DTX/Σ_W ratio tol
    // because the aux trace is a sub-dominant observable (pred is the
    // *free-field* saddle; gauge perturbations shift the actual saddle).
    // The 3σ check kicks in once HMC explores the aux fluctuation cone.
    RealD vev_abs_tol_sigma = 5.0 * pass_tol * std::fabs(pred_tr_sigma);
    RealD vev_abs_tol_s     = 5.0 * pass_tol * std::fabs(pred_tr_s);
    bool pass_tr_sigma_abs  = dev_tr_sigma < vev_abs_tol_sigma;
    bool pass_tr_s_abs      = dev_tr_s     < vev_abs_tol_s;
    bool pass_tr_sigma_3sig = dev_tr_sigma < 3.0 * se_tr_sigma;
    bool pass_tr_s_3sig     = dev_tr_s     < 3.0 * se_tr_s;
    bool pass_tr_sigma = pass_tr_sigma_abs && pass_tr_sigma_3sig;
    bool pass_tr_s     = pass_tr_s_abs     && pass_tr_s_3sig;
    pass = pass && pass_tr_sigma && pass_tr_s;

    std::cout << GridLogMessage << std::endl
              << "===== DTXQCD Fierz averaging summary (N=" << N
              << " samples, " << n_noise_per_traj_ << " noise/traj) ====="
              << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr σ⟩ = " << mean(tr_sigma_) << " ± " << stdev(tr_sigma_, mean(tr_sigma_)) << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr π⟩ = " << mean(tr_pi_)    << " ± " << stdev(tr_pi_, mean(tr_pi_)) << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr s⟩ = " << mean(tr_s_)     << " ± " << stdev(tr_s_, mean(tr_s_)) << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr p⟩ = " << mean(tr_p_)     << " ± " << stdev(tr_p_, mean(tr_p_)) << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr d⟩ = " << mean(tr_d_)     << " ± " << stdev(tr_d_, mean(tr_d_)) << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr n⟩ = " << mean(tr_n_)     << " ± " << stdev(tr_n_, mean(tr_n_)) << std::endl;
    std::cout << GridLogMessage
              << "‖σ‖²/V = " << mean(n_sigma_sq_) << "   ‖s‖²/V = " << mean(n_s_sq_)
              << "   ‖d‖²/V = " << mean(n_d_sq_) << std::endl;
    std::cout << GridLogMessage
              << "Σ_DTXQCD = " << mean_sigma_dtx
              << " ± " << stdev(sigma_dtx_, mean_sigma_dtx) / std::sqrt((RealD)N)
              << " (SE)" << std::endl;
    std::cout << GridLogMessage
              << "Σ_W      = " << mean_sigma_w
              << " ± " << stdev(sigma_w_, mean_sigma_w) / std::sqrt((RealD)N)
              << " (SE)" << std::endl;
    std::cout << GridLogMessage
              << "ratio  Σ_DTXQCD/Σ_W = " << ratio
              << " ± " << ratio_se << " (SE)"
              << "   |dev| = " << dev
              << "   tol = " << pass_tol
              << "   abs=" << (pass_abs ? "PASS" : "FAIL")
              << "   3σ=" << (pass_3sig ? "PASS" : "FAIL")
              << std::endl;
    const char *pred_sigma_str = DtxqcdDnComplexSymmetric() ? "Nf·Σ/λ²" : "0 (traceless)";
    const char *pred_s_str     = DtxqcdDnComplexSymmetric() ? "Nf·Σ/λ²" : "2·Nf·Σ/λ²";
    std::cout << GridLogMessage
              << "⟨Tr σ⟩ saddle: obs=" << mean_tr_sigma
              << " ± " << se_tr_sigma << " (SE)  pred=" << pred_sigma_str
              << "=" << pred_tr_sigma
              << "  |dev|=" << dev_tr_sigma
              << "  abs_tol=" << vev_abs_tol_sigma
              << "  abs=" << (pass_tr_sigma_abs ? "PASS" : "FAIL")
              << "  3σ=" << (pass_tr_sigma_3sig ? "PASS" : "FAIL") << std::endl;
    std::cout << GridLogMessage
              << "⟨s⟩ saddle: obs=" << mean_tr_s
              << " ± " << se_tr_s << " (SE)  pred=" << pred_s_str
              << "=" << pred_tr_s
              << "  |dev|=" << dev_tr_s
              << "  abs_tol=" << vev_abs_tol_s
              << "  abs=" << (pass_tr_s_abs ? "PASS" : "FAIL")
              << "  3σ=" << (pass_tr_s_3sig ? "PASS" : "FAIL") << std::endl;
    std::cout << GridLogMessage
              << "[" << test_name << " AVG] " << (pass ? "PASS" : "FAIL")
              << std::endl;
    return {mean_sigma_dtx, mean_sigma_w, ratio, dev, pass};
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_, csw_, lambda_, cg_tol_;
  int n_skip_, n_noise_per_traj_;
  std::vector<RealD> sigma_dtx_, sigma_w_, ratio_;
  std::vector<RealD> tr_sigma_, tr_pi_, tr_s_, tr_p_, tr_d_, tr_n_;
  std::vector<RealD> n_sigma_sq_, n_s_sq_, n_d_sq_;
};

inline DtxqcdFierzCheckResult DtxqcdFierzCheck(DTXQCDField &U,
                                                 GridCartesian &Grid_,
                                                 GridRedBlackCartesian &RBGrid,
                                                 RealD mass, RealD csw,
                                                 int n_noise, RealD cg_tol,
                                                 RealD pass_tol,
                                                 const std::string &test_name) {
  RealD V = (RealD)Grid_.gSites();
  auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
  auto tr_pi    = TensorRemove(sum(trace(U.pi)));
  auto tr_s     = TensorRemove(sum(trace(U.s)));
  auto tr_p     = TensorRemove(sum(trace(U.p)));
  auto tr_d     = TensorRemove(sum(trace(U.d)));
  auto tr_n     = TensorRemove(sum(trace(U.n)));
  std::cout << GridLogMessage << std::endl
            << "===== DTXQCD Fierz check on final HMC state =====" << std::endl;
  std::cout << GridLogMessage << "Aux trace VEVs:"
            << "\n   ⟨Tr σ⟩ = " << (tr_sigma.real()/V) << " + i·" << (tr_sigma.imag()/V)
            << "\n   ⟨Tr π⟩ = " << (tr_pi.real()/V)    << " + i·" << (tr_pi.imag()/V)
            << "\n   ⟨Tr s⟩ = " << (tr_s.real()/V)     << " + i·" << (tr_s.imag()/V)
            << "\n   ⟨Tr p⟩ = " << (tr_p.real()/V)     << " + i·" << (tr_p.imag()/V)
            << "\n   ⟨Tr d⟩ = " << (tr_d.real()/V)     << " + i·" << (tr_d.imag()/V)
            << "\n   ⟨Tr n⟩ = " << (tr_n.real()/V)     << " + i·" << (tr_n.imag()/V)
            << std::endl;
  std::cout << GridLogMessage << "Aux ‖·‖²/V:"
            << "  ‖σ‖²=" << (norm2(U.sigma) / V)
            << "  ‖π‖²=" << (norm2(U.pi)    / V)
            << "  ‖s‖²=" << (norm2(U.s)     / V)
            << "  ‖p‖²=" << (norm2(U.p)     / V)
            << "  ‖d‖²=" << (norm2(U.d)     / V)
            << "  ‖n‖²=" << (norm2(U.n)     / V)
            << std::endl;

  GridParallelRNG noisePRNG(&Grid_);
  noisePRNG.SeedFixedIntegers({1001, 1002, 1003, 1004, 1005});
  RealD sigma_dtx = DtxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG,
                                         mass, csw, n_noise, cg_tol);
  noisePRNG.SeedFixedIntegers({2001, 2002, 2003, 2004, 2005});
  RealD sigma_w = DtxqcdFierzPlainWilsonTrminv(U.U, Grid_, RBGrid, noisePRNG,
                                                 mass, csw, n_noise, cg_tol);
  RealD ratio = sigma_dtx / sigma_w;
  RealD dev = std::fabs(ratio - 1.0);
  bool pass = (dev < pass_tol);

  std::cout << GridLogMessage << "Σ_DTXQCD (with aux) = " << sigma_dtx << std::endl;
  std::cout << GridLogMessage << "Σ_W      (plain D_W) = " << sigma_w
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage << "ratio Σ_DTXQCD / Σ_W = " << ratio
            << "   |dev| = " << dev
            << "   tol = " << pass_tol << std::endl;
  std::cout << GridLogMessage
            << "[" << test_name << "] " << (pass ? "PASS" : "FAIL") << std::endl;

  return {sigma_dtx, sigma_w, ratio, dev, pass};
}

NAMESPACE_END(Grid);
