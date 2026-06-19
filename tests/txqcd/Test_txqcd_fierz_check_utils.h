// Shared Fierz-check helpers for the free-field unit tests:
//
//   • Σ_TX = Tr[M_TX^{-1}] / (V · Nf)   (per-quark, /(2V) Hutchinson convention)
//   • Σ_W  = Tr[D_W^{-1}] / V           (plain Wilson at the same U)
//   • do_fierz_check: stochastic ratio + PASS/FAIL on |ratio - 1| < tol
//
// All operators are constructed with APBC time (chroma physics convention,
// matching TXQCDWilsonOp::DefaultImplParams() and
// TXQCDWilsonCloverFermionEO::DefaultImplParams()).
//
// Header so it can be shared between Test_txqcd_freefield_qbarq and
// Test_txqcd_freefield_qbarq_light (and any future variants).

#pragma once

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/utils/GaugeGroup.h>
#include <Grid/parallelIO/NerscIO.h>

NAMESPACE_BEGIN(Grid);

// Initialize the frozen gauge field for the Fierz unit tests.  Aux fields
// are assumed already initialized by the caller (TXQCDCompositeImpl::
// ColdConfiguration + FillAuxFields are the usual sequence).
//
// Mode is taken from the GAUGE_INIT env knob:
//   "cold"        — U_μ(x) = I (default, current behaviour)
//   "tepid"       — Grid's TepidConfiguration (small Lie-algebra noise,
//                    amp=0.01); a weak-field stress test
//   "tepid:AMP"   — explicit Lie-randomize amplitude (e.g. tepid:0.05)
//   "hot"         — Grid's HotConfiguration (uniform random SU(3))
//   "nersc:PATH"  — read NERSC-format config at PATH
//
// Returns the average plaquette so the caller can log it.
inline RealD TxqcdInitFrozenGauge(GridParallelRNG &pRNG, TXQCDField &U) {
  const char *env = std::getenv("GAUGE_INIT");
  std::string mode = (env && *env) ? env : "cold";
  if (mode == "cold") {
    U.U = 1.0;
  } else if (mode == "hot") {
    SU<Nc>::HotConfiguration(pRNG, U.U);
  } else if (mode.rfind("tepid", 0) == 0) {
    RealD amp = 0.01;  // Grid default
    if (mode.size() > 5 && mode[5] == ':') amp = std::atof(mode.c_str() + 6);
    typedef LatticeColourMatrix LCM;
    LCM Umu(U.U.Grid());
    for (int mu = 0; mu < Nd; ++mu) {
      SU<Nc>::LieRandomize(pRNG, Umu, amp);
      PokeIndex<LorentzIndex>(U.U, Umu, mu);
    }
    std::cout << GridLogMessage << "TxqcdInitFrozenGauge: tepid amp=" << amp << std::endl;
  } else if (mode.rfind("nersc:", 0) == 0) {
    std::string path = mode.substr(6);
    FieldMetaData header;
    NerscIO::readConfiguration(U.U, header, path);
    std::cout << GridLogMessage
              << "TxqcdInitFrozenGauge: loaded NERSC " << path << std::endl;
  } else {
    std::cerr << "TxqcdInitFrozenGauge: unknown GAUGE_INIT='"
              << mode << "'.  Falling back to cold." << std::endl;
    U.U = 1.0;
  }
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage
            << "TxqcdInitFrozenGauge: mode=" << mode << " plaq=" << plaq
            << " (1.0 = cold, < 1 = perturbed)" << std::endl;
  return plaq;
}

inline WilsonImplR::ImplParams TxqcdFierzApbcImplParams() {
  WilsonImplR::ImplParams p;
  p.boundary_phases.resize(Nd, 1.0);
  p.boundary_phases[Nd - 1] = -1.0;
  return p;
}

// Stochastic Tr[D_W^{-1}] / V on plain Wilson at the given U, APBC time.
// At csw != 0, dispatches to WilsonCloverFermion (csw_r = csw_t = csw).
template <class Op>
inline RealD TxqcdFierzWilsonLikeTrminv_(Op &Dw, GridCartesian &Grid_,
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

inline RealD TxqcdFierzPlainWilsonTrminv(LatticeGaugeField &U,
                                          GridCartesian &Grid_,
                                          GridRedBlackCartesian &RBGrid,
                                          GridParallelRNG &prng,
                                          RealD mass, RealD csw,
                                          int n_noise, RealD cg_tol) {
  auto impl_p = TxqcdFierzApbcImplParams();
  if (csw == 0.0) {
    WilsonFermionD Dw(U, Grid_, RBGrid, mass, impl_p);
    return TxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  } else {
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(U, Grid_, RBGrid, mass, csw, csw,
            WilsonAnisotropyCoefficients(), impl_p);
    return TxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  }
}

// Stochastic Tr[M_TX^{-1}] / (V · Nf) on the doubled TXQCD Wilson-clover.
// Per-quark normalization → matches plain_wilson_trminv at aux=0.
inline RealD TxqcdFierzOpTrminv(TXQCDField &U,
                                 GridCartesian &Grid_,
                                 GridRedBlackCartesian &RBGrid,
                                 GridParallelRNG &prng,
                                 RealD mass, RealD csw,
                                 int n_noise, RealD cg_tol) {
  std::array<RealD, TxqcdNf> mass_arr;
  for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass;
  TXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const int cg_max_iter = 30000;
  for (int h = 0; h < n_noise; ++h) {
    TXQCDFermionNf eta(&Grid_), b(&Grid_), x(&Grid_);
    TXQCDFermionNf r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(prng, eta.f[a]);
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < TxqcdNf; ++a) {
      r.f[a] = b.f[a];
      p.f[a] = b.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max_iter; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < TxqcdNf; ++a) {
        x.f[a] = x.f[a] + alpha * p.f[a];
        r.f[a] = r.f[a] - alpha * Ap.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta * p.f[a];
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / (n_noise * (RealD)TxqcdNf);
}

struct TxqcdFierzCheckResult {
  RealD sigma_tx;
  RealD sigma_w;
  RealD ratio;
  RealD dev;
  bool pass;
};

// In-line averaging observer.  Runs a small Fierz check on every HMC
// trajectory once burn-in is past, accumulating per-traj samples of
// Σ_TX, Σ_W, and the aux trace VEVs.  Finalize() returns the
// ensemble-averaged ratio with cfg-fluctuation noise folded in.
class TxqcdFierzAveragingObserver : public HmcObservable<TXQCDField> {
 public:
  TxqcdFierzAveragingObserver(GridCartesian &grid,
                               GridRedBlackCartesian &rbgrid,
                               RealD mass, RealD csw, RealD lambda,
                               int n_skip, int n_noise_per_traj,
                               RealD cg_tol = 1e-8)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
        lambda_(lambda), cg_tol_(cg_tol), n_skip_(n_skip),
        n_noise_per_traj_(n_noise_per_traj) {}

  void TrajectoryComplete(int traj, TXQCDField &U,
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
    tr_sigma_.push_back(tr_sigma.real() / V);
    tr_pi_.push_back(tr_pi.real() / V);
    tr_s_.push_back(tr_s.real() / V);
    tr_p_.push_back(tr_p.real() / V);
    n_sigma_sq_.push_back(norm2(U.sigma) / V);
    n_s_sq_.push_back(norm2(U.s) / V);
    n_t_sq_.push_back(norm2(U.t) / V);

    GridParallelRNG noisePRNG(&grid_);
    noisePRNG.SeedFixedIntegers({1000 + 7 * traj, 1100 + 7 * traj,
                                  1200 + 7 * traj, 1300 + 7 * traj,
                                  1400 + 7 * traj});
    RealD sigma_tx = TxqcdFierzOpTrminv(U, grid_, rbgrid_, noisePRNG,
                                         mass_, csw_, n_noise_per_traj_,
                                         cg_tol_);
    noisePRNG.SeedFixedIntegers({2000 + 7 * traj, 2100 + 7 * traj,
                                  2200 + 7 * traj, 2300 + 7 * traj,
                                  2400 + 7 * traj});
    // Use 4× the noise count for the plain-Wilson reference so its
    // stochastic error is sub-dominant to the TXQCD aux-side error.
    RealD sigma_w = TxqcdFierzPlainWilsonTrminv(U.U, grid_, rbgrid_,
                                                 noisePRNG, mass_, csw_,
                                                 4 * n_noise_per_traj_, cg_tol_);
    sigma_tx_.push_back(sigma_tx);
    sigma_w_.push_back(sigma_w);
    ratio_.push_back(sigma_tx / sigma_w);
    std::cout << GridLogMessage
              << "[FierzAvg traj " << traj << "] Σ_TX=" << sigma_tx
              << " Σ_W=" << sigma_w << " ratio=" << (sigma_tx / sigma_w)
              << "  ⟨Tr s⟩=" << (tr_s.real() / V)
              << "  ⟨Tr σ⟩=" << (tr_sigma.real() / V) << std::endl;
  }

  TxqcdFierzCheckResult finalize(RealD pass_tol,
                                  const std::string &test_name) {
    int N = (int)sigma_tx_.size();
    if (N == 0) {
      TxqcdFierzCheckResult r{0, 0, 0, 0, false};
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
    RealD mean_sigma_tx = mean(sigma_tx_);
    RealD mean_sigma_w  = mean(sigma_w_);
    RealD ratio = mean_sigma_tx / mean_sigma_w;
    RealD ratio_std = stdev(ratio_, mean(ratio_));
    RealD ratio_se = ratio_std / std::sqrt((RealD)N);
    RealD dev = std::fabs(ratio - 1.0);
    // PASS requires BOTH: |dev| < absolute tol AND dev < 3·SE (statistical).
    bool pass_abs   = dev < pass_tol;
    bool pass_3sig  = dev < 3.0 * ratio_se;
    bool pass       = pass_abs && pass_3sig;

    // Saddle predictions for aux VEVs (free-field, isotropic flavor/color):
    //   ⟨Tr σ⟩ = Nf · Σ_W / λ²            (flavor Hermitian Nf×Nf)
    //   ⟨Tr s⟩ = Nf · Σ_W / (√2 · λ²)     (color Hermitian Nc×Nc)
    //          = ⟨Tr σ⟩ / √2
    RealD lam2 = lambda_ * lambda_;
    RealD pred_tr_sigma = (RealD)TxqcdNf * mean_sigma_w / lam2;
    RealD pred_tr_s     = pred_tr_sigma / std::sqrt(2.0);
    RealD mean_tr_sigma = mean(tr_sigma_);
    RealD mean_tr_s     = mean(tr_s_);
    RealD se_tr_sigma = stdev(tr_sigma_, mean_tr_sigma) / std::sqrt((RealD)N);
    RealD se_tr_s     = stdev(tr_s_, mean_tr_s)         / std::sqrt((RealD)N);
    RealD dev_tr_sigma = std::fabs(mean_tr_sigma - pred_tr_sigma);
    RealD dev_tr_s     = std::fabs(mean_tr_s     - pred_tr_s);
    // Pass tolerance for VEV saddle: max(5·SE, pass_tol·|pred|)
    RealD tol_tr_sigma = std::max(5.0 * se_tr_sigma, pass_tol * std::fabs(pred_tr_sigma));
    RealD tol_tr_s     = std::max(5.0 * se_tr_s,     pass_tol * std::fabs(pred_tr_s));
    bool pass_tr_sigma = dev_tr_sigma < tol_tr_sigma;
    bool pass_tr_s     = dev_tr_s     < tol_tr_s;
    pass = pass && pass_tr_sigma && pass_tr_s;

    std::cout << GridLogMessage << std::endl
              << "===== TXQCD Fierz averaging summary (N=" << N
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
              << "‖σ‖²/V = " << mean(n_sigma_sq_) << "   ‖s‖²/V = " << mean(n_s_sq_)
              << "   ‖t‖²/V = " << mean(n_t_sq_) << std::endl;
    std::cout << GridLogMessage
              << "Σ_TX = " << mean_sigma_tx
              << " ± " << stdev(sigma_tx_, mean_sigma_tx) / std::sqrt((RealD)N)
              << " (SE)" << std::endl;
    std::cout << GridLogMessage
              << "Σ_W  = " << mean_sigma_w
              << " ± " << stdev(sigma_w_, mean_sigma_w) / std::sqrt((RealD)N)
              << " (SE)" << std::endl;
    std::cout << GridLogMessage
              << "ratio  Σ_TX/Σ_W = " << ratio
              << " ± " << ratio_se << " (SE)"
              << "   |dev| = " << dev
              << "   tol = " << pass_tol
              << "   abs=" << (pass_abs ? "PASS" : "FAIL")
              << "   3σ=" << (pass_3sig ? "PASS" : "FAIL")
              << std::endl;
    // Aux VEV saddle checks (lambda=" << lambda_ << ", Nf=" << TxqcdNf << ")
    std::cout << GridLogMessage
              << "⟨Tr σ⟩ saddle: obs=" << mean_tr_sigma
              << " ± " << se_tr_sigma << " (SE)  pred=Nf·Σ/λ²=" << pred_tr_sigma
              << "  |dev|=" << dev_tr_sigma << "  tol=" << tol_tr_sigma
              << "  [" << (pass_tr_sigma ? "PASS" : "FAIL") << "]" << std::endl;
    std::cout << GridLogMessage
              << "⟨Tr s⟩ saddle: obs=" << mean_tr_s
              << " ± " << se_tr_s << " (SE)  pred=Nf·Σ/(√2·λ²)=" << pred_tr_s
              << "  |dev|=" << dev_tr_s << "  tol=" << tol_tr_s
              << "  [" << (pass_tr_s ? "PASS" : "FAIL") << "]" << std::endl;
    std::cout << GridLogMessage
              << "[" << test_name << " AVG] " << (pass ? "PASS" : "FAIL")
              << std::endl;
    return {mean_sigma_tx, mean_sigma_w, ratio, dev, pass};
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_, csw_, lambda_, cg_tol_;
  int n_skip_, n_noise_per_traj_;
  std::vector<RealD> sigma_tx_, sigma_w_, ratio_;
  std::vector<RealD> tr_sigma_, tr_pi_, tr_s_, tr_p_;
  std::vector<RealD> n_sigma_sq_, n_s_sq_, n_t_sq_;
};

// Run the full Σ_TX vs Σ_W check on the equilibrated state.  Prints all
// diagnostics (aux trace VEVs, ‖·‖²/V per channel, Σ_TX, Σ_W, ratio, PASS/FAIL).
inline TxqcdFierzCheckResult TxqcdFierzCheck(TXQCDField &U,
                                              GridCartesian &Grid_,
                                              GridRedBlackCartesian &RBGrid,
                                              RealD mass, RealD csw,
                                              int n_noise, RealD cg_tol,
                                              RealD pass_tol,
                                              const std::string &test_name) {
  RealD V = (RealD)Grid_.gSites();

  // Aux trace VEVs (real + imag for parity / Hermitian canaries).
  auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
  auto tr_pi    = TensorRemove(sum(trace(U.pi)));
  auto tr_s     = TensorRemove(sum(trace(U.s)));
  auto tr_p     = TensorRemove(sum(trace(U.p)));
  std::cout << GridLogMessage << std::endl
            << "===== Fierz check on final HMC state =====" << std::endl;
  std::cout << GridLogMessage << "Aux trace VEVs:"
            << "\n   ⟨Tr σ⟩ = " << (tr_sigma.real()/V) << " + i·" << (tr_sigma.imag()/V)
            << "\n   ⟨Tr π⟩ = " << (tr_pi.real()/V)    << " + i·" << (tr_pi.imag()/V)
            << "\n   ⟨Tr s⟩ = " << (tr_s.real()/V)     << " + i·" << (tr_s.imag()/V)
            << "\n   ⟨Tr p⟩ = " << (tr_p.real()/V)     << " + i·" << (tr_p.imag()/V)
            << std::endl;
  std::cout << GridLogMessage << "Aux ‖·‖²/V:"
            << "  ‖σ‖²=" << (norm2(U.sigma) / V)
            << "  ‖π‖²=" << (norm2(U.pi)    / V)
            << "  ‖s‖²=" << (norm2(U.s)     / V)
            << "  ‖p‖²=" << (norm2(U.p)     / V)
            << "  ‖t‖²=" << (norm2(U.t)     / V)
            << std::endl;

  // Independent noise streams for σ_TX and σ_W.
  GridParallelRNG noisePRNG(&Grid_);
  noisePRNG.SeedFixedIntegers({1001, 1002, 1003, 1004, 1005});
  RealD sigma_tx = TxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG,
                                       mass, csw, n_noise, cg_tol);
  noisePRNG.SeedFixedIntegers({2001, 2002, 2003, 2004, 2005});
  RealD sigma_w = TxqcdFierzPlainWilsonTrminv(U.U, Grid_, RBGrid, noisePRNG,
                                                mass, csw, n_noise, cg_tol);
  RealD ratio = sigma_tx / sigma_w;
  RealD dev = std::fabs(ratio - 1.0);
  bool pass = (dev < pass_tol);

  std::cout << GridLogMessage << "Σ_TX (with aux)  = " << sigma_tx << std::endl;
  std::cout << GridLogMessage << "Σ_W  (plain D_W) = " << sigma_w
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage << "ratio Σ_TX / Σ_W = " << ratio
            << "   |dev| = " << dev
            << "   tol = " << pass_tol << std::endl;
  std::cout << GridLogMessage
            << "[" << test_name << "] " << (pass ? "PASS" : "FAIL") << std::endl;

  return {sigma_tx, sigma_w, ratio, dev, pass};
}

NAMESPACE_END(Grid);
