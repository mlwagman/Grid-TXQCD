#pragma once
// Shared utilities for the DTXQCD 2pt Fierz-equivalence test suite.
// Mirror of Test_txqcd_2pt_utils.h with the field roster updated for the
// diquark-tensor variant.  Re-uses the QCD reference ensemble that
// Test_txqcd_2pt_gencfgs generates -- the DTXQCD vs QCD comparison shares
// the same QCD configs (configs_2pt_qcd_nf2), so a 2pt suite run only needs
// to generate the DTXQCD half.
//
// Configs land in configs_2pt_dtxqcd/ with the standard NERSC gauge file
// (ckpoint_lat.N) plus the DTXQCDCheckpointer aux sidecar
// (ckpoint_lat_daux.N, magic = DTXA = 0x44545841) and Grid RNG state
// (ckpoint_rng.N).

#include "../txqcd/Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxCorrelator.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/Eigen/Eigenvalues>

namespace DtxqcdTest2pt {

using namespace TxqcdTest2pt;  // shared params (beta, lambda, mass, etc.) +
                                // file helpers + correlator math + QCD diag

// DTXQCD-specific config directory.  Sits beside configs_2pt_txqcd /
// configs_2pt_qcd_nf2 so a single run of the 2pt suite produces all three
// ensembles at the same trajectory range.
inline std::string dtxqcd_cfg_dir() {
  if (const char *d = std::getenv("CFG_DIR"); d && *d) return std::string(d);
  return "configs_2pt_dtxqcd";
}

// Existence check: in addition to gauge + rng, the DTXQCD sidecar has the
// "_daux" suffix (not "_aux", which is the TXQCD sidecar).
inline bool dtxqcd_configs_exist(const std::string &dir) {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(dir + "/ckpoint_lat."      + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_lat_daux." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng."      + std::to_string(t))) return false;
  }
  return true;
}
inline bool dtxqcd_configs_exist() { return dtxqcd_configs_exist(dtxqcd_cfg_dir()); }

inline int latest_dtxqcd_checkpoint(const std::string &dir) {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    if (file_exists(dir + "/ckpoint_lat."      + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_lat_daux." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng."      + std::to_string(t)))
      latest = t;
  }
  return latest;
}
inline int latest_dtxqcd_checkpoint() {
  return latest_dtxqcd_checkpoint(dtxqcd_cfg_dir());
}

inline void LoadDtxqcdConfig(DTXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG, int traj,
                              const std::string &dir) {
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  dir + "/ckpoint_lat",
                                  dir + "/ckpoint_rng", traj);
}
inline void LoadDtxqcdConfig(DTXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG, int traj) {
  LoadDtxqcdConfig(U, sRNG, pRNG, traj, dtxqcd_cfg_dir());
}

// Per-trajectory HMC diagnostics for DTXQCD.  Mirrors TxqcdDiagnostics but
// the aux field roster is (sigma^A, pi^A, t^A, d, n) instead of TXQCD's
// (sigma, pi, s, p, t) -- so the recorded VEVs are scalar norm-squared
// sums per slot (the Pauli triplet sigma^A is not a flavor matrix where
// Tr applies, and there's no s/p field).  Tr M^{-1} on the gauge field
// works identically to QCD/TXQCD's path.
struct DtxqcdDiagnostics : HmcDiagWriter<DTXQCDField> {
  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG       &prng_;
  RealD mass_;
  RealD lambda_;
  int   n_vev_noise_;
  std::vector<RealD> norm_sigma_, norm_pi_, norm_d_, norm_n_, norm_s_, norm_p_;
  std::vector<RealD> vev_trminv_;
  // Per-traj aux wall-wall correlators (length T or Nf²·T each).  Outer dim
  // accumulates trajectories since last flush; flushed at meas_skip intervals.
  std::vector<std::vector<ComplexD>>
      aux_C_pi_plus_, aux_C_pi_minus_, aux_C_pi_zero_,
      aux_C_a0_plus_, aux_C_a0_minus_, aux_C_a0_zero_,
      aux_C_s_, aux_C_p_, aux_C_trsig_, aux_C_trpi_,
      aux_C_trsig_s_, aux_C_trpi_p_;
  std::vector<std::vector<ComplexD>>
      aux_wall_sig_ab_, aux_wall_pi_ab_,
      aux_wall_d_ab_,   aux_wall_n_ab_,
      aux_wall_s_, aux_wall_p_, aux_wall_trsig_, aux_wall_trpi_;
  // Per-traj lowest signed eigenvalues of γ5·M48 (Hermitian), tracking
  // Pfaffian sign changes.  Each row is the lowest N_EV_TRACK |λ| Ritz
  // values from a Lanczos with N_KRYLOV_TRACK Krylov dimension.
  std::vector<std::vector<RealD>> g5M_evals_;

  DtxqcdDiagnostics(const std::string &prefix, int interval,
                    std::vector<ActionRef> acts,
                    GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                    GridParallelRNG &prng, RealD mass, RealD lambda,
                    int n_vev_noise)
      : HmcDiagWriter(prefix, interval, std::move(acts), true),
        grid_(grid), rbgrid_(rbgrid), prng_(prng),
        mass_(mass), lambda_(lambda), n_vev_noise_(n_vev_noise) {}

  // Lanczos on γ5·M48 (Hermitian for our γ5-Hermitian doubled M48).
  // Returns the lowest |λ| Ritz values (signed).  Krylov dim ~40-60 is
  // sufficient at 4³×8 for the lowest 6-10 eigenvalues; cost ≈ Nm × |M|
  // applies ≈ same as ~Nm CG iter, ~0.1-0.3 s/cfg on small lattice.
  std::vector<RealD> compute_g5M_evals(DTXQCDField &U, int Nev, int Nm) {
    DTXQCDWilsonCloverFermionEO Dw(U.U, grid_, rbgrid_, mass_, /*csw=*/0.0,
                                    U.sigma, U.pi, U.d, U.n, U.s, U.p);
    auto g5m = [&](const DTXQCDFermionDoubled &x, DTXQCDFermionDoubled &y) {
      Dw.M(const_cast<DTXQCDFermionDoubled&>(x), y);
      for (int a = 0; a < DtxqcdNf; ++a) {
        y.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * y.upper.f[a];
        y.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * y.lower.f[a];
      }
    };
    auto normalize = [&](DTXQCDFermionDoubled &x) {
      RealD n = std::sqrt(norm2(x));
      if (n < 1e-30) return;
      RealD inv = 1.0 / n;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = inv * x.upper.f[a];
        x.lower.f[a] = inv * x.lower.f[a];
      }
    };
    std::vector<DTXQCDFermionDoubled> V;
    V.reserve(Nm + 1);
    V.emplace_back(&grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(prng_, V[0].upper.f[a]);
      gaussian(prng_, V[0].lower.f[a]);
    }
    normalize(V[0]);
    std::vector<RealD> alpha(Nm), beta(Nm + 1, 0.0);
    DTXQCDFermionDoubled w(&grid_);
    int Nactual = Nm;
    for (int j = 0; j < Nm; ++j) {
      g5m(V[j], w);
      if (j > 0) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          w.upper.f[a] = w.upper.f[a] - beta[j] * V[j-1].upper.f[a];
          w.lower.f[a] = w.lower.f[a] - beta[j] * V[j-1].lower.f[a];
        }
      }
      alpha[j] = innerProduct(V[j], w).real();
      for (int a = 0; a < DtxqcdNf; ++a) {
        w.upper.f[a] = w.upper.f[a] - alpha[j] * V[j].upper.f[a];
        w.lower.f[a] = w.lower.f[a] - alpha[j] * V[j].lower.f[a];
      }
      // One-pass reorthogonalization.
      for (int i = 0; i < j; ++i) {
        ComplexD c = innerProduct(V[i], w);
        for (int a = 0; a < DtxqcdNf; ++a) {
          w.upper.f[a] = w.upper.f[a] - c * V[i].upper.f[a];
          w.lower.f[a] = w.lower.f[a] - c * V[i].lower.f[a];
        }
      }
      RealD bnext = std::sqrt(norm2(w));
      if (bnext < 1e-12) { Nactual = j + 1; break; }
      beta[j + 1] = bnext;
      if (j < Nm - 1) {
        V.emplace_back(&grid_);
        RealD inv = 1.0 / bnext;
        for (int a = 0; a < DtxqcdNf; ++a) {
          V[j + 1].upper.f[a] = inv * w.upper.f[a];
          V[j + 1].lower.f[a] = inv * w.lower.f[a];
        }
      }
    }
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(Nactual, Nactual);
    for (int i = 0; i < Nactual; ++i) {
      T(i, i) = alpha[i];
      if (i + 1 < Nactual) { T(i, i+1) = beta[i+1]; T(i+1, i) = beta[i+1]; }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    Eigen::VectorXd evals = es.eigenvalues();
    std::vector<RealD> all(evals.data(), evals.data() + evals.size());
    std::sort(all.begin(), all.end(),
              [](RealD a, RealD b){ return std::abs(a) < std::abs(b); });
    int K = std::min(Nev, (int)all.size());
    return std::vector<RealD>(all.begin(), all.begin() + K);
  }

  RealD get_plaq(DTXQCDField &U) override {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  }

  // Σ_DTXQCD via Hutchinson stochastic estimator on the FULL doubled M48
  // operator (built from the current gauge + aux configuration).  This is
  // the physical ⟨q̄q⟩ that the fermion measure actually sees, in contrast
  // to the pre-2026-06-15 implementation that used plain Wilson on the
  // gauge field alone (which is only correct at λ → ∞ where aux fields
  // → 0).  Cost: one MdagM CG per Hutchinson noise.
  //
  // Normalisation: Tr M48⁻¹ runs over the doubled fermion space (2·V·Ns·Nc·Nf
  // entries for upper+lower blocks).  We return acc / (2·V) to match the
  // earlier convention where /2 accounted for the Nf factor of WilsonFermion;
  // for the doubled M48 the "Nf factor" is 2·Nf because there are 2 doubled
  // blocks each of Nf flavor, so the same /2 captures the per-quark Σ.
  // Reader: this is the natural value to compare with Σ_bare (plain QCD
  // ⟨q̄q⟩ on the same gauge) — they agree as λ → ∞.
  // Σ_DTXQCD per quark via Hutchinson on M48.  Returns
  //   Σ ≡ (1/V·2·N_F) Tr M48⁻¹
  // i.e. the full doubled trace divided out by the doubling factor
  // (2 = upper+lower blocks) and N_F (flavor copies).  This matches the
  // plain-Wilson Σ convention at aux=0 (verified by
  // Test_dtxqcd_trminv_zeroaux: ratio = 2·N_F up to noise).  The /(2V)
  // factor inside the noise loop is the standard plain-Wilson
  // normalization; the extra /(2·N_F) here makes it per-quark.
  RealD compute_trminv(DTXQCDField &U) {
    DTXQCDWilsonCloverFermionEO Dw(U.U, grid_, rbgrid_, mass_, /*csw=*/0.0,
                                    U.sigma, U.pi, U.d, U.n, U.s, U.p);
    RealD V = (RealD)grid_.gSites();
    RealD acc = 0.0;
    const RealD cg_tol = 1e-8;
    for (int h = 0; h < n_vev_noise_; ++h) {
      DTXQCDFermionDoubled eta(&grid_), b(&grid_), x(&grid_);
      DTXQCDFermionDoubled r(&grid_), p(&grid_), Ap(&grid_), tmp(&grid_);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(prng_, eta.upper.f[a]);
        gaussian(prng_, eta.lower.f[a]);
      }
      Dw.Mdag(eta, b);
      x = Zero();
      for (int a = 0; a < DtxqcdNf; ++a) {
        r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
        p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
      }
      RealD r2 = norm2(r), b2 = norm2(b);
      RealD tol2 = cg_tol * cg_tol * b2;
      // Custom CG on M†M (same pattern as the AUX_INIT bisection's helper —
      // DTXQCDFermionDoubled doesn't satisfy the Lattice<vobj> concept that
      // Grid's stock ConjugateGradient requires).
      for (int k = 0; k < cg_max; ++k) {
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
      // η† M⁻¹ η = η† x stochastically estimates Tr M⁻¹.  Divide by 2V to
      // match the prior convention (factor 2 captures the doubled-block
      // count, V is volume).
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    // Per-quark normalization: full doubled trace ÷ (2·N_F).
    return acc / (n_vev_noise_ * 2.0 * DtxqcdNf);
  }

  void record_aux(DTXQCDField &U) override {
    RealD V = (RealD)U.Grid()->gSites();
    norm_sigma_.push_back(norm2(U.sigma) / V);
    norm_pi_   .push_back(norm2(U.pi)    / V);
    norm_d_    .push_back(norm2(U.d)     / V);
    norm_n_    .push_back(norm2(U.n)     / V);
    norm_s_    .push_back(norm2(U.s)     / V);
    norm_p_    .push_back(norm2(U.p)     / V);
    vev_trminv_.push_back(compute_trminv(U));
    // Aux wall-wall correlators (collective sliceSum — must run on every
    // rank).  Cost ≪1% of a trajectory.
    DtxqcdAuxWallCorrelators awc =
        DtxqcdComputeAuxWallCorrelators(U, lambda_);
    aux_C_pi_plus_  .push_back(std::move(awc.C_pi_plus));
    aux_C_pi_minus_ .push_back(std::move(awc.C_pi_minus));
    aux_C_pi_zero_  .push_back(std::move(awc.C_pi_zero));
    aux_C_a0_plus_  .push_back(std::move(awc.C_a0_plus));
    aux_C_a0_minus_ .push_back(std::move(awc.C_a0_minus));
    aux_C_a0_zero_  .push_back(std::move(awc.C_a0_zero));
    aux_C_s_        .push_back(std::move(awc.C_s));
    aux_C_p_        .push_back(std::move(awc.C_p));
    aux_C_trsig_    .push_back(std::move(awc.C_trsig));
    aux_C_trpi_     .push_back(std::move(awc.C_trpi));
    aux_C_trsig_s_  .push_back(std::move(awc.C_trsig_s));
    aux_C_trpi_p_   .push_back(std::move(awc.C_trpi_p));
    aux_wall_sig_ab_.push_back(std::move(awc.wall_sig_ab_flat));
    aux_wall_pi_ab_ .push_back(std::move(awc.wall_pi_ab_flat));
    aux_wall_d_ab_  .push_back(std::move(awc.wall_d_ab_flat));
    aux_wall_n_ab_  .push_back(std::move(awc.wall_n_ab_flat));
    aux_wall_s_     .push_back(std::move(awc.wall_s));
    aux_wall_p_     .push_back(std::move(awc.wall_p));
    aux_wall_trsig_ .push_back(std::move(awc.wall_trsig));
    aux_wall_trpi_  .push_back(std::move(awc.wall_trpi));
    // Per-traj γ5·M48 lowest signed eigenvalues for Pfaffian sign tracking.
    // Skip if G5M_EVALS_OFF=1 (debug bypass).
    bool g5m_off = std::getenv("G5M_EVALS_OFF")
                && std::atoi(std::getenv("G5M_EVALS_OFF")) != 0;
    if (!g5m_off) {
      int Nev = 6;
      if (const char *e = std::getenv("G5M_NEV"); e && *e) Nev = std::atoi(e);
      int Nm = 40;
      if (const char *e = std::getenv("G5M_NKRYLOV"); e && *e) Nm = std::atoi(e);
      auto evs = compute_g5M_evals(U, Nev, Nm);
      g5M_evals_.push_back(evs);
      // n_neg in lowest |λ| is sort-order-stable: counts physical Pf
      // parity contribution from the K lowest eigenvalues.  |λ|_min is
      // the distance-to-zero indicator (small ⇒ Pf sign-flip imminent).
      int n_neg_low = 0;
      RealD abs_min = std::numeric_limits<RealD>::infinity();
      for (auto e : evs) {
        if (e < 0) ++n_neg_low;
        if (std::abs(e) < abs_min) abs_min = std::abs(e);
      }
      // signPf_lowK = (-1)^{n_neg_lowK}; the full signPf factor used in
      // sign reweighting is the same provided no eval outside the
      // lowest K has crossed zero (verify with wider Lanczos at
      // thermalization end if |λ|_min approaches 0).
      int signPf_lowK = (n_neg_low & 1) ? -1 : +1;
      std::cout << GridLogMessage << "[γ5M evals]";
      for (auto e : evs) std::cout << "  " << std::showpos << e << std::noshowpos;
      std::cout << "  |λ|_min=" << abs_min
                << "  n_neg_lowK=" << n_neg_low
                << "  signPf_lowK=" << std::showpos << signPf_lowK
                << std::noshowpos << std::endl;
    }
    // Per-traj aux VEV summary to stdout — singlet means and channel norms.
    // wall_* vectors have length T; mean over t gives ⟨op⟩ at λ⁰ scaling.
    auto avg = [](const std::vector<ComplexD> &v) {
      RealD acc = 0.0;
      for (const auto &z : v) acc += z.real();
      return v.empty() ? 0.0 : acc / static_cast<RealD>(v.size());
    };
    RealD V3 = static_cast<RealD>(U.Grid()->gSites()) / static_cast<RealD>(awc.T);
    RealD vev_s     = avg(aux_wall_s_.back())     / V3;
    RealD vev_p     = avg(aux_wall_p_.back())     / V3;
    RealD vev_trsig = avg(aux_wall_trsig_.back()) / V3;
    RealD vev_trpi  = avg(aux_wall_trpi_.back())  / V3;
    std::cout << GridLogMessage
              << "[DTXQCD aux VEVs]  ⟨s⟩=" << vev_s
              << "  ⟨Trσ⟩=" << vev_trsig
              << "  ⟨p⟩=" << vev_p
              << "  ⟨Trπ⟩=" << vev_trpi
              << "  ||σ||²/V=" << norm_sigma_.back()
              << "  ||π||²/V=" << norm_pi_.back()
              << "  ||d||²/V=" << norm_d_.back()
              << "  ||n||²/V=" << norm_n_.back()
              << std::endl;
  }

  void flush(int traj) override {
    if (traj_.empty()) return;
    std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
    Hdf5Writer wr(fname);
    write(wr, "traj", traj_);
    write(wr, "plaq", plaq_);
    write(wr, "force_avg", force_avg_);
    write(wr, "force_max", force_max_);
    write(wr, "fdt_avg",   fdt_avg_);
    write(wr, "fdt_max",   fdt_max_);
    write(wr, "norm_sigma", norm_sigma_);
    write(wr, "norm_pi",    norm_pi_);
    write(wr, "norm_d",     norm_d_);
    write(wr, "norm_n",     norm_n_);
    write(wr, "norm_s",     norm_s_);
    write(wr, "norm_p",     norm_p_);
    write(wr, "vev_trminv", vev_trminv_);
    // Aux wall-wall correlators (one row per traj since last flush)
    write(wr, "aux_C_pi_plus",  aux_C_pi_plus_);
    write(wr, "aux_C_pi_minus", aux_C_pi_minus_);
    write(wr, "aux_C_pi_zero",  aux_C_pi_zero_);
    write(wr, "aux_C_a0_plus",  aux_C_a0_plus_);
    write(wr, "aux_C_a0_minus", aux_C_a0_minus_);
    write(wr, "aux_C_a0_zero",  aux_C_a0_zero_);
    write(wr, "aux_C_s",        aux_C_s_);
    write(wr, "aux_C_p",        aux_C_p_);
    write(wr, "aux_C_trsig",    aux_C_trsig_);
    write(wr, "aux_C_trpi",     aux_C_trpi_);
    write(wr, "aux_C_trsig_s",  aux_C_trsig_s_);
    write(wr, "aux_C_trpi_p",   aux_C_trpi_p_);
    write(wr, "aux_wall_sig_ab", aux_wall_sig_ab_);
    write(wr, "aux_wall_pi_ab",  aux_wall_pi_ab_);
    write(wr, "aux_wall_d_ab",   aux_wall_d_ab_);
    write(wr, "aux_wall_n_ab",   aux_wall_n_ab_);
    write(wr, "aux_wall_s",      aux_wall_s_);
    write(wr, "aux_wall_p",      aux_wall_p_);
    write(wr, "aux_wall_trsig",  aux_wall_trsig_);
    write(wr, "aux_wall_trpi",   aux_wall_trpi_);
    write(wr, "g5M_evals",       g5M_evals_);
    std::vector<std::string> names;
    for (auto &a : actions_) names.push_back(a.name);
    write(wr, "action_names", names);

    traj_.clear(); plaq_.clear();
    force_avg_.clear(); force_max_.clear();
    fdt_avg_.clear();  fdt_max_.clear();
    norm_sigma_.clear(); norm_pi_.clear();
    norm_d_.clear();     norm_n_.clear();
    norm_s_.clear();     norm_p_.clear();
    vev_trminv_.clear();
    aux_C_pi_plus_.clear();  aux_C_pi_minus_.clear(); aux_C_pi_zero_.clear();
    aux_C_a0_plus_.clear();  aux_C_a0_minus_.clear(); aux_C_a0_zero_.clear();
    aux_C_s_.clear();        aux_C_p_.clear();
    aux_C_trsig_.clear();    aux_C_trpi_.clear();
    aux_C_trsig_s_.clear();  aux_C_trpi_p_.clear();
    aux_wall_sig_ab_.clear(); aux_wall_pi_ab_.clear();
    aux_wall_d_ab_.clear();   aux_wall_n_ab_.clear();
    aux_wall_s_.clear();      aux_wall_p_.clear();
    aux_wall_trsig_.clear();  aux_wall_trpi_.clear();
    g5M_evals_.clear();

    std::cout << GridLogMessage << "HMC diagnostics written to " << fname
              << std::endl;
  }
};

}  // namespace DtxqcdTest2pt
