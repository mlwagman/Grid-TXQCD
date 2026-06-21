#pragma once
// Production per-trajectory HMC observer for the DTXQCD generator.
//
// Self-contained port of DtxqcdTest2pt::DtxqcdDiagnostics (tests/dtxqcd/
// Test_dtxqcd_2pt_utils.h) into production-grade, csw-parametrized form.  It is
// included by production/gen_dtxqcd_cfgs.cc and writes one HDF5 file per
// checkpoint interval at  out_prefix + "." + traj + ".h5"  with the dataset
// layout consumed by production/analyze_sign_reweighting.py
// (load_diagnostics reads `traj`, `plaq`, `g5M_evals`).
//
// What it records each trajectory:
//   - plaquette, per-action force/Fdt norm summaries (from Action base
//     accessors deriv_norm_average / deriv_max_average / Fdt_*_average);
//   - aux-field VEV diagnostics (||sigma||^2/V, ..., singlet means via the
//     wall slices);
//   - Tr M48^{-1} per quark (Hutchinson on the full doubled operator);
//   - aux wall-wall correlators (DtxqcdComputeAuxWallCorrelators);
//   - the lowest signed eigenvalues of gamma5 . M48 (Lanczos) -- the
//     Pfaffian-sign tracker.  signPf_n = (-1)^{n_neg(g5M_evals[n])} is the
//     reweighting observable; analyze_sign_reweighting.py derives <signPf>.
//
// Unlike the test version (which hardcodes csw=0 in the operator
// construction), this observer takes csw as a constructor argument and builds
// DTXQCDWilsonCloverFermionEO with the real csw.  The EO operator defaults to
// APBC in time (DTXQCDMeooeDoubled::DefaultImplParams, the DTXQCD physics
// convention), so the operator constructed here matches the HMC fermion action.
//
// nvcc-compat: under GRID_CUDA, Grid's ComplexD is thrust::complex, which has
// no std::conj/abs/arg and cannot be mixed with Eigen std::complex matrices.
// The gamma5.M48 Lanczos uses Eigen only for the small real symmetric
// tridiagonal solve (Eigen::MatrixXd / SelfAdjointEigenSolver, all real), so
// no complex<->Eigen conversion is needed here.  No peekSite inside any
// thread_for.  These follow the same patterns as the committed DTXQCD GPU port.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxCorrelator.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace TXQCDProduction {

using namespace Grid;

// Per-trajectory DTXQCD HMC diagnostics observer.  Self-contained: derives
// directly from Grid::HmcObservable<DTXQCDField> (the test-tree HmcDiagWriter
// base functionality -- plaquette + force/Fdt norm bookkeeping + the HDF5
// flush -- is lifted in here, so nothing from tests/ is included).
class DtxqcdDiagnostics : public Grid::HmcObservable<Grid::DTXQCDField> {
 public:
  // Maximum CG iterations for the Tr M48^{-1} Hutchinson estimator.  Local to
  // this header to keep it self-contained (the test version pulled `cg_max`
  // from the shared test params namespace).
  static constexpr int kCgMax = 30000;

  DtxqcdDiagnostics(const std::string &out_prefix, int interval,
                    std::vector<Grid::Action<Grid::DTXQCDField> *> actions,
                    Grid::GridCartesian &Grid_,
                    Grid::GridRedBlackCartesian &RBGrid_,
                    Grid::GridParallelRNG &pRNG_,
                    Grid::RealD mass, Grid::RealD csw, Grid::RealD lambda,
                    int n_vev_noise)
      : prefix_(out_prefix), interval_(interval),
        actions_(std::move(actions)),
        grid_(Grid_), rbgrid_(RBGrid_), prng_(pRNG_),
        mass_(mass), csw_(csw), lambda_(lambda),
        n_vev_noise_(n_vev_noise) {}

  // ---- HmcObservable interface --------------------------------------------
  void TrajectoryComplete(int traj, Grid::DTXQCDField &U,
                          Grid::GridSerialRNG &sRNG,
                          Grid::GridParallelRNG &pRNG) override {
    (void)sRNG;
    (void)pRNG;
    traj_.push_back(traj);
    RealD plaq_now = get_plaq(U);
    plaq_.push_back(plaq_now);
    std::cout << GridLogMessage << "Traj " << traj
              << " plaq = " << plaq_now << std::endl;

    int na = (int)actions_.size();
    std::vector<RealD> fa(na), fm(na), fdta(na), fdtm(na);
    for (int i = 0; i < na; ++i) {
      fa[i]   = actions_[i]->deriv_norm_average();
      fm[i]   = actions_[i]->deriv_max_average();
      fdta[i] = actions_[i]->Fdt_norm_average();
      fdtm[i] = actions_[i]->Fdt_max_average();
    }
    force_avg_.push_back(std::move(fa));
    force_max_.push_back(std::move(fm));
    fdt_avg_.push_back(std::move(fdta));
    fdt_max_.push_back(std::move(fdtm));

    record_aux(U);

    if (interval_ > 0 && traj % interval_ == 0) flush(traj);
  }

 protected:
  RealD get_plaq(DTXQCDField &U) {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  }

  // Lowest signed eigenvalues of gamma5 . M48 (Hermitian for the gamma5-
  // Hermitian doubled operator).  Lanczos with one-pass reorthogonalization;
  // Eigen used only for the small real tridiagonal eigen-solve.  Ported from
  // DtxqcdTest2pt::DtxqcdDiagnostics::compute_g5M_evals, with csw passed
  // through (the test hardcoded csw=0).
  std::vector<RealD> compute_g5M_evals(DTXQCDField &U, int Nev, int Nm) {
    DTXQCDWilsonCloverFermionEO Dw(U.U, grid_, rbgrid_, mass_, csw_,
                                   U.sigma, U.pi, U.d, U.n, U.s, U.p);
    auto g5m = [&](const DTXQCDFermionDoubled &x, DTXQCDFermionDoubled &y) {
      Dw.M(const_cast<DTXQCDFermionDoubled &>(x), y);
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
          w.upper.f[a] = w.upper.f[a] - beta[j] * V[j - 1].upper.f[a];
          w.lower.f[a] = w.lower.f[a] - beta[j] * V[j - 1].lower.f[a];
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
      if (i + 1 < Nactual) { T(i, i + 1) = beta[i + 1]; T(i + 1, i) = beta[i + 1]; }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    Eigen::VectorXd evals = es.eigenvalues();
    std::vector<RealD> all(evals.data(), evals.data() + evals.size());
    std::sort(all.begin(), all.end(),
              [](RealD a, RealD b) { return std::abs(a) < std::abs(b); });
    int K = std::min(Nev, (int)all.size());
    return std::vector<RealD>(all.begin(), all.begin() + K);
  }

  // Sigma_DTXQCD per quark via Hutchinson on the full doubled M48 operator.
  // Returns (1/V . 2 . N_F) Tr M48^{-1}.  Ported verbatim from the test
  // version, with csw passed through.
  RealD compute_trminv(DTXQCDField &U) {
    DTXQCDWilsonCloverFermionEO Dw(U.U, grid_, rbgrid_, mass_, csw_,
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
      // Custom CG on M^dag M -- DTXQCDFermionDoubled is not a Lattice<vobj>,
      // so Grid's stock ConjugateGradient cannot be used directly.
      for (int k = 0; k < kCgMax; ++k) {
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
    return acc / (n_vev_noise_ * 2.0 * DtxqcdNf);
  }

  void record_aux(DTXQCDField &U) {
    RealD V = (RealD)U.Grid()->gSites();
    norm_sigma_.push_back(norm2(U.sigma) / V);
    norm_pi_   .push_back(norm2(U.pi)    / V);
    norm_d_    .push_back(norm2(U.d)     / V);
    norm_n_    .push_back(norm2(U.n)     / V);
    norm_s_    .push_back(norm2(U.s)     / V);
    norm_p_    .push_back(norm2(U.p)     / V);
    vev_trminv_.push_back(compute_trminv(U));

    // Aux wall-wall correlators (collective sliceSum -- runs on every rank).
    DtxqcdAuxWallCorrelators awc = DtxqcdComputeAuxWallCorrelators(U, lambda_);
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

    // Per-traj gamma5 . M48 lowest signed eigenvalues for Pfaffian-sign
    // tracking.  Skip if G5M_EVALS_OFF=1 (debug bypass).
    bool g5m_off = std::getenv("G5M_EVALS_OFF")
                && std::atoi(std::getenv("G5M_EVALS_OFF")) != 0;
    if (!g5m_off) {
      int Nev = 6;
      if (const char *e = std::getenv("G5M_NEV"); e && *e) Nev = std::atoi(e);
      int Nm = 40;
      if (const char *e = std::getenv("G5M_NKRYLOV"); e && *e) Nm = std::atoi(e);
      auto evs = compute_g5M_evals(U, Nev, Nm);
      g5M_evals_.push_back(evs);
      int n_neg_low = 0;
      RealD abs_min = std::numeric_limits<RealD>::infinity();
      for (auto e : evs) {
        if (e < 0) ++n_neg_low;
        if (std::abs(e) < abs_min) abs_min = std::abs(e);
      }
      int signPf_lowK = (n_neg_low & 1) ? -1 : +1;
      std::cout << GridLogMessage << "[gamma5M evals]";
      for (auto e : evs) std::cout << "  " << std::showpos << e << std::noshowpos;
      std::cout << "  |lambda|_min=" << abs_min
                << "  n_neg_lowK=" << n_neg_low
                << "  signPf_lowK=" << std::showpos << signPf_lowK
                << std::noshowpos << std::endl;
    }

    // Per-traj aux VEV summary to stdout.
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
              << "[DTXQCD aux VEVs]  <s>=" << vev_s
              << "  <Trsig>=" << vev_trsig
              << "  <p>=" << vev_p
              << "  <Trpi>=" << vev_trpi
              << "  ||sigma||^2/V=" << norm_sigma_.back()
              << "  ||pi||^2/V=" << norm_pi_.back()
              << "  ||d||^2/V=" << norm_d_.back()
              << "  ||n||^2/V=" << norm_n_.back()
              << std::endl;
  }

  void flush(int traj) {
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
    for (auto *a : actions_) names.push_back(a->action_name());
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

 private:
  std::string prefix_;
  int interval_;
  std::vector<Grid::Action<Grid::DTXQCDField> *> actions_;
  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG       &prng_;
  RealD mass_, csw_, lambda_;
  int   n_vev_noise_;

  // Per-traj scalar series (flushed every `interval_` trajectories).
  std::vector<int>   traj_;
  std::vector<RealD> plaq_;
  std::vector<std::vector<RealD>> force_avg_, force_max_;
  std::vector<std::vector<RealD>> fdt_avg_, fdt_max_;
  std::vector<RealD> norm_sigma_, norm_pi_, norm_d_, norm_n_, norm_s_, norm_p_;
  std::vector<RealD> vev_trminv_;
  std::vector<std::vector<ComplexD>>
      aux_C_pi_plus_, aux_C_pi_minus_, aux_C_pi_zero_,
      aux_C_a0_plus_, aux_C_a0_minus_, aux_C_a0_zero_,
      aux_C_s_, aux_C_p_, aux_C_trsig_, aux_C_trpi_,
      aux_C_trsig_s_, aux_C_trpi_p_;
  std::vector<std::vector<ComplexD>>
      aux_wall_sig_ab_, aux_wall_pi_ab_,
      aux_wall_d_ab_,   aux_wall_n_ab_,
      aux_wall_s_, aux_wall_p_, aux_wall_trsig_, aux_wall_trpi_;
  std::vector<std::vector<RealD>> g5M_evals_;
};

}  // namespace TXQCDProduction
