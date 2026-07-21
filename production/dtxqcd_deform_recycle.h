// dtxqcd_deform_recycle.h -- incremental subspace-RECYCLING CGNR for the
// deformed EO wall-prop solves (the eigCG-class accelerator for deep-tent
// eta >= 0.7, 2026-07-20).
//
// PHYSICS TARGET: the measured cost of deep-tent CGNR is the dense
// MID-BAND (1e-6..1e-3 tower) ground decade-by-decade -- individually
// unresolvable (Ritz mush) but SPAN-representable.  Late-solve CG
// residual snapshots ARE approximate band vectors; accumulated across
// the 192-solve per-config workload they span the band without any
// eigensolve.  New solves start from the Galerkin projection over the
// accumulated space: x0 = U (U^dag A U)^{-1} U^dag b_n with
// A = Mpc~^dag Mpc~, removing the recycled components before CG grinds.
//
// SAFETY: init-guess deflation is UNCONDITIONALLY exact -- CG converges
// to the true solution from any x0 under the true-residual criterion.
// Approximations (t0-translated vectors, snapshot quality) cost only
// iterations, never correctness.
//
// SOURCE-CENTERED CONTOURS: the per-t_src operator changes (profile
// shift).  OnT0Change(dt) translates U by Cshift in time (the contour
// moved; per-config non-covariance is absorbed) and RECOMPUTES AU and
// H = U^dag A U exactly under the new operator (2|U| fine applies ~ 1%
// of one deep solve) -- the Galerkin data is exact at every solve.
//
// Env: DTXQCD_RECYCLE_MMAX (64) accumulated-vector cap (~150 MB each:
// U + AU pair); DTXQCD_RECYCLE_MAXADD (3) snapshots kept per solve;
// DTXQCD_RECYCLE_ORTHTOL (0.05) redundancy drop threshold;
// DTXQCD_RECYCLE_SHIFTSIGN (+1) t-translation sign (self-diagnosing via
// the capture print: if capture degrades after t0 steps, flip).
//
// Prints per solve: [recycle] dim=<|U|> capture=<|r0|/|b_n|> iters=<n>
// -- capture < 1 measures the pre-removed residual fraction.

#pragma once

#include <Eigen/Dense>

namespace Grid {

class DtxqcdDeformRecycle {
 public:
  typedef DTXQCDFermionDoubled F;

  DtxqcdDeformRecycle(DtxqcdDeformQuda &q, GridCartesian &grid,
                      GridRedBlackCartesian &rb)
      : q_(q), grid_(grid), rb_(rb) {
    mmax_ = EnvInt_("DTXQCD_RECYCLE_MMAX", 64);
    maxadd_ = EnvInt_("DTXQCD_RECYCLE_MAXADD", 3);
    orthtol_ = EnvReal_("DTXQCD_RECYCLE_ORTHTOL", 0.05);
    shiftsign_ = EnvInt_("DTXQCD_RECYCLE_SHIFTSIGN", 1);
    cur_t0_ = -1;
  }

  // A v = Mpc~^dag Mpc~ v (2 fine applies)
  void A_(const F &v, F &Av) {
    F t(&rb_);
    q_.MpcTil(const_cast<F &>(v), t);
    q_.MpcTilDag(t, Av);
  }

  static void Axpy_(F &y, ComplexD a, const F &x) {
    for (int c = 0; c < DtxqcdNf; ++c) {
      y.upper.f[c] = y.upper.f[c] + a * x.upper.f[c];
      y.lower.f[c] = y.lower.f[c] + a * x.lower.f[c];
    }
  }
  static void Scale_(F &y, RealD a) {
    for (int c = 0; c < DtxqcdNf; ++c) {
      y.upper.f[c] = a * y.upper.f[c];
      y.lower.f[c] = a * y.lower.f[c];
    }
  }
  static void Zero_(F &y, int cb) {
    for (int c = 0; c < DtxqcdNf; ++c) {
      y.upper.f[c] = Grid::Zero();
      y.lower.f[c] = Grid::Zero();
      y.upper.f[c].Checkerboard() = cb;
      y.lower.f[c].Checkerboard() = cb;
    }
  }

  // Contour stepped t0_old -> t0_new: translate the subspace with the
  // contour and rebuild the exact Galerkin data under the new operator.
  void OnT0Change(int t0_new) {
    if (cur_t0_ >= 0 && !U_.empty()) {
      int dt = (t0_new - cur_t0_);
      const int T = rb_.GlobalDimensions()[Nd - 1];
      dt = ((dt % T) + T) % T;
      // An ODD t-shift flips every site's parity: the odd-CB restriction
      // of the shifted field is EXACTLY ZERO (measured: dim wiped at the
      // first t_src step, gate 1303439).  Accumulate steps and translate
      // only by the EVEN part (a deferred 1-slice envelope offset is
      // within the translation approximation, which the exact Galerkin
      // rebuild + CG absorb regardless).
      pending_ += dt;
      pending_ %= rb_.GlobalDimensions()[Nd - 1];
      const int ev2 = pending_ - (pending_ % 2);
      if (ev2 != 0) {
        TranslateAll_(shiftsign_ * ev2);
        pending_ -= ev2;
      }
      RebuildGalerkin_();
    }
    cur_t0_ = t0_new;
  }

  // Odd-block solve, plugs into DtxqcdDeformQuda::SetOddSolver.
  int Solve(const F &b_o, F &x_o, RealD tol, int maxit, RealD &rel_out) {
    // normal-equation RHS
    F bn(&rb_), r(&rb_), p(&rb_), Ap(&rb_), t(&rb_);
    q_.MpcTilDag(const_cast<F &>(b_o), bn);
    const RealD bn2 = std::max(Fn2_(bn), 1e-300);
    // Galerkin init guess over U
    Zero_(x_o, b_o.upper.f[0].Checkerboard());
    r = bn;
    if (!U_.empty()) {
      const int M = (int)U_.size();
      Eigen::VectorXcd c(M);
      for (int k = 0; k < M; ++k) {
        const ComplexD z = Fip_(U_[k], bn);
        c(k) = std::complex<double>(DtxqcdToStd(z).real(),
                                    DtxqcdToStd(z).imag());
      }
      Eigen::VectorXcd d = Hllt_.solve(c);
      for (int k = 0; k < M; ++k) {
        const ComplexD dk(d(k).real(), d(k).imag());
        Axpy_(x_o, dk, U_[k]);
        Axpy_(r, -dk, AU_[k]);
      }
    }
    const RealD cap = std::sqrt(Fn2_(r) / bn2);
    // TRUE DEFLATED CG (Def-PCG): the projector acts INSIDE the
    // iteration (z = (1 - U H^{-1} (AU)^dag) r each step) -- init-guess
    // alone measured useless at kappa~1e8 (gate 1303508: iters flat at
    // dim=64; slow modes re-excited unless removed from the search).
    F z(&rb_);
    auto Proj_ = [&](const F &rin, F &zout) {
      zout = rin;
      const int M = (int)U_.size();
      if (M == 0) return;
      Eigen::VectorXcd c(M);
      for (int j = 0; j < M; ++j) {
        const ComplexD ip = Fip_(AU_[j], rin);
        c(j) = std::complex<double>(DtxqcdToStd(ip).real(),
                                    DtxqcdToStd(ip).imag());
      }
      Eigen::VectorXcd d = Hllt_.solve(c);
      for (int j = 0; j < M; ++j)
        Axpy_(zout, ComplexD(-d(j).real(), -d(j).imag()), U_[j]);
    };
    Proj_(r, z);
    p = z;
    RealD r2 = Fn2_(r);
    RealD rz = DtxqcdToStd(Fip_(r, z)).real();
    const RealD tol2 = tol * tol * bn2;
    int k = 0;
    int nsnap = 0;
    RealD next_decade = 1e-4;   // skip the bulk-rich first decade
    std::vector<F> snaps;
    for (; k < maxit && r2 > tol2; ++k) {
      A_(p, Ap);
      const RealD pAp = DtxqcdToStd(
          DtxqcdDeformQuda::finner_(p, Ap)).real();
      const RealD alpha = rz / std::max(pAp, 1e-300);
      Axpy_(x_o, ComplexD(alpha, 0.0), p);
      Axpy_(r, ComplexD(-alpha, 0.0), Ap);
      const RealD rn = Fn2_(r);
      const RealD relr = std::sqrt(rn / bn2);
      if (relr < next_decade && nsnap < maxadd_ &&
          (int)U_.size() + (int)snaps.size() < mmax_) {
        snaps.push_back(r);
        Scale_(snaps.back(), 1.0 / std::sqrt(rn));
        ++nsnap;
        next_decade *= 1e-2;
      }
      Proj_(r, z);
      const RealD rzn = DtxqcdToStd(Fip_(r, z)).real();
      const RealD beta = rzn / std::max(rz, 1e-300);
      for (int c2 = 0; c2 < DtxqcdNf; ++c2) {
        p.upper.f[c2] = z.upper.f[c2] + beta * p.upper.f[c2];
        p.lower.f[c2] = z.lower.f[c2] + beta * p.lower.f[c2];
      }
      r2 = rn;
      rz = rzn;
    }
    rel_out = std::sqrt(r2 / bn2);
    std::cout << GridLogMessage << "[recycle] dim=" << U_.size()
              << " capture=" << cap << " iters=" << k << std::endl;
    // harvest snapshots
    for (auto &s : snaps) Append_(s);
    return k;
  }

 private:
  void TranslateAll_(int sh) {
    GridBase *full = &grid_;
    for (auto &u : U_) {
      for (int c = 0; c < DtxqcdNf; ++c) {
        LatticeFermion fu(full), fs(full);
        fu = Zero();
        setCheckerboard(fu, u.upper.f[c]);
        fs = Cshift(fu, Nd - 1, sh);
        pickCheckerboard(Odd, u.upper.f[c], fs);
        fu = Zero();
        setCheckerboard(fu, u.lower.f[c]);
        fs = Cshift(fu, Nd - 1, sh);
        pickCheckerboard(Odd, u.lower.f[c], fs);
      }
      // renormalize (odd-dt shifts halve the surviving support)
      const RealD n = Fn2_(u);
      if (n > 1e-12) Scale_(u, 1.0 / std::sqrt(n));
    }
    // drop near-null survivors
    std::vector<F> keep;
    for (auto &u : U_)
      if (Fn2_(u) > 0.5) keep.push_back(u);
    U_.swap(keep);
  }
  void RebuildGalerkin_() {
    const int M = (int)U_.size();
    AU_.clear();
    if (M == 0) {
      H_.resize(0, 0);   // stale-H crash guard (Eigen assert, 1303500)
      return;
    }
    // re-orthonormalize U (modified Gram-Schmidt) under the plain inner
    // product, then exact AU and H under the CURRENT operator.
    std::vector<F> V;
    for (auto &u : U_) {
      F v = u;
      for (auto &w : V) {
        const ComplexD ip = Fip_(w, v);
        Axpy_(v, -ip, w);
      }
      const RealD n = std::sqrt(Fn2_(v));
      if (n > orthtol_) {
        Scale_(v, 1.0 / n);
        V.push_back(v);
      }
    }
    U_.swap(V);
    const int M2 = (int)U_.size();
    H_.resize(M2, M2);
    for (int k = 0; k < M2; ++k) {
      F Av(&rb_);
      A_(U_[k], Av);
      AU_.push_back(Av);
      for (int j = 0; j <= k; ++j) {
        const ComplexD z = Fip_(U_[j], AU_[k]);
        H_(j, k) = std::complex<double>(DtxqcdToStd(z).real(),
                                        DtxqcdToStd(z).imag());
        H_(k, j) = std::conj(H_(j, k));
      }
    }
    Hllt_.compute(H_);
    std::cout << GridLogMessage << "[recycle] Galerkin rebuilt: dim=" << M2
              << std::endl;
  }
  void Append_(F &s) {
    // orthogonalize vs U, drop if redundant, extend AU/H incrementally
    F v = s;
    for (auto &w : U_) {
      const ComplexD ip = Fip_(w, v);
      Axpy_(v, -ip, w);
    }
    const RealD n = std::sqrt(Fn2_(v));
    if (n < orthtol_) return;
    Scale_(v, 1.0 / n);
    F Av(&rb_);
    A_(v, Av);
    U_.push_back(v);
    AU_.push_back(Av);
    const int M = (int)U_.size();
    Eigen::MatrixXcd Hn(M, M);
    Hn.topLeftCorner(M - 1, M - 1) = H_;
    for (int j = 0; j < M; ++j) {
      const ComplexD z = Fip_(U_[j], AU_[M - 1]);
      Hn(j, M - 1) = std::complex<double>(DtxqcdToStd(z).real(),
                                          DtxqcdToStd(z).imag());
      Hn(M - 1, j) = std::conj(Hn(j, M - 1));
    }
    H_ = Hn;
    Hllt_.compute(H_);
  }
  static RealD Fn2_(const F &v) { return DtxqcdDeformQuda::fnorm2_(v); }
  static ComplexD Fip_(const F &u, const F &v) {
    return DtxqcdDeformQuda::finner_(u, v);
  }
  static int EnvInt_(const char *n, int d) {
    const char *v = std::getenv(n);
    return (v && *v) ? std::atoi(v) : d;
  }
  static RealD EnvReal_(const char *n, RealD d) {
    const char *v = std::getenv(n);
    return (v && *v) ? std::atof(v) : d;
  }

  DtxqcdDeformQuda &q_;
  GridCartesian &grid_;
  GridRedBlackCartesian &rb_;
  std::vector<F> U_, AU_;
  Eigen::MatrixXcd H_;
  Eigen::LLT<Eigen::MatrixXcd> Hllt_;
  int mmax_, maxadd_, shiftsign_, cur_t0_;
  int pending_ = 0;
  RealD orthtol_;
};

}  // namespace Grid
