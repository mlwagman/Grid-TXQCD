#pragma once
// CG and multi-shift CG solvers for TXQCDFermionNf.
//
// Grid's stock ConjugateGradient uses autoView/accelerator_for which require
// Lattice<vobj> internals. TXQCDFermionNf is a struct of Nf LatticeFermion,
// not a Lattice type, so these solvers provide equivalent functionality using
// the free functions defined for TXQCDFermionNf (innerProduct, norm2, etc.).
//
// Both solvers operate through LinearOperatorBase<TXQCDFermionNf>, so they
// work with TXQCDSchurOp (Mpc†Mpc on the half-volume RB odd sublattice).

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/txqcd/TXQCDFusedShiftKernels.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>

NAMESPACE_BEGIN(Grid);

class TXQCDConjugateGradient {
 public:
  RealD Tolerance;
  int MaxIterations;
  int IterationsToComplete;
  RealD TrueResidual;

  TXQCDConjugateGradient(RealD tol, int maxit)
      : Tolerance(tol), MaxIterations(maxit) {}

  void operator()(LinearOperatorBase<TXQCDFermionNf> &LinOp,
                  const TXQCDFermionNf &src, TXQCDFermionNf &psi) {
    GridBase *grid = src.Grid();
    TXQCDFermionNf p(grid), mmp(grid), r(grid);

    RealD ssq = norm2(src);
    if (ssq == 0.0) {
      psi = Zero();
      IterationsToComplete = 0;
      TrueResidual = 0.0;
      return;
    }

    psi = Zero();
    for (int a = 0; a < TxqcdNf; ++a)
      psi.f[a].Checkerboard() = src.f[0].Checkerboard();

    r = src;
    p = r;
    RealD cp = norm2(r);
    RealD rsq = Tolerance * Tolerance * ssq;

    if (cp <= rsq) {
      IterationsToComplete = 0;
      TrueResidual = std::sqrt(cp / ssq);
      return;
    }

    int k;
    for (k = 1; k <= MaxIterations; ++k) {
      RealD c = cp;
      LinOp.HermOp(p, mmp);

      ComplexD dc = innerProduct(p, mmp);
      RealD d = dc.real();
      RealD a = c / d;

      // r -= a * mmp
      for (int f = 0; f < TxqcdNf; ++f)
        r.f[f] = r.f[f] - a * mmp.f[f];
      cp = norm2(r);

      RealD b = cp / c;

      // psi += a * p; p = r + b * p
      for (int f = 0; f < TxqcdNf; ++f) {
        psi.f[f] = psi.f[f] + a * p.f[f];
        p.f[f] = r.f[f] + b * p.f[f];
      }

      if (cp <= rsq) break;
    }

    LinOp.HermOp(psi, mmp);
    TXQCDFermionNf residual(grid);
    for (int f = 0; f < TxqcdNf; ++f)
      residual.f[f] = mmp.f[f] - src.f[f];
    TrueResidual = std::sqrt(norm2(residual) / ssq);
    IterationsToComplete = k;

    std::cout << GridLogMessage << "TXQCDConjugateGradient: converged iter=" << k
              << " computed=" << std::sqrt(cp / ssq)
              << " true=" << TrueResidual << std::endl;
  }
};

// Multi-RHS CG: solves LinOp.HermOp · psi_b = src_b for each b ∈ [0, N) where
// all RHS share the same Hermitian operator (= Mpc†Mpc for TXQCDCloverSchurOp).
//
// Each RHS gets its own α, β, residual; the algorithm is N independent CGs
// run in lockstep so that operator applications can be amortized across RHS
// (via TXQCDMultiRHSLinearOperatorBase::HermOpN if available — falls back to
// looping over single-RHS HermOp otherwise).
//
// Use case: connected propagator measurement with Nf×Ns×Nc = 24 RHS at the
// same mass and gauge.  Each iteration shares the operator's gauge/clover/aux
// fetches across columns when HermOpN is fused.
class TXQCDMultiRHSConjugateGradient {
 public:
  RealD Tolerance;
  int MaxIterations;
  std::vector<int> IterationsToComplete;
  std::vector<RealD> TrueResidual;

  TXQCDMultiRHSConjugateGradient(RealD tol, int maxit)
      : Tolerance(tol), MaxIterations(maxit) {}

  // Naive multi-RHS dispatch: loop over RHS calling LinOp.HermOp for each.
  // Subclassing LinOp to implement HermOpN(std::vector, std::vector) is the
  // path to amortizing operator-application cost; the algorithm above is
  // unchanged when that's swapped in.
  void operator()(LinearOperatorBase<TXQCDFermionNf> &LinOp,
                  const std::vector<TXQCDFermionNf> &src,
                  std::vector<TXQCDFermionNf> &psi) {
    int N = (int)src.size();
    GRID_ASSERT(N > 0);
    GRID_ASSERT((int)psi.size() == N);

    GridBase *grid = src[0].Grid();

    std::vector<TXQCDFermionNf> p, mmp, r;
    p.reserve(N); mmp.reserve(N); r.reserve(N);
    for (int b = 0; b < N; ++b) {
      p.emplace_back(grid);
      mmp.emplace_back(grid);
      r.emplace_back(grid);
    }

    std::vector<RealD> ssq(N), cp(N), rsq(N);
    std::vector<bool> done(N, false);
    int n_active = 0;

    for (int b = 0; b < N; ++b) {
      ssq[b] = norm2(src[b]);
      psi[b] = Zero();
      for (int f = 0; f < TxqcdNf; ++f)
        psi[b].f[f].Checkerboard() = src[b].f[0].Checkerboard();
      r[b] = src[b];
      p[b] = r[b];
      cp[b] = norm2(r[b]);
      rsq[b] = Tolerance * Tolerance * std::max(ssq[b], 1e-30);
      if (ssq[b] == 0.0 || cp[b] <= rsq[b]) {
        done[b] = true;
      } else {
        n_active++;
      }
    }

    IterationsToComplete.assign(N, 0);
    TrueResidual.assign(N, 0);
    if (n_active == 0) return;

    // If LinOp is a TXQCDCloverSchurOp, use its fused HermOpN — single
    // gemmBatched per Mooee/MooeeInv across all active RHS.
    auto *schur_op = dynamic_cast<TXQCDCloverSchurOp*>(&LinOp);

    // Phase M.4.c profiling: per-bucket timers in the inner loop.
    uint64_t t_hermop = 0, t_inner = 0, t_axpy_r = 0, t_norm2_r = 0,
             t_axpy_psi_p = 0, t_pack = 0;

    // Phase M.4.c scratch for the rare partial-pack iterations (last few iters
    // when some RHS converged ahead of others).  Allocated lazily.
    std::vector<TXQCDFermionNf> p_active_scratch, mmp_active_scratch;

    int k;
    for (k = 1; k <= MaxIterations; ++k) {
      auto t_pack0 = usecond();
      if (schur_op) {
        // Fast path: all RHS still active → call HermOpN directly on p/mmp,
        // no deep copies.  This is true for >95% of iterations at NRHS=24
        // (single-mass, similar conditioning).
        int n_done = 0;
        for (int b = 0; b < N; ++b) if (done[b]) ++n_done;
        if (n_done == 0) {
          t_pack += usecond() - t_pack0;
          auto t_h0 = usecond();
          schur_op->HermOpN(p, mmp);
          t_hermop += usecond() - t_h0;
        } else {
          // Slow path: pack only the un-converged RHS.  Deep copies here are
          // unavoidable because HermOpN takes contiguous vectors; cost is
          // amortized over the few partial-pack iterations.
          if ((int)p_active_scratch.size() < N) {
            p_active_scratch.reserve(N);
            mmp_active_scratch.reserve(N);
            while ((int)p_active_scratch.size() < N) {
              p_active_scratch.emplace_back(grid);
              mmp_active_scratch.emplace_back(grid);
            }
          }
          std::vector<int> active_idx;
          active_idx.reserve(N);
          int nact = 0;
          for (int b = 0; b < N; ++b) {
            if (done[b]) continue;
            p_active_scratch[nact] = p[b];
            for (int a = 0; a < TxqcdNf; ++a)
              mmp_active_scratch[nact].f[a].Checkerboard() = p[b].f[a].Checkerboard();
            active_idx.push_back(b);
            ++nact;
          }
          // Trim views to active count (HermOpN reads ins.size()).
          std::vector<TXQCDFermionNf> p_active(p_active_scratch.begin(),
                                                p_active_scratch.begin() + nact);
          std::vector<TXQCDFermionNf> mmp_active(mmp_active_scratch.begin(),
                                                  mmp_active_scratch.begin() + nact);
          t_pack += usecond() - t_pack0;
          if (!active_idx.empty()) {
            auto t_h0 = usecond();
            schur_op->HermOpN(p_active, mmp_active);
            t_hermop += usecond() - t_h0;
            auto t_p1 = usecond();
            for (size_t i = 0; i < active_idx.size(); ++i)
              mmp[active_idx[i]] = mmp_active[i];
            t_pack += usecond() - t_p1;
          }
        }
      } else {
        // Generic fallback: per-RHS HermOp.
        auto t_h0 = usecond();
        for (int b = 0; b < N; ++b) {
          if (done[b]) continue;
          LinOp.HermOp(p[b], mmp[b]);
        }
        t_hermop += usecond() - t_h0;
      }

      bool any_progress = false;
      for (int b = 0; b < N; ++b) {
        if (done[b]) continue;
        any_progress = true;

        RealD c = cp[b];
        auto t_i0 = usecond();
        ComplexD dc = innerProduct(p[b], mmp[b]);
        t_inner += usecond() - t_i0;
        RealD d = dc.real();
        RealD a = c / d;

        auto t_a0 = usecond();
        for (int f = 0; f < TxqcdNf; ++f)
          r[b].f[f] = r[b].f[f] - a * mmp[b].f[f];
        t_axpy_r += usecond() - t_a0;
        auto t_n0 = usecond();
        cp[b] = norm2(r[b]);
        t_norm2_r += usecond() - t_n0;

        RealD bv = cp[b] / c;
        auto t_x0 = usecond();
        for (int f = 0; f < TxqcdNf; ++f) {
          psi[b].f[f] = psi[b].f[f] + a * p[b].f[f];
          p[b].f[f]   = r[b].f[f] + bv * p[b].f[f];
        }
        t_axpy_psi_p += usecond() - t_x0;

        if (cp[b] <= rsq[b]) {
          done[b] = true;
          IterationsToComplete[b] = k;
        }
      }
      if (!any_progress) break;
    }
    if (std::getenv("TXQCD_MULTIRHS_TIMERS")) {
      std::cout << GridLogMessage
                << "[MultiRHSCG breakdown N=" << N << "]"
                << " HermOpN=" << t_hermop / 1e6 << "s"
                << " inner=" << t_inner / 1e6 << "s"
                << " axpy(r)=" << t_axpy_r / 1e6 << "s"
                << " norm2(r)=" << t_norm2_r / 1e6 << "s"
                << " axpy(psi,p)=" << t_axpy_psi_p / 1e6 << "s"
                << " pack=" << t_pack / 1e6 << "s"
                << std::endl;
    }

    // Fill un-converged iter counts and compute true residuals.
    for (int b = 0; b < N; ++b) {
      if (IterationsToComplete[b] == 0) IterationsToComplete[b] = k;
      LinOp.HermOp(psi[b], mmp[b]);
      TXQCDFermionNf res(grid);
      for (int f = 0; f < TxqcdNf; ++f)
        res.f[f] = mmp[b].f[f] - src[b].f[f];
      TrueResidual[b] = (ssq[b] > 0) ? std::sqrt(norm2(res) / ssq[b]) : 0;
    }

    int max_iter = 0; for (int b = 0; b < N; ++b)
      max_iter = std::max(max_iter, IterationsToComplete[b]);
    int min_iter = max_iter; for (int b = 0; b < N; ++b)
      if (IterationsToComplete[b] > 0)
        min_iter = std::min(min_iter, IterationsToComplete[b]);
    RealD max_resid = 0; for (int b = 0; b < N; ++b)
      max_resid = std::max(max_resid, TrueResidual[b]);
    std::cout << GridLogMessage
              << "TXQCDMultiRHSCG (N=" << N << "): iter min=" << min_iter
              << " max=" << max_iter << " max_resid=" << max_resid << std::endl;
  }
};

// Multi-shift CG: solves (A + sigma_s) x_s = src for all shifts simultaneously,
// where A is the HermOp of the LinearOperatorBase (= Mpc†Mpc for TXQCDSchurOp).
class TXQCDMultiShiftCGSchur {
 public:
  int MaxIterations;
  int IterationsToComplete;

  TXQCDMultiShiftCGSchur(int maxit) : MaxIterations(maxit) {}

  void operator()(LinearOperatorBase<TXQCDFermionNf> &LinOp,
                  const std::vector<RealD> &shifts,
                  const std::vector<RealD> &tol,
                  const TXQCDFermionNf &src,
                  std::vector<TXQCDFermionNf> &psi) {
    const int nshift = static_cast<int>(shifts.size());
    GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
    GRID_ASSERT(static_cast<int>(tol.size()) == nshift);

    GridBase *grid = src.Grid();

    TXQCDFermionNf r(grid), p(grid), mmp(grid);
    std::vector<TXQCDFermionNf> ps;
    ps.reserve(nshift);
    for (int s = 0; s < nshift; ++s) ps.emplace_back(grid);

    std::vector<RealD> alpha(nshift, 1.0);
    std::vector<RealD> bs(nshift);
    std::vector<RealD> rsq_target(nshift);
    std::vector<std::array<RealD, 2>> z(nshift);
    std::vector<int> converged(nshift, 0);

    for (int s = 0; s < nshift; ++s) {
      GRID_ASSERT(shifts[s] >= shifts[0]);
    }

    RealD cp = norm2(src);
    if (cp == 0.0) {
      for (int s = 0; s < nshift; ++s) psi[s] = Zero();
      return;
    }

    for (int s = 0; s < nshift; ++s) {
      rsq_target[s] = cp * tol[s] * tol[s];
      ps[s] = src;
      psi[s] = Zero();
    }
    r = src;
    p = src;

    // mmp = (A + shifts[0]) p
    LinOp.HermOp(p, mmp);
    RealD d = real(innerProduct(p, mmp));
    for (int a = 0; a < TxqcdNf; ++a)
      mmp.f[a] = mmp.f[a] + shifts[0] * p.f[a];
    RealD rn = norm2(p);
    d += rn * shifts[0];

    RealD b = -cp / d;

    int iz = 0;
    z[0][0] = 1.0;
    z[0][1] = 1.0;
    bs[0] = b;
    for (int s = 1; s < nshift; ++s) {
      z[s][0] = 1.0;
      z[s][1] = 1.0;
      z[s][iz] = 1.0 / (1.0 - b * (shifts[s] - shifts[0]));
      bs[s] = b * z[s][iz];
    }

    for (int a = 0; a < TxqcdNf; ++a) r.f[a] = r.f[a] + b * mmp.f[a];
    RealD c = norm2(r);

    static int use_fused = []() {
      const char *e = std::getenv("TXQCD_CG_FUSED");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();

    // Build fused-shift context once per CG (opens views, uploads pointer
    // arrays).  Destructor at scope exit closes views.
    using vobj = typename LatticeFermion::vector_object;
    TxqcdFusedShift::FusedShiftCtx<vobj> fctx;
    if (use_fused) TxqcdFusedShift::BuildContext(fctx, ps, psi, grid);

    if (use_fused) {
      std::vector<RealD> psi_init_scale(nshift);
      for (int s = 0; s < nshift; ++s) psi_init_scale[s] = -bs[s] * alpha[s];
      TxqcdFusedShift::psiInit(fctx, psi, src, psi_init_scale);
    } else {
      for (int s = 0; s < nshift; ++s) {
        RealD coef = -bs[s] * alpha[s];
        for (int a = 0; a < TxqcdNf; ++a) psi[s].f[a] = coef * src.f[a];
      }
    }

    uint64_t t_book_us = 0;
    uint64_t t_hermop_us = 0;
    int k;
    for (k = 1; k <= MaxIterations; ++k) {
      RealD aa = c / cp;
      auto t_b0 = usecond();

      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = aa * p.f[a] + r.f[a];

      if (use_fused) {
        // Fused per-shift ps update.  Coefficients depend on s: for s=0 it's
        // (alpha=1, beta=aa); for s>0 it's (alpha=zc[s], beta=as[s]).
        std::vector<RealD> alpha_arr(nshift), beta_arr(nshift);
        alpha_arr[0] = 1.0;
        beta_arr[0]  = aa;
        for (int s = 1; s < nshift; ++s) {
          RealD as = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
          RealD zc = z[s][iz];
          alpha_arr[s] = zc;
          beta_arr[s]  = as;
        }
        TxqcdFusedShift::psUpdate(fctx, ps, r, converged, alpha_arr, beta_arr);
      } else {
        for (int s = 0; s < nshift; ++s) {
          if (converged[s]) continue;
          if (s == 0) {
            for (int a = 0; a < TxqcdNf; ++a)
              ps[s].f[a] = aa * ps[s].f[a] + r.f[a];
          } else {
            RealD as = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
            RealD zc = z[s][iz];
            for (int a = 0; a < TxqcdNf; ++a)
              ps[s].f[a] = zc * r.f[a] + as * ps[s].f[a];
          }
        }
      }
      t_book_us += usecond() - t_b0;

      RealD cp_prev = c;
      auto t_h0 = usecond();
      LinOp.HermOp(p, mmp);
      t_hermop_us += usecond() - t_h0;
      d = real(innerProduct(p, mmp));
      for (int a = 0; a < TxqcdNf; ++a)
        mmp.f[a] = mmp.f[a] + shifts[0] * p.f[a];
      rn = norm2(p);
      d += rn * shifts[0];

      RealD bp = b;
      b = -cp_prev / d;

      for (int a = 0; a < TxqcdNf; ++a) r.f[a] = r.f[a] + b * mmp.f[a];
      RealD c_new = norm2(r);
      cp = cp_prev;
      c = c_new;

      bs[0] = b;
      iz = 1 - iz;
      for (int s = 1; s < nshift; ++s) {
        if (converged[s]) continue;
        RealD z0 = z[s][1 - iz];
        RealD z1 = z[s][iz];
        z[s][iz] = z0 * z1 * bp /
                   (b * aa * (z1 - z0) +
                    z1 * bp * (1.0 - (shifts[s] - shifts[0]) * b));
        bs[s] = b * z[s][iz] / z0;
      }

      auto t_b1 = usecond();
      if (use_fused) {
        std::vector<RealD> coef_arr(nshift);
        for (int s = 0; s < nshift; ++s) coef_arr[s] = -bs[s] * alpha[s];
        TxqcdFusedShift::psiUpdate(fctx, psi, ps, converged, coef_arr);
      } else {
        for (int s = 0; s < nshift; ++s) {
          if (converged[s]) continue;
          RealD coef = -bs[s] * alpha[s];
          for (int a = 0; a < TxqcdNf; ++a)
            psi[s].f[a] = psi[s].f[a] + coef * ps[s].f[a];
        }
      }
      t_book_us += usecond() - t_b1;

      int all_converged = 1;
      for (int s = 0; s < nshift; ++s) {
        if (converged[s]) continue;
        RealD css = c * z[s][iz] * z[s][iz];
        if (css < rsq_target[s]) {
          converged[s] = 1;
        } else {
          all_converged = 0;
        }
      }
      if (all_converged) {
        IterationsToComplete = k;
        std::cout << GridLogMessage
                  << "TXQCDMultiShiftCGSchur: converged iter=" << k
                  << " nshift=" << nshift
                  << " bookkeeping=" << double(t_book_us)*1e-6 << " s"
                  << " hermop=" << double(t_hermop_us)*1e-6 << " s"
                  << std::endl;
        return;
      }
    }
    IterationsToComplete = k;
    std::cout << GridLogMessage
              << "TXQCDMultiShiftCGSchur: did not converge in " << MaxIterations
              << " iterations (c=" << c << ")" << std::endl;
  }
};

NAMESPACE_END(Grid);
