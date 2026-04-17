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

    for (int s = 0; s < nshift; ++s) {
      RealD coef = -bs[s] * alpha[s];
      for (int a = 0; a < TxqcdNf; ++a) psi[s].f[a] = coef * src.f[a];
    }

    int k;
    for (k = 1; k <= MaxIterations; ++k) {
      RealD aa = c / cp;

      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = aa * p.f[a] + r.f[a];

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

      RealD cp_prev = c;
      LinOp.HermOp(p, mmp);
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

      for (int s = 0; s < nshift; ++s) {
        if (converged[s]) continue;
        RealD coef = -bs[s] * alpha[s];
        for (int a = 0; a < TxqcdNf; ++a)
          psi[s].f[a] = psi[s].f[a] + coef * ps[s].f[a];
      }

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
                  << " nshift=" << nshift << std::endl;
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
