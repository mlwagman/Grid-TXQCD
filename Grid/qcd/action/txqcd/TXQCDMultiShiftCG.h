#pragma once
// Hand-rolled multi-shift conjugate gradient for TXQCDFermionNf + TXQCDWilsonOp.
//
// Solves (M^dag M + sigma_s I) x_s = src for all shifts s in one pass, using
// the standard shifted-CG recurrence. This is a direct port of Grid's
// ConjugateGradientMultiShift (see algorithms/iterative/ConjugateGradientMultiShift.h)
// specialized so it operates on TXQCDFermionNf without requiring the field to
// conform to LinearOperatorBase<Field>. Kept local to txqcd/ to match the
// existing hand-rolled CG style in TXQCDWilsonPseudoFermionAction.h.
//
// Caller provides:
//   - the TXQCDWilsonOp (already carrying gauge + aux references),
//   - poles[s] (>=0, with poles[0] the smallest) and per-shift tol[s],
//   - src, and an output vector psi of length nshift preallocated on the grid.
//
// On return, psi[s] ~= (M^dag M + poles[s])^{-1} src.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>

NAMESPACE_BEGIN(Grid);

// Templated on the TXQCD Dirac operator type (TXQCDWilsonOp,
// TXQCDWilsonCloverOp, TXQCDMobiusOp, ...): only requires M(in,out) and
// Mdag(in,out) on TXQCDFermionNf.  Existing call sites that pass a
// TXQCDWilsonOp continue to work unchanged.
template <class TXQCDOp>
inline void TXQCDMultiShiftCG(TXQCDOp &Mop,
                              const std::vector<RealD> &poles,
                              const std::vector<RealD> &tol,
                              const TXQCDFermionNf &src,
                              std::vector<TXQCDFermionNf> &psi,
                              int MaxIter) {
  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);

  GridBase *grid = src.Grid();

  TXQCDFermionNf r(grid), p(grid), mmp(grid), tmp(grid);
  std::vector<TXQCDFermionNf> ps;
  ps.reserve(nshift);
  for (int s = 0; s < nshift; ++s) ps.emplace_back(grid);

  std::vector<RealD> alpha(nshift, 1.0);
  std::vector<RealD> bs(nshift);
  std::vector<RealD> rsq_target(nshift);
  std::vector<std::array<RealD, 2>> z(nshift);
  std::vector<int> converged(nshift, 0);

  for (int s = 0; s < nshift; ++s) {
    GRID_ASSERT(poles[s] >= poles[0]);
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

  // mmp = (M^dag M + poles[0]) p ; d = p^H mmp
  Mop.M(p, tmp);
  Mop.Mdag(tmp, mmp);
  RealD d = real(innerProduct(p, mmp));
  for (int a = 0; a < TxqcdNf; ++a) mmp.f[a] = mmp.f[a] + poles[0] * p.f[a];
  RealD rn = norm2(p);
  d += rn * poles[0];

  RealD b = -cp / d;

  int iz = 0;
  z[0][0] = 1.0; z[0][1] = 1.0;
  bs[0] = b;
  for (int s = 1; s < nshift; ++s) {
    z[s][0] = 1.0; z[s][1] = 1.0;
    z[s][iz] = 1.0 / (1.0 - b * (poles[s] - poles[0]));
    bs[s] = b * z[s][iz];
  }

  // r = r + b*mmp ; c = ||r||^2
  for (int a = 0; a < TxqcdNf; ++a) r.f[a] = r.f[a] + b * mmp.f[a];
  RealD c = norm2(r);

  // psi[s] = -bs[s]*alpha[s] * src
  for (int s = 0; s < nshift; ++s) {
    RealD coef = -bs[s] * alpha[s];
    for (int a = 0; a < TxqcdNf; ++a) psi[s].f[a] = coef * src.f[a];
  }

  int k;
  for (k = 1; k <= MaxIter; ++k) {
    RealD aa = c / cp;

    // p = aa*p + r
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = aa * p.f[a] + r.f[a];

    // ps[s] = z[s][iz]*r + as*ps[s]  (s=0 special: as = aa, z=1)
    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      if (s == 0) {
        for (int a = 0; a < TxqcdNf; ++a) ps[s].f[a] = aa * ps[s].f[a] + r.f[a];
      } else {
        RealD as = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
        RealD zc = z[s][iz];
        for (int a = 0; a < TxqcdNf; ++a) ps[s].f[a] = zc * r.f[a] + as * ps[s].f[a];
      }
    }

    RealD cp_prev = c;
    // mmp = (M^dag M + poles[0]) p
    Mop.M(p, tmp);
    Mop.Mdag(tmp, mmp);
    d = real(innerProduct(p, mmp));
    for (int a = 0; a < TxqcdNf; ++a) mmp.f[a] = mmp.f[a] + poles[0] * p.f[a];
    rn = norm2(p);
    d += rn * poles[0];

    RealD bp = b;
    b = -cp_prev / d;

    // r = r + b*mmp
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
                 (b * aa * (z1 - z0) + z1 * bp * (1.0 - (poles[s] - poles[0]) * b));
      bs[s] = b * z[s][iz] / z0;
    }

    // psi[s] += -bs[s]*alpha[s] * ps[s]
    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      RealD coef = -bs[s] * alpha[s];
      for (int a = 0; a < TxqcdNf; ++a) psi[s].f[a] = psi[s].f[a] + coef * ps[s].f[a];
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
      std::cout << GridLogMessage << "[TXQCDMultiShiftCG] converged iter=" << k
                << " nshift=" << nshift << std::endl;
      return;
    }
  }
  std::cout << GridLogMessage
            << "[TXQCDMultiShiftCG] did not converge in " << MaxIter
            << " iterations (c=" << c << ")" << std::endl;
}

NAMESPACE_END(Grid);
