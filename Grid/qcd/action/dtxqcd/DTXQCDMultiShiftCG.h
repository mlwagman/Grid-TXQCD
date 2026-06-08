#pragma once
// Multi-shift CG for the DTXQCD doubled fermion type.
//
// Solves (M^dag M + sigma_s I) psi_s = src for all shifts s in one Krylov
// pass.  Standard shifted-CG recurrence, ported from Grid's
// ConjugateGradientMultiShift but specialized to DTXQCDFermionDoubled (which
// does not conform to LinearOperatorBase<Field>).  Direct sibling of
// TXQCDMultiShiftCG -- same algorithm, just operating on the doubled
// (upper, lower) × Nf flavor structure.
//
// Caller provides:
//   - any operator Mop with M(in, out) and Mdag(in, out) on DTXQCDFermionDoubled
//     (typically DTXQCDMpcOp),
//   - poles[s] (>= 0, sorted ascending with poles[0] the smallest),
//   - per-shift tolerances tol[s],
//   - src, and an output vector psi of length nshift preallocated on the grid.
//
// On return, psi[s] ~= (M^dag M + poles[s])^{-1} src.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>

NAMESPACE_BEGIN(Grid);

namespace dtxqcd_msshift_detail {

// Helpers operating on DTXQCDFermionDoubled per (block, flavor).
inline void Zero_(DTXQCDFermionDoubled &x) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = Zero();
    x.lower.f[a] = Zero();
  }
}

// y = scale * y + add  (axpy with scale on LHS)
inline void Scale_Add(DTXQCDFermionDoubled &y, RealD scale,
                      const DTXQCDFermionDoubled &add) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    y.upper.f[a] = scale * y.upper.f[a] + add.upper.f[a];
    y.lower.f[a] = scale * y.lower.f[a] + add.lower.f[a];
  }
}

// y += a * x
inline void Axpy(DTXQCDFermionDoubled &y, RealD a,
                 const DTXQCDFermionDoubled &x) {
  for (int aa = 0; aa < DtxqcdNf; ++aa) {
    y.upper.f[aa] = y.upper.f[aa] + a * x.upper.f[aa];
    y.lower.f[aa] = y.lower.f[aa] + a * x.lower.f[aa];
  }
}

// y = coef * src
inline void Set_Scaled(DTXQCDFermionDoubled &y, RealD coef,
                       const DTXQCDFermionDoubled &src) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    y.upper.f[a] = coef * src.upper.f[a];
    y.lower.f[a] = coef * src.lower.f[a];
  }
}

}  // namespace dtxqcd_msshift_detail

template <class DTXQCDMop>
inline void DTXQCDMultiShiftCG(DTXQCDMop &Mop,
                                const std::vector<RealD> &poles,
                                const std::vector<RealD> &tol,
                                const DTXQCDFermionDoubled &src,
                                std::vector<DTXQCDFermionDoubled> &psi,
                                int MaxIter) {
  using namespace dtxqcd_msshift_detail;

  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);

  GridBase *grid = src.Grid();
  DTXQCDFermionDoubled r(grid), p(grid), mmp(grid), tmp(grid);
  std::vector<DTXQCDFermionDoubled> ps;
  ps.reserve(nshift);
  for (int s = 0; s < nshift; ++s) ps.emplace_back(grid);

  std::vector<RealD> alpha(nshift, 1.0);
  std::vector<RealD> bs(nshift);
  std::vector<RealD> rsq_target(nshift);
  std::vector<std::array<RealD, 2>> z(nshift);
  std::vector<int> converged(nshift, 0);

  for (int s = 0; s < nshift; ++s) GRID_ASSERT(poles[s] >= poles[0]);

  RealD cp = norm2(src);
  if (cp == 0.0) {
    for (int s = 0; s < nshift; ++s) Zero_(psi[s]);
    return;
  }

  for (int s = 0; s < nshift; ++s) {
    rsq_target[s] = cp * tol[s] * tol[s];
    ps[s] = src;
    Zero_(psi[s]);
  }
  r = src;
  p = src;

  // mmp = (M^dag M + poles[0]) p ; d = p^H mmp + poles[0] * ||p||^2
  Mop.M(p, tmp);
  Mop.Mdag(tmp, mmp);
  RealD d = real(innerProduct(p, mmp));
  Axpy(mmp, poles[0], p);
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

  Axpy(r, b, mmp);
  RealD c = norm2(r);

  // psi[s] = -bs[s] * alpha[s] * src
  for (int s = 0; s < nshift; ++s) {
    Set_Scaled(psi[s], -bs[s] * alpha[s], src);
  }

  for (int k = 1; k <= MaxIter; ++k) {
    RealD aa = c / cp;

    Scale_Add(p, aa, r);

    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      if (s == 0) {
        Scale_Add(ps[s], aa, r);
      } else {
        RealD as = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
        RealD zc = z[s][iz];
        // ps[s] = zc * r + as * ps[s]
        for (int aa_idx = 0; aa_idx < DtxqcdNf; ++aa_idx) {
          ps[s].upper.f[aa_idx] = zc * r.upper.f[aa_idx]
                                + as * ps[s].upper.f[aa_idx];
          ps[s].lower.f[aa_idx] = zc * r.lower.f[aa_idx]
                                + as * ps[s].lower.f[aa_idx];
        }
      }
    }

    RealD cp_prev = c;
    Mop.M(p, tmp);
    Mop.Mdag(tmp, mmp);
    d = real(innerProduct(p, mmp));
    Axpy(mmp, poles[0], p);
    rn = norm2(p);
    d += rn * poles[0];

    RealD bp = b;
    b = -cp_prev / d;

    Axpy(r, b, mmp);
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

    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      Axpy(psi[s], -bs[s] * alpha[s], ps[s]);
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
      std::cout << GridLogMessage << "[DTXQCDMultiShiftCG] converged iter=" << k
                << " nshift=" << nshift << std::endl;
      return;
    }
  }
  std::cout << GridLogMessage
            << "[DTXQCDMultiShiftCG] did not converge in " << MaxIter
            << " iterations (c=" << c << ")" << std::endl;
}

NAMESPACE_END(Grid);
