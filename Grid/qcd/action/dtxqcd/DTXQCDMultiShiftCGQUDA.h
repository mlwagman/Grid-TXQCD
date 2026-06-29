#pragma once
// M-wrap.5b: shifted-CG multi-shift solver where the inner mat-vec routes
// through QUDA's Wilson-clover MatQuda (via DTXQCDMpcOpQUDA) instead of the
// all-Grid path.  Direct sibling of DTXQCDMultiShiftCG (the all-DP Grid
// baseline) and DTXQCDMultiShiftCGMixedPrec (DP outer + SP-Grid inner);
// this is "DP outer + QUDA-DP inner."
//
// Stage A (this version): use the host-mode DTXQCDMpcOpQUDA.M / .Mdag for
// the inner HermOp.  Per (M·v) call internally:
//     pack Grid -> flat EO host -> QUDA MatQuda(device) -> flat EO host
//     -> apply_C/conj basis-correction for the lower block -> Grid aux kernel
//     -> unpack to Grid layout.
// The outer recurrence state (r, p, ps[s], psi[s]) lives in Grid SIMD layout
// throughout; per-iter BLAS uses Grid's existing Axpy/Scale_Add/norm2/
// innerProduct on DTXQCDFermionDoubled.  This validates the multishift
// architecture end-to-end and gives bit-equivalent results to
// DTXQCDMultiShiftCG modulo FP reordering inside QUDA's MatQuda.
//
// Stage B (deferred, see M-wrap.5b.2 if perf demands): keep r, p, ps[s],
// psi[s] in QUDA ColorSpinorField device-resident; swap the Grid BLAS for
// blas_quda fused primitives (axpyCGNorm + block::axpy + block::axpby);
// use DTXQCDMpcOpQUDA::M_device to avoid per-call host pack/unpack.  Stage B
// is the perf path; Stage A is the correctness path.
//
// Reliable update: every ReliableUpdateFreq iters, recompute the primary
// residual via the Grid-DP HermOp (Mop_d).  Mostly a safety net at DP — the
// drift between QUDA-DP and Grid-DP is small (FP reordering only), so the
// dominant role of the reliable update here is to bound any cumulative
// rounding error rather than flush SP-side drift like the MixedPrec
// counterpart does.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>

NAMESPACE_BEGIN(Grid);

// HermOp helper: (M† M)·p via Mop_quda.M then Mop_quda.Mdag.  Returns the
// inner product Re<p, (M†M)·p> which the shifted-CG recurrence wants for the
// 'd' update.  Mirrors DtxqcdHermOpD's role in MixedPrec but routes through
// the QUDA wrapper.
inline RealD DtxqcdHermOpQUDA(DTXQCDMpcOpQUDA &Mop_quda,
                              const DTXQCDFermionDoubled &p,
                              DTXQCDFermionDoubled &mmp,
                              DTXQCDFermionDoubled &tmp) {
  Mop_quda.M(p, tmp);
  Mop_quda.Mdag(tmp, mmp);
  return real(innerProduct(p, mmp));
}

template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA(
    DTXQCDMopD &Mop_d,                 // Grid-DP HermOp for reliable update
    DTXQCDMpcOpQUDA &Mop_quda,         // QUDA-DP HermOp for inner mat-vec
    const std::vector<RealD> &poles,
    const std::vector<RealD> &tol,
    const DTXQCDFermionDoubled &src,
    std::vector<DTXQCDFermionDoubled> &psi,
    int MaxIter,
    int ReliableUpdateFreq = 50) {
  using namespace dtxqcd_msshift_detail;

  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);
  for (int s = 0; s < nshift; ++s) GRID_ASSERT(poles[s] >= poles[0]);

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

  // First HermOp via QUDA.
  RealD d = DtxqcdHermOpQUDA(Mop_quda, p, mmp, tmp);
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

  for (int s = 0; s < nshift; ++s) {
    Set_Scaled(psi[s], -bs[s] * alpha[s], src);
  }

  GridStopWatch MatrixTimer, ReliableTimer;
  int reliable_updates = 0;

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
        for (int aa_idx = 0; aa_idx < DtxqcdNf; ++aa_idx) {
          ps[s].upper.f[aa_idx] = zc * r.upper.f[aa_idx]
                                + as * ps[s].upper.f[aa_idx];
          ps[s].lower.f[aa_idx] = zc * r.lower.f[aa_idx]
                                + as * ps[s].lower.f[aa_idx];
        }
      }
    }

    RealD cp_prev = c;

    MatrixTimer.Start();
    d = DtxqcdHermOpQUDA(Mop_quda, p, mmp, tmp);
    MatrixTimer.Stop();
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

    // Reliable update: every ReliableUpdateFreq iters, recompute the primary
    // residual via the Grid-DP HermOp (Mop_d).  Bounds cumulative rounding
    // error from QUDA pack/unpack roundtrips.
    if (k % ReliableUpdateFreq == 0) {
      ReliableTimer.Start();
      RealD c_old = c;
      RealD dd = DtxqcdHermOpD(Mop_d, psi[0], mmp, tmp);
      (void)dd;
      Axpy(mmp, poles[0], psi[0]);
      for (int a = 0; a < DtxqcdNf; ++a) {
        r.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
        r.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
      }
      c = norm2(r);
      ReliableTimer.Stop();
      ++reliable_updates;
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA] reliable update k=" << k
                << " |r|^2: " << c_old << " -> " << c << std::endl;
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
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA] converged iter=" << k
                << " nshift=" << nshift
                << " reliable_updates=" << reliable_updates
                << "  QUDA-matrix=" << MatrixTimer.Elapsed()
                << "  reliable=" << ReliableTimer.Elapsed() << std::endl;
      return;
    }
  }
  std::cout << GridLogMessage
            << "[DTXQCDMultiShiftCGQUDA] did not converge in " << MaxIter
            << " iterations (c=" << c << ")" << std::endl;
}

NAMESPACE_END(Grid);
