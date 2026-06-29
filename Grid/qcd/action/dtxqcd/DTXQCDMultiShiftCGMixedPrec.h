#pragma once
// Reliable-update mixed-precision multi-shift CG for the DTXQCD doubled
// fermion type.
//
// Solves (M^dag M + poles_s I) psi_s = src for all shifts s in one Krylov
// pass, with the per-iteration matrix multiply done in SINGLE precision
// (DTXQCDMpcOpF) while the search directions, solutions, and the periodic
// reliable-update residual stay in DOUBLE precision (DTXQCDMpcOp).  This is
// the doubled-field analogue of Grid's
// ConjugateGradientMultiShiftMixedPrec (which cannot be used directly because
// DTXQCDFermionDoubled does not conform to LinearOperatorBase<Field>).
//
// Algorithm (CK reliable-update, mirroring the Grid reference):
//   - r_d, p_d, ps_d[s], psi_d[s] : double precision (DTXQCDFermionDoubled)
//   - the shifted-CG recurrence is identical to the all-double
//     DTXQCDMultiShiftCG; only the operator apply MdagM p changes:
//       * down-cast p_d -> p_f (single)
//       * mmp_f = (Mop_f^dag Mop_f) p_f           [single precision]
//       * up-cast mmp_f -> mmp_d                  [the inner-product d and the
//                                                  residual recurrence use mmp_d]
//   - every ReliableUpdateFreq iterations, recompute the true primary
//     residual in double precision:
//       r_d = src - (Mop_d^dag Mop_d + poles_0) psi_d[0]
//     so accumulated single-precision error is flushed.
//
// On return psi[s] ~= (M^dag M + poles_s)^{-1} src to the requested tol[s]
// (the same double-precision accuracy as DTXQCDMultiShiftCG).
//
// Caller provides DP and SP Schur operators wrapping the SAME imported gauge +
// aux (the SP one built from the DP one's downcast caches via DTXQCDMpcOpF).

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubledF.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpF.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>

NAMESPACE_BEGIN(Grid);

// MdagM_d + poles[0]: full double-precision primary HermOp, used for the
// reliable-update true-residual recompute.  Returns mmp = (M^dag M) p and
// d = Re<p, mmp> (the inner product the recurrence needs).
template <class DTXQCDMopD>
inline RealD DtxqcdHermOpD(DTXQCDMopD &Mop,
                           const DTXQCDFermionDoubled &p,
                           DTXQCDFermionDoubled &mmp,
                           DTXQCDFermionDoubled &tmp) {
  Mop.M(p, tmp);
  Mop.Mdag(tmp, mmp);
  return real(innerProduct(p, mmp));
}

// MdagM_f + poles[0]: single-precision primary HermOp on the SP doubled field.
template <class DTXQCDMopF>
inline void DtxqcdHermOpF(DTXQCDMopF &Mop_f,
                          const DTXQCDFermionDoubledF &p_f,
                          DTXQCDFermionDoubledF &mmp_f,
                          DTXQCDFermionDoubledF &tmp_f) {
  Mop_f.M(p_f, tmp_f);
  Mop_f.Mdag(tmp_f, mmp_f);
}

// Pure-SP single-shift CG cleanup:
//   solve (M^dag M + pole) x = src to tol with SP matvec + SP residual
//   accumulator, NO mid-call DP reliable update, warm-started from the
//   multishift's approximate x.  Caller decides whether to call this or the
//   legacy MP version; this one drops DP HermOps entirely from the inner.
//
//   Returns true on success (SP residual reached target); false if MaxIter
//   exhausted without convergence.  The caller is expected to verify the
//   answer with one DP HermOp at the end and fall back to the legacy MP
//   cleanup if needed.
template <class DTXQCDMopF>
inline bool DTXQCDSingleShiftCGSpCleanup(
    DTXQCDMopF &Mop_f, RealD pole, RealD tol,
    const DTXQCDFermionDoubled &src, DTXQCDFermionDoubled &x,
    int MaxIter) {
  using namespace dtxqcd_msshift_detail;
  GridBase *grid   = src.Grid();
  GridBase *grid_f = &Mop_f.RbGrid();
  DTXQCDFermionDoubled r(grid), p(grid), Ap(grid), mmp(grid);
  DTXQCDFermionDoubledF p_f(grid_f), mmp_f(grid_f), tmp_f(grid_f);

  auto ApplyShiftSP = [&](const DTXQCDFermionDoubled &in,
                          DTXQCDFermionDoubled &out) {
    DtxqcdPrecisionChange(p_f, in);
    DtxqcdHermOpF(Mop_f, p_f, mmp_f, tmp_f);
    DtxqcdPrecisionChange(out, mmp_f);
    Axpy(out, pole, in);                 // out = (MdagM + pole) in  (SP matvec)
  };

  // Initial residual via SP matvec (warm start from x).  This is the only
  // place we differ from DTXQCDSingleShiftCGMixedPrec: no DP starting probe.
  ApplyShiftSP(x, mmp);
  for (int a = 0; a < DtxqcdNf; ++a) {
    r.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
    r.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
  }
  p = r;
  RealD rsq = norm2(r);
  RealD ssq = norm2(src);
  RealD target = ssq * tol * tol;
  if (rsq < target) return true;

  for (int k = 1; k <= MaxIter; ++k) {
    ApplyShiftSP(p, Ap);                 // Ap = (MdagM + pole) p
    RealD pAp = real(innerProduct(p, Ap));
    RealD alpha = rsq / pAp;
    Axpy(x, alpha, p);                   // x += alpha p
    Axpy(r, -alpha, Ap);                 // r -= alpha Ap
    RealD rsq_new = norm2(r);
    if (rsq_new < target) return true;
    RealD beta = rsq_new / rsq;
    rsq = rsq_new;
    Scale_Add(p, beta, r);               // p = beta p + r
  }
  return false;
}

// Single-shift reliable-update mixed-precision CG cleanup:
//   solve (M^dag M + pole) x = src to tol, SP matvec + DP reliable update,
//   warm-started from the multishift's approximate x.  Used to polish the
//   per-shift solutions that the SP multishift Krylov pass left just above
//   tol (SP residual floor ~1e-7).  Much cheaper than a full DP multishift:
//   one shift, warm-started, SP matvec.  Mirrors Grid's
//   MixedPrecisionConjugateGradient cleanup but on the doubled field.
template <class DTXQCDMopD, class DTXQCDMopF>
inline void DTXQCDSingleShiftCGMixedPrec(
    DTXQCDMopD &Mop_d, DTXQCDMopF &Mop_f, RealD pole, RealD tol,
    const DTXQCDFermionDoubled &src, DTXQCDFermionDoubled &x,
    int MaxIter, int ReliableUpdateFreq) {
  using namespace dtxqcd_msshift_detail;
  GridBase *grid   = src.Grid();
  GridBase *grid_f = &Mop_f.RbGrid();
  DTXQCDFermionDoubled r(grid), p(grid), Ap(grid), mmp(grid), tmp(grid);
  DTXQCDFermionDoubledF p_f(grid_f), mmp_f(grid_f), tmp_f(grid_f);

  auto ApplyShiftSP = [&](const DTXQCDFermionDoubled &in,
                          DTXQCDFermionDoubled &out) {
    DtxqcdPrecisionChange(p_f, in);
    DtxqcdHermOpF(Mop_f, p_f, mmp_f, tmp_f);
    DtxqcdPrecisionChange(out, mmp_f);
    Axpy(out, pole, in);                 // out = (MdagM + pole) in  (SP matvec)
  };
  auto ApplyShiftDP = [&](const DTXQCDFermionDoubled &in,
                          DTXQCDFermionDoubled &out) {
    DtxqcdHermOpD(Mop_d, in, out, tmp);
    Axpy(out, pole, in);                 // out = (MdagM + pole) in  (DP matvec)
  };

  // True initial residual in DP (warm start from x).
  ApplyShiftDP(x, mmp);
  for (int a = 0; a < DtxqcdNf; ++a) {
    r.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
    r.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
  }
  p = r;
  RealD rsq = norm2(r);
  RealD ssq = norm2(src);
  RealD target = ssq * tol * tol;
  if (rsq < target) return;

  for (int k = 1; k <= MaxIter; ++k) {
    ApplyShiftSP(p, Ap);                 // Ap = (MdagM + pole) p
    RealD pAp = real(innerProduct(p, Ap));
    RealD alpha = rsq / pAp;
    Axpy(x, alpha, p);                   // x += alpha p
    Axpy(r, -alpha, Ap);                 // r -= alpha Ap
    RealD rsq_new = norm2(r);

    if (k % ReliableUpdateFreq == 0) {
      ApplyShiftDP(x, mmp);              // flush SP drift: true DP residual
      for (int a = 0; a < DtxqcdNf; ++a) {
        r.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
        r.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
      }
      rsq_new = norm2(r);
    }
    if (rsq_new < target) {
      // Confirm with a true DP residual before declaring done.
      ApplyShiftDP(x, mmp);
      for (int a = 0; a < DtxqcdNf; ++a) {
        tmp.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
        tmp.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
      }
      if (norm2(tmp) < target) return;
      r = tmp; rsq_new = norm2(r);
    }
    RealD beta = rsq_new / rsq;
    rsq = rsq_new;
    Scale_Add(p, beta, r);               // p = beta p + r
  }
}

template <class DTXQCDMopD, class DTXQCDMopF>
inline void DTXQCDMultiShiftCGMixedPrec(
    DTXQCDMopD &Mop_d,
    DTXQCDMopF &Mop_f,
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

  GridBase *grid   = src.Grid();          // double-precision RB grid
  GridBase *grid_f = &Mop_f.RbGrid();     // single-precision RB grid

  DTXQCDFermionDoubled  r(grid), p(grid), mmp(grid), tmp(grid);
  DTXQCDFermionDoubledF p_f(grid_f), mmp_f(grid_f), tmp_f(grid_f);
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

  // First HermOp at single precision.
  DtxqcdPrecisionChange(p_f, p);
  DtxqcdHermOpF(Mop_f, p_f, mmp_f, tmp_f);
  DtxqcdPrecisionChange(mmp, mmp_f);
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

  for (int s = 0; s < nshift; ++s) {
    Set_Scaled(psi[s], -bs[s] * alpha[s], src);
  }

  GridStopWatch MatrixTimer, PrecChangeTimer, ReliableTimer;
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

    // ---- Single-precision MdagM apply (the dominant per-iteration cost) ----
    PrecChangeTimer.Start();
    DtxqcdPrecisionChange(p_f, p);
    PrecChangeTimer.Stop();
    MatrixTimer.Start();
    DtxqcdHermOpF(Mop_f, p_f, mmp_f, tmp_f);
    MatrixTimer.Stop();
    PrecChangeTimer.Start();
    DtxqcdPrecisionChange(mmp, mmp_f);
    PrecChangeTimer.Stop();

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

    // ---- Reliable update: recompute the true primary residual in DP ----
    if (k % ReliableUpdateFreq == 0) {
      ReliableTimer.Start();
      RealD c_old = c;
      RealD dd = DtxqcdHermOpD(Mop_d, psi[0], mmp, tmp);  // mmp = MdagM psi[0]
      (void)dd;
      Axpy(mmp, poles[0], psi[0]);                        // + poles[0] psi[0]
      // r = src - mmp
      for (int a = 0; a < DtxqcdNf; ++a) {
        r.upper.f[a] = src.upper.f[a] - mmp.upper.f[a];
        r.lower.f[a] = src.lower.f[a] - mmp.lower.f[a];
      }
      c = norm2(r);
      ReliableTimer.Stop();
      ++reliable_updates;
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGMixedPrec] reliable update k=" << k
                << " replaced |r|^2=" << c_old << " with |r|^2=" << c
                << std::endl;
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
      // The SP multishift Krylov pass typically converges the primary shift
      // but leaves the tighter shifts just above tol (SP residual floor
      // ~1e-7).  Polish each under-converged shift with a single-shift
      // reliable-update MP CG, warm-started from psi[s] -- still SP matvec, so
      // the speedup is preserved; far cheaper than a full DP multishift.
      //
      // DTXQCD_MP_CG_CLEANUP=1: skip the per-shift DP probe + DP reliable
      // updates inside cleanup; run pure-SP cleanup; verify with ONE DP
      // HermOp per shift at the end.  If any shift fails verification, fall
      // back to legacy MP cleanup for that shift.
      const char *cleanup_env = std::getenv("DTXQCD_MP_CG_CLEANUP");
      const int cleanup_sp_only = (cleanup_env && *cleanup_env) ? std::atoi(cleanup_env) : 0;
      int n_cleanup = 0;
      int n_fallback = 0;
      if (cleanup_sp_only) {
        // Skip the per-shift DP probe.  Use the SP residual bound
        // c * z[s][iz]^2 (already enforced by the convergence check above) as
        // the proxy for "is shift s done?".  Then run pure-SP cleanup for
        // shifts whose SP bound exceeds the per-shift target; do a single DP
        // verify per shift at the end and fall back if needed.
        for (int s = 0; s < nshift; ++s) {
          // Trust the multishift SP recurrence: every shift's residual proxy
          // already passed its target.  Run the SP-only single-shift cleanup
          // anyway as a safety polish; the warm start means it usually
          // returns at iter 0 (cheap).
          ++n_cleanup;
          bool sp_ok = DTXQCDSingleShiftCGSpCleanup(Mop_f, poles[s], tol[s],
                                                    src, psi[s], MaxIter);
          // One DP HermOp to verify.  This is the only DP work per shift in
          // the SP-only cleanup path.
          RealD dd = DtxqcdHermOpD(Mop_d, psi[s], mmp, tmp);
          (void)dd;
          Axpy(mmp, poles[s], psi[s]);
          for (int a = 0; a < DtxqcdNf; ++a) {
            r.upper.f[a] = mmp.upper.f[a] - src.upper.f[a];
            r.lower.f[a] = mmp.lower.f[a] - src.lower.f[a];
          }
          RealD true_resid = std::sqrt(norm2(r) / norm2(src));
          if (!sp_ok || true_resid > tol[s]) {
            // SP cleanup wasn't enough.  Fall back to the legacy MP cleanup
            // (DP reliable update inside) for this shift only.
            ++n_fallback;
            DTXQCDSingleShiftCGMixedPrec(Mop_d, Mop_f, poles[s], tol[s], src,
                                         psi[s], MaxIter, ReliableUpdateFreq);
          }
        }
        std::cout << GridLogMessage
                  << "[DTXQCDMultiShiftCGMixedPrec] converged iter=" << k
                  << " nshift=" << nshift
                  << " reliable_updates=" << reliable_updates
                  << " sp_cleanup_shifts=" << n_cleanup
                  << " sp_cleanup_fallback=" << n_fallback
                  << "  SP-matrix=" << MatrixTimer.Elapsed()
                  << "  precChange=" << PrecChangeTimer.Elapsed()
                  << "  reliable=" << ReliableTimer.Elapsed()
                  << "  [MP_CG_CLEANUP=1]" << std::endl;
        return;
      }
      // Legacy path: per-shift DP probe + MP cleanup (with DP reliable update inside).
      for (int s = 0; s < nshift; ++s) {
        RealD dd = DtxqcdHermOpD(Mop_d, psi[s], mmp, tmp);
        (void)dd;
        Axpy(mmp, poles[s], psi[s]);
        for (int a = 0; a < DtxqcdNf; ++a) {
          r.upper.f[a] = mmp.upper.f[a] - src.upper.f[a];
          r.lower.f[a] = mmp.lower.f[a] - src.lower.f[a];
        }
        RealD true_resid = std::sqrt(norm2(r) / norm2(src));
        if (true_resid > tol[s]) {
          ++n_cleanup;
          DTXQCDSingleShiftCGMixedPrec(Mop_d, Mop_f, poles[s], tol[s], src,
                                       psi[s], MaxIter, ReliableUpdateFreq);
        }
      }
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGMixedPrec] converged iter=" << k
                << " nshift=" << nshift
                << " reliable_updates=" << reliable_updates
                << " sp_cleanup_shifts=" << n_cleanup
                << "  SP-matrix=" << MatrixTimer.Elapsed()
                << "  precChange=" << PrecChangeTimer.Elapsed()
                << "  reliable=" << ReliableTimer.Elapsed() << std::endl;
      return;
    }
  }
  // Primary never converged in MaxIter: fall back to the all-DP solver to
  // guarantee correctness (should be rare; indicates a conditioning problem).
  std::cout << GridLogMessage
            << "[DTXQCDMultiShiftCGMixedPrec] did not converge in " << MaxIter
            << " iterations (c=" << c << ") -- DP cleanup" << std::endl;
  DTXQCDMultiShiftCG(Mop_d, poles, tol, src, psi, MaxIter);
}

NAMESPACE_END(Grid);
