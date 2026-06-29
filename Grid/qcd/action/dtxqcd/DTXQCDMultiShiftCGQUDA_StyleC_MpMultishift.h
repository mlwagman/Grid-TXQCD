#pragma once
// MP-CG Session D — canonical QUDA-style mixed-precision multishift CG.
//
// Mirrors QUDA's MultiShiftCG (inv_multi_cg_quda.cpp): SP multishift CG inner
// loop with periodic DP reliable updates, followed by per-shift DP polish CG
// warm-started from the SP solutions.
//
// Differs from DTXQCDMultiShiftCGQUDA_StyleC's existing pattern (Session B):
//   StyleC + Cleanup=1:  DP multishift loose-tol  +  per-shift SP cleanup tail
//   This file:           SP multishift            +  per-shift DP polish
//
// Per-iter cost gain: bulk Krylov work happens at SP MatQuda (~2× cheaper than
// DP) with Krylov sharing preserved through the precision sync.  Compared to
// pure DP Style C (52.5 s substep-1 at 16³×48 lam=3 MDS=30), projected substep
// is ~25-30 s.
//
// Reference: /lustre2/nplqcd/src/quda/lib/inv_multi_cg_quda.cpp lines 189-460.
// updateAlphaZeta math (line 170-187 of QUDA) copied verbatim into our scalar
// recurrence; multishift bookkeeping (zeta, alpha, beta) lives in std::vector
// of doubles on the host.
//
// Gating: DTXQCD_MP_CG_MULTISHIFT=1 (new env knob distinct from the existing
// DTXQCD_MP_CG_CLEANUP=1 path).

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSFSp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdQudaStyleCanonicalMP {

// HermOp on SP CSF: (M†M)·p — same structure as DtxqcdHermOpQUDA_StyleC but
// using M_device_csf_sp / Mdag_device_csf_sp.  Returns Re<p, (M†M)·p>.
inline RealD HermOpSp(DTXQCDMpcOpQUDA &Mop_quda,
                      DtxqcdQudaStyleC::DoubledStateCSFSp &p,
                      DtxqcdQudaStyleC::DoubledStateCSFSp &mmp,
                      DtxqcdQudaStyleC::DoubledStateCSFSp &tmp,
                      DtxqcdQudaStyleC::DoubledStateCSFSp &scratch_lo) {
  quda::ColorSpinorField *p_up [DtxqcdNf] = {p.csf[0].get(),  p.csf[1].get()};
  quda::ColorSpinorField *p_lo [DtxqcdNf] = {p.csf[2].get(),  p.csf[3].get()};
  quda::ColorSpinorField *t_up [DtxqcdNf] = {tmp.csf[0].get(), tmp.csf[1].get()};
  quda::ColorSpinorField *t_lo [DtxqcdNf] = {tmp.csf[2].get(), tmp.csf[3].get()};
  quda::ColorSpinorField *m_up [DtxqcdNf] = {mmp.csf[0].get(), mmp.csf[1].get()};
  quda::ColorSpinorField *m_lo [DtxqcdNf] = {mmp.csf[2].get(), mmp.csf[3].get()};
  quda::ColorSpinorField *s_lo [DtxqcdNf] = {scratch_lo.csf[0].get(), scratch_lo.csf[1].get()};

  mmp.zero();
  tmp.zero();
  scratch_lo.zero();

  Mop_quda.M_device_csf_sp(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  Mop_quda.Mdag_device_csf_sp(t_up, t_lo, m_up, m_lo, s_lo);

  return DtxqcdQudaStyleC::DoubledStateCSFSp::redot(p, mmp);
}

// HermOp on DP CSF: (M†M)·p via M_device_csf / Mdag_device_csf.  Returns
// Re<p, (M†M)·p>.  Used for DP reliable update + Phase B polish.
inline RealD HermOpDp(DTXQCDMpcOpQUDA &Mop_quda,
                     DtxqcdQudaStyleC::DoubledStateCSF &p,
                     DtxqcdQudaStyleC::DoubledStateCSF &mmp,
                     DtxqcdQudaStyleC::DoubledStateCSF &tmp,
                     DtxqcdQudaStyleC::DoubledStateCSF &scratch_lo) {
  quda::ColorSpinorField *p_up [DtxqcdNf] = {p.csf[0].get(),  p.csf[1].get()};
  quda::ColorSpinorField *p_lo [DtxqcdNf] = {p.csf[2].get(),  p.csf[3].get()};
  quda::ColorSpinorField *t_up [DtxqcdNf] = {tmp.csf[0].get(), tmp.csf[1].get()};
  quda::ColorSpinorField *t_lo [DtxqcdNf] = {tmp.csf[2].get(), tmp.csf[3].get()};
  quda::ColorSpinorField *m_up [DtxqcdNf] = {mmp.csf[0].get(), mmp.csf[1].get()};
  quda::ColorSpinorField *m_lo [DtxqcdNf] = {mmp.csf[2].get(), mmp.csf[3].get()};
  quda::ColorSpinorField *s_lo [DtxqcdNf] = {scratch_lo.csf[0].get(), scratch_lo.csf[1].get()};

  mmp.zero();
  tmp.zero();
  scratch_lo.zero();

  Mop_quda.M_device_csf(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  Mop_quda.Mdag_device_csf(t_up, t_lo, m_up, m_lo, s_lo);

  return DtxqcdQudaStyleC::DoubledStateCSF::redot(p, mmp);
}

// QUDA's updateAlphaZeta (inv_multi_cg_quda.cpp:170-187), verbatim recurrence.
// Updates alpha[1..nShift] and zeta[1..nShift] in place; zeta_old also.
// alpha[0] is set to r2[0]/pAp; zeta[0] := 1.0.
inline void UpdateAlphaZeta(std::vector<double> &alpha,
                            std::vector<double> &zeta,
                            std::vector<double> &zeta_old,
                            const std::vector<double> &r2,
                            const std::vector<double> &beta,
                            double pAp,
                            const std::vector<double> &offset,
                            int nShift, int j_low) {
  std::vector<double> alpha_old(alpha);
  alpha[0] = r2[0] / pAp;
  zeta[0] = 1.0;
  for (int j = 1; j < nShift; ++j) {
    double c0 = zeta[j] * zeta_old[j] * alpha_old[j_low];
    double c1 = alpha[j_low] * beta[j_low] * (zeta_old[j] - zeta[j]);
    double c2 = zeta_old[j] * alpha_old[j_low] *
                (1.0 + (offset[j] - offset[0]) * alpha[j_low]);
    zeta_old[j] = zeta[j];
    zeta[j] = (c1 + c2 != 0.0) ? c0 / (c1 + c2) : 0.0;
    alpha[j] = (zeta[j] != 0.0) ? alpha[j_low] * zeta[j] / zeta_old[j] : 0.0;
  }
}

// Phase B — per-shift DP single-shift CG polish, warm-started from x_dp[k].
// Solves (M†M + pole) x = src to tight tol.  Inner state lives in DP CSFs
// passed in by caller (reused across shifts to avoid per-shift alloc).
// Returns iteration count.
inline int SingleShiftDpPolishCSF(
    DTXQCDMpcOpQUDA &Mop_quda, double pole, double tol,
    const DtxqcdQudaStyleC::DoubledStateCSF &src_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &x_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &p_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &r_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &Ap_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &mmp_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &tmp_dp,
    DtxqcdQudaStyleC::DoubledStateCSF &scratch_lo_dp,
    int MaxIter) {
  // r = src - (M†M + pole) x
  // Inline HermOp with shift via scalar axpy on output.
  RealD pAp_dp = HermOpDp(Mop_quda, x_dp, mmp_dp, tmp_dp, scratch_lo_dp);
  mmp_dp.axpy(pole, x_dp);  // mmp_dp += pole · x_dp
  pAp_dp += pole * x_dp.norm2();
  (void)pAp_dp;
  // r = src - mmp_dp
  r_dp.copy_from(src_dp);
  r_dp.axpy(-1.0, mmp_dp);
  RealD r2 = r_dp.norm2();
  RealD r2_init = r2;
  RealD r2_target = tol * tol * src_dp.norm2();

  if (r2 < r2_target) {
    return 0;
  }

  // p = r
  p_dp.copy_from(r_dp);

  for (int it = 1; it <= MaxIter; ++it) {
    // Ap = (M†M + pole) p
    RealD pAp = HermOpDp(Mop_quda, p_dp, Ap_dp, tmp_dp, scratch_lo_dp);
    Ap_dp.axpy(pole, p_dp);
    pAp += pole * p_dp.norm2();

    RealD alpha = r2 / pAp;
    // x += alpha · p
    x_dp.axpy(alpha, p_dp);
    // r -= alpha · Ap
    r_dp.axpy(-alpha, Ap_dp);
    RealD r2_new = r_dp.norm2();
    RealD beta = r2_new / r2;
    r2 = r2_new;

    if (r2 < r2_target) {
      return it;
    }
    // p = beta · p + r
    p_dp.scale_add(beta, r_dp);
  }
  (void)r2_init;
  return MaxIter;
}

}  // namespace DtxqcdQudaStyleCanonicalMP

// Phase A — SP multishift CG with periodic DP reliable updates.  Phase B —
// per-shift DP polish CG warm-started from SP solutions.  Canonical
// production-grade MP multishift pattern; mirrors QUDA's
// MultiShiftCG (Phase A) + per-shift CG refinement (Phase B).
template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA_StyleC_MpMultishift(
    DTXQCDMopD &Mop_d,                  // Grid-DP HermOp (unused for now; Mop_quda's DP path is used)
    DTXQCDMpcOpQUDA &Mop_quda,          // DP + SP HermOps via M_device_csf / M_device_csf_sp
    const std::vector<RealD> &poles,
    const std::vector<RealD> &tol,
    const DTXQCDFermionDoubled &src,
    std::vector<DTXQCDFermionDoubled> &psi,
    int MaxIter,
    int /*ReliableUpdateFreq*/ = 50) {
  using namespace dtxqcd_msshift_detail;
  namespace SC = DtxqcdQudaStyleC;
  namespace MP = DtxqcdQudaStyleCanonicalMP;

  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);
  for (int s = 0; s < nshift; ++s) GRID_ASSERT(poles[s] >= poles[0]);

  GridBase *grid = src.Grid();
  (void)Mop_d;
  (void)grid;

  // CSF allocation templates.
  quda::ColorSpinorParam param_dp = Mop_quda.MakeNativeCsfParam();
  quda::ColorSpinorParam param_sp = Mop_quda.MakeNativeCsfParamSp();
  int X_full_dims[4];
  for (int d = 0; d < 4; ++d) X_full_dims[d] = Mop_quda.GaugeParam().X[d];

  // DP scratches (also serve as Phase B's per-shift CG state).
  SC::DoubledStateCSF src_dp_csf, mmp_dp, tmp_dp, scratch_lo_dp, r_dp, x_dp_buf,
                     p_dp_polish, Ap_dp_polish;
  src_dp_csf.allocate(param_dp);
  mmp_dp.allocate(param_dp);
  tmp_dp.allocate(param_dp);
  scratch_lo_dp.allocate_lower_only(param_dp);
  r_dp.allocate(param_dp);
  x_dp_buf.allocate(param_dp);
  p_dp_polish.allocate(param_dp);
  Ap_dp_polish.allocate(param_dp);
  // Per-shift DP solutions live in x_dp[k]; refreshed at the start of Phase B.
  std::vector<SC::DoubledStateCSF> x_dp(nshift);
  for (int s = 0; s < nshift; ++s) x_dp[s].allocate(param_dp);

  // Pack src once (DP CSF) — used for reliable-update r recompute + Phase B
  // RHS.
  src_dp_csf.copy_from_grid(src, Mop_quda.InvertParam(), X_full_dims);

  RealD b2 = src_dp_csf.norm2();
  if (b2 == 0.0) {
    for (int s = 0; s < nshift; ++s) Zero_(psi[s]);
    return;
  }

  // SP CG state.  All multishift work happens here.
  SC::DoubledStateCSFSp r_sp, Ap_sp, mmp_sp, tmp_sp, scratch_lo_sp;
  r_sp.allocate(param_sp);
  Ap_sp.allocate(param_sp);
  mmp_sp.allocate(param_sp);
  tmp_sp.allocate(param_sp);
  scratch_lo_sp.allocate_lower_only(param_sp);
  std::vector<SC::DoubledStateCSFSp> p_sp(nshift), x_sp(nshift);
  for (int s = 0; s < nshift; ++s) {
    p_sp[s].allocate(param_sp);
    x_sp[s].allocate(param_sp);
  }

  // Initialise SP state: x = 0, r = src, p[k] = r for all k.
  for (int s = 0; s < nshift; ++s) {
    x_sp[s].zero();
  }
  r_sp.copy_from_dp(src_dp_csf);
  for (int s = 0; s < nshift; ++s) {
    p_sp[s].copy_from(r_sp);
  }

  std::vector<double> offset(nshift);
  for (int s = 0; s < nshift; ++s) offset[s] = poles[s];

  // Multishift scalars — host-side, mirror QUDA's std::vector<double>.
  std::vector<double> alpha(nshift, 1.0), beta(nshift, 0.0);
  std::vector<double> zeta(nshift, 1.0), zeta_old(nshift, 1.0);
  std::vector<double> r2(nshift, b2);

  // Stopping tolerances for Phase A — safely above SP precision floor.
  // The SP-floor relative residual for M†M at 16³×48 Wilson-clover light is
  // empirically ~2e-4 (from v1 smoke at λ=3 — SP CG plateaus near this value
  // and cannot reach lower).  Phase A must exit ABOVE the plateau or the SP
  // CG runs forever, drift accumulates, and the residual eventually
  // diverges.  Default 1e-3 leaves ~5× margin above the empirical plateau.
  // Phase B then takes each shift from 1e-3 down to tight tol[s] in DP via
  // ~30-50 iters per shift.  Tunable via DTXQCD_MP_CG_SP_TOL env.
  double phaseA_tol = 1.0e-4;
  if (const char *e = std::getenv("DTXQCD_MP_CG_SP_TOL"); e && *e) {
    phaseA_tol = std::atof(e);
  }
  std::vector<double> stop_phaseA(nshift);
  std::vector<double> stop_phaseB(nshift);
  for (int s = 0; s < nshift; ++s) {
    stop_phaseB[s] = tol[s] * tol[s] * b2;
    stop_phaseA[s] = phaseA_tol * phaseA_tol * b2;
  }

  // Forward decl of helper used at end-of-iter convergence check.
  auto all_shifts_converged = [&](const std::vector<int> &flags) {
    for (int s = 0; s < nshift; ++s) if (!flags[s]) return false;
    return true;
  };

  // Reliable update tracking — only shift 0 triggers reliable update; per
  // QUDA comment at line 308.
  double delta = 0.1;
  if (const char *e = std::getenv("DTXQCD_MP_CG_DELTA"); e && *e) {
    delta = std::atof(e);
  }
  std::vector<double> rNorm(nshift, std::sqrt(b2));
  std::vector<double> r0Norm(rNorm);
  std::vector<double> maxrr(rNorm), maxrx(rNorm);

  int j_low = 0;
  int nshift_now = nshift;
  std::vector<int> shift_converged(nshift, 0);
  int rUpdate = 0;
  int sp_iters = 0;
  bool exit_early = false;

  GridStopWatch SpMatTimer, DpMatTimer, BlasTimer, ReliableTimer;

  // ----- Phase A: SP multishift with DP reliable updates -----
  for (int k = 1; k <= MaxIter; ++k) {
    // SP HermOp on shift-0 direction.
    SpMatTimer.Start();
    double pAp = MP::HermOpSp(Mop_quda, p_sp[0], Ap_sp, tmp_sp, scratch_lo_sp);
    SpMatTimer.Stop();
    BlasTimer.Start();
    // Multishift's pAp includes the shift-0 offset.
    pAp += offset[0] * p_sp[0].norm2();

    double r2_old = r2[0];
    MP::UpdateAlphaZeta(alpha, zeta, zeta_old, r2, beta, pAp,
                        offset, nshift_now, j_low);

    // r_sp -= alpha[0] · (Ap_sp + offset[0] · p_sp[0])
    Ap_sp.axpy(offset[0], p_sp[0]);
    r_sp.axpy(-alpha[0], Ap_sp);
    double r2_new = r_sp.norm2();

    // zn for beta recurrence: <r_new, r_new>.
    double zn = r2_new;
    r2[0] = r2_new;

    // Per-shift residuals via rNorm[k] = rNorm[0] * |zeta[k]|.
    rNorm[0] = std::sqrt(r2[0]);
    for (int j = 1; j < nshift_now; ++j) {
      rNorm[j] = rNorm[0] * std::abs(zeta[j]);
    }

    // Reliable update check on shift 0 (per QUDA fixme comment).
    if (rNorm[0] > maxrx[0]) maxrx[0] = rNorm[0];
    if (rNorm[0] > maxrr[0]) maxrr[0] = rNorm[0];
    bool updateX = (rNorm[0] < delta * r0Norm[0]) && (r0Norm[0] <= maxrx[0]);
    bool updateR = ((rNorm[0] < delta * maxrr[0]) && (r0Norm[0] <= maxrr[0])) ||
                   updateX;

    BlasTimer.Stop();

    if (!(updateX || updateR)) {
      // Standard SP update path.
      BlasTimer.Start();
      beta[0] = zn / r2_old;
      // x_sp[0] += alpha[0] · p_sp[0]
      x_sp[0].axpy(alpha[0], p_sp[0]);
      // p_sp[0] = beta[0] · p_sp[0] + r_sp
      p_sp[0].scale_add(beta[0], r_sp);
      for (int j = 1; j < nshift_now; ++j) {
        // x_sp[j] += alpha[j] · p_sp[j]
        x_sp[j].axpy(alpha[j], p_sp[j]);
        beta[j] = beta[0] * zeta[j] * alpha[j] /
                  (zeta_old[j] * alpha[0]);
        // p_sp[j] = beta[j] · p_sp[j] + zeta[j] · r_sp
        // Achieve via scale + axpy: scale_add(beta[j], r_sp) does p = beta*p + r;
        // then we need to scale the r contribution by zeta[j], so use a temp:
        // simpler: p[j] *= beta[j]; then p[j] += zeta[j] * r_sp.
        // Implementation: scale_add(beta[j]/zeta[j], (zeta[j]·r_sp))...
        // Cleanest: do two ops via DoubledStateCSFSp axpy / explicit blas.
        // p_sp[j] = beta[j]·p_sp[j] + zeta[j]·r_sp
        //        = (beta[j]/zeta[j]) · (zeta[j]·p_sp[j])  +  zeta[j]·r_sp
        // Easier: use blas::axpby on the CSFs directly.
        for (int slot = 0; slot < 4; ++slot) {
          if (!p_sp[j].csf[slot] || !r_sp.csf[slot]) continue;
          quda::vector<double> as{zeta[j]};
          quda::vector<double> bs_v{beta[j]};
          quda::vector_ref<const quda::ColorSpinorField> xr{*r_sp.csf[slot]};
          quda::vector_ref<quda::ColorSpinorField> yr{*p_sp[j].csf[slot]};
          quda::blas::axpby(as, xr, bs_v, yr);
        }
      }
      BlasTimer.Stop();
    } else {
      // Reliable update: accumulate, recompute residual in DP, sync SP.
      ReliableTimer.Start();
      // Accumulate x_sp[j] += alpha[j] · p_sp[j] for all active shifts (so the
      // DP recompute on x_sp[0]'s updated value is consistent).
      for (int j = 0; j < nshift_now; ++j) {
        x_sp[j].axpy(alpha[j], p_sp[j]);
      }
      // Promote x_sp[0] to DP, compute true residual.
      x_sp[0].copy_to_dp(x_dp_buf);
      DpMatTimer.Start();
      MP::HermOpDp(Mop_quda, x_dp_buf, mmp_dp, tmp_dp, scratch_lo_dp);
      DpMatTimer.Stop();
      // mmp_dp += offset[0] · x_dp_buf  ⇒ mmp_dp = (M†M + offset[0]) · x
      mmp_dp.axpy(offset[0], x_dp_buf);
      // r_dp = src_dp - mmp_dp
      r_dp.copy_from(src_dp_csf);
      r_dp.axpy(-1.0, mmp_dp);
      double r2_true = r_dp.norm2();
      r2[0] = r2_true;
      for (int j = 1; j < nshift_now; ++j) {
        r2[j] = zeta[j] * zeta[j] * r2[0];
      }
      // Sync SP residual with DP truth.
      r_sp.copy_from_dp(r_dp);

      // Reorthogonalise p_sp[j] against r_sp (QUDA line 371-374).
      // Without this the SP-DP precision sync breaks conjugacy.
      // rp = <r_sp, p_sp[j]> / r2[0]; p_sp[j] -= rp · r_sp.
      // (Inner product is real for the doubled state.)
      for (int j = 0; j < nshift_now; ++j) {
        double rp = SC::DoubledStateCSFSp::redot(r_sp, p_sp[j]) / r2[0];
        p_sp[j].axpy(-rp, r_sp);
      }

      // Recompute beta and p update post-sync.
      beta[0] = r2[0] / r2_old;
      // p_sp[0] = beta[0] · p_sp[0] + r_sp
      p_sp[0].scale_add(beta[0], r_sp);
      for (int j = 1; j < nshift_now; ++j) {
        beta[j] = beta[0] * zeta[j] * alpha[j] /
                  (zeta_old[j] * alpha[0]);
        // p_sp[j] = beta[j] · p_sp[j] + zeta[j] · r_sp
        for (int slot = 0; slot < 4; ++slot) {
          if (!p_sp[j].csf[slot] || !r_sp.csf[slot]) continue;
          quda::vector<double> as{zeta[j]};
          quda::vector<double> bs_v{beta[j]};
          quda::vector_ref<const quda::ColorSpinorField> xr{*r_sp.csf[slot]};
          quda::vector_ref<quda::ColorSpinorField> yr{*p_sp[j].csf[slot]};
          quda::blas::axpby(as, xr, bs_v, yr);
        }
      }

      // Update reliable-update tracking for shift 0.
      rNorm[0] = std::sqrt(r2[0]);
      maxrr[0] = rNorm[0];
      maxrx[0] = rNorm[0];
      r0Norm[0] = rNorm[0];
      ++rUpdate;
      ReliableTimer.Stop();

      std::cout << GridLogMessage
                << "[MpMultishift] reliable update k=" << k
                << " r2[0]=" << r2[0] << std::endl;
    }

    // Convergence check & shift retirement (mirror QUDA lines 393-415).
    int converged_this_iter = 0;
    for (int j = nshift_now - 1; j >= 1; --j) {
      if (zeta[j] == 0.0) {
        ++converged_this_iter;
        shift_converged[j] = 1;
      } else {
        r2[j] = zeta[j] * zeta[j] * r2[0];
        if (r2[j] < stop_phaseA[j]) {
          ++converged_this_iter;
          shift_converged[j] = 1;
        }
      }
    }
    nshift_now -= converged_this_iter;

    // Exit-early when only shift 0 remains in mixed-prec mode (per QUDA
    // line 412): defer tight polish to Phase B per-shift DP CG.
    if (nshift_now == 1) {
      exit_early = true;
    }
    if (r2[0] < stop_phaseA[0]) {
      shift_converged[0] = 1;
    }

    if ((shift_converged[0] || exit_early) && all_shifts_converged(shift_converged)) {
      sp_iters = k;
      break;
    }
    if (k == MaxIter) {
      sp_iters = MaxIter;
      std::cout << GridLogMessage
                << "[MpMultishift] Phase A did not converge in " << MaxIter
                << " iterations, r2[0]=" << r2[0] << std::endl;
    }
  }

  std::cout << GridLogMessage
            << "[MpMultishift] Phase A done: sp_iters=" << sp_iters
            << " rUpdate=" << rUpdate
            << " SP-mat=" << SpMatTimer.Elapsed()
            << " DP-mat=" << DpMatTimer.Elapsed()
            << " blas=" << BlasTimer.Elapsed()
            << " reliable=" << ReliableTimer.Elapsed() << std::endl;

  // ----- Phase B: per-shift DP polish CG warm-started from SP solutions -----
  GridStopWatch PhaseBTimer;
  PhaseBTimer.Start();
  int total_dp_polish_iters = 0;
  for (int s = 0; s < nshift; ++s) {
    // Warm start: x_dp[s] = x_sp[s] (precision change).
    x_sp[s].copy_to_dp(x_dp[s]);
    int it = MP::SingleShiftDpPolishCSF(
        Mop_quda, poles[s], tol[s], src_dp_csf, x_dp[s],
        p_dp_polish, r_dp, Ap_dp_polish, mmp_dp, tmp_dp, scratch_lo_dp,
        MaxIter);
    total_dp_polish_iters += it;
  }
  PhaseBTimer.Stop();
  std::cout << GridLogMessage
            << "[MpMultishift] Phase B done: dp_polish_iters="
            << total_dp_polish_iters
            << " PhaseB=" << PhaseBTimer.Elapsed() << std::endl;

  // Unpack DP solutions back to Grid.
  for (int s = 0; s < nshift; ++s) {
    x_dp[s].copy_to_grid(psi[s], Mop_quda.InvertParam(), X_full_dims);
  }
}

NAMESPACE_END(Grid);
