#pragma once
// Phase β Session B — SP-only single-shift CG cleanup on native CSF.
//
// After Style C's DP multishift converges, per-shift refinement: load each
// pole's solution into SP CSF, run single-shift CG entirely on SP CSF via
// M_device_csf_sp / Mdag_device_csf_sp, periodically flush with a DP HermOp
// (reliable update) to correct SP drift, return refined DP solution.
//
// Pattern mirrors DTXQCDSingleShiftCGMixedPrec in DTXQCDMultiShiftCGMixedPrec.h
// (lines 127-194) but operates on native DoubledStateCSFSp / DoubledStateCSF.
// Reliable-update interval defaults to 50 iters.
//
// Gate: target SP precision floor (~1e-6 rel) — caller's `tol` is the DP
// outer target (1e-8 typically).  SP iterations bring us within SP floor;
// DP reliable updates promote to DP floor.

#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSFSp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaStyleC {

// Apply (M†M + pole)·in on SP CSF.  Mirrors DtxqcdHermOpQUDA_StyleC but with
// SP M_device_csf_sp + Mdag_device_csf_sp.
inline void HermOpSp(DTXQCDMpcOpQUDA &Mop_quda,
                     DoubledStateCSFSp &in,
                     DoubledStateCSFSp &mmp,
                     DoubledStateCSFSp &tmp,
                     DoubledStateCSFSp &scratch_lo,
                     double pole) {
  quda::ColorSpinorField *p_up [DtxqcdNf] = {in.csf[0].get(),  in.csf[1].get()};
  quda::ColorSpinorField *p_lo [DtxqcdNf] = {in.csf[2].get(),  in.csf[3].get()};
  quda::ColorSpinorField *t_up [DtxqcdNf] = {tmp.csf[0].get(), tmp.csf[1].get()};
  quda::ColorSpinorField *t_lo [DtxqcdNf] = {tmp.csf[2].get(), tmp.csf[3].get()};
  quda::ColorSpinorField *m_up [DtxqcdNf] = {mmp.csf[0].get(), mmp.csf[1].get()};
  quda::ColorSpinorField *m_lo [DtxqcdNf] = {mmp.csf[2].get(), mmp.csf[3].get()};
  quda::ColorSpinorField *s_lo [DtxqcdNf] = {scratch_lo.csf[0].get(),
                                              scratch_lo.csf[1].get()};

  mmp.zero();
  tmp.zero();
  scratch_lo.zero();
  Mop_quda.M_device_csf_sp(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  Mop_quda.Mdag_device_csf_sp(t_up, t_lo, m_up, m_lo, s_lo);
  // mmp += pole · in  (single shift add)
  mmp.axpy(pole, in);
}

// Apply (M†M + pole)·in on DP CSF.  Used for reliable updates + final verify.
inline void HermOpDp(DTXQCDMpcOpQUDA &Mop_quda,
                     DoubledStateCSF &in,
                     DoubledStateCSF &mmp,
                     DoubledStateCSF &tmp,
                     DoubledStateCSF &scratch_lo,
                     double pole) {
  quda::ColorSpinorField *p_up [DtxqcdNf] = {in.csf[0].get(),  in.csf[1].get()};
  quda::ColorSpinorField *p_lo [DtxqcdNf] = {in.csf[2].get(),  in.csf[3].get()};
  quda::ColorSpinorField *t_up [DtxqcdNf] = {tmp.csf[0].get(), tmp.csf[1].get()};
  quda::ColorSpinorField *t_lo [DtxqcdNf] = {tmp.csf[2].get(), tmp.csf[3].get()};
  quda::ColorSpinorField *m_up [DtxqcdNf] = {mmp.csf[0].get(), mmp.csf[1].get()};
  quda::ColorSpinorField *m_lo [DtxqcdNf] = {mmp.csf[2].get(), mmp.csf[3].get()};
  quda::ColorSpinorField *s_lo [DtxqcdNf] = {scratch_lo.csf[0].get(),
                                              scratch_lo.csf[1].get()};

  mmp.zero();
  tmp.zero();
  scratch_lo.zero();
  Mop_quda.M_device_csf(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  Mop_quda.Mdag_device_csf(t_up, t_lo, m_up, m_lo, s_lo);
  mmp.axpy(pole, in);
}

// Single-shift CG cleanup on native CSF, mirrors DTXQCDSingleShiftCGMixedPrec.
//
//   src_dp, x_dp           : DP CSF source and warm-start solution
//   p_sp, r_sp, Ap_sp, x_sp: SP CSF scratch
//   mmp_dp, tmp_dp, scratch_lo_dp : DP scratch for reliable updates + verify
//   mmp_sp, tmp_sp, scratch_lo_sp : SP scratch for inner HermOpSp
//
// Returns iter count.  On convergence, x_dp holds the refined DP solution.
inline int SingleShiftCGSpCleanupCSF(
    DTXQCDMpcOpQUDA &Mop_quda,
    double pole, double tol,
    const DoubledStateCSF &src_dp,
    DoubledStateCSF &x_dp,
    DoubledStateCSFSp &p_sp,  DoubledStateCSFSp &r_sp,
    DoubledStateCSFSp &Ap_sp, DoubledStateCSFSp &x_sp,
    DoubledStateCSFSp &mmp_sp, DoubledStateCSFSp &tmp_sp,
    DoubledStateCSFSp &scratch_lo_sp,
    DoubledStateCSF &mmp_dp, DoubledStateCSF &tmp_dp,
    DoubledStateCSF &scratch_lo_dp,
    DoubledStateCSF &r_dp,
    int MaxIter, int ReliableUpdateFreq = 50) {
  // True initial residual in DP (warm start from x_dp).
  HermOpDp(Mop_quda, x_dp, mmp_dp, tmp_dp, scratch_lo_dp, pole);
  // r_dp = src_dp - mmp_dp  (use axpy: r_dp = src_dp; r_dp -= mmp_dp)
  r_dp.zero();
  r_dp.axpy( 1.0, src_dp);
  r_dp.axpy(-1.0, mmp_dp);

  double rsq = r_dp.norm2();
  double ssq = src_dp.norm2();
  double target = ssq * tol * tol;
  if (rsq < target) return 0;

  // Initialize SP state: x_sp ← x_dp, p_sp ← r_sp ← r_dp
  x_sp.copy_from_dp(x_dp);
  r_sp.copy_from_dp(r_dp);
  p_sp.copy_from(r_sp);
  // rsq is the DP residual (tighter); SP CG inner uses SP rsq computed below.
  double rsq_sp = r_sp.norm2();

  for (int k = 1; k <= MaxIter; ++k) {
    HermOpSp(Mop_quda, p_sp, Ap_sp, tmp_sp, scratch_lo_sp, pole);
    double pAp = DoubledStateCSFSp::redot(p_sp, Ap_sp);
    if (pAp <= 0.0) {
      // Pathological; abort.
      std::cout << GridLogMessage
                << "[SP_CLEANUP] pAp <= 0 at k=" << k << " — abort" << std::endl;
      return k;
    }
    double alpha = rsq_sp / pAp;
    x_sp.axpy( alpha, p_sp);
    r_sp.axpy(-alpha, Ap_sp);
    double rsq_new = r_sp.norm2();

    if (k % ReliableUpdateFreq == 0) {
      // Flush: pull x_sp → x_dp, true DP residual, push refreshed r back to SP.
      x_sp.copy_to_dp(x_dp);
      HermOpDp(Mop_quda, x_dp, mmp_dp, tmp_dp, scratch_lo_dp, pole);
      r_dp.zero();
      r_dp.axpy( 1.0, src_dp);
      r_dp.axpy(-1.0, mmp_dp);
      double rsq_true = r_dp.norm2();
      if (rsq_true < target) {
        return k;
      }
      // Reset SP r from DP true residual, recompute SP rsq, reset p_sp = r_sp.
      r_sp.copy_from_dp(r_dp);
      rsq_new = r_sp.norm2();
      p_sp.copy_from(r_sp);
      rsq_sp = rsq_new;
      // (β skipped for this iter — fresh restart.)
      continue;
    }

    if (rsq_new < target) {
      // Try to confirm with a true DP residual before declaring done.
      x_sp.copy_to_dp(x_dp);
      HermOpDp(Mop_quda, x_dp, mmp_dp, tmp_dp, scratch_lo_dp, pole);
      r_dp.zero();
      r_dp.axpy( 1.0, src_dp);
      r_dp.axpy(-1.0, mmp_dp);
      double rsq_true = r_dp.norm2();
      if (rsq_true < target) return k;
      // Drift: reset and continue.
      r_sp.copy_from_dp(r_dp);
      rsq_new = r_sp.norm2();
      p_sp.copy_from(r_sp);
      rsq_sp = rsq_new;
      continue;
    }

    double beta = rsq_new / rsq_sp;
    rsq_sp = rsq_new;
    p_sp.scale_add(beta, r_sp);
  }
  // MaxIter exhausted; final flush.
  x_sp.copy_to_dp(x_dp);
  return MaxIter;
}

}  // namespace DtxqcdQudaStyleC
NAMESPACE_END(Grid);
