#pragma once
// MP-CG Session H — SP multishift → DP multishift refinement.
//
// PROBLEM with canonical Session D pattern: per-pole DP polish in Phase B
// loses multishift Krylov sharing across shifts.  At 16³×48 lam=3 we measured:
//   v3 (SP_TOL=1e-3): 311 SP + 1417 per-pole DP = 83.6 s
//   v4 (SP_TOL=1e-4): 456 SP + 1142 per-pole DP = 73.2 s
//   Pure DP Style C : 838 multishift DP        = 52.5 s   ← faster than MP-CG
//
// Session H trades per-pole polish for a single DP MULTISHIFT CG that
// preserves Krylov sharing.  Initial state for Phase B is the warm-started
// SP solutions, with zeta(σ) MEASURED from actual DP residual ratios at the
// precision boundary (not predicted from the SP recurrence).
//
// In exact arithmetic, after SP multishift converges the DP-truth residuals
// satisfy r(σ)_DP = ζ(σ) · r(0)_DP exactly (the multishift CG invariant).  SP
// roundoff breaks this by ~ε_SP (~1.19e-7).  Measuring ζ at Phase B init
// handles the SP error; the DP multishift then refines to DP precision in
// a small number of iters (~100-300) because residuals are already small.
//
// Gating: DTXQCD_MP_CG_MULTISHIFT_REFINE=1 (mutually exclusive with the
// canonical DTXQCD_MP_CG_MULTISHIFT=1 — refine takes precedence if both set).

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSFSp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>
// Reuse HermOpSp / HermOpDp / UpdateAlphaZeta from the canonical pattern.
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC_MpMultishift.h>

NAMESPACE_BEGIN(Grid);

// Phase A — SP multishift CG with periodic DP reliable updates (same as
//   canonical pattern, Session D).  Output: x(σ)_SP for all σ.
// Phase B — DP MULTISHIFT CG (NOT per-pole), warm-started from x(σ)_SP.
//   Krylov sharing is preserved through the precision boundary by measuring
//   the multishift ζ at Phase B init from actual DP residual ratios.
template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA_StyleC_MpMultishiftRefine(
    DTXQCDMopD &Mop_d,                  // Grid-DP HermOp (currently unused)
    DTXQCDMpcOpQUDA &Mop_quda,          // DP + SP HermOps via M_device_csf
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

  // DP scratches (shared between Phase A reliable update + Phase B).
  SC::DoubledStateCSF src_dp_csf, mmp_dp, tmp_dp, scratch_lo_dp, r_dp_buf,
                     x_dp_buf;
  src_dp_csf.allocate(param_dp);
  mmp_dp.allocate(param_dp);
  tmp_dp.allocate(param_dp);
  scratch_lo_dp.allocate_lower_only(param_dp);
  r_dp_buf.allocate(param_dp);
  x_dp_buf.allocate(param_dp);

  // Per-shift DP storage for Phase B (warm-started from SP).
  std::vector<SC::DoubledStateCSF> x_dp(nshift), p_dp(nshift), r_dp_init(nshift);
  for (int s = 0; s < nshift; ++s) {
    x_dp[s].allocate(param_dp);
    p_dp[s].allocate(param_dp);
    r_dp_init[s].allocate(param_dp);
  }
  // Phase B shared seed residual + Ap scratch.
  SC::DoubledStateCSF r_seed_dp, Ap_dp;
  r_seed_dp.allocate(param_dp);
  Ap_dp.allocate(param_dp);

  // Pack src once.
  src_dp_csf.copy_from_grid(src, Mop_quda.InvertParam(), X_full_dims);

  RealD b2 = src_dp_csf.norm2();
  if (b2 == 0.0) {
    for (int s = 0; s < nshift; ++s) Zero_(psi[s]);
    return;
  }

  // SP CG state — Phase A only.
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
  for (int s = 0; s < nshift; ++s) x_sp[s].zero();
  r_sp.copy_from_dp(src_dp_csf);
  for (int s = 0; s < nshift; ++s) p_sp[s].copy_from(r_sp);

  std::vector<double> offset(nshift);
  for (int s = 0; s < nshift; ++s) offset[s] = poles[s];

  // Multishift scalars — Phase A.
  std::vector<double> alpha(nshift, 1.0), beta(nshift, 0.0);
  std::vector<double> zeta(nshift, 1.0), zeta_old(nshift, 1.0);
  std::vector<double> r2(nshift, b2);

  // Phase A stop tol (same logic as canonical).
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

  auto all_shifts_converged_a = [&](const std::vector<int> &flags) {
    for (int s = 0; s < nshift; ++s) if (!flags[s]) return false;
    return true;
  };

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
  // (verbatim from canonical Session D pattern — proven to converge cleanly
  // when stop_phaseA is above the SP precision floor.)
  for (int k = 1; k <= MaxIter; ++k) {
    SpMatTimer.Start();
    double pAp = MP::HermOpSp(Mop_quda, p_sp[0], Ap_sp, tmp_sp, scratch_lo_sp);
    SpMatTimer.Stop();
    BlasTimer.Start();
    pAp += offset[0] * p_sp[0].norm2();

    double r2_old = r2[0];
    MP::UpdateAlphaZeta(alpha, zeta, zeta_old, r2, beta, pAp,
                        offset, nshift_now, j_low);

    Ap_sp.axpy(offset[0], p_sp[0]);
    r_sp.axpy(-alpha[0], Ap_sp);
    double r2_new = r_sp.norm2();
    double zn = r2_new;
    r2[0] = r2_new;

    rNorm[0] = std::sqrt(r2[0]);
    for (int j = 1; j < nshift_now; ++j) {
      rNorm[j] = rNorm[0] * std::abs(zeta[j]);
    }

    if (rNorm[0] > maxrx[0]) maxrx[0] = rNorm[0];
    if (rNorm[0] > maxrr[0]) maxrr[0] = rNorm[0];
    bool updateX = (rNorm[0] < delta * r0Norm[0]) && (r0Norm[0] <= maxrx[0]);
    bool updateR = ((rNorm[0] < delta * maxrr[0]) && (r0Norm[0] <= maxrr[0])) ||
                   updateX;

    BlasTimer.Stop();

    if (!(updateX || updateR)) {
      BlasTimer.Start();
      beta[0] = zn / r2_old;
      x_sp[0].axpy(alpha[0], p_sp[0]);
      p_sp[0].scale_add(beta[0], r_sp);
      for (int j = 1; j < nshift_now; ++j) {
        x_sp[j].axpy(alpha[j], p_sp[j]);
        beta[j] = beta[0] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[0]);
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
      ReliableTimer.Start();
      for (int j = 0; j < nshift_now; ++j) {
        x_sp[j].axpy(alpha[j], p_sp[j]);
      }
      x_sp[0].copy_to_dp(x_dp_buf);
      DpMatTimer.Start();
      MP::HermOpDp(Mop_quda, x_dp_buf, mmp_dp, tmp_dp, scratch_lo_dp);
      DpMatTimer.Stop();
      mmp_dp.axpy(offset[0], x_dp_buf);
      r_dp_buf.copy_from(src_dp_csf);
      r_dp_buf.axpy(-1.0, mmp_dp);
      double r2_true = r_dp_buf.norm2();
      r2[0] = r2_true;
      for (int j = 1; j < nshift_now; ++j) {
        r2[j] = zeta[j] * zeta[j] * r2[0];
      }
      r_sp.copy_from_dp(r_dp_buf);

      for (int j = 0; j < nshift_now; ++j) {
        double rp = SC::DoubledStateCSFSp::redot(r_sp, p_sp[j]) / r2[0];
        p_sp[j].axpy(-rp, r_sp);
      }

      beta[0] = r2[0] / r2_old;
      p_sp[0].scale_add(beta[0], r_sp);
      for (int j = 1; j < nshift_now; ++j) {
        beta[j] = beta[0] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[0]);
        for (int slot = 0; slot < 4; ++slot) {
          if (!p_sp[j].csf[slot] || !r_sp.csf[slot]) continue;
          quda::vector<double> as{zeta[j]};
          quda::vector<double> bs_v{beta[j]};
          quda::vector_ref<const quda::ColorSpinorField> xr{*r_sp.csf[slot]};
          quda::vector_ref<quda::ColorSpinorField> yr{*p_sp[j].csf[slot]};
          quda::blas::axpby(as, xr, bs_v, yr);
        }
      }

      rNorm[0] = std::sqrt(r2[0]);
      maxrr[0] = rNorm[0];
      maxrx[0] = rNorm[0];
      r0Norm[0] = rNorm[0];
      ++rUpdate;
      ReliableTimer.Stop();

      std::cout << GridLogMessage
                << "[MpMultishiftRefine] Phase A reliable update k=" << k
                << " r2[0]=" << r2[0] << std::endl;
    }

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
    if (nshift_now == 1) exit_early = true;
    if (r2[0] < stop_phaseA[0]) shift_converged[0] = 1;

    if ((shift_converged[0] || exit_early) && all_shifts_converged_a(shift_converged)) {
      sp_iters = k;
      break;
    }
    if (k == MaxIter) {
      sp_iters = MaxIter;
      std::cout << GridLogMessage
                << "[MpMultishiftRefine] Phase A did not converge in " << MaxIter
                << " iterations, r2[0]=" << r2[0] << std::endl;
    }
  }

  std::cout << GridLogMessage
            << "[MpMultishiftRefine] Phase A done: sp_iters=" << sp_iters
            << " rUpdate=" << rUpdate
            << " SP-mat=" << SpMatTimer.Elapsed()
            << " DP-mat=" << DpMatTimer.Elapsed()
            << " blas=" << BlasTimer.Elapsed()
            << " reliable=" << ReliableTimer.Elapsed() << std::endl;

  // ----- Phase B: DP multishift refinement with measured zeta -----
  GridStopWatch PhaseBTimer, PhaseBMatTimer, PhaseBBlasTimer;
  PhaseBTimer.Start();

  // (1) Warm-start: copy x_sp[s] → x_dp[s] (precision change).
  for (int s = 0; s < nshift; ++s) {
    x_sp[s].copy_to_dp(x_dp[s]);
  }

  // (2) Compute initial DP residuals: r(s)_DP = src - (M†M + σ_s) · x(s)_DP.
  //     One HermOp per shift (N HermOps total).
  std::vector<double> r2_dp(nshift, 0.0);
  for (int s = 0; s < nshift; ++s) {
    PhaseBMatTimer.Start();
    MP::HermOpDp(Mop_quda, x_dp[s], mmp_dp, tmp_dp, scratch_lo_dp);
    PhaseBMatTimer.Stop();
    PhaseBBlasTimer.Start();
    mmp_dp.axpy(poles[s], x_dp[s]);           // mmp = (M†M + σ_s) · x
    r_dp_init[s].copy_from(src_dp_csf);
    r_dp_init[s].axpy(-1.0, mmp_dp);          // r = src - mmp
    r2_dp[s] = r_dp_init[s].norm2();
    PhaseBBlasTimer.Stop();
  }

  // (3) Measure initial zeta from actual DP residual ratios.
  //     In exact arithmetic at Phase A exit, r(s)_DP = ζ(s)_exact · r(0)_DP.
  //     SP roundoff breaks this by ~ε_SP; we use the MEASURED ratio so the
  //     multishift CG starts from a self-consistent state.
  PhaseBBlasTimer.Start();
  std::vector<double> zeta_B(nshift, 1.0), zeta_old_B(nshift, 1.0);
  std::vector<double> alpha_B(nshift, 1.0), beta_B(nshift, 0.0);
  std::vector<double> r2_B(nshift, 0.0);

  zeta_B[0] = 1.0;
  r2_B[0] = r2_dp[0];
  for (int s = 1; s < nshift; ++s) {
    // Sign of zeta from real inner product of r(s) with r(0).
    double rdot = SC::DoubledStateCSF::redot(r_dp_init[0], r_dp_init[s]);
    double norm_s = std::sqrt(r2_dp[s]);
    double norm_0 = std::sqrt(r2_dp[0]);
    double sgn = (rdot >= 0.0) ? 1.0 : -1.0;
    zeta_B[s] = (norm_0 > 0.0) ? sgn * norm_s / norm_0 : 0.0;
    r2_B[s] = zeta_B[s] * zeta_B[s] * r2_B[0];
  }

  // Diagnostic: report measured zeta deviations from SP-predicted (Phase A end).
  double max_zeta_dev = 0.0;
  for (int s = 1; s < nshift; ++s) {
    double dev = std::abs(zeta_B[s] - zeta[s]);
    if (dev > max_zeta_dev) max_zeta_dev = dev;
  }
  std::cout << GridLogMessage
            << "[MpMultishiftRefine] Phase B init: r2_0=" << r2_dp[0]
            << "  max|zeta_meas - zeta_SP|=" << max_zeta_dev
            << "  (small = SP roundoff bounded)" << std::endl;

  // (4) Initialise Phase B search directions and remaining state.
  //     p(s) = r(s)_init.  We adopt the convention that the multishift CG
  //     tracks r(0)_DP as the seed residual via r_seed_dp; per-shift residuals
  //     stay consistent via the zeta recurrence.
  r_seed_dp.copy_from(r_dp_init[0]);
  for (int s = 0; s < nshift; ++s) {
    p_dp[s].copy_from(r_dp_init[s]);
  }

  // Convergence target: tight tol per shift on the multishift-tracked r2_B.
  std::vector<int> shift_converged_B(nshift, 0);
  // The shifted residual that we monitor: |r(s)|² = ζ(s)² · |r_seed|².
  // We declare convergence on each shift when its tracked residual reaches
  // stop_phaseB[s].  If a shift is already below target on init, mark it
  // converged immediately.
  int n_converged_init = 0;
  for (int s = 0; s < nshift; ++s) {
    if (r2_B[s] < stop_phaseB[s]) {
      shift_converged_B[s] = 1;
      ++n_converged_init;
    }
  }
  PhaseBBlasTimer.Stop();

  // (5) DP multishift CG loop.
  int dp_iters = 0;
  if (n_converged_init < nshift) {
    // Standard multishift CG using QUDA's UpdateAlphaZeta recurrence, on DP.
    // r_seed_dp is the shared Krylov residual (= r(0)_DP).
    // p_dp[s] are the per-shift search directions.
    // x_dp[s] are the per-shift solutions (warm-started, additively refined).
    double r2_seed = r2_B[0];
    int nshift_now_B = nshift;
    for (int k = 1; k <= MaxIter; ++k) {
      PhaseBMatTimer.Start();
      double pAp = MP::HermOpDp(Mop_quda, p_dp[0], Ap_dp, tmp_dp, scratch_lo_dp);
      PhaseBMatTimer.Stop();
      PhaseBBlasTimer.Start();
      pAp += poles[0] * p_dp[0].norm2();

      double r2_old = r2_B[0];
      MP::UpdateAlphaZeta(alpha_B, zeta_B, zeta_old_B, r2_B, beta_B, pAp,
                          offset, nshift_now_B, j_low);

      // r_seed -= alpha(0) · (Ap + σ_0 · p(0))
      Ap_dp.axpy(poles[0], p_dp[0]);
      r_seed_dp.axpy(-alpha_B[0], Ap_dp);
      double r2_seed_new = r_seed_dp.norm2();
      r2_B[0] = r2_seed_new;
      r2_seed = r2_seed_new;

      // Per-shift residual norms via zeta.
      for (int j = 1; j < nshift_now_B; ++j) {
        if (!shift_converged_B[j])
          r2_B[j] = zeta_B[j] * zeta_B[j] * r2_seed;
      }

      // x updates: x(s) += alpha(s) · p(s)
      for (int s = 0; s < nshift_now_B; ++s) {
        if (shift_converged_B[s]) continue;
        x_dp[s].axpy(alpha_B[s], p_dp[s]);
      }

      // p(s) update: p(s) = beta(s) · p(s) + zeta(s) · r_seed
      beta_B[0] = r2_seed_new / r2_old;
      p_dp[0].scale_add(beta_B[0], r_seed_dp);
      for (int j = 1; j < nshift_now_B; ++j) {
        if (shift_converged_B[j]) continue;
        beta_B[j] = beta_B[0] * zeta_B[j] * alpha_B[j] /
                    (zeta_old_B[j] * alpha_B[0]);
        // p_dp[j] = beta(j) · p_dp[j] + zeta(j) · r_seed_dp.
        for (int slot = 0; slot < 4; ++slot) {
          if (!p_dp[j].csf[slot] || !r_seed_dp.csf[slot]) continue;
          quda::vector<double> as{zeta_B[j]};
          quda::vector<double> bs_v{beta_B[j]};
          quda::vector_ref<const quda::ColorSpinorField> xr{*r_seed_dp.csf[slot]};
          quda::vector_ref<quda::ColorSpinorField> yr{*p_dp[j].csf[slot]};
          quda::blas::axpby(as, xr, bs_v, yr);
        }
      }
      PhaseBBlasTimer.Stop();

      // Convergence check + shift retirement (smallest pole = shift 0 is
      // slowest; check larger poles for convergence and retire).
      int new_converged = 0;
      for (int j = nshift_now_B - 1; j >= 1; --j) {
        if (shift_converged_B[j]) continue;
        if (zeta_B[j] == 0.0 || r2_B[j] < stop_phaseB[j]) {
          shift_converged_B[j] = 1;
          ++new_converged;
        }
      }
      nshift_now_B -= new_converged;
      if (r2_B[0] < stop_phaseB[0]) shift_converged_B[0] = 1;

      bool all_done = true;
      for (int s = 0; s < nshift; ++s) {
        if (!shift_converged_B[s]) { all_done = false; break; }
      }
      if (all_done) {
        dp_iters = k;
        break;
      }
      if (k == MaxIter) {
        dp_iters = MaxIter;
        std::cout << GridLogMessage
                  << "[MpMultishiftRefine] Phase B did not converge in "
                  << MaxIter << " iters, r2_seed=" << r2_seed << std::endl;
      }
    }
  }
  PhaseBTimer.Stop();

  std::cout << GridLogMessage
            << "[MpMultishiftRefine] Phase B done: dp_iters=" << dp_iters
            << " (init_converged=" << n_converged_init << "/" << nshift << ")"
            << "  PhaseB=" << PhaseBTimer.Elapsed()
            << "  DP-mat=" << PhaseBMatTimer.Elapsed()
            << "  blas=" << PhaseBBlasTimer.Elapsed() << std::endl;

  // (6) Unpack DP solutions back to Grid.
  for (int s = 0; s < nshift; ++s) {
    x_dp[s].copy_to_grid(psi[s], Mop_quda.InvertParam(), X_full_dims);
  }
}

NAMESPACE_END(Grid);
