#pragma once
// Style B Phase 2.5 Session 3b — Style C multishift CG.
//
// Mirrors DTXQCDMultiShiftCGQUDA_StageB.h structure but with all state on
// native ColorSpinorField (DoubledStateCSF) instead of raw cudaMalloc'd flat
// EO buffers.  Per-iter mat-vec via DTXQCDMpcOpQUDA::M_device_csf /
// Mdag_device_csf — no per-call csf.copy in the inner loop.
//
// Per-iter cost (expected at 16³×48 mpi=1.1.1.4):
//   dirac->M × 8 (M + Mdag, 4 flavor blocks each) : ~8-24 ms
//   aux native (Session 1 v2) + Pre/Post/Gamma5 native (Session 2) : ~5-15 ms
//   quda::blas norm2/reDot/axpy/axpby/caxpy : ~3-10 ms
//   total : ~20-50 ms/iter → substep-1 25-50 s at 1030 iters
//
// vs Stage B Option β (2026-06-27): ~225 ms/iter → 228 s/substep
// vs Stage A baseline: ~77 ms/iter → 80 s/substep
// Target gate: substep-1 ≤ 65 s.
//
// Layout convert is amortized over the whole CG:
//   entry  : Grid SIMD → host flat-24V → native CSF  (once per CG)
//   exit   : native CSF → host flat-24V → Grid SIMD  (once per CG)
//   reliable update: native CSF → host flat-24V → Grid SIMD → HermOpD →
//                    flat-24V → native CSF (~once per ReliableUpdateFreq iters)
//
// Bit-equiv reference: Stage B (DTXQCDMultiShiftCGQUDA_StageB).  Per-shift gate
// ≤ 1e-9 rel diff vs Stage B's psi[s].

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DoubledStateCSF.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>

NAMESPACE_BEGIN(Grid);

// HermOp (M†M)·p via M_device_csf then Mdag_device_csf, on caller's native
// DoubledStateCSF.  Returns Re<p, (M†M)·p>.
inline RealD DtxqcdHermOpQUDA_StyleC(DTXQCDMpcOpQUDA &Mop_quda,
                                     DtxqcdQudaStyleC::DoubledStateCSF &p,
                                     DtxqcdQudaStyleC::DoubledStateCSF &mmp,
                                     DtxqcdQudaStyleC::DoubledStateCSF &tmp,
                                     DtxqcdQudaStyleC::DoubledStateCSF &scratch_lo) {
  quda::ColorSpinorField *p_up [DtxqcdNf] = {p.csf[0].get(), p.csf[1].get()};
  quda::ColorSpinorField *p_lo [DtxqcdNf] = {p.csf[2].get(), p.csf[3].get()};
  quda::ColorSpinorField *t_up [DtxqcdNf] = {tmp.csf[0].get(), tmp.csf[1].get()};
  quda::ColorSpinorField *t_lo [DtxqcdNf] = {tmp.csf[2].get(), tmp.csf[3].get()};
  quda::ColorSpinorField *m_up [DtxqcdNf] = {mmp.csf[0].get(), mmp.csf[1].get()};
  quda::ColorSpinorField *m_lo [DtxqcdNf] = {mmp.csf[2].get(), mmp.csf[3].get()};
  quda::ColorSpinorField *s_lo [DtxqcdNf] = {scratch_lo.csf[0].get(), scratch_lo.csf[1].get()};

  // ZERO ALL output / intermediate buffers, including their PADDING / halo.
  // NULL_FIELD_CREATE leaves uninitialised bytes that QUDA's dirac->M halo
  // gather can read from (even single-rank, the FloatNOrder accessor may
  // touch padded slots).  Without this the FIRST HermOp call after fresh
  // allocations gives a different result than subsequent calls.
  mmp.zero();
  tmp.zero();
  scratch_lo.zero();
  Mop_quda.M_device_csf(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  // Mdag·tmp → mmp.  Mdag_device_csf wraps M_device_csf with γ_5 in/out.
  Mop_quda.Mdag_device_csf(t_up, t_lo, m_up, m_lo, s_lo);

  return DtxqcdQudaStyleC::DoubledStateCSF::redot(p, mmp);
}

template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA_StyleC(
    DTXQCDMopD &Mop_d,                 // Grid-DP HermOp for reliable update
    DTXQCDMpcOpQUDA &Mop_quda,         // QUDA-DP HermOp for inner mat-vec
    const std::vector<RealD> &poles,
    const std::vector<RealD> &tol,
    const DTXQCDFermionDoubled &src,
    std::vector<DTXQCDFermionDoubled> &psi,
    int MaxIter,
    int ReliableUpdateFreq = 50) {
  using namespace dtxqcd_msshift_detail;
  namespace SC = DtxqcdQudaStyleC;
  namespace SB = DtxqcdQudaStageB;

  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);
  for (int s = 0; s < nshift; ++s) GRID_ASSERT(poles[s] >= poles[0]);

  GridBase *grid = src.Grid();

  // Native-CSF allocation template + dim list for csf.copy calls.
  quda::ColorSpinorParam param_tmpl = Mop_quda.MakeNativeCsfParam();
  int X_full_dims[4];
  for (int d = 0; d < 4; ++d) X_full_dims[d] = Mop_quda.GaugeParam().X[d];

  // Allocate device state on native CSF.
  SC::DoubledStateCSF r_q, p_q, mmp_q, tmp_q, scratch_lo;
  r_q.allocate(param_tmpl);
  p_q.allocate(param_tmpl);
  mmp_q.allocate(param_tmpl);
  tmp_q.allocate(param_tmpl);
  scratch_lo.allocate_lower_only(param_tmpl);
  std::vector<SC::DoubledStateCSF> ps_q(nshift), psi_q(nshift);
  for (int s = 0; s < nshift; ++s) {
    ps_q[s].allocate(param_tmpl);
    psi_q[s].allocate(param_tmpl);
  }

  // Reliable-update + final-unpack Grid scratches.
  DTXQCDFermionDoubled r_grid(grid), mmp_grid(grid), tmp_grid(grid),
      psi0_grid(grid);

  std::vector<RealD> alpha(nshift, 1.0);
  std::vector<RealD> bs(nshift);
  std::vector<RealD> rsq_target(nshift);
  std::vector<std::array<RealD, 2>> z(nshift);
  std::vector<int> converged(nshift, 0);

  RealD cp = norm2(src);
  if (cp == 0.0) {
    for (int s = 0; s < nshift; ++s) Zero_(psi[s]);
    return;  // RAII frees CSFs
  }

  // CG entry: pack src onto device, replicate to r, p, ps[s]; zero psi[s].
  r_q.copy_from_grid(src, Mop_quda.InvertParam(), X_full_dims);
  p_q.copy_from(r_q);
  for (int s = 0; s < nshift; ++s) {
    rsq_target[s] = cp * tol[s] * tol[s];
    ps_q[s].copy_from(r_q);
    psi_q[s].zero();
  }

  // First HermOp on device.
  RealD d = DtxqcdHermOpQUDA_StyleC(Mop_quda, p_q, mmp_q, tmp_q, scratch_lo);
  // mmp += poles[0] · p
  mmp_q.axpy(poles[0], p_q);
  RealD rn = p_q.norm2();
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

  // r += b · mmp
  r_q.axpy(b, mmp_q);
  RealD c = r_q.norm2();

  // psi[s] = -bs[s] · alpha[s] · src.  Grid-side scalar multiply then pack.
  DTXQCDFermionDoubled scaled_src(grid);
  for (int s = 0; s < nshift; ++s) {
    RealD scale = -bs[s] * alpha[s];
    for (int a = 0; a < DtxqcdNf; ++a) {
      scaled_src.upper.f[a] = scale * src.upper.f[a];
      scaled_src.lower.f[a] = scale * src.lower.f[a];
    }
    psi_q[s].copy_from_grid(scaled_src, Mop_quda.InvertParam(), X_full_dims);
  }

  GridStopWatch MatrixTimer, ReliableTimer, BlasTimer;
  int reliable_updates = 0;

  for (int k = 1; k <= MaxIter; ++k) {
    RealD aa = c / cp;

    BlasTimer.Start();
    // p = aa · p + r
    p_q.scale_add(aa, r_q);

    // ps[s] updates.  Per active shift, ps[s] = b_arr · ps[s] + a_arr · r
    // with b_arr = aa·z[s][iz]·bs[s]/(z[s][1-iz]·b), a_arr = z[s][iz].
    std::vector<double> a_arr(nshift), b_arr(nshift);
    int n_active = 0;
    std::vector<int> active_idx;
    active_idx.reserve(nshift);
    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      active_idx.push_back(s);
      if (s == 0) {
        b_arr[n_active] = aa;
        a_arr[n_active] = 1.0;
      } else {
        b_arr[n_active] = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
        a_arr[n_active] = z[s][iz];
      }
      ++n_active;
    }
    // ps[s] = b_arr[s] · ps[s] + a_arr[s] · r  (r shared across shifts)
    if (n_active > 0) {
      std::vector<SC::DoubledStateCSF *> ps_ptrs(n_active);
      for (int i = 0; i < n_active; ++i) ps_ptrs[i] = &ps_q[active_idx[i]];
      SC::BlockAxpbyShared(ps_ptrs, b_arr.data(), a_arr.data(), r_q, n_active);
    }
    BlasTimer.Stop();

    RealD cp_prev = c;

    MatrixTimer.Start();
    d = DtxqcdHermOpQUDA_StyleC(Mop_quda, p_q, mmp_q, tmp_q, scratch_lo);
    MatrixTimer.Stop();
    BlasTimer.Start();
    mmp_q.axpy(poles[0], p_q);
    rn = p_q.norm2();
    d += rn * poles[0];

    RealD bp = b;
    b = -cp_prev / d;

    r_q.axpy(b, mmp_q);
    RealD c_new = r_q.norm2();
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

    // psi[s] += -bs[s]·alpha[s] · ps[s]  per active shift.
    if (n_active > 0) {
      std::vector<double> psi_a_arr(n_active);
      std::vector<SC::DoubledStateCSF *> psi_ptrs(n_active), ps_ptrs(n_active);
      for (int i = 0; i < n_active; ++i) {
        int s = active_idx[i];
        psi_a_arr[i] = -bs[s] * alpha[s];
        psi_ptrs[i] = &psi_q[s];
        ps_ptrs[i]  = &ps_q[s];
      }
      SC::BlockAxpy(psi_ptrs, psi_a_arr.data(), ps_ptrs, n_active);
    }
    BlasTimer.Stop();

    if (k % ReliableUpdateFreq == 0) {
      ReliableTimer.Start();
      RealD c_old = c;
      psi_q[0].copy_to_grid(psi0_grid, Mop_quda.InvertParam(), X_full_dims);
      RealD dd = DtxqcdHermOpD(Mop_d, psi0_grid, mmp_grid, tmp_grid);
      (void)dd;
      Axpy(mmp_grid, poles[0], psi0_grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        r_grid.upper.f[a] = src.upper.f[a] - mmp_grid.upper.f[a];
        r_grid.lower.f[a] = src.lower.f[a] - mmp_grid.lower.f[a];
      }
      RealD c_true = ::Grid::norm2(r_grid);
      r_q.copy_from_grid(r_grid, Mop_quda.InvertParam(), X_full_dims);
      c = c_true;
      ReliableTimer.Stop();
      ++reliable_updates;
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA_StyleC] reliable update k=" << k
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
                << "[DTXQCDMultiShiftCGQUDA_StyleC] converged iter=" << k
                << " nshift=" << nshift
                << " reliable_updates=" << reliable_updates
                << "  QUDA-matrix=" << MatrixTimer.Elapsed()
                << "  blas=" << BlasTimer.Elapsed()
                << "  reliable=" << ReliableTimer.Elapsed() << std::endl;
      break;
    }
    if (k == MaxIter) {
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA_StyleC] did not converge in "
                << MaxIter << " iterations (c=" << c << ")" << std::endl;
    }
  }

  // Unpack solutions native → flat-24V → Grid.
  for (int s = 0; s < nshift; ++s) {
    psi_q[s].copy_to_grid(psi[s], Mop_quda.InvertParam(), X_full_dims);
  }

  // RAII frees the DoubledStateCSF members at scope exit.
}

NAMESPACE_END(Grid);
