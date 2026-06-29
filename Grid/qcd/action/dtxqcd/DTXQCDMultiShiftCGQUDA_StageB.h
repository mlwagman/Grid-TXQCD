#pragma once
// M-wrap.5b.2 Stage B: device-resident multishift CG for DTXQCD doubled fermion.
//
// Outer state (r, p, ps[s], psi[s], mmp, tmp) lives in raw device buffers
// (4 doubles*flat-EO buffers per state vector, one per upper/lower × flavor).
// Per-iter mat-vec via DTXQCDMpcOpQUDA::M_device / Mdag_device — no host
// pack/unpack inside the inner loop.  BLAS via accelerator_for kernels on
// flat double buffers (DtxqcdQudaStageB helpers).
//
// Differs from Stage A (DTXQCDMultiShiftCGQUDA.h):
//   - Stage A: outer state on Grid SIMD (DTXQCDFermionDoubled), per-iter
//     Mop.M / Mop.Mdag does host-mode pack→QUDA→unpack→aux→repack.  Phase I
//     overhead pattern.  Production scale: 80 ms/iter (= production parity).
//   - Stage B: outer state on device, per-iter M_device / Mdag_device skip
//     host roundtrip.  Aux contribution still goes via host (Option α from
//     5a, baked into M_device).  Production target: 25-40 ms/iter.
//
// Reliable update: every K iters, unpack psi[0] to Grid → Mop_d.M/Mdag →
// recompute true residual r = src - (M†M + poles[0])·psi[0] → repack r.
// Bounds cumulative rounding error from QUDA pack roundtrips inside M_device.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>

NAMESPACE_BEGIN(Grid);

// HermOp (M†M)·p via M_device then Mdag_device.  Returns Re<p, (M†M)·p>.
inline RealD DtxqcdHermOpQUDA_StageB(DTXQCDMpcOpQUDA &Mop_quda,
                                    DtxqcdQudaStageB::DoubledStateBuf &p,
                                    DtxqcdQudaStageB::DoubledStateBuf &mmp,
                                    DtxqcdQudaStageB::DoubledStateBuf &tmp,
                                    DtxqcdQudaStageB::DoubledStateBuf &scratch_lo,
                                    std::size_t V, std::size_t Nbuf,
                                    GridBase *grid) {
  double *p_up[DtxqcdNf] = {p.d[0], p.d[1]};
  double *p_lo[DtxqcdNf] = {p.d[2], p.d[3]};
  double *t_up[DtxqcdNf] = {tmp.d[0], tmp.d[1]};
  double *t_lo[DtxqcdNf] = {tmp.d[2], tmp.d[3]};
  double *m_up[DtxqcdNf] = {mmp.d[0], mmp.d[1]};
  double *m_lo[DtxqcdNf] = {mmp.d[2], mmp.d[3]};
  double *s_lo[DtxqcdNf] = {scratch_lo.d[0], scratch_lo.d[1]};

  // M·p → tmp
  Mop_quda.M_device(p_up, p_lo, t_up, t_lo, s_lo, /*dagger=*/false);
  // Mdag·tmp → mmp
  Mop_quda.Mdag_device(t_up, t_lo, m_up, m_lo, s_lo);

  return DtxqcdQudaStageB::DoubledStateBuf::redot(p, mmp, Nbuf, grid);
}

template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA_StageB(
    DTXQCDMopD &Mop_d,                 // Grid-DP HermOp for reliable update
    DTXQCDMpcOpQUDA &Mop_quda,         // QUDA-DP HermOp for inner mat-vec
    const std::vector<RealD> &poles,
    const std::vector<RealD> &tol,
    const DTXQCDFermionDoubled &src,
    std::vector<DTXQCDFermionDoubled> &psi,
    int MaxIter,
    int ReliableUpdateFreq = 50) {
  using namespace dtxqcd_msshift_detail;
  namespace SB = DtxqcdQudaStageB;

  const int nshift = static_cast<int>(poles.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  GRID_ASSERT(static_cast<int>(tol.size()) == nshift);
  for (int s = 0; s < nshift; ++s) GRID_ASSERT(poles[s] >= poles[0]);

  GridBase *grid = src.Grid();
  std::size_t V = Quda::local_volume(grid);
  std::size_t Nbuf = SB::buf_doubles(V);

  // Allocate device state.  scratch_lo is the M_device γ_2·conj round-trip
  // scratch — only the lower-block slots (d[0], d[1]) are ever used (see
  // HermOp's s_lo[DtxqcdNf] = {scratch_lo.d[0], scratch_lo.d[1]} pass).  Use
  // allocate_lower_only_state to avoid wasting 2 buffers worth of device
  // memory (~19 MB/rank at 16³×48).
  SB::DoubledStateBuf r_q, p_q, mmp_q, tmp_q, scratch_lo;
  SB::allocate_state(r_q, Nbuf);
  SB::allocate_state(p_q, Nbuf);
  SB::allocate_state(mmp_q, Nbuf);
  SB::allocate_state(tmp_q, Nbuf);
  SB::allocate_lower_only_state(scratch_lo, Nbuf);
  std::vector<SB::DoubledStateBuf> ps_q(nshift), psi_q(nshift);
  for (int s = 0; s < nshift; ++s) {
    SB::allocate_state(ps_q[s], Nbuf);
    SB::allocate_state(psi_q[s], Nbuf);
  }

  // For reliable update + final unpack: Grid-side scratches.
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
    // Free device state.
    SB::free_state(r_q); SB::free_state(p_q); SB::free_state(mmp_q);
    SB::free_state(tmp_q); SB::free_state(scratch_lo);
    for (int s = 0; s < nshift; ++s) {
      SB::free_state(ps_q[s]); SB::free_state(psi_q[s]);
    }
    return;
  }

  // Initial state: pack src onto device, replicate to r, p, ps[s]; zero psi[s].
  SB::pack_grid_to_state(src, r_q, V);
  p_q.copy_from(r_q, Nbuf);
  for (int s = 0; s < nshift; ++s) {
    rsq_target[s] = cp * tol[s] * tol[s];
    ps_q[s].copy_from(r_q, Nbuf);
    psi_q[s].zero(Nbuf);
  }

  // First HermOp on device.
  RealD d = DtxqcdHermOpQUDA_StageB(Mop_quda, p_q, mmp_q, tmp_q, scratch_lo,
                                    V, Nbuf, grid);
  // mmp += poles[0] · p   (per-buffer axpy across all 4 flavor/block slots)
  mmp_q.axpy(poles[0], p_q, Nbuf);
  RealD rn = p_q.norm2(Nbuf, grid);
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
  r_q.axpy(b, mmp_q, Nbuf);
  RealD c = r_q.norm2(Nbuf, grid);

  // psi[s] = -bs[s] · alpha[s] · src   (single shot, since psi_q starts at 0).
  // Drop the device-side src_q (saved ~38 MB/rank at 16³×48); do the scalar
  // multiply Grid-side then pack into psi_q[s] directly.
  DTXQCDFermionDoubled scaled_src(grid);
  for (int s = 0; s < nshift; ++s) {
    RealD scale = -bs[s] * alpha[s];
    for (int a = 0; a < DtxqcdNf; ++a) {
      scaled_src.upper.f[a] = scale * src.upper.f[a];
      scaled_src.lower.f[a] = scale * src.lower.f[a];
    }
    SB::pack_grid_to_state(scaled_src, psi_q[s], V);
  }

  GridStopWatch MatrixTimer, ReliableTimer, BlasTimer;
  int reliable_updates = 0;

  for (int k = 1; k <= MaxIter; ++k) {
    RealD aa = c / cp;

    BlasTimer.Start();
    // p = aa · p + r
    p_q.scale_add(aa, r_q, Nbuf);

    // ps_q updates: per shift, ps[s] = aa·z[s][iz]·bs[s]/(z[s][1-iz]·b) · ps + z[s][iz] · r
    // Build per-shift coefficient arrays for the fused block kernel.
    std::vector<double> a_arr(nshift), z_arr(nshift);
    int n_active = 0;
    std::vector<int> active_idx;
    active_idx.reserve(nshift);
    for (int s = 0; s < nshift; ++s) {
      if (converged[s]) continue;
      active_idx.push_back(s);
      if (s == 0) {
        a_arr[n_active] = aa;
        z_arr[n_active] = 1.0;
      } else {
        a_arr[n_active] = aa * z[s][iz] * bs[s] / (z[s][1 - iz] * b);
        z_arr[n_active] = z[s][iz];
      }
      ++n_active;
    }
    // Fused block axpby: for each active shift, ps[s] = a_arr[s] · ps[s] + z_arr[s] · r
    // (rewritten as y = b·y + a·x_shared with b = a_arr, a = z_arr).
    if (n_active > 0) {
      std::vector<double *> ps_ptrs(4 * n_active);
      for (int slot = 0; slot < 4; ++slot) {
        for (int i = 0; i < n_active; ++i) {
          ps_ptrs[slot * n_active + i] = ps_q[active_idx[i]].d[slot];
        }
        SB::BlockAxpbyShared(ps_ptrs.data() + slot * n_active,
                             a_arr.data(), z_arr.data(),
                             r_q.d[slot], n_active, Nbuf);
      }
    }
    BlasTimer.Stop();

    RealD cp_prev = c;

    MatrixTimer.Start();
    d = DtxqcdHermOpQUDA_StageB(Mop_quda, p_q, mmp_q, tmp_q, scratch_lo,
                                V, Nbuf, grid);
    MatrixTimer.Stop();
    BlasTimer.Start();
    mmp_q.axpy(poles[0], p_q, Nbuf);
    rn = p_q.norm2(Nbuf, grid);
    d += rn * poles[0];

    RealD bp = b;
    b = -cp_prev / d;

    r_q.axpy(b, mmp_q, Nbuf);
    RealD c_new = r_q.norm2(Nbuf, grid);
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

    // Fused block axpy: psi[s] += -bs[s]·alpha[s] · ps[s]  per active shift.
    if (n_active > 0) {
      std::vector<double> psi_a_arr(n_active);
      std::vector<double *> psi_ptrs(4 * n_active), ps_ptrs(4 * n_active);
      for (int i = 0; i < n_active; ++i) {
        int s = active_idx[i];
        psi_a_arr[i] = -bs[s] * alpha[s];
      }
      for (int slot = 0; slot < 4; ++slot) {
        for (int i = 0; i < n_active; ++i) {
          psi_ptrs[slot * n_active + i] = psi_q[active_idx[i]].d[slot];
          ps_ptrs [slot * n_active + i] = ps_q [active_idx[i]].d[slot];
        }
        SB::BlockAxpy(psi_ptrs.data() + slot * n_active, psi_a_arr.data(),
                      ps_ptrs.data() + slot * n_active, n_active, Nbuf);
      }
    }
    BlasTimer.Stop();

    if (k % ReliableUpdateFreq == 0) {
      ReliableTimer.Start();
      RealD c_old = c;
      SB::unpack_state_to_grid(psi_q[0], psi0_grid, V);
      RealD dd = DtxqcdHermOpD(Mop_d, psi0_grid, mmp_grid, tmp_grid);
      (void)dd;
      Axpy(mmp_grid, poles[0], psi0_grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        r_grid.upper.f[a] = src.upper.f[a] - mmp_grid.upper.f[a];
        r_grid.lower.f[a] = src.lower.f[a] - mmp_grid.lower.f[a];
      }
      RealD c_true = ::Grid::norm2(r_grid);
      SB::pack_grid_to_state(r_grid, r_q, V);
      c = c_true;
      ReliableTimer.Stop();
      ++reliable_updates;
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA_StageB] reliable update k=" << k
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
                << "[DTXQCDMultiShiftCGQUDA_StageB] converged iter=" << k
                << " nshift=" << nshift
                << " reliable_updates=" << reliable_updates
                << "  QUDA-matrix=" << MatrixTimer.Elapsed()
                << "  blas=" << BlasTimer.Elapsed()
                << "  reliable=" << ReliableTimer.Elapsed() << std::endl;
      break;
    }
    if (k == MaxIter) {
      std::cout << GridLogMessage
                << "[DTXQCDMultiShiftCGQUDA_StageB] did not converge in "
                << MaxIter << " iterations (c=" << c << ")" << std::endl;
    }
  }

  // Unpack solutions.
  for (int s = 0; s < nshift; ++s) {
    SB::unpack_state_to_grid(psi_q[s], psi[s], V);
  }

  // Free device state.
  SB::free_state(r_q); SB::free_state(p_q); SB::free_state(mmp_q);
  SB::free_state(tmp_q); SB::free_state(scratch_lo);
  for (int s = 0; s < nshift; ++s) {
    SB::free_state(ps_q[s]); SB::free_state(psi_q[s]);
  }
}

NAMESPACE_END(Grid);
