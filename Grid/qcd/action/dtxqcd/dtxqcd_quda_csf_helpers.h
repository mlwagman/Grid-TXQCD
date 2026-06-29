#pragma once
// M-wrap.5b.2 Stage B helpers: device-resident multishift CG support.
//
// Strategy:
//   The wrapper's M_device takes raw `double*` device pointers (24·V doubles
//   per buffer, EO-permuted layout matching Quda::fermion_to_eo_buffer).  For
//   the Stage B multishift loop we keep r/p/ps[s]/psi[s]/mmp/tmp as collections
//   of those raw buffers (4 per state vector: upper.f[0,1] + lower.f[0,1]).
//
//   QUDA's blas_quda.h API takes ColorSpinorField references with a specific
//   internal layout that does NOT match our flat EO buffers byte-for-byte.
//   Bridging via ColorSpinorParam{create=REFERENCE} is fragile (precision /
//   ghost / SoA-order mismatches).  Instead we write thin accelerator_for
//   BLAS kernels operating directly on the flat double buffers — exactly the
//   memory the wrapper's M_device already produces and consumes.
//
//   The fused block::axpy / block::axpby ops are implemented as single
//   accelerator_for calls that loop over (site, shift) — same launch
//   amortisation as QUDA's block ops.

#include <Grid/Grid.h>
#include <Grid/util/QudaFieldConvert.h>

#ifndef GRID_HAVE_QUDA
#  error "dtxqcd_quda_csf_helpers requires GRID_HAVE_QUDA"
#endif

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaStageB {

// Per-buffer length: 24 doubles per site (12 complex × 2 reals).
inline std::size_t buf_doubles(std::size_t V) { return std::size_t{24} * V; }

// y[i] += a · x[i]  on a single flat double buffer of length N.
inline void Axpy(double *y_d, double a, const double *x_d, std::size_t N) {
  accelerator_for(i, N, 1, {
    y_d[i] += a * x_d[i];
  });
}

// y[i] = scale · y[i] + add[i]  (Grid's Scale_Add semantics).
inline void Scale_Add(double *y_d, double scale, const double *add_d, std::size_t N) {
  accelerator_for(i, N, 1, {
    y_d[i] = scale * y_d[i] + add_d[i];
  });
}

// y[i] = a · x[i] + b · y[i]
inline void Axpby(double *y_d, double a, const double *x_d, double b, std::size_t N) {
  accelerator_for(i, N, 1, {
    y_d[i] = a * x_d[i] + b * y_d[i];
  });
}

// y[i] = coef · src[i]
inline void Set_Scaled(double *y_d, double coef, const double *src_d, std::size_t N) {
  accelerator_for(i, N, 1, {
    y_d[i] = coef * src_d[i];
  });
}

// dst = src  (device-to-device copy).
inline void Copy(double *dst_d, const double *src_d, std::size_t N) {
  accelerator_for(i, N, 1, {
    dst_d[i] = src_d[i];
  });
}

inline void Zero(double *y_d, std::size_t N) {
  accelerator_for(i, N, 1, {
    y_d[i] = 0.0;
  });
}

// |x|² as a sum over flat doubles.  Treats the buffer as a flat real vector;
// since |c|² = Re² + Im² for complex c stored as (Re, Im) pairs, the sum of
// squares over flat doubles equals the standard squared L² norm of the
// complex spinor field.
//
// Implementation: download buffer to host + reduce on CPU.  For 24·V doubles
// at 16³ per GPU this is ~10 MB / call * 4 buffers = ~40 MB ⇒ ~50 μs on
// PCIe-4 (~30 GB/s) — well under the per-iter cost.  Future optimisation:
// cuBLAS Dnrm2 / Ddot on the device pointer.
inline RealD Norm2(const double *x_d, std::size_t N, GridBase *grid) {
  std::vector<double> host(N);
  acceleratorCopyFromDevice(const_cast<double *>(x_d), host.data(), N * sizeof(double));
  RealD acc = 0.0;
  for (std::size_t i = 0; i < N; ++i) acc += host[i] * host[i];
  // CRITICAL: MPI Allreduce across ranks.  Without this, multi-rank CG
  // gets local-only inner products and diverges (M-wrap.5b.2 16³ smoke 2026-06-26).
  grid->GlobalSum(acc);
  return acc;
}

// Re<x, y> = Σ_i x[i]·y[i].  See Norm2 note: complex inner product's real
// part is exactly the flat real-vector dot.
inline RealD ReDot(const double *x_d, const double *y_d, std::size_t N,
                   GridBase *grid) {
  std::vector<double> hx(N), hy(N);
  acceleratorCopyFromDevice(const_cast<double *>(x_d), hx.data(), N * sizeof(double));
  acceleratorCopyFromDevice(const_cast<double *>(y_d), hy.data(), N * sizeof(double));
  RealD acc = 0.0;
  for (std::size_t i = 0; i < N; ++i) acc += hx[i] * hy[i];
  grid->GlobalSum(acc);
  return acc;
}

// Fused block axpy: for each shift s in 0..K-1,  y_s[i] += a[s] · x_s[i].
// Single kernel launch over (i, s).  K is a runtime arg; we use a small
// device-side coefficient array.
inline void BlockAxpy(double **y_arr_d, const double *a_arr_host,
                      double **x_arr_d, int K, std::size_t N) {
  // Copy coefficient array to device-accessible storage.
  Vector<double> a_buf(K);
  for (int s = 0; s < K; ++s) a_buf[s] = a_arr_host[s];
  double *a_d = a_buf.data();

  // Copy pointer arrays so the kernel can index them.
  Vector<double *> yp(K), xp(K);
  for (int s = 0; s < K; ++s) { yp[s] = y_arr_d[s]; xp[s] = x_arr_d[s]; }
  double **y_d = yp.data();
  double **x_d = xp.data();

  accelerator_for(idx, (std::size_t)K * N, 1, {
    std::size_t s = idx / N;
    std::size_t i = idx % N;
    y_d[s][i] += a_d[s] * x_d[s][i];
  });
}

// Fused block axpby: for each shift s,  y_s[i] = b[s] · y_s[i] + a[s] · x[i].
// `x` is a SINGLE buffer (shared across shifts) — matches the shifted-CG
// recurrence ps_s = z_s · r + bs_s · ps_s.
inline void BlockAxpbyShared(double **y_arr_d, const double *b_arr_host,
                             const double *a_arr_host,
                             const double *x_d, int K, std::size_t N) {
  Vector<double> a_buf(K), b_buf(K);
  for (int s = 0; s < K; ++s) { a_buf[s] = a_arr_host[s]; b_buf[s] = b_arr_host[s]; }
  double *a_d = a_buf.data();
  double *b_d = b_buf.data();

  Vector<double *> yp(K);
  for (int s = 0; s < K; ++s) yp[s] = y_arr_d[s];
  double **y_d = yp.data();

  accelerator_for(idx, (std::size_t)K * N, 1, {
    std::size_t s = idx / N;
    std::size_t i = idx % N;
    y_d[s][i] = b_d[s] * y_d[s][i] + a_d[s] * x_d[i];
  });
}

// γ_5 application on a flat EO buffer (Grid-convention layout from
// fermion_to_eo_buffer): per site, 24 doubles laid out as
//   spin 0: doubles 0-5    (3 colors × re/im)
//   spin 1: doubles 6-11
//   spin 2: doubles 12-17
//   spin 3: doubles 18-23
//
// Grid's γ_5 (DR convention) negates spin 0 and 1, leaves 2 and 3 unchanged.
// Implemented as in-place sign flip on doubles 0-11 per site.  If the sign
// convention is wrong we'd see the Mdag test fail at FD; toggling via env
// `DTXQCD_STAGEB_G5_SIGN={A,B}` lets us probe variants from the test driver.
inline void ApplyGamma5Inplace(double *buf_d, std::size_t V, int variant = 0) {
  // variant 0: negate spin 0, 1 (γ_5 = diag(-1,-1,+1,+1))
  // variant 1: negate spin 2, 3 (γ_5 = diag(+1,+1,-1,-1))
  std::size_t lo = (variant == 0) ? 0 : 12;
  accelerator_for(site, V, 1, {
    for (int k = 0; k < 12; ++k) {
      buf_d[site * 24 + lo + k] = -buf_d[site * 24 + lo + k];
    }
  });
}

// ---- Allocation helpers for the 4-buffer-per-state-vector pattern ----

// A "doubled state buffer" is 4 device-resident flat double buffers (one per
// upper/lower × flavor).  Sized at 24·V doubles each.
struct DoubledStateBuf {
  double *d[4] = {nullptr, nullptr, nullptr, nullptr};
  // d[0] = upper.f[0]; d[1] = upper.f[1]; d[2] = lower.f[0]; d[3] = lower.f[1].

  // Sum of |·|² over all 4 buffers — full DTXQCDFermionDoubled norm.
  // Globally reduced via GridBase::GlobalSum.
  RealD norm2(std::size_t N_per_buf, GridBase *grid) const {
    return Norm2(d[0], N_per_buf, grid) + Norm2(d[1], N_per_buf, grid)
         + Norm2(d[2], N_per_buf, grid) + Norm2(d[3], N_per_buf, grid);
  }
  // Re<x, y> over all 4 buffers.  Globally reduced.
  static RealD redot(const DoubledStateBuf &x, const DoubledStateBuf &y,
                     std::size_t N_per_buf, GridBase *grid) {
    return ReDot(x.d[0], y.d[0], N_per_buf, grid)
         + ReDot(x.d[1], y.d[1], N_per_buf, grid)
         + ReDot(x.d[2], y.d[2], N_per_buf, grid)
         + ReDot(x.d[3], y.d[3], N_per_buf, grid);
  }
  void zero(std::size_t N_per_buf) {
    for (int i = 0; i < 4; ++i) DtxqcdQudaStageB::Zero(d[i], N_per_buf);
  }
  void axpy(double a, const DoubledStateBuf &x, std::size_t N_per_buf) {
    for (int i = 0; i < 4; ++i) DtxqcdQudaStageB::Axpy(d[i], a, x.d[i], N_per_buf);
  }
  void scale_add(double scale, const DoubledStateBuf &add, std::size_t N_per_buf) {
    for (int i = 0; i < 4; ++i)
      DtxqcdQudaStageB::Scale_Add(d[i], scale, add.d[i], N_per_buf);
  }
  void set_scaled(double coef, const DoubledStateBuf &src, std::size_t N_per_buf) {
    for (int i = 0; i < 4; ++i)
      DtxqcdQudaStageB::Set_Scaled(d[i], coef, src.d[i], N_per_buf);
  }
  void copy_from(const DoubledStateBuf &src, std::size_t N_per_buf) {
    for (int i = 0; i < 4; ++i) DtxqcdQudaStageB::Copy(d[i], src.d[i], N_per_buf);
  }
};

inline void allocate_state(DoubledStateBuf &s, std::size_t N_per_buf) {
  size_t bytes = N_per_buf * sizeof(double);
  for (int i = 0; i < 4; ++i) {
    s.d[i] = (double *)acceleratorAllocDevice(bytes);
  }
}

// Allocate only the lower-block slots (d[0], d[1]).  Used for scratch_lo in
// DTXQCDMultiShiftCGQUDA_StageB, where the HermOp passes scratch_lo.d[0,1]
// as the 2-slot s_lo[] array to M_device/Mdag_device.  Cuts state memory
// usage by ~50 MB/rank at 16³×48.
inline void allocate_lower_only_state(DoubledStateBuf &s, std::size_t N_per_buf) {
  size_t bytes = N_per_buf * sizeof(double);
  s.d[0] = (double *)acceleratorAllocDevice(bytes);
  s.d[1] = (double *)acceleratorAllocDevice(bytes);
  s.d[2] = nullptr;
  s.d[3] = nullptr;
}

inline void free_state(DoubledStateBuf &s) {
  for (int i = 0; i < 4; ++i) {
    if (s.d[i]) acceleratorFreeDevice(s.d[i]);
    s.d[i] = nullptr;
  }
}

// Pack a DTXQCDFermionDoubled (Grid SIMD) into a DoubledStateBuf (4 device
// flat EO buffers).  4 fermion_to_eo_buffer host-side packs + 4 H2D copies.
inline void pack_grid_to_state(const DTXQCDFermionDoubled &src,
                               DoubledStateBuf &dst, std::size_t V) {
  std::size_t N = buf_doubles(V);
  std::vector<double> host(N);
  Quda::fermion_to_eo_buffer(src.upper.f[0], host.data());
  acceleratorCopyToDevice(host.data(), dst.d[0], N * sizeof(double));
  Quda::fermion_to_eo_buffer(src.upper.f[1], host.data());
  acceleratorCopyToDevice(host.data(), dst.d[1], N * sizeof(double));
  Quda::fermion_to_eo_buffer(src.lower.f[0], host.data());
  acceleratorCopyToDevice(host.data(), dst.d[2], N * sizeof(double));
  Quda::fermion_to_eo_buffer(src.lower.f[1], host.data());
  acceleratorCopyToDevice(host.data(), dst.d[3], N * sizeof(double));
}

inline void unpack_state_to_grid(const DoubledStateBuf &src,
                                 DTXQCDFermionDoubled &dst, std::size_t V) {
  std::size_t N = buf_doubles(V);
  std::vector<double> host(N);
  acceleratorCopyFromDevice(src.d[0], host.data(), N * sizeof(double));
  Quda::eo_buffer_to_fermion(host.data(), dst.upper.f[0]);
  acceleratorCopyFromDevice(src.d[1], host.data(), N * sizeof(double));
  Quda::eo_buffer_to_fermion(host.data(), dst.upper.f[1]);
  acceleratorCopyFromDevice(src.d[2], host.data(), N * sizeof(double));
  Quda::eo_buffer_to_fermion(host.data(), dst.lower.f[0]);
  acceleratorCopyFromDevice(src.d[3], host.data(), N * sizeof(double));
  Quda::eo_buffer_to_fermion(host.data(), dst.lower.f[1]);
}

}  // namespace DtxqcdQudaStageB
NAMESPACE_END(Grid);
