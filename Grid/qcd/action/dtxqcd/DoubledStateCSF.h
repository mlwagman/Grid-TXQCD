#pragma once
// Style B Phase 2.5 Session 3b — Style C device-resident CG state.
//
// 4-slot RAII wrapper holding native-layout (FLOAT2, halo-padded, UKQCD-basis)
// quda::ColorSpinorField allocations.  Mirrors DoubledStateBuf's
// upper.f[0,1] + lower.f[0,1] slot convention but with QUDA-native CSF
// storage instead of raw cudaMalloc'd flat 24·V buffers.
//
// Inner-loop blas operations dispatch to quda::blas::{norm2,reDotProduct,axpy,
// caxpy,ax,axpby} on the native CSFs — proper device-side tree reductions, no
// host roundtrip per blas.  copy_from_buf / copy_to_buf bridge to flat-24V at
// CG entry / exit only.
//
// Allocation budget at 16³×48 mpi=1.1.1.4 (per rank, 4 GPUs):
//   per CSF ~9.4 MB (24·V_local·8B FLOAT2-padded) × 4 slots/state
//   × (4 single states r,p,mmp,tmp + 24 shift slots for 12 shifts × ps/psi)
//   = 112 CSF × 9.4 MB ≈ 1 GB device — fits the 38 GB device-mem budget.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>
#include <Grid/util/QudaFieldConvert.h>

#ifndef GRID_HAVE_QUDA
#  error "DoubledStateCSF requires GRID_HAVE_QUDA"
#endif

#include <quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <memory>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaStyleC {

// 4-slot doubled state on native CSF.  slot[0]=upper.f[0], [1]=upper.f[1],
// [2]=lower.f[0], [3]=lower.f[1] — matching DoubledStateBuf's convention.
struct DoubledStateCSF {
  std::unique_ptr<quda::ColorSpinorField> csf[4];

  // Allocate all 4 slots from a cuda-native ColorSpinorParam template
  // (typically obtained via DTXQCDMpcOpQUDA::MakeNativeCsfParam()).
  void allocate(const quda::ColorSpinorParam &param_tmpl) {
    auto p = param_tmpl;
    p.create = QUDA_NULL_FIELD_CREATE;
    for (int i = 0; i < 4; ++i) {
      csf[i] = std::make_unique<quda::ColorSpinorField>(p);
    }
  }

  // Allocate only the lower-block slots [0],[1] for scratch_lo use.  Matches
  // DoubledStateBuf::allocate_lower_only_state — scratch_lo is only ever
  // indexed at slots 0,1 (passed to M_device_csf as scratch_lower[0,1]).
  void allocate_lower_only(const quda::ColorSpinorParam &param_tmpl) {
    auto p = param_tmpl;
    p.create = QUDA_NULL_FIELD_CREATE;
    csf[0] = std::make_unique<quda::ColorSpinorField>(p);
    csf[1] = std::make_unique<quda::ColorSpinorField>(p);
    csf[2].reset();
    csf[3].reset();
  }

  // Sum of |·|² over all 4 slots — full doubled-fermion norm.  quda::blas's
  // norm2 does the device tree reduction + MPI Allreduce internally.
  RealD norm2() const {
    RealD acc = 0.0;
    for (int i = 0; i < 4; ++i)
      if (csf[i]) acc += quda::blas::norm2(*csf[i]);
    return acc;
  }

  // Re<x, y> over all 4 slots.  Globally reduced via QUDA.
  static RealD redot(const DoubledStateCSF &x, const DoubledStateCSF &y) {
    RealD acc = 0.0;
    for (int i = 0; i < 4; ++i) {
      if (x.csf[i] && y.csf[i])
        acc += quda::blas::reDotProduct(*x.csf[i], *y.csf[i]);
    }
    return acc;
  }

  // Set all slots to zero.  Use ColorSpinorField::zero() directly: blas::ax(0)
  // would propagate NaN if the buffer is uninitialised (NULL_FIELD_CREATE),
  // since 0*NaN = NaN.
  void zero() {
    for (int i = 0; i < 4; ++i) {
      if (csf[i]) csf[i]->zero();
    }
  }

  // y += a · x  (4-slot axpy).  quda::blas::axpy takes cvector<double>.
  void axpy(double a, const DoubledStateCSF &x) {
    quda::vector<double> a_vec{a};
    for (int i = 0; i < 4; ++i) {
      if (!csf[i] || !x.csf[i]) continue;
      quda::vector_ref<const quda::ColorSpinorField> xr{*x.csf[i]};
      quda::vector_ref<quda::ColorSpinorField> yr{*csf[i]};
      quda::blas::axpy(a_vec, xr, yr);
    }
  }

  // y = scale · y + add  (4-slot).  Note: DoubledStateBuf::scale_add is
  //   y = scale * y + add (i.e. add is the source-with-coeff-1).  We use
  //   quda::blas::axpby(a=1, x=add, b=scale, y) which is y = 1·add + scale·y.
  void scale_add(double scale, const DoubledStateCSF &add) {
    quda::vector<double> ones{1.0};
    quda::vector<double> bs{scale};
    for (int i = 0; i < 4; ++i) {
      if (!csf[i] || !add.csf[i]) continue;
      quda::vector_ref<const quda::ColorSpinorField> xr{*add.csf[i]};
      quda::vector_ref<quda::ColorSpinorField> yr{*csf[i]};
      quda::blas::axpby(ones, xr, bs, yr);
    }
  }

  // y = coef · src.  Field-to-field copy with scale (used by initial psi_q[s]
  // = scale·src in Stage B's CG init).  Done as zero+axpy when needed.
  void set_scaled(double coef, const DoubledStateCSF &src) {
    zero();
    axpy(coef, src);
  }

  // Plain field-to-field copy across all 4 slots.  ColorSpinorField::copy
  // dispatches to the right layout-aware copy kernel.
  void copy_from(const DoubledStateCSF &src) {
    for (int i = 0; i < 4; ++i) {
      if (csf[i] && src.csf[i]) csf[i]->copy(*src.csf[i]);
    }
  }

  // CG entry: Grid SIMD doubled fermion → host flat-24V (per slot) → csf
  // (csf.copy lifts SPACE_SPIN_COLOR+EO → NATIVE FLOAT2+UKQCD).
  // The 4 host-bound `make_native_from_flat` calls per state are the only
  // explicit layout converts in the Style C path — once per CG entry/exit.
  void copy_from_grid(const DTXQCDFermionDoubled &src,
                      QudaInvertParam &inv_param,
                      const int X_full_dims[4]) {
    int V = csf[0]->Volume();  // full-site volume on native CSF
    std::size_t Nbytes = static_cast<std::size_t>(24) * V * sizeof(double);
    std::vector<double> host(24 * V);
    double *d_flat = (double *)acceleratorAllocDevice(Nbytes);
    quda::lat_dim_t X_full;
    for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
    for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
    QudaInvertParam inv = inv_param;
    inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
    inv.output_location = QUDA_CUDA_FIELD_LOCATION;
    auto pack_one = [&](const LatticeFermion &grid_fld, int slot) {
      Quda::fermion_to_eo_buffer(grid_fld, host.data());
      acceleratorCopyToDevice(host.data(), d_flat, Nbytes);
      quda::ColorSpinorParam cpuParam(d_flat, inv, X_full, /*pc=*/false,
                                      QUDA_CUDA_FIELD_LOCATION);
      quda::ColorSpinorField cpu_csf(cpuParam);
      csf[slot]->copy(cpu_csf);
      cudaDeviceSynchronize();
    };
    if (csf[0]) pack_one(src.upper.f[0], 0);
    if (csf[1]) pack_one(src.upper.f[1], 1);
    if (csf[2]) pack_one(src.lower.f[0], 2);
    if (csf[3]) pack_one(src.lower.f[1], 3);
    acceleratorFreeDevice(d_flat);
  }

  // CG exit (or reliable update): csf (native UKQCD) → flat-24V → Grid SIMD.
  void copy_to_grid(DTXQCDFermionDoubled &dst,
                    QudaInvertParam &inv_param,
                    const int X_full_dims[4]) const {
    int V = csf[0]->Volume();
    std::size_t Nbytes = static_cast<std::size_t>(24) * V * sizeof(double);
    std::vector<double> host(24 * V);
    double *d_flat = (double *)acceleratorAllocDevice(Nbytes);
    quda::lat_dim_t X_full;
    for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
    for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
    QudaInvertParam inv = inv_param;
    inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
    inv.output_location = QUDA_CUDA_FIELD_LOCATION;
    auto unpack_one = [&](LatticeFermion &grid_fld, int slot) {
      quda::ColorSpinorParam cpuParam(d_flat, inv, X_full, /*pc=*/false,
                                      QUDA_CUDA_FIELD_LOCATION);
      quda::ColorSpinorField cpu_csf(cpuParam);
      cpu_csf.copy(*csf[slot]);
      cudaDeviceSynchronize();
      acceleratorCopyFromDevice(d_flat, host.data(), Nbytes);
      Quda::eo_buffer_to_fermion(host.data(), grid_fld);
    };
    if (csf[0]) unpack_one(dst.upper.f[0], 0);
    if (csf[1]) unpack_one(dst.upper.f[1], 1);
    if (csf[2]) unpack_one(dst.lower.f[0], 2);
    if (csf[3]) unpack_one(dst.lower.f[1], 3);
    acceleratorFreeDevice(d_flat);
  }
};

// Per-shift y[s] = b[s] · y[s] + a[s] · x  with `x` SHARED across shifts.
// Matches DtxqcdQudaStageB::BlockAxpbyShared semantics (line 134-151 of
// dtxqcd_quda_csf_helpers.h).  Implemented as one quda::blas::axpby call
// per (slot, shift); blas API doesn't have a single-shared-x batched form
// directly so we loop, but each call is a fully-fused device kernel.
inline void BlockAxpbyShared(std::vector<DoubledStateCSF *> &y_arr,
                              const double *b_arr_host,
                              const double *a_arr_host,
                              const DoubledStateCSF &x,
                              int K) {
  for (int slot = 0; slot < 4; ++slot) {
    if (!x.csf[slot]) continue;
    for (int k = 0; k < K; ++k) {
      if (!y_arr[k] || !y_arr[k]->csf[slot]) continue;
      quda::vector<double> a_vec{a_arr_host[k]};
      quda::vector<double> b_vec{b_arr_host[k]};
      quda::vector_ref<const quda::ColorSpinorField> xr{*x.csf[slot]};
      quda::vector_ref<quda::ColorSpinorField> yr{*y_arr[k]->csf[slot]};
      quda::blas::axpby(a_vec, xr, b_vec, yr);
    }
  }
}

// Per-shift y[s] += a[s] · x[s] (1:1).  Matches BlockAxpy.  Per-(slot,shift)
// quda::blas::axpy.
inline void BlockAxpy(std::vector<DoubledStateCSF *> &y_arr,
                       const double *a_arr_host,
                       std::vector<DoubledStateCSF *> &x_arr,
                       int K) {
  for (int slot = 0; slot < 4; ++slot) {
    for (int k = 0; k < K; ++k) {
      if (!y_arr[k] || !y_arr[k]->csf[slot]) continue;
      if (!x_arr[k] || !x_arr[k]->csf[slot]) continue;
      quda::vector<double> a_vec{a_arr_host[k]};
      quda::vector_ref<const quda::ColorSpinorField> xr{*x_arr[k]->csf[slot]};
      quda::vector_ref<quda::ColorSpinorField> yr{*y_arr[k]->csf[slot]};
      quda::blas::axpy(a_vec, xr, yr);
    }
  }
}

}  // namespace DtxqcdQudaStyleC
NAMESPACE_END(Grid);
