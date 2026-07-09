#pragma once
// Aux-kernel V3 (2026-07-04, DTXQCD_AUX_FUSION_DESIGN.md): bandwidth-fixed
// rewrite of the NativeV2 aux contraction.  The V2 kernel was measured at
// 8.6 ms/apply = 75% of the full-volume M_device_csf (job 1297599 j0),
// ~10x its bandwidth bound, from two compounding defects:
//   1. sigma/pi/d/n re-loaded from global memory for EVERY spin component
//      (the (b,j) load loops sit inside the alpha loop): ~6x redundant
//      traffic including the transpose orientation;
//   2. site-major AoS cache layout (72 contiguous doubles/site): warp
//      threads read the same matrix element at 576 B stride — every complex
//      load is its own memory transaction.
// V3 fixes both without touching the arithmetic:
//   - a ONE-TIME device repack (RepackAuxV3) permutes the existing
//     DeviceAuxCache into QUDA-(parity,x_cb)-ordered structure-of-arrays:
//     buf[(lin*2+ri)*N + idx] — warp loads become fully coalesced and the
//     perm indirection leaves the hot loop.  Under SetAuxStatic the repack
//     runs once per solve (like the pack itself).
//   - the apply kernel hoists each (a,i) row of sigma/pi (both
//     orientations)/d/n into registers BEFORE the alpha loop.  The
//     per-output-element arithmetic sequence is IDENTICAL to V2 (only the
//     operand loads move), so results are bit-identical.
// Env: DTXQCD_AUX_KERNEL_V3=1 (dispatch lives in DTXQCDMpcOpQUDA.h;
// default OFF = byte-identical V2 path).
//
// Forward-decl TU pattern as in dtxqcd_quda_aux_kernel_native_v2_decl.h.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // DeviceAuxCache

namespace quda {
  class ColorSpinorField;
}

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelNativeV3 {

// QUDA-ordered SoA copy of the aux cache.  Matrix fields: 72 doubles/site at
// buf[(lin*2+ri)*N + idx], lin = ((a*Nf+b)*Nc+i)*Nc+j, idx = parity*VolCB
// + x_cb.  Scalars: re part only, buf[idx].
struct AuxCacheV3 {
  double *sigma_v3 = nullptr;
  double *pi_v3    = nullptr;
  double *d_v3     = nullptr;
  double *n_v3     = nullptr;
  double *s_v3     = nullptr;
  double *p_v3     = nullptr;
  std::size_t N = 0;
  bool allocated() const { return sigma_v3 != nullptr; }
};

void free_aux_cache_v3(AuxCacheV3 &c);

// Permute+transpose the packed AoS cache (our_eo order) into the SoA layout
// (QUDA idx order).  Allocates lazily; device kernel, ~2.3 KB/site each way.
void RepackAuxV3(const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
                 const int *perm_d, std::size_t N, AuxCacheV3 &v3);

// DP apply: same signature/math as NativeV2::ApplyAuxKernelNative minus the
// perm table (baked into the repack).
void ApplyAuxKernelNativeV3(
    const AuxCacheV3 &aux,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true);

// V4 decomposition probe (DP only): thread-per-(site, output-flavor) on the
// UNCHANGED V2 AoS cache + perm table — 2x threads, ~half the per-thread
// state, disjoint output CSFs per thread.  Coordinator-directed occupancy
// experiment after V2 8.60 / V3b 12.24 / V3a 21.62 ms/apply established the
// per-thread-state ordering.  Env DTXQCD_AUX_KERNEL_V4=1 (DP call site only;
// SP falls back to V2 unless V4 graduates).
void ApplyAuxKernelNativeV4(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true);

// SP twin (float spinor I/O, double accumulation — mirrors NativeV2Sp).
void ApplyAuxKernelNativeV3Sp(
    const AuxCacheV3 &aux,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true);

}  // namespace DtxqcdQudaAuxKernelNativeV3
NAMESPACE_END(Grid);
