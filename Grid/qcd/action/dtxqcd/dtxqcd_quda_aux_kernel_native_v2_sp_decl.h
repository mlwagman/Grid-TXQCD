#pragma once
// Phase β Session A — SP variant decl for aux native v2 kernel.
//
// Same interface shape as the DP decl (dtxqcd_quda_aux_kernel_native_v2_decl.h)
// but operates on SP CSF storage (FloatNOrder<float,4,3,4>).  The perm table is
// reused from the DP version — site enumeration is identical between DP and SP
// CSF on the same lattice; only the per-site stride differs (FLOAT4 vs FLOAT2).
// Aux fields (sigma, pi, d, n, s, p) stay double-precision via the existing
// DeviceAuxCache; only the spinor I/O is SP.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // DeviceAuxCache

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelNativeV2Sp {

// Runtime dispatcher (handles the 4 template combos internally).
void ApplyAuxKernelNativeSp(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true);

}  // namespace DtxqcdQudaAuxKernelNativeV2Sp
NAMESPACE_END(Grid);
