#pragma once
// Style B Phase 2.5 Session 3d — forward declaration ONLY.
//
// See dtxqcd_quda_gamma5_native_decl.h for the trove-isolation rationale.
// Implementation lives in dtxqcd_quda_aux_kernel_native_v2_impl.cc.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // DeviceAuxCache

// Forward declarations for QUDA types we only need by-reference.
namespace quda {
  class ColorSpinorField;
}
struct QudaInvertParam_s;
typedef struct QudaInvertParam_s QudaInvertParam;

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelNativeV2 {

// Build the (parity, x_cb) → our_eo_idx perm table at setup time.
std::vector<int> build_perm_table(quda::ColorSpinorField &native_csf,
                                  QudaInvertParam &inv_param,
                                  const int X_full_dims[4],
                                  GridCartesian *grid);

// Runtime dispatcher (handles the 4 template combos internally).
void ApplyAuxKernelNative(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true);

}  // namespace DtxqcdQudaAuxKernelNativeV2
NAMESPACE_END(Grid);
