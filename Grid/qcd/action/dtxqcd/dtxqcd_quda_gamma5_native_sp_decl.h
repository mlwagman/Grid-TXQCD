#pragma once
// Phase β Session A — SP variant decl for Gamma5 native kernel.
// Same shape as the DP decl in dtxqcd_quda_gamma5_native_decl.h, but acts on
// SP CSF (QUDA_SINGLE_PRECISION storage, FloatNOrder<float,4,3,4>).

#include <Grid/Grid.h>

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaGamma5NativeSp {

void ApplyGamma5InplaceNativeSp(quda::ColorSpinorField &io_csf, int variant = 0);

}  // namespace DtxqcdQudaGamma5NativeSp
NAMESPACE_END(Grid);
