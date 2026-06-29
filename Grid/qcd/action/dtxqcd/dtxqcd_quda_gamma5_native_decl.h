#pragma once
// Style B Phase 2.5 Session 3d — forward declaration ONLY.
//
// Why: the full implementation includes <color_spinor_field_order.h> which
// transitively includes QUDA's trove externals. The trove externals define a
// `detail` namespace at global scope, which collides with `Grid::detail`
// (introduced via `using namespace Grid;` in production main()s). Header-split:
// this decl header carries only function signatures + plain types so it can
// be safely included by host code; the full implementation lives in
// dtxqcd_quda_gamma5_native_impl.cc compiled in a clean TU.

#include <Grid/Grid.h>

// Forward-declare QUDA's ColorSpinorField without including its header (which
// would pull in trove). We only need the reference type for the signature.
namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaGamma5Native {

void ApplyGamma5InplaceNative(quda::ColorSpinorField &io_csf, int variant = 0);

}  // namespace DtxqcdQudaGamma5Native
NAMESPACE_END(Grid);
