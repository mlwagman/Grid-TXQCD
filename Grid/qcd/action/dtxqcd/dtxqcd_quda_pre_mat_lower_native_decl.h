#pragma once
// Style B Phase 2.5 Session 3d — forward declaration ONLY (see gamma5 decl for why).

#include <Grid/Grid.h>

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPreMatLowerNative {

void ApplyPreMatLowerNative(quda::ColorSpinorField &in_csf,
                            quda::ColorSpinorField &out_csf);

}  // namespace DtxqcdQudaPreMatLowerNative
NAMESPACE_END(Grid);
