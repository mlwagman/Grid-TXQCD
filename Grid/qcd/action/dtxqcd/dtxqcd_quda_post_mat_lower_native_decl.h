#pragma once
// Style B Phase 2.5 Session 3d — forward declaration ONLY (see gamma5 decl for why).

#include <Grid/Grid.h>

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPostMatLowerNative {

void ApplyPostMatLowerNative(quda::ColorSpinorField &io_csf);

}  // namespace DtxqcdQudaPostMatLowerNative
NAMESPACE_END(Grid);
