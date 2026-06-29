#pragma once
// Phase β Session A — SP variant decl for PostMatLower native kernel.

#include <Grid/Grid.h>

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPostMatLowerNativeSp {

void ApplyPostMatLowerNativeSp(quda::ColorSpinorField &io_csf);

}  // namespace DtxqcdQudaPostMatLowerNativeSp
NAMESPACE_END(Grid);
