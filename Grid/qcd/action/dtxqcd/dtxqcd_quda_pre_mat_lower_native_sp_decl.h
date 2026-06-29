#pragma once
// Phase β Session A — SP variant decl for PreMatLower native kernel.

#include <Grid/Grid.h>

namespace quda { class ColorSpinorField; }

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPreMatLowerNativeSp {

void ApplyPreMatLowerNativeSp(quda::ColorSpinorField &in_csf,
                              quda::ColorSpinorField &out_csf);

}  // namespace DtxqcdQudaPreMatLowerNativeSp
NAMESPACE_END(Grid);
