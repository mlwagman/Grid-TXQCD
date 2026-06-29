#pragma once
// MP-CG Session A — SP twin of dtxqcd_quda_native_helpers.h.
//
// FloatNOrder accessor type for single-precision Wilson spinors (Ns=4, Nc=3,
// N_=4 FLOAT4 — QUDA canonical SP layout, see color_spinor_field_order.h:1863)
// + normalized inline basis transforms in float arithmetic.
//
// Use pattern inside `accelerator_for`:
//   AccessorTySp in_acc(in_csf_sp);   AccessorTySp out_acc(out_csf_sp);
//   accelerator_for(idx, N, 1, {
//     quda::complex<float> spinor_UK[12], spinor_DR[12];
//     in_acc.load(spinor_UK, x_cb, parity);
//     toRelInlineSp(spinor_UK, spinor_DR);
//     // ... per-site DR math in float ...
//     quda::complex<float> out_UK[12];
//     toNonRelHalfInlineSp(spinor_DR, out_UK);
//     out_acc.save(out_UK, x_cb, parity);
//   });
//
// Same normalized convention as the DP twin: T_NR_norm·T_R_norm = I.

#include <Grid/Grid.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaNativeHelpersSp {

using AccessorTySp = quda::colorspinor::FloatNOrder<float, 4, 3, 4>;

accelerator_inline void toRelInlineSp(const quda::complex<float> in[12],
                                      quda::complex<float> out[12]) {
  const float inv_sqrt2 = 0.70710678118654752440f;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

accelerator_inline void toNonRelHalfInlineSp(const quda::complex<float> in[12],
                                             quda::complex<float> out[12]) {
  const float inv_sqrt2 = 0.70710678118654752440f;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
  }
}

accelerator_inline float Gamma5SignDRSp(int spin) {
  return (spin < 2) ? 1.0f : -1.0f;
}

}  // namespace DtxqcdQudaNativeHelpersSp
NAMESPACE_END(Grid);
