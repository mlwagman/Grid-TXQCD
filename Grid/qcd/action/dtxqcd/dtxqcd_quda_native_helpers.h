#pragma once
// Style B Path B Session 2 — shared helpers for DTXQCD kernels operating on
// QUDA NATIVE (FLOAT2, halo-padded, UKQCD basis) layout via the
// quda::colorspinor::FloatNOrder accessor.
//
// Inline normalized basis transforms (UKQCD ↔ DR) and a γ_5 sign convention
// that matches the flat-24V kernels (Grid γ_5 = diag(+1,+1,-1,-1) in DR basis,
// or diag(-1,-1,+1,+1) — see Gamma.h:90 + DtxqcdQudaStageB::ApplyGamma5Inplace).
//
// Use pattern inside `accelerator_for`:
//   AccTy in_acc(in_csf);
//   AccTy out_acc(out_csf);
//   accelerator_for(idx, N, 1, {
//     quda::complex<double> spinor_UK[12], spinor_DR[12];
//     in_acc.load(spinor_UK, x_cb, parity);
//     toRelInline(spinor_UK, spinor_DR);            // UKQCD → DR
//     // ... per-site DR math ...
//     quda::complex<double> out_UK[12];
//     toNonRelHalfInline(spinor_DR, out_UK);        // DR → UKQCD (normalized inv)
//     out_acc.save(out_UK, x_cb, parity);
//   });
//
// Both transforms are NORMALIZED (entries ±1/√2): T_NR · T_R = I.

#include <Grid/Grid.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaNativeHelpers {

// FloatNOrder accessor type for double-precision Wilson spinors (Ns=4, Nc=3,
// N_=2 FLOAT2 pairs).  Explicit 4th template arg required by this older QUDA.
using AccessorTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;

// ----------------------------------------------------------------------------
// Normalized inline basis transforms operating on complex[12] arrays indexed
// by (spin*Nc + color), with Nc=3.
//
// toRelInline:        UKQCD → DR        (entries ±1/√2)
// toNonRelHalfInline: DR    → UKQCD     (entries ±1/√2)
//
// T_NR_norm · T_R_norm = I (normalized round-trip = identity), which matches
// what csf.copy applies internally between cpu and native fields.
// ----------------------------------------------------------------------------
accelerator_inline void toRelInline(const quda::complex<double> in[12],
                                    quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

accelerator_inline void toNonRelHalfInline(const quda::complex<double> in[12],
                                           quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
  }
}

// ----------------------------------------------------------------------------
// γ_5 sign in DR basis matching the flat-24V kernels (DtxqcdQudaAuxKernelDevice
// convention): diag(+1,+1,-1,-1).  spin 0,1 keep sign; spin 2,3 flip.
// ----------------------------------------------------------------------------
accelerator_inline double Gamma5SignDR(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

// ----------------------------------------------------------------------------
// Per-(parity, x_cb) iteration count helper.
// ----------------------------------------------------------------------------
inline std::size_t native_total_sites(const quda::ColorSpinorField &csf) {
  int volumeCB = csf.VolumeCB();
  int nParity = (csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  return (std::size_t)nParity * (std::size_t)volumeCB;
}

}  // namespace DtxqcdQudaNativeHelpers
NAMESPACE_END(Grid);
