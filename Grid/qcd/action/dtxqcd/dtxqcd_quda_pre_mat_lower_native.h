#pragma once
// Style B Path B Session 2 — PreMatLowerKernel ported to QUDA NATIVE/FLOAT2
// layout via FloatNOrder accessor.
//
// Counterpart of Quda::PreMatLowerKernel (dtxqcd_quda_apply_C.h:148) which
// works on flat 24·V_local SPACE_SPIN_COLOR DR-basis buffer.  Per-site math:
//   scratch[0,c] = conj(-in[1,c])  = (-in[1,c]_re, +in[1,c]_im)
//   scratch[1,c] = conj(+in[0,c])  = (+in[0,c]_re, -in[0,c]_im)
//   scratch[2,c] = conj(+in[3,c])  = (+in[3,c]_re, -in[3,c]_im)
//   scratch[3,c] = conj(-in[2,c])  = (-in[2,c]_re, +in[2,c]_im)
// In complex form (DR basis):
//   scratch[0,c] = -conj(in[1,c])
//   scratch[1,c] = +conj(in[0,c])
//   scratch[2,c] = +conj(in[3,c])
//   scratch[3,c] = -conj(in[2,c])
//
// Storage in native is UKQCD; we round-trip:
//   ψ_DR = T_R_norm · ψ_UKQCD
//   ψ_DR_out = PreMat(ψ_DR)
//   ψ_UKQCD_out = T_NR_norm · ψ_DR_out
// With normalized T's, T_NR_norm · T_R_norm = I, so a no-op middle = identity.

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPreMatLowerNative {

using DtxqcdQudaNativeHelpers::AccessorTy;
using DtxqcdQudaNativeHelpers::toRelInline;
using DtxqcdQudaNativeHelpers::toNonRelHalfInline;
using DtxqcdQudaNativeHelpers::native_total_sites;

// ----------------------------------------------------------------------------
// out = conj(C · in)  with C = lower-block spin matrix (see flat ref).
//
// in_csf and out_csf must be distinct (not the same field) — flat ref takes
// a separate `scratch_d` output buffer.
// ----------------------------------------------------------------------------
inline void ApplyPreMatLowerNative(quda::ColorSpinorField &in_csf,
                                   quda::ColorSpinorField &out_csf) {
  AccessorTy in_acc(in_csf);
  AccessorTy out_acc(out_csf);

  int volumeCB = in_csf.VolumeCB();
  int nParity = (in_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

    // Load UKQCD storage and convert to DR.
    quda::complex<double> in_UK[12], in_DR[12];
    in_acc.load(in_UK, x_cb, parity);
    toRelInline(in_UK, in_DR);

    // Apply PreMatLower in DR basis (per-site spin permutation+conj).
    quda::complex<double> out_DR[12];
    for (int c = 0; c < Nc; ++c) {
      // spin 0: -conj(in[1, c])
      out_DR[0 * Nc + c] = quda::complex<double>(-in_DR[1 * Nc + c].real(),
                                                  +in_DR[1 * Nc + c].imag());
      // spin 1: +conj(in[0, c])
      out_DR[1 * Nc + c] = quda::complex<double>(+in_DR[0 * Nc + c].real(),
                                                  -in_DR[0 * Nc + c].imag());
      // spin 2: +conj(in[3, c])
      out_DR[2 * Nc + c] = quda::complex<double>(+in_DR[3 * Nc + c].real(),
                                                  -in_DR[3 * Nc + c].imag());
      // spin 3: -conj(in[2, c])
      out_DR[3 * Nc + c] = quda::complex<double>(-in_DR[2 * Nc + c].real(),
                                                  +in_DR[2 * Nc + c].imag());
    }

    // Convert DR → UKQCD and save.
    quda::complex<double> out_UK[12];
    toNonRelHalfInline(out_DR, out_UK);
    out_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaPreMatLowerNative
NAMESPACE_END(Grid);
