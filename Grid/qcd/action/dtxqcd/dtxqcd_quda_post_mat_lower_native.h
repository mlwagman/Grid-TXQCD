#pragma once
// Style B Path B Session 2 — PostMatLowerKernel ported to QUDA NATIVE/FLOAT2
// layout via FloatNOrder accessor.
//
// Counterpart of Quda::PostMatLowerKernel (dtxqcd_quda_apply_C.h:176) which
// works in-place on flat 24·V_local SPACE_SPIN_COLOR DR-basis buffer.
// Per-site math (DR basis, in-place):
//   new[0,c] = +conj(old[1,c]) = (+old[1,c]_re, -old[1,c]_im)
//   new[1,c] = -conj(old[0,c]) = (-old[0,c]_re, +old[0,c]_im)
//   new[2,c] = -conj(old[3,c]) = (-old[3,c]_re, +old[3,c]_im)
//   new[3,c] = +conj(old[2,c]) = (+old[2,c]_re, -old[2,c]_im)
//
// Storage in native is UKQCD; we round-trip via toRel / toNonRelHalf.
// In-place safe per site (whole 12-complex array loaded into local regs).

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPostMatLowerNative {

using DtxqcdQudaNativeHelpers::AccessorTy;
using DtxqcdQudaNativeHelpers::toRelInline;
using DtxqcdQudaNativeHelpers::toNonRelHalfInline;

inline void ApplyPostMatLowerNative(quda::ColorSpinorField &io_csf) {
  AccessorTy io_acc(io_csf);

  int volumeCB = io_csf.VolumeCB();
  int nParity = (io_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

    // Load UKQCD storage and convert to DR.
    quda::complex<double> in_UK[12], in_DR[12];
    io_acc.load(in_UK, x_cb, parity);
    toRelInline(in_UK, in_DR);

    // Apply PostMatLower in DR basis (per-site, in-place via locals).
    quda::complex<double> out_DR[12];
    for (int c = 0; c < Nc; ++c) {
      // spin 0: +conj(in[1, c])
      out_DR[0 * Nc + c] = quda::complex<double>(+in_DR[1 * Nc + c].real(),
                                                  -in_DR[1 * Nc + c].imag());
      // spin 1: -conj(in[0, c])
      out_DR[1 * Nc + c] = quda::complex<double>(-in_DR[0 * Nc + c].real(),
                                                  +in_DR[0 * Nc + c].imag());
      // spin 2: -conj(in[3, c])
      out_DR[2 * Nc + c] = quda::complex<double>(-in_DR[3 * Nc + c].real(),
                                                  +in_DR[3 * Nc + c].imag());
      // spin 3: +conj(in[2, c])
      out_DR[3 * Nc + c] = quda::complex<double>(+in_DR[2 * Nc + c].real(),
                                                  -in_DR[2 * Nc + c].imag());
    }

    // Convert DR → UKQCD and save back in-place.
    quda::complex<double> out_UK[12];
    toNonRelHalfInline(out_DR, out_UK);
    io_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaPostMatLowerNative
NAMESPACE_END(Grid);
