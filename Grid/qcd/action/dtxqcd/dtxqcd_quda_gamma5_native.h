#pragma once
// Style B Path B Session 2 — ApplyGamma5Inplace ported to QUDA NATIVE/FLOAT2
// layout via FloatNOrder accessor.
//
// Counterpart of DtxqcdQudaStageB::ApplyGamma5Inplace
// (dtxqcd_quda_csf_helpers.h:164) which works in-place on flat 24·V_local
// SPACE_SPIN_COLOR DR-basis buffer.  Per-site math (DR basis, in-place):
//   variant 0: γ_5 = diag(-1,-1,+1,+1)  → negate spin 0, 1
//   variant 1: γ_5 = diag(+1,+1,-1,-1)  → negate spin 2, 3
//
// Storage in native is UKQCD; we round-trip via toRel / toNonRelHalf.

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaGamma5Native {

using DtxqcdQudaNativeHelpers::AccessorTy;
using DtxqcdQudaNativeHelpers::toRelInline;
using DtxqcdQudaNativeHelpers::toNonRelHalfInline;

inline void ApplyGamma5InplaceNative(quda::ColorSpinorField &io_csf,
                                     int variant = 0) {
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

    // Apply γ_5 in DR basis (per-site spin sign flip).
    quda::complex<double> out_DR[12];
    if (variant == 0) {
      // variant 0: negate spin 0, 1; leave spin 2, 3
      for (int c = 0; c < Nc; ++c) {
        out_DR[0 * Nc + c] = -in_DR[0 * Nc + c];
        out_DR[1 * Nc + c] = -in_DR[1 * Nc + c];
        out_DR[2 * Nc + c] = +in_DR[2 * Nc + c];
        out_DR[3 * Nc + c] = +in_DR[3 * Nc + c];
      }
    } else {
      // variant 1: leave spin 0, 1; negate spin 2, 3
      for (int c = 0; c < Nc; ++c) {
        out_DR[0 * Nc + c] = +in_DR[0 * Nc + c];
        out_DR[1 * Nc + c] = +in_DR[1 * Nc + c];
        out_DR[2 * Nc + c] = -in_DR[2 * Nc + c];
        out_DR[3 * Nc + c] = -in_DR[3 * Nc + c];
      }
    }

    // Convert DR → UKQCD and save back in-place.
    quda::complex<double> out_UK[12];
    toNonRelHalfInline(out_DR, out_UK);
    io_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaGamma5Native
NAMESPACE_END(Grid);
