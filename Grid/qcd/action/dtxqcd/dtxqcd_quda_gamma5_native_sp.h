#pragma once
// MP-CG Session A — SP twin of dtxqcd_quda_gamma5_native.h.
//
// Identical math to DP version, with float types throughout.
// Storage: SP CSF (FLOAT2 + UKQCD basis). Round-trip via normalized SP toRel /
// toNonRelHalf so the math runs in DR.

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers_sp.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaGamma5NativeSp {

using DtxqcdQudaNativeHelpersSp::AccessorTySp;
using DtxqcdQudaNativeHelpersSp::toRelInlineSp;
using DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp;

inline void ApplyGamma5InplaceNativeSp(quda::ColorSpinorField &io_csf,
                                       int variant = 0) {
  AccessorTySp io_acc(io_csf);

  int volumeCB = io_csf.VolumeCB();
  int nParity = (io_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

    quda::complex<float> in_UK[12], in_DR[12];
    io_acc.load(in_UK, x_cb, parity);
    toRelInlineSp(in_UK, in_DR);

    quda::complex<float> out_DR[12];
    if (variant == 0) {
      for (int c = 0; c < Nc; ++c) {
        out_DR[0 * Nc + c] = -in_DR[0 * Nc + c];
        out_DR[1 * Nc + c] = -in_DR[1 * Nc + c];
        out_DR[2 * Nc + c] = +in_DR[2 * Nc + c];
        out_DR[3 * Nc + c] = +in_DR[3 * Nc + c];
      }
    } else {
      for (int c = 0; c < Nc; ++c) {
        out_DR[0 * Nc + c] = +in_DR[0 * Nc + c];
        out_DR[1 * Nc + c] = +in_DR[1 * Nc + c];
        out_DR[2 * Nc + c] = -in_DR[2 * Nc + c];
        out_DR[3 * Nc + c] = -in_DR[3 * Nc + c];
      }
    }

    quda::complex<float> out_UK[12];
    toNonRelHalfInlineSp(out_DR, out_UK);
    io_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaGamma5NativeSp
NAMESPACE_END(Grid);
