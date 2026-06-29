#pragma once
// MP-CG Session A — SP twin of dtxqcd_quda_post_mat_lower_native.h.

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers_sp.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaPostMatLowerNativeSp {

using DtxqcdQudaNativeHelpersSp::AccessorTySp;
using DtxqcdQudaNativeHelpersSp::toRelInlineSp;
using DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp;

inline void ApplyPostMatLowerNativeSp(quda::ColorSpinorField &io_csf) {
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
    for (int c = 0; c < Nc; ++c) {
      out_DR[0 * Nc + c] = quda::complex<float>(+in_DR[1 * Nc + c].real(),
                                                 -in_DR[1 * Nc + c].imag());
      out_DR[1 * Nc + c] = quda::complex<float>(-in_DR[0 * Nc + c].real(),
                                                 +in_DR[0 * Nc + c].imag());
      out_DR[2 * Nc + c] = quda::complex<float>(-in_DR[3 * Nc + c].real(),
                                                 +in_DR[3 * Nc + c].imag());
      out_DR[3 * Nc + c] = quda::complex<float>(+in_DR[2 * Nc + c].real(),
                                                 -in_DR[2 * Nc + c].imag());
    }

    quda::complex<float> out_UK[12];
    toNonRelHalfInlineSp(out_DR, out_UK);
    io_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaPostMatLowerNativeSp
NAMESPACE_END(Grid);
