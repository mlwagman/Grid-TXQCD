// Phase β Session A — SP variant impl for Gamma5 native kernel.
// Pure-SP precision drop of dtxqcd_quda_gamma5_native_impl.cc.
//
// AccessorTy is FloatNOrder<float,4,3,4> (FLOAT4 ordering for SP Wilson per
// QUDA convention; see color_spinor_field_order.h:1863).
// Per-site math runs in complex<float> — Gamma5 is sign-flip only (no inner
// product), so SP precision loss is limited to one basis-transform round trip.

#include <Grid/Grid.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma5_native_sp_decl.h>

namespace Grid {
namespace DtxqcdQudaGamma5NativeSp {

using AccessorTy = quda::colorspinor::FloatNOrder<float, 4, 3, 4>;

static accelerator_inline void toRel_(const quda::complex<float> in[12],
                                      quda::complex<float> out[12]) {
  const float inv_sqrt2 = 0.70710678118654752440f;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

static accelerator_inline void toNonRelHalf_(const quda::complex<float> in[12],
                                             quda::complex<float> out[12]) {
  const float inv_sqrt2 = 0.70710678118654752440f;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
  }
}

void ApplyGamma5InplaceNativeSp(quda::ColorSpinorField &io_csf, int variant) {
  AccessorTy io_acc(io_csf);

  int volumeCB = io_csf.VolumeCB();
  int nParity = (io_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

    quda::complex<float> in_UK[12], in_DR[12];
    io_acc.load(in_UK, x_cb, parity);
    toRel_(in_UK, in_DR);

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
    toNonRelHalf_(out_DR, out_UK);
    io_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaGamma5NativeSp
}  // namespace Grid
