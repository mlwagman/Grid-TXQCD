// Style B Phase 2.5 Session 3d — out-of-line strong definition for PreMatLower
// native kernel.  See gamma5_impl.cc for the trove-isolation rationale.

#include <Grid/Grid.h>
#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native_decl.h>

namespace Grid {
namespace DtxqcdQudaPreMatLowerNative {

using AccessorTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;

static accelerator_inline void toRel_(const quda::complex<double> in[12],
                                      quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

static accelerator_inline void toNonRelHalf_(const quda::complex<double> in[12],
                                             quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
  }
}

void ApplyPreMatLowerNative(quda::ColorSpinorField &in_csf,
                            quda::ColorSpinorField &out_csf) {
  AccessorTy in_acc(in_csf);
  AccessorTy out_acc(out_csf);

  int volumeCB = in_csf.VolumeCB();
  int nParity = (in_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

    quda::complex<double> in_UK[12], in_DR[12];
    in_acc.load(in_UK, x_cb, parity);
    toRel_(in_UK, in_DR);

    quda::complex<double> out_DR[12];
    for (int c = 0; c < Nc; ++c) {
      out_DR[0 * Nc + c] = quda::complex<double>(-in_DR[1 * Nc + c].real(),
                                                  +in_DR[1 * Nc + c].imag());
      out_DR[1 * Nc + c] = quda::complex<double>(+in_DR[0 * Nc + c].real(),
                                                  -in_DR[0 * Nc + c].imag());
      out_DR[2 * Nc + c] = quda::complex<double>(+in_DR[3 * Nc + c].real(),
                                                  -in_DR[3 * Nc + c].imag());
      out_DR[3 * Nc + c] = quda::complex<double>(-in_DR[2 * Nc + c].real(),
                                                  +in_DR[2 * Nc + c].imag());
    }

    quda::complex<double> out_UK[12];
    toNonRelHalf_(out_DR, out_UK);
    out_acc.save(out_UK, x_cb, parity);
  });
}

}  // namespace DtxqcdQudaPreMatLowerNative
}  // namespace Grid
