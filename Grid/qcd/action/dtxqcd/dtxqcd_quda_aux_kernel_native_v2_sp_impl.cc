// Phase β Session A — SP variant impl for aux native v2 kernel.
//
// Precision drop strategy: spinor I/O is SP (FloatNOrder<float,4,3,4>), per-site
// math runs in complex<double> to keep aux-sum accumulator precision near DP
// floor.  Aux fields stay double via existing DeviceAuxCache.  Perm table is
// reused from the DP build_perm_table — site enumeration is identical between
// DP and SP CSF (only the per-site stride differs: FLOAT4 vs FLOAT2).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_sp_decl.h>

namespace Grid {
namespace DtxqcdQudaAuxKernelNativeV2Sp {

static constexpr std::size_t SigmaDoublesPerSite =
    DtxqcdQudaAuxKernelDevice::SigmaDoublesPerSite;
static constexpr std::size_t ScalarDoublesPerSite =
    DtxqcdQudaAuxKernelDevice::ScalarDoublesPerSite;

accelerator_inline std::size_t MatIdxNative_(std::size_t our_eo_idx, int a,
                                             int b, int i, int j, int ri) {
  std::size_t lin = ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
  return our_eo_idx * SigmaDoublesPerSite + lin * 2 + ri;
}

accelerator_inline double Gamma5SignDR_(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

// SP-input normalized basis transform (operates in float; followed by
// promotion to double for the per-site sum).
static accelerator_inline void toRel_f_(const quda::complex<float> in[12],
                                        quda::complex<float> out[12]) {
  const float inv_sqrt2 = 0.70710678118654752440f;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

// DP→SP normalized basis transform (operates in double on accumulator output,
// emits float for accessor.save).
static accelerator_inline void toNonRelHalf_d2f_(
    const quda::complex<double> in[12], quda::complex<float> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    quda::complex<double> a = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    quda::complex<double> b = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    quda::complex<double> d = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    quda::complex<double> e = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
    out[0*3 + c] = quda::complex<float>((float)a.real(), (float)a.imag());
    out[1*3 + c] = quda::complex<float>((float)b.real(), (float)b.imag());
    out[2*3 + c] = quda::complex<float>((float)d.real(), (float)d.imag());
    out[3*3 + c] = quda::complex<float>((float)e.real(), (float)e.imag());
  }
}

// ---- ApplyAuxKernelNativeSp (templated impl + runtime dispatcher) -----------
template <bool TransposeAux, bool UseDnConj>
static void ApplyAuxKernelNativeImplSp_(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1) {
  using AccTy = quda::colorspinor::FloatNOrder<float, 4, 3, 4>;
  AccTy iu0_acc(in_u0),  iu1_acc(in_u1),  il0_acc(in_l0),  il1_acc(in_l1);
  AccTy ou0_acc(out_u0), ou1_acc(out_u1), ol0_acc(out_l0), ol1_acc(out_l1);

  int volumeCB = in_u0.VolumeCB();
  int nParity = (in_u0.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  size_t N = (size_t)nParity * (size_t)volumeCB;

  const double *sigma_d = aux.sigma_d;
  const double *pi_d    = aux.pi_d;
  const double *d_d     = aux.d_d;
  const double *n_d     = aux.n_d;
  const double *s_d     = aux.s_d;
  const double *p_d     = aux.p_d;
  const double scale_dn = DtxqcdOffdiagFactor();

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;
    std::size_t our_eo = (std::size_t)perm_d[parity * volumeCB + x_cb];

    // Load SP spinors and apply normalized basis transform in float.
    quda::complex<float> spinor_iu0_UK[12], spinor_iu1_UK[12];
    quda::complex<float> spinor_il0_UK[12], spinor_il1_UK[12];
    iu0_acc.load(spinor_iu0_UK, x_cb, parity);
    iu1_acc.load(spinor_iu1_UK, x_cb, parity);
    il0_acc.load(spinor_il0_UK, x_cb, parity);
    il1_acc.load(spinor_il1_UK, x_cb, parity);

    quda::complex<float> iu0_DR_f[12], iu1_DR_f[12];
    quda::complex<float> il0_DR_f[12], il1_DR_f[12];
    toRel_f_(spinor_iu0_UK, iu0_DR_f);
    toRel_f_(spinor_iu1_UK, iu1_DR_f);
    toRel_f_(spinor_il0_UK, il0_DR_f);
    toRel_f_(spinor_il1_UK, il1_DR_f);

    // Promote to double for the per-site sum (precision-sensitive).
    quda::complex<double> spinor_iu0_DR[12], spinor_iu1_DR[12];
    quda::complex<double> spinor_il0_DR[12], spinor_il1_DR[12];
    for (int k = 0; k < 12; ++k) {
      spinor_iu0_DR[k] = quda::complex<double>(iu0_DR_f[k].real(),
                                                iu0_DR_f[k].imag());
      spinor_iu1_DR[k] = quda::complex<double>(iu1_DR_f[k].real(),
                                                iu1_DR_f[k].imag());
      spinor_il0_DR[k] = quda::complex<double>(il0_DR_f[k].real(),
                                                il0_DR_f[k].imag());
      spinor_il1_DR[k] = quda::complex<double>(il1_DR_f[k].real(),
                                                il1_DR_f[k].imag());
    }

    quda::complex<double> spinor_ou0_DR_inc[12] = {};
    quda::complex<double> spinor_ou1_DR_inc[12] = {};
    quda::complex<double> spinor_ol0_DR_inc[12] = {};
    quda::complex<double> spinor_ol1_DR_inc[12] = {};

    double s_val = s_d[our_eo * 2];
    double p_val = p_d[our_eo * 2];

    for (int a = 0; a < DtxqcdNf; ++a) {
      auto *ou_a_DR = (a == 0) ? spinor_ou0_DR_inc : spinor_ou1_DR_inc;
      auto *ol_a_DR = (a == 0) ? spinor_ol0_DR_inc : spinor_ol1_DR_inc;
      auto *iu_a_DR = (a == 0) ? spinor_iu0_DR : spinor_iu1_DR;
      auto *il_a_DR = (a == 0) ? spinor_il0_DR : spinor_il1_DR;

      for (int alpha = 0; alpha < 4; ++alpha) {
        double g5_sign_a = Gamma5SignDR_(alpha);

        for (int i = 0; i < Nc; ++i) {
          quda::complex<double> valu(0, 0), vall(0, 0);

          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<double> sig(
                  sigma_d[MatIdxNative_(our_eo, a, b, i, j, 0)],
                  sigma_d[MatIdxNative_(our_eo, a, b, i, j, 1)]);
              quda::complex<double> pii(
                  pi_d[MatIdxNative_(our_eo, a, b, i, j, 0)],
                  pi_d[MatIdxNative_(our_eo, a, b, i, j, 1)]);

              quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];

              valu += sig * iu_bj + g5_sign_a * pii * iu_bj;

              if constexpr (TransposeAux) {
                quda::complex<double> sig_t(
                    sigma_d[MatIdxNative_(our_eo, b, a, j, i, 0)],
                    sigma_d[MatIdxNative_(our_eo, b, a, j, i, 1)]);
                quda::complex<double> pii_t(
                    pi_d[MatIdxNative_(our_eo, b, a, j, i, 0)],
                    pi_d[MatIdxNative_(our_eo, b, a, j, i, 1)]);
                vall += sig_t * il_bj + g5_sign_a * pii_t * il_bj;
              } else {
                vall += sig * il_bj + g5_sign_a * pii * il_bj;
              }
            }
          }

          quda::complex<double> iu_a_alpha_i = iu_a_DR[alpha * Nc + i];
          quda::complex<double> il_a_alpha_i = il_a_DR[alpha * Nc + i];
          valu += s_val * iu_a_alpha_i + g5_sign_a * p_val * iu_a_alpha_i;
          vall += s_val * il_a_alpha_i + g5_sign_a * p_val * il_a_alpha_i;

          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<double> dd(
                  d_d[MatIdxNative_(our_eo, a, b, i, j, 0)],
                  d_d[MatIdxNative_(our_eo, a, b, i, j, 1)]);
              quda::complex<double> nn(
                  n_d[MatIdxNative_(our_eo, a, b, i, j, 0)],
                  n_d[MatIdxNative_(our_eo, a, b, i, j, 1)]);

              quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];

              valu += scale_dn * (g5_sign_a * (dd * il_bj));
              valu += scale_dn * (nn * il_bj);

              if constexpr (UseDnConj) {
                vall += scale_dn * (g5_sign_a * (conj(dd) * iu_bj));
                vall += scale_dn * (conj(nn) * iu_bj);
              } else {
                vall += scale_dn * (g5_sign_a * (dd * iu_bj));
                vall += scale_dn * (nn * iu_bj);
              }
            }
          }

          ou_a_DR[alpha * Nc + i] += valu;
          ol_a_DR[alpha * Nc + i] += vall;
        }
      }
    }

    // Demote accumulator (DR-double) → SP UK via combined basis transform + cast.
    quda::complex<float> ou0_inc_UK[12], ou1_inc_UK[12];
    quda::complex<float> ol0_inc_UK[12], ol1_inc_UK[12];
    toNonRelHalf_d2f_(spinor_ou0_DR_inc, ou0_inc_UK);
    toNonRelHalf_d2f_(spinor_ou1_DR_inc, ou1_inc_UK);
    toNonRelHalf_d2f_(spinor_ol0_DR_inc, ol0_inc_UK);
    toNonRelHalf_d2f_(spinor_ol1_DR_inc, ol1_inc_UK);

    quda::complex<float> ou0_cur[12], ou1_cur[12], ol0_cur[12], ol1_cur[12];
    ou0_acc.load(ou0_cur, x_cb, parity);
    ou1_acc.load(ou1_cur, x_cb, parity);
    ol0_acc.load(ol0_cur, x_cb, parity);
    ol1_acc.load(ol1_cur, x_cb, parity);
    for (int k = 0; k < 12; ++k) {
      ou0_cur[k] += ou0_inc_UK[k];
      ou1_cur[k] += ou1_inc_UK[k];
      ol0_cur[k] += ol0_inc_UK[k];
      ol1_cur[k] += ol1_inc_UK[k];
    }
    ou0_acc.save(ou0_cur, x_cb, parity);
    ou1_acc.save(ou1_cur, x_cb, parity);
    ol0_acc.save(ol0_cur, x_cb, parity);
    ol1_acc.save(ol1_cur, x_cb, parity);
  });
}

void ApplyAuxKernelNativeSp(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux, bool use_dn_conj) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeImplSp_<true, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeImplSp_<true, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeImplSp_<false, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeImplSp_<false, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}

}  // namespace DtxqcdQudaAuxKernelNativeV2Sp
}  // namespace Grid
