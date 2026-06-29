#pragma once
// MP-CG Session A — SP twin of dtxqcd_quda_aux_kernel_native_v2.h.
//
// Identical math to DP version, with float arithmetic and
// FloatNOrder<float, 4, 3, 4> accessor.  Aux fields stay packed as doubles
// (DeviceAuxCache from the DP path); the kernel casts each scalar to float at
// the use site.  This avoids maintaining a parallel SP aux cache for Session A;
// when Session B wires SP M_device_csf, an SP cache can be added (or kept in
// DP — the per-site cast cost is small compared to the matvec).
//
// Perm table: REUSE the DP build_perm_table from
// dtxqcd_quda_aux_kernel_native_v2_impl.cc — it operates on the FloatNOrder
// layout-detection probe (which only cares about site enumeration, not
// precision).  Tests pass the same perm_d (built once on a DP CSF) into the SP
// kernel; the (parity, x_cb) → our_eo_idx mapping is precision-agnostic
// because QUDA's NATIVE FLOAT2 ordering uses the same site enumeration for SP
// and DP fields.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // reuse aux pack + DeviceAuxCache
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_native_helpers_sp.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelNativeV2Sp {

using SigmaDoublesPerSite =
    std::integral_constant<std::size_t,
                           DtxqcdQudaAuxKernelDevice::SigmaDoublesPerSite>;
using ScalarDoublesPerSite =
    std::integral_constant<std::size_t,
                           DtxqcdQudaAuxKernelDevice::ScalarDoublesPerSite>;

// Aux index inside the production EO-packed buffer (same layout as DP).
accelerator_inline std::size_t MatIdxNativeSp(std::size_t our_eo_idx, int a,
                                              int b, int i, int j, int ri) {
  std::size_t lin = ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
  return our_eo_idx * SigmaDoublesPerSite::value + lin * 2 + ri;
}

// ----------------------------------------------------------------------------
// Apply the fused aux contraction on SP NATIVE-layout CSFs.
//
// in_*/out_* are SP native ColorSpinorFields (upper.f[0,1], lower.f[0,1]).
// aux is the DP DeviceAuxCache (sigma_d, pi_d, d_d, n_d, s_d, p_d as double*).
// perm_d is the perm table built once on a DP CSF; SP/DP NATIVE site
// enumeration is precision-agnostic so the same perm table works for both.
//
// Same template params as DP: TransposeAux, UseDnConj.
// Accumulates into existing output values (matching the DP kernel).
// ----------------------------------------------------------------------------
template <bool TransposeAux, bool UseDnConj>
inline void ApplyAuxKernelNativeSpImpl(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1) {
  using AccTy = DtxqcdQudaNativeHelpersSp::AccessorTySp;
  AccTy iu0_acc(in_u0),  iu1_acc(in_u1),  il0_acc(in_l0),  il1_acc(in_l1);
  AccTy ou0_acc(out_u0), ou1_acc(out_u1), ol0_acc(out_l0), ol1_acc(out_l1);

  int volumeCB = in_u0.VolumeCB();
  int nParity = (in_u0.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  std::size_t N = (std::size_t)nParity * (std::size_t)volumeCB;

  const double *sigma_d = aux.sigma_d;
  const double *pi_d    = aux.pi_d;
  const double *d_d     = aux.d_d;
  const double *n_d     = aux.n_d;
  const double *s_d     = aux.s_d;
  const double *p_d     = aux.p_d;
  const float scale_dn  = (float)DtxqcdOffdiagFactor();

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;
    std::size_t our_eo = (std::size_t)perm_d[parity * volumeCB + x_cb];

    // ---------- Load 4 input spinors in UKQCD basis, convert to DR ----------
    quda::complex<float> spinor_iu0_UK[12], spinor_iu1_UK[12];
    quda::complex<float> spinor_il0_UK[12], spinor_il1_UK[12];
    iu0_acc.load(spinor_iu0_UK, x_cb, parity);
    iu1_acc.load(spinor_iu1_UK, x_cb, parity);
    il0_acc.load(spinor_il0_UK, x_cb, parity);
    il1_acc.load(spinor_il1_UK, x_cb, parity);

    quda::complex<float> spinor_iu0_DR[12], spinor_iu1_DR[12];
    quda::complex<float> spinor_il0_DR[12], spinor_il1_DR[12];
    DtxqcdQudaNativeHelpersSp::toRelInlineSp(spinor_iu0_UK, spinor_iu0_DR);
    DtxqcdQudaNativeHelpersSp::toRelInlineSp(spinor_iu1_UK, spinor_iu1_DR);
    DtxqcdQudaNativeHelpersSp::toRelInlineSp(spinor_il0_UK, spinor_il0_DR);
    DtxqcdQudaNativeHelpersSp::toRelInlineSp(spinor_il1_UK, spinor_il1_DR);

    quda::complex<float> spinor_ou0_DR_inc[12] = {};
    quda::complex<float> spinor_ou1_DR_inc[12] = {};
    quda::complex<float> spinor_ol0_DR_inc[12] = {};
    quda::complex<float> spinor_ol1_DR_inc[12] = {};

    float s_val = (float)s_d[our_eo * 2];
    float p_val = (float)p_d[our_eo * 2];

    for (int a = 0; a < DtxqcdNf; ++a) {
      auto *ou_a_DR = (a == 0) ? spinor_ou0_DR_inc : spinor_ou1_DR_inc;
      auto *ol_a_DR = (a == 0) ? spinor_ol0_DR_inc : spinor_ol1_DR_inc;
      auto *iu_a_DR = (a == 0) ? spinor_iu0_DR : spinor_iu1_DR;
      auto *il_a_DR = (a == 0) ? spinor_il0_DR : spinor_il1_DR;

      for (int alpha = 0; alpha < 4; ++alpha) {
        float g5_sign_a = DtxqcdQudaNativeHelpersSp::Gamma5SignDRSp(alpha);

        for (int i = 0; i < Nc; ++i) {
          quda::complex<float> valu(0, 0), vall(0, 0);

          // ===== Pass A: σ + π + s + p =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<float> sig(
                  (float)sigma_d[MatIdxNativeSp(our_eo, a, b, i, j, 0)],
                  (float)sigma_d[MatIdxNativeSp(our_eo, a, b, i, j, 1)]);
              quda::complex<float> pii(
                  (float)pi_d[MatIdxNativeSp(our_eo, a, b, i, j, 0)],
                  (float)pi_d[MatIdxNativeSp(our_eo, a, b, i, j, 1)]);

              quda::complex<float> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<float> il_bj = il_b_DR[alpha * Nc + j];

              valu += sig * iu_bj + g5_sign_a * pii * iu_bj;

              if constexpr (TransposeAux) {
                quda::complex<float> sig_t(
                    (float)sigma_d[MatIdxNativeSp(our_eo, b, a, j, i, 0)],
                    (float)sigma_d[MatIdxNativeSp(our_eo, b, a, j, i, 1)]);
                quda::complex<float> pii_t(
                    (float)pi_d[MatIdxNativeSp(our_eo, b, a, j, i, 0)],
                    (float)pi_d[MatIdxNativeSp(our_eo, b, a, j, i, 1)]);
                vall += sig_t * il_bj + g5_sign_a * pii_t * il_bj;
              } else {
                vall += sig * il_bj + g5_sign_a * pii * il_bj;
              }
            }
          }

          quda::complex<float> iu_a_alpha_i = iu_a_DR[alpha * Nc + i];
          quda::complex<float> il_a_alpha_i = il_a_DR[alpha * Nc + i];
          valu += s_val * iu_a_alpha_i + g5_sign_a * p_val * iu_a_alpha_i;
          vall += s_val * il_a_alpha_i + g5_sign_a * p_val * il_a_alpha_i;

          // ===== Pass B: d + n cross =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<float> dd(
                  (float)d_d[MatIdxNativeSp(our_eo, a, b, i, j, 0)],
                  (float)d_d[MatIdxNativeSp(our_eo, a, b, i, j, 1)]);
              quda::complex<float> nn(
                  (float)n_d[MatIdxNativeSp(our_eo, a, b, i, j, 0)],
                  (float)n_d[MatIdxNativeSp(our_eo, a, b, i, j, 1)]);

              quda::complex<float> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<float> il_bj = il_b_DR[alpha * Nc + j];

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

    quda::complex<float> ou0_inc_UK[12], ou1_inc_UK[12];
    quda::complex<float> ol0_inc_UK[12], ol1_inc_UK[12];
    DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp(spinor_ou0_DR_inc, ou0_inc_UK);
    DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp(spinor_ou1_DR_inc, ou1_inc_UK);
    DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp(spinor_ol0_DR_inc, ol0_inc_UK);
    DtxqcdQudaNativeHelpersSp::toNonRelHalfInlineSp(spinor_ol1_DR_inc, ol1_inc_UK);

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

inline void ApplyAuxKernelNativeSp(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeSpImpl<true, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeSpImpl<true, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeSpImpl<false, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeSpImpl<false, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}

}  // namespace DtxqcdQudaAuxKernelNativeV2Sp
NAMESPACE_END(Grid);
