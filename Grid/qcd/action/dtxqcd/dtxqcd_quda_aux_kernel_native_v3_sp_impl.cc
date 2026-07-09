// Aux-kernel V3 — SP twin (float spinor I/O, double accumulation, exactly
// the NativeV2Sp arithmetic with the V3 SoA operand loads).  See the decl
// header + the DP impl for the design; the repack lives in the DP TU.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>  // constexpr helpers used by DTXQCDDeltaOp.h
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v3_decl.h>

namespace Grid {
namespace DtxqcdQudaAuxKernelNativeV3 {

accelerator_inline int LinIdxSp_(int a, int b, int i, int j) {
  return ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
}

accelerator_inline double Gamma5SignDRSp_(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

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

template <bool TransposeAux, bool UseDnConj>
static void ApplyAuxKernelNativeV3ImplSp_(
    const AuxCacheV3 &aux,
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

  const double *sigma_v3 = aux.sigma_v3;
  const double *pi_v3    = aux.pi_v3;
  const double *d_v3     = aux.d_v3;
  const double *n_v3     = aux.n_v3;
  const double *s_v3     = aux.s_v3;
  const double *p_v3     = aux.p_v3;
  const double scale_dn = DtxqcdOffdiagFactor();

  accelerator_for(idx, N, 1, {
    int parity = idx / volumeCB;
    int x_cb   = idx % volumeCB;

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

    double s_val = s_v3[idx];
    double p_val = p_v3[idx];

    // V3b: no runtime-indexed operand arrays (they force local memory — the
    // V3a row-hoist regressed 8.6 -> 21.6 ms/apply).  Operands are SCALARS
    // loaded once per (a,i,b,j); the alpha dependence is handled by 4
    // register accumulators in fully-unrolled loops, so every array index
    // below is a compile-time constant after unrolling.  Per output element
    // the arithmetic sequence (sigma/pi over (b,j), then s/p, then d/n over
    // (b,j)) is IDENTICAL to V2 — results stay bit-identical.
#pragma unroll
    for (int a = 0; a < DtxqcdNf; ++a) {
      auto *ou_a_DR = (a == 0) ? spinor_ou0_DR_inc : spinor_ou1_DR_inc;
      auto *ol_a_DR = (a == 0) ? spinor_ol0_DR_inc : spinor_ol1_DR_inc;
      auto *iu_a_DR = (a == 0) ? spinor_iu0_DR : spinor_iu1_DR;
      auto *il_a_DR = (a == 0) ? spinor_il0_DR : spinor_il1_DR;

#pragma unroll
      for (int i = 0; i < Nc; ++i) {
        quda::complex<double> valu_a[4] = {quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0)};
        quda::complex<double> vall_a[4] = {quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0),
                                           quda::complex<double>(0, 0)};

        // Pass 1: sigma/pi (+ transpose orientation), scalar operands.
#pragma unroll
        for (int b = 0; b < DtxqcdNf; ++b) {
          auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
          auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;
#pragma unroll
          for (int j = 0; j < Nc; ++j) {
            const std::size_t lin = (std::size_t)LinIdxSp_(a, b, i, j);
            const quda::complex<double> sig(
                sigma_v3[(lin * 2 + 0) * N + idx],
                sigma_v3[(lin * 2 + 1) * N + idx]);
            const quda::complex<double> pii(
                pi_v3[(lin * 2 + 0) * N + idx],
                pi_v3[(lin * 2 + 1) * N + idx]);
            quda::complex<double> sig_t(0, 0), pii_t(0, 0);
            if constexpr (TransposeAux) {
              const std::size_t lin_t = (std::size_t)LinIdxSp_(b, a, j, i);
              sig_t = quda::complex<double>(
                  sigma_v3[(lin_t * 2 + 0) * N + idx],
                  sigma_v3[(lin_t * 2 + 1) * N + idx]);
              pii_t = quda::complex<double>(
                  pi_v3[(lin_t * 2 + 0) * N + idx],
                  pi_v3[(lin_t * 2 + 1) * N + idx]);
            }
#pragma unroll
            for (int alpha = 0; alpha < 4; ++alpha) {
              const double g5_sign_a = Gamma5SignDRSp_(alpha);
              const quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              const quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];
              valu_a[alpha] += sig * iu_bj + g5_sign_a * pii * iu_bj;
              if constexpr (TransposeAux) {
                vall_a[alpha] += sig_t * il_bj + g5_sign_a * pii_t * il_bj;
              } else {
                vall_a[alpha] += sig * il_bj + g5_sign_a * pii * il_bj;
              }
            }
          }
        }

        // s/p singlets.
#pragma unroll
        for (int alpha = 0; alpha < 4; ++alpha) {
          const double g5_sign_a = Gamma5SignDRSp_(alpha);
          const quda::complex<double> iu_a_alpha_i = iu_a_DR[alpha * Nc + i];
          const quda::complex<double> il_a_alpha_i = il_a_DR[alpha * Nc + i];
          valu_a[alpha] +=
              s_val * iu_a_alpha_i + g5_sign_a * p_val * iu_a_alpha_i;
          vall_a[alpha] +=
              s_val * il_a_alpha_i + g5_sign_a * p_val * il_a_alpha_i;
        }

        // Pass 2: d/n cross terms, scalar operands.
#pragma unroll
        for (int b = 0; b < DtxqcdNf; ++b) {
          auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
          auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;
#pragma unroll
          for (int j = 0; j < Nc; ++j) {
            const std::size_t lin = (std::size_t)LinIdxSp_(a, b, i, j);
            const quda::complex<double> dd(
                d_v3[(lin * 2 + 0) * N + idx],
                d_v3[(lin * 2 + 1) * N + idx]);
            const quda::complex<double> nn(
                n_v3[(lin * 2 + 0) * N + idx],
                n_v3[(lin * 2 + 1) * N + idx]);
#pragma unroll
            for (int alpha = 0; alpha < 4; ++alpha) {
              const double g5_sign_a = Gamma5SignDRSp_(alpha);
              const quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              const quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];
              valu_a[alpha] += scale_dn * (g5_sign_a * (dd * il_bj));
              valu_a[alpha] += scale_dn * (nn * il_bj);
              if constexpr (UseDnConj) {
                vall_a[alpha] += scale_dn * (g5_sign_a * (conj(dd) * iu_bj));
                vall_a[alpha] += scale_dn * (conj(nn) * iu_bj);
              } else {
                vall_a[alpha] += scale_dn * (g5_sign_a * (dd * iu_bj));
                vall_a[alpha] += scale_dn * (nn * iu_bj);
              }
            }
          }
        }

#pragma unroll
        for (int alpha = 0; alpha < 4; ++alpha) {
          ou_a_DR[alpha * Nc + i] += valu_a[alpha];
          ol_a_DR[alpha * Nc + i] += vall_a[alpha];
        }
      }
    }

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

void ApplyAuxKernelNativeV3Sp(
    const AuxCacheV3 &aux,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux, bool use_dn_conj) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeV3ImplSp_<true, true>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeV3ImplSp_<true, false>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeV3ImplSp_<false, true>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeV3ImplSp_<false, false>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}

}  // namespace DtxqcdQudaAuxKernelNativeV3
}  // namespace Grid
