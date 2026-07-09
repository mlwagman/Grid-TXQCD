// Aux-kernel V3 — DP apply + the shared SoA repack.  See the decl header for
// the diagnosis and design; math is copied from NativeV2 with the operand
// loads hoisted per (a,i) row so the per-element arithmetic sequence (and
// hence every bit of the result) is unchanged.

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

static constexpr std::size_t SigmaDoublesPerSite =
    DtxqcdQudaAuxKernelDevice::SigmaDoublesPerSite;

accelerator_inline int LinIdx_(int a, int b, int i, int j) {
  return ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
}

accelerator_inline double Gamma5SignDR_(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

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

void free_aux_cache_v3(AuxCacheV3 &c) {
  if (c.sigma_v3) acceleratorFreeDevice(c.sigma_v3);
  if (c.pi_v3)    acceleratorFreeDevice(c.pi_v3);
  if (c.d_v3)     acceleratorFreeDevice(c.d_v3);
  if (c.n_v3)     acceleratorFreeDevice(c.n_v3);
  if (c.s_v3)     acceleratorFreeDevice(c.s_v3);
  if (c.p_v3)     acceleratorFreeDevice(c.p_v3);
  c = AuxCacheV3{};
}

void RepackAuxV3(const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
                 const int *perm_d, std::size_t N, AuxCacheV3 &v3) {
  if (!v3.allocated()) {
    std::size_t mat_bytes = SigmaDoublesPerSite * N * sizeof(double);
    std::size_t sc_bytes  = N * sizeof(double);
    v3.sigma_v3 = (double *)acceleratorAllocDevice(mat_bytes);
    v3.pi_v3    = (double *)acceleratorAllocDevice(mat_bytes);
    v3.d_v3     = (double *)acceleratorAllocDevice(mat_bytes);
    v3.n_v3     = (double *)acceleratorAllocDevice(mat_bytes);
    v3.s_v3     = (double *)acceleratorAllocDevice(sc_bytes);
    v3.p_v3     = (double *)acceleratorAllocDevice(sc_bytes);
    v3.N = N;
  }
  const double *sig_src = aux.sigma_d;
  const double *pi_src  = aux.pi_d;
  const double *d_src   = aux.d_d;
  const double *n_src   = aux.n_d;
  const double *s_src   = aux.s_d;
  const double *p_src   = aux.p_d;
  double *sig_dst = v3.sigma_v3;
  double *pi_dst  = v3.pi_v3;
  double *d_dst   = v3.d_v3;
  double *n_dst   = v3.n_v3;
  double *s_dst   = v3.s_v3;
  double *p_dst   = v3.p_v3;

  accelerator_for(idx, N, 1, {
    std::size_t our_eo = (std::size_t)perm_d[idx];
    const std::size_t base = our_eo * SigmaDoublesPerSite;
    for (int w = 0; w < (int)SigmaDoublesPerSite; ++w) {
      sig_dst[(std::size_t)w * N + idx] = sig_src[base + w];
      pi_dst [(std::size_t)w * N + idx] = pi_src [base + w];
      d_dst  [(std::size_t)w * N + idx] = d_src  [base + w];
      n_dst  [(std::size_t)w * N + idx] = n_src  [base + w];
    }
    s_dst[idx] = s_src[our_eo * 2];
    p_dst[idx] = p_src[our_eo * 2];
  });
}


accelerator_inline std::size_t MatIdxV4_(std::size_t our_eo_idx, int a,
                                         int b, int i, int j, int ri) {
  std::size_t lin = ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
  return our_eo_idx * SigmaDoublesPerSite + lin * 2 + ri;
}
// ---- DP apply ---------------------------------------------------------------
template <bool TransposeAux, bool UseDnConj>
static void ApplyAuxKernelNativeV3Impl_(
    const AuxCacheV3 &aux,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1) {
  using AccTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;
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

    quda::complex<double> spinor_iu0_UK[12], spinor_iu1_UK[12];
    quda::complex<double> spinor_il0_UK[12], spinor_il1_UK[12];
    iu0_acc.load(spinor_iu0_UK, x_cb, parity);
    iu1_acc.load(spinor_iu1_UK, x_cb, parity);
    il0_acc.load(spinor_il0_UK, x_cb, parity);
    il1_acc.load(spinor_il1_UK, x_cb, parity);

    quda::complex<double> spinor_iu0_DR[12], spinor_iu1_DR[12];
    quda::complex<double> spinor_il0_DR[12], spinor_il1_DR[12];
    toRel_(spinor_iu0_UK, spinor_iu0_DR);
    toRel_(spinor_iu1_UK, spinor_iu1_DR);
    toRel_(spinor_il0_UK, spinor_il0_DR);
    toRel_(spinor_il1_UK, spinor_il1_DR);

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
            const std::size_t lin = (std::size_t)LinIdx_(a, b, i, j);
            const quda::complex<double> sig(
                sigma_v3[(lin * 2 + 0) * N + idx],
                sigma_v3[(lin * 2 + 1) * N + idx]);
            const quda::complex<double> pii(
                pi_v3[(lin * 2 + 0) * N + idx],
                pi_v3[(lin * 2 + 1) * N + idx]);
            quda::complex<double> sig_t(0, 0), pii_t(0, 0);
            if constexpr (TransposeAux) {
              const std::size_t lin_t = (std::size_t)LinIdx_(b, a, j, i);
              sig_t = quda::complex<double>(
                  sigma_v3[(lin_t * 2 + 0) * N + idx],
                  sigma_v3[(lin_t * 2 + 1) * N + idx]);
              pii_t = quda::complex<double>(
                  pi_v3[(lin_t * 2 + 0) * N + idx],
                  pi_v3[(lin_t * 2 + 1) * N + idx]);
            }
#pragma unroll
            for (int alpha = 0; alpha < 4; ++alpha) {
              const double g5_sign_a = Gamma5SignDR_(alpha);
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
          const double g5_sign_a = Gamma5SignDR_(alpha);
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
            const std::size_t lin = (std::size_t)LinIdx_(a, b, i, j);
            const quda::complex<double> dd(
                d_v3[(lin * 2 + 0) * N + idx],
                d_v3[(lin * 2 + 1) * N + idx]);
            const quda::complex<double> nn(
                n_v3[(lin * 2 + 0) * N + idx],
                n_v3[(lin * 2 + 1) * N + idx]);
#pragma unroll
            for (int alpha = 0; alpha < 4; ++alpha) {
              const double g5_sign_a = Gamma5SignDR_(alpha);
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

    quda::complex<double> ou0_inc_UK[12], ou1_inc_UK[12];
    quda::complex<double> ol0_inc_UK[12], ol1_inc_UK[12];
    toNonRelHalf_(spinor_ou0_DR_inc, ou0_inc_UK);
    toNonRelHalf_(spinor_ou1_DR_inc, ou1_inc_UK);
    toNonRelHalf_(spinor_ol0_DR_inc, ol0_inc_UK);
    toNonRelHalf_(spinor_ol1_DR_inc, ol1_inc_UK);

    quda::complex<double> ou0_cur[12], ou1_cur[12], ol0_cur[12], ol1_cur[12];
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

void ApplyAuxKernelNativeV3(
    const AuxCacheV3 &aux,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux, bool use_dn_conj) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeV3Impl_<true, true>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeV3Impl_<true, false>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeV3Impl_<false, true>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeV3Impl_<false, false>(aux,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}


// ---- V4: thread-per-(site, output-flavor) ----------------------------------
// Coordinator-directed decomposition experiment (V2 8.60 / V3b 12.24 / V3a
// 21.6 ms is monotonic in per-thread state => occupancy-limited).  V4 keeps
// V2's AoS cache + perm indirection UNCHANGED (only the decomposition
// varies): 2x threads, each computing ONE output flavor a — the two flavors
// write disjoint CSFs, so no cross-thread cooperation is needed.  Per-thread
// state: half the accumulators, half the aux rows; input spinors are loaded
// by both site-threads (coalesced + L2-warm).  Arithmetic per output element
// is IDENTICAL to V2.
template <bool TransposeAux, bool UseDnConj>
static void ApplyAuxKernelNativeV4Impl_(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1) {
  using AccTy = quda::colorspinor::FloatNOrder<double, 4, 3, 2>;
  AccTy iu0_acc(in_u0),  iu1_acc(in_u1),  il0_acc(in_l0),  il1_acc(in_l1);
  AccTy ou0_acc(out_u0), ou1_acc(out_u1), ol0_acc(out_l0), ol1_acc(out_l1);

  int volumeCB = in_u0.VolumeCB();
  int nParity = (in_u0.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  size_t N = (size_t)nParity * (size_t)volumeCB;
  size_t N4 = 2 * N;

  const double *sigma_d = aux.sigma_d;
  const double *pi_d    = aux.pi_d;
  const double *d_d     = aux.d_d;
  const double *n_d     = aux.n_d;
  const double *s_d     = aux.s_d;
  const double *p_d     = aux.p_d;
  const double scale_dn = DtxqcdOffdiagFactor();

  accelerator_for(idx4, N4, 1, {
    const int a = (int)(idx4 & 1);
    const size_t site = idx4 >> 1;
    int parity = site / volumeCB;
    int x_cb   = site % volumeCB;
    std::size_t our_eo = (std::size_t)perm_d[parity * volumeCB + x_cb];

    quda::complex<double> spinor_iu0_UK[12], spinor_iu1_UK[12];
    quda::complex<double> spinor_il0_UK[12], spinor_il1_UK[12];
    iu0_acc.load(spinor_iu0_UK, x_cb, parity);
    iu1_acc.load(spinor_iu1_UK, x_cb, parity);
    il0_acc.load(spinor_il0_UK, x_cb, parity);
    il1_acc.load(spinor_il1_UK, x_cb, parity);

    quda::complex<double> spinor_iu0_DR[12], spinor_iu1_DR[12];
    quda::complex<double> spinor_il0_DR[12], spinor_il1_DR[12];
    toRel_(spinor_iu0_UK, spinor_iu0_DR);
    toRel_(spinor_iu1_UK, spinor_iu1_DR);
    toRel_(spinor_il0_UK, spinor_il0_DR);
    toRel_(spinor_il1_UK, spinor_il1_DR);

    quda::complex<double> ou_a_DR_inc[12] = {};
    quda::complex<double> ol_a_DR_inc[12] = {};

    double s_val = s_d[our_eo * 2];
    double p_val = p_d[our_eo * 2];

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
                sigma_d[MatIdxV4_(our_eo, a, b, i, j, 0)],
                sigma_d[MatIdxV4_(our_eo, a, b, i, j, 1)]);
            quda::complex<double> pii(
                pi_d[MatIdxV4_(our_eo, a, b, i, j, 0)],
                pi_d[MatIdxV4_(our_eo, a, b, i, j, 1)]);

            quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
            quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];

            valu += sig * iu_bj + g5_sign_a * pii * iu_bj;

            if constexpr (TransposeAux) {
              quda::complex<double> sig_t(
                  sigma_d[MatIdxV4_(our_eo, b, a, j, i, 0)],
                  sigma_d[MatIdxV4_(our_eo, b, a, j, i, 1)]);
              quda::complex<double> pii_t(
                  pi_d[MatIdxV4_(our_eo, b, a, j, i, 0)],
                  pi_d[MatIdxV4_(our_eo, b, a, j, i, 1)]);
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
                d_d[MatIdxV4_(our_eo, a, b, i, j, 0)],
                d_d[MatIdxV4_(our_eo, a, b, i, j, 1)]);
            quda::complex<double> nn(
                n_d[MatIdxV4_(our_eo, a, b, i, j, 0)],
                n_d[MatIdxV4_(our_eo, a, b, i, j, 1)]);

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

        ou_a_DR_inc[alpha * Nc + i] += valu;
        ol_a_DR_inc[alpha * Nc + i] += vall;
      }
    }

    quda::complex<double> ou_inc_UK[12], ol_inc_UK[12];
    toNonRelHalf_(ou_a_DR_inc, ou_inc_UK);
    toNonRelHalf_(ol_a_DR_inc, ol_inc_UK);

    AccTy &ou_acc = (a == 0) ? ou0_acc : ou1_acc;
    AccTy &ol_acc = (a == 0) ? ol0_acc : ol1_acc;
    quda::complex<double> ou_cur[12], ol_cur[12];
    ou_acc.load(ou_cur, x_cb, parity);
    ol_acc.load(ol_cur, x_cb, parity);
    for (int k = 0; k < 12; ++k) {
      ou_cur[k] += ou_inc_UK[k];
      ol_cur[k] += ol_inc_UK[k];
    }
    ou_acc.save(ou_cur, x_cb, parity);
    ol_acc.save(ol_cur, x_cb, parity);
  });
}

void ApplyAuxKernelNativeV4(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux, bool use_dn_conj) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeV4Impl_<true, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeV4Impl_<true, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeV4Impl_<false, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeV4Impl_<false, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}
}  // namespace DtxqcdQudaAuxKernelNativeV3
}  // namespace Grid
