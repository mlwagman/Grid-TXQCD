// Style B Phase 2.5 Session 3d — out-of-line strong definitions for the aux
// native v2 kernel.  See gamma5_impl.cc for the trove-isolation rationale.
//
// DO NOT include dtxqcd_quda_aux_kernel_native_v2.h here — that header defines
// `build_perm_table` and `ApplyAuxKernelNative` as `inline`, which would cause
// an in-TU redefinition.  Standalone strong definitions below; bodies are
// copied verbatim from the original header to keep the math identical.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>  // DtxqcdSigmaPiHermitianOnly / DnComplexSymmetric constexpr helpers
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // DeviceAuxCache
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_decl.h>

namespace Grid {
namespace DtxqcdQudaAuxKernelNativeV2 {

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

// ---- build_perm_table -------------------------------------------------------
std::vector<int> build_perm_table(quda::ColorSpinorField &native_csf,
                                  QudaInvertParam &inv_param,
                                  const int X_full_dims[4],
                                  GridCartesian *grid) {
  int volumeCB = native_csf.VolumeCB();
  int nParity = (native_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  int V = nParity * volumeCB;

  std::vector<double> h_probe(24 * V, 0.0);
  Coordinate lc = grid->LocalDimensions();
  int lc0 = lc[0], lc1 = lc[1], lc2 = lc[2], lc3 = lc[3];
  int V_eo = V / 2;
  for (int t = 0; t < lc3; ++t) {
    for (int z = 0; z < lc2; ++z) {
      for (int y = 0; y < lc1; ++y) {
        for (int x = 0; x < lc0; ++x) {
          int lex = x + lc0 * (y + lc1 * (z + lc2 * t));
          int parity = (x + y + z + t) & 1;
          int our_eo = parity * V_eo + (lex >> 1);
          h_probe[our_eo * 24 + 0] = (double)lex * 1.0e6;
        }
      }
    }
  }

  size_t buf_bytes = 24 * V * sizeof(double);
  double *d_probe = (double *)acceleratorAllocDevice(buf_bytes);
  acceleratorCopyToDevice(h_probe.data(), d_probe, buf_bytes);
  acceleratorCopySynchronise();

  bool pc = false;
  quda::lat_dim_t X_full;
  for (int d = 0; d < 4; ++d) X_full[d] = X_full_dims[d];
  for (int d = 4; d < QUDA_MAX_DIM; ++d) X_full[d] = 1;
  QudaInvertParam inv = inv_param;
  inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
  inv.output_location = QUDA_CUDA_FIELD_LOCATION;
  quda::ColorSpinorParam cpuParam(d_probe, inv, X_full, pc,
                                  QUDA_CUDA_FIELD_LOCATION);
  quda::ColorSpinorField cpu_csf(cpuParam);
  native_csf.copy(cpu_csf);
  cudaDeviceSynchronize();

  std::vector<double> h_native(native_csf.Bytes() / sizeof(double));
  acceleratorCopyFromDevice(native_csf.data<double *>(), h_native.data(),
                            native_csf.Bytes());
  acceleratorCopySynchronise();

  std::size_t offset_doubles = native_csf.Bytes() / (2 * sizeof(double));

  std::vector<int> perm((size_t)nParity * volumeCB, -1);
  double min_nonzero_sum = 1.0e30;
  for (int parity = 0; parity < nParity; ++parity) {
    for (int x_cb = 0; x_cb < volumeCB; ++x_cb) {
      double sum_abs = 0.0;
      for (int complex_idx = 0; complex_idx < 12; ++complex_idx) {
        size_t idx = (size_t)parity * offset_doubles
                   + (size_t)volumeCB * complex_idx * 2
                   + (size_t)x_cb * 2;
        sum_abs += std::fabs(h_native[idx + 0]);
        sum_abs += std::fabs(h_native[idx + 1]);
      }
      if (sum_abs > 0.5e6 && sum_abs < min_nonzero_sum) {
        min_nonzero_sum = sum_abs;
      }
    }
  }
  double divisor;
  if (min_nonzero_sum < 1.2e6) divisor = 1.0;
  else if (min_nonzero_sum < 1.6e6) divisor = std::sqrt(2.0);
  else divisor = 2.0;

  for (int parity = 0; parity < nParity; ++parity) {
    for (int x_cb = 0; x_cb < volumeCB; ++x_cb) {
      double sum_abs = 0.0;
      for (int complex_idx = 0; complex_idx < 12; ++complex_idx) {
        size_t idx = (size_t)parity * offset_doubles
                   + (size_t)volumeCB * complex_idx * 2
                   + (size_t)x_cb * 2;
        sum_abs += std::fabs(h_native[idx + 0]);
        sum_abs += std::fabs(h_native[idx + 1]);
      }
      double lex_d = sum_abs / divisor / 1.0e6;
      int lex = (int)std::round(lex_d);
      int our_eo = parity * V_eo + (lex >> 1);
      perm[(size_t)parity * volumeCB + x_cb] = our_eo;
    }
  }
  std::cout << "[NativeV2::build_perm] divisor=" << divisor
            << " min_nonzero_sum=" << min_nonzero_sum << std::endl;

  acceleratorFreeDevice(d_probe);
  return perm;
}

// ---- ApplyAuxKernelNative (templated impl + runtime dispatcher) -------------
template <bool TransposeAux, bool UseDnConj>
static void ApplyAuxKernelNativeImpl_(
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

void ApplyAuxKernelNative(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux, bool use_dn_conj) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeImpl_<true, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeImpl_<true, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeImpl_<false, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeImpl_<false, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}

}  // namespace DtxqcdQudaAuxKernelNativeV2
}  // namespace Grid
