#pragma once
// Style B Path B Session 1 v2 — DTXQCD aux contraction kernel operating on
// QUDA NATIVE (FLOAT2, halo-padded, UKQCD basis) layout via the
// quda::colorspinor::FloatNOrder accessor inside a Grid `accelerator_for`.
//
// This is the native-CSF counterpart of dtxqcd_quda_aux_kernel_device.h
// (which works on flat 24·V SPACE_SPIN_COLOR/EO buffers).  The math is the
// same per-site contraction; the only differences are:
//
//   1. Spinor I/O: accessor.load/save instead of raw `buf[24·site + ...]`.
//   2. Storage basis: NATIVE is UKQCD-basis; flat-24V is QUDA-DR-basis.
//      Round trip via FloatNOrder is `toRel` (UKQCD→DR) then `toNonRel/2`
//      (DR→UKQCD).  Per QUDA's color_spinor.h:606-631:
//         T_NR · T_R = 2·I   ⇒   T^{-1} = T_NR / 2
//      Math runs in DR basis (matches the flat-24V kernel's γ_5 sign convention
//      = diag(+1,+1,-1,-1)).  Output is divided by 2 inside the toNonRel-style
//      store helper to compensate for the round-trip 2×.
//   3. Site mapping: NATIVE uses QUDA's (parity, x_cb) — a different enumeration
//      from production's `parity*V_eo + (lex>>1)`.  Aux fields are packed in
//      production EO order; we need a perm table that maps
//         (parity, x_cb) → our_eo_idx  (= parity*V_eo + (lex>>1))
//      so the kernel can index aux per-site.  The perm is derived at setup time
//      via the same probe technique as Test_dtxqcd_native_layout_probe.cc:
//      write encoded lex_site values into a cpu CSF, csf.copy → native, read
//      the native buffer and decode each (parity, x_cb) → lex → our_eo_idx.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>  // reuse aux pack
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelNativeV2 {

using SigmaDoublesPerSite =
    std::integral_constant<std::size_t,
                           DtxqcdQudaAuxKernelDevice::SigmaDoublesPerSite>;
using ScalarDoublesPerSite =
    std::integral_constant<std::size_t,
                           DtxqcdQudaAuxKernelDevice::ScalarDoublesPerSite>;

// ----------------------------------------------------------------------------
// Per-site aux index inside the production EO-packed buffer.
// ----------------------------------------------------------------------------
accelerator_inline std::size_t MatIdxNative(std::size_t our_eo_idx, int a, int b,
                                            int i, int j, int ri) {
  std::size_t lin = ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
  return our_eo_idx * SigmaDoublesPerSite::value + lin * 2 + ri;
}

// ----------------------------------------------------------------------------
// Build the perm table:  perm[parity * volumeCB + x_cb] → our_eo_idx
//
// Recipe: write encoded values `1.0e6 * lex_site` into a cpu CSF in
// DIRAC_ORDER+EO layout (using our `parity*V_eo + (lex>>1)` enumeration).
// csf.copy → native.  Read the native buffer at (parity, x_cb, complex_idx=0,
// slot=0) — the value decodes to OUR lex_site at that QUDA (parity, x_cb).
// Then our_eo_idx = parity * V_eo + (lex >> 1).
// ----------------------------------------------------------------------------
inline std::vector<int> build_perm_table(quda::ColorSpinorField &native_csf,
                                         QudaInvertParam &inv_param,
                                         const int X_full_dims[4],
                                         GridCartesian *grid) {
  int volumeCB = native_csf.VolumeCB();
  int nParity = (native_csf.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? 2 : 1;
  int V = nParity * volumeCB;

  // Build encoded host buffer in our EO layout.
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
          // Encode lex into spinor[(spin=0, color=0, re)] = lex * 1e6.
          // Keep all other 23 components at 0.
          h_probe[our_eo * 24 + 0] = (double)lex * 1.0e6;
        }
      }
    }
  }

  // Upload to a temp device buffer.
  size_t buf_bytes = 24 * V * sizeof(double);
  double *d_probe = (double *)acceleratorAllocDevice(buf_bytes);
  acceleratorCopyToDevice(h_probe.data(), d_probe, buf_bytes);
  acceleratorCopySynchronise();

  // Build cpu CSF wrapping d_probe (DIRAC_ORDER + EVEN_ODD), then promote to
  // native via the cudaParam ctor + copy.
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

  // Pull native bytes back to host for decoding.
  std::vector<double> h_native(native_csf.Bytes() / sizeof(double));
  acceleratorCopyFromDevice(native_csf.data<double *>(), h_native.data(),
                            native_csf.Bytes());
  acceleratorCopySynchronise();

  // Per-parity offset (in doubles).
  std::size_t offset_doubles = native_csf.Bytes() / (2 * sizeof(double));

  // For each (parity, x_cb), read at complex_idx=0, slot=0 and (slot=1) to
  // recover the lex.  The DR↔UKQCD basis convert rotates non-zero entries
  // across spins/colors, but the value-encoding `lex*1e6` lives in a 4D
  // unit-vector spinor that distributes across multiple complex slots in
  // UKQCD basis.  Robust decode: sum |native[i]| over all 24 slots at this
  // site; the sum equals 4·|lex*1e6| because T·e_(spin=0,color=0) has 4 unit
  // components (by inspection of toRel/toNonRel rows).  Use 1/4 of the sum
  // of magnitudes to get back |lex*1e6|.
  //
  // Even simpler: use the value at any single non-zero slot.  We can find
  // it by inspecting magnitudes.  But cleanest: sum absolute values, divide
  // by 4, round to integer = lex.
  std::vector<int> perm((size_t)nParity * volumeCB, -1);
  // Sum |native| over all 12 complex (24 doubles) at this (parity, x_cb).
  // After csf.copy, for a probe with only spin=0,color=0,re = lex*1e6 set:
  //   - if csf.copy applies T_NR basis transform: 2 non-zero entries each
  //     magnitude lex*1e6 → sum_abs = 2 * lex * 1e6
  //   - if csf.copy is pure reorder (no basis transform): 1 non-zero entry
  //     magnitude lex*1e6 → sum_abs = lex * 1e6
  // Detect divisor empirically by finding the smallest non-zero sum_abs at a
  // known parity-1 site (where lex >= 1), and inferring whether sum_abs is
  // 1·lex_min or 2·lex_min·1e6.
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
  // Smallest non-zero lex is 1 (since lex=0 site encodes 0).
  //   no transform → sum_abs = 1·1e6 → divisor=1
  //   normalized   → sum_abs = √2·1e6 → divisor=√2 (csf.copy in this QUDA)
  //   unnormalized → sum_abs = 2·1e6  → divisor=2
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

// ----------------------------------------------------------------------------
// γ_5 in DR basis (matches flat-24V kernel): diag(+1,+1,-1,-1) → sign at spin.
// ----------------------------------------------------------------------------
accelerator_inline double Gamma5SignDR(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

// ----------------------------------------------------------------------------
// Inline basis transforms operating on complex[12] arrays indexed by
// (spin*Nc + color).
//
// toRel:    UKQCD → DR        (psi_DR = T_R · psi_UKQCD)
// toNonRel: DR → UKQCD        (psi_UKQCD_partial = T_NR · psi_DR)
//                              where T_NR · T_R = 2·I
//
// To produce the UKQCD-basis output that, after csf.copy(UKQCD→DR), gives the
// desired DR-basis result, we apply T_NR · psi_DR / 2.
// ----------------------------------------------------------------------------
// Normalized inline transforms.  Each scales the unnormalized T_R/T_NR
// (entries ±1) by 1/√2, so that T_norm_NR · T_norm_R = I (matching what
// csf.copy applies internally).
accelerator_inline void toRelInline(const quda::complex<double> in[12],
                                    quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * (-in[1*3 + c] - in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * ( in[0*3 + c] + in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * (-in[1*3 + c] + in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * ( in[0*3 + c] - in[2*3 + c]);
  }
}

accelerator_inline void toNonRelHalfInline(const quda::complex<double> in[12],
                                           quda::complex<double> out[12]) {
  const double inv_sqrt2 = 0.70710678118654752440;
  for (int c = 0; c < 3; ++c) {
    out[0*3 + c] = inv_sqrt2 * ( in[1*3 + c] + in[3*3 + c]);
    out[1*3 + c] = inv_sqrt2 * (-in[0*3 + c] - in[2*3 + c]);
    out[2*3 + c] = inv_sqrt2 * ( in[1*3 + c] - in[3*3 + c]);
    out[3*3 + c] = inv_sqrt2 * (-in[0*3 + c] + in[2*3 + c]);
  }
}

// ----------------------------------------------------------------------------
// Apply the fused aux contraction on native-layout CSFs.
//
// in_*_acc, out_*_acc are FloatNOrder<double, 4, 3, 2> accessors built from
// 4 input + 4 output ColorSpinorFields (upper.f[0,1], lower.f[0,1]).
//
// The kernel ACCUMULATES into existing output values (matching the flat
// kernel's `out += aux_contribution` semantics).
//
// Template params match the flat kernel: TransposeAux, UseDnConj.
// ----------------------------------------------------------------------------
template <bool TransposeAux, bool UseDnConj>
inline void ApplyAuxKernelNativeImpl(
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

    // ---------- Load 4 input spinors in UKQCD basis, convert to DR ----------
    quda::complex<double> spinor_iu0_UK[12], spinor_iu1_UK[12];
    quda::complex<double> spinor_il0_UK[12], spinor_il1_UK[12];
    iu0_acc.load(spinor_iu0_UK, x_cb, parity);
    iu1_acc.load(spinor_iu1_UK, x_cb, parity);
    il0_acc.load(spinor_il0_UK, x_cb, parity);
    il1_acc.load(spinor_il1_UK, x_cb, parity);

    quda::complex<double> spinor_iu0_DR[12], spinor_iu1_DR[12];
    quda::complex<double> spinor_il0_DR[12], spinor_il1_DR[12];
    toRelInline(spinor_iu0_UK, spinor_iu0_DR);
    toRelInline(spinor_iu1_UK, spinor_iu1_DR);
    toRelInline(spinor_il0_UK, spinor_il0_DR);
    toRelInline(spinor_il1_UK, spinor_il1_DR);

    // Output accumulators in DR (initialised to 0; aux contribution will be
    // added; final round-trip handles existing output via separate load+add).
    quda::complex<double> spinor_ou0_DR_inc[12] = {};
    quda::complex<double> spinor_ou1_DR_inc[12] = {};
    quda::complex<double> spinor_ol0_DR_inc[12] = {};
    quda::complex<double> spinor_ol1_DR_inc[12] = {};

    // ---------- Aux contraction in DR basis ----------
    double s_val = s_d[our_eo * 2];     // real
    double p_val = p_d[our_eo * 2];

    for (int a = 0; a < DtxqcdNf; ++a) {
      auto *ou_a_DR = (a == 0) ? spinor_ou0_DR_inc : spinor_ou1_DR_inc;
      auto *ol_a_DR = (a == 0) ? spinor_ol0_DR_inc : spinor_ol1_DR_inc;
      auto *iu_a_DR = (a == 0) ? spinor_iu0_DR : spinor_iu1_DR;
      auto *il_a_DR = (a == 0) ? spinor_il0_DR : spinor_il1_DR;

      for (int alpha = 0; alpha < 4; ++alpha) {
        double g5_sign_a = Gamma5SignDR(alpha);

        for (int i = 0; i < Nc; ++i) {
          quda::complex<double> valu(0, 0), vall(0, 0);

          // ===== Pass A: σ + π + s + p =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<double> sig(
                  sigma_d[MatIdxNative(our_eo, a, b, i, j, 0)],
                  sigma_d[MatIdxNative(our_eo, a, b, i, j, 1)]);
              quda::complex<double> pii(
                  pi_d[MatIdxNative(our_eo, a, b, i, j, 0)],
                  pi_d[MatIdxNative(our_eo, a, b, i, j, 1)]);

              quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];

              valu += sig * iu_bj + g5_sign_a * pii * iu_bj;

              if constexpr (TransposeAux) {
                quda::complex<double> sig_t(
                    sigma_d[MatIdxNative(our_eo, b, a, j, i, 0)],
                    sigma_d[MatIdxNative(our_eo, b, a, j, i, 1)]);
                quda::complex<double> pii_t(
                    pi_d[MatIdxNative(our_eo, b, a, j, i, 0)],
                    pi_d[MatIdxNative(our_eo, b, a, j, i, 1)]);
                vall += sig_t * il_bj + g5_sign_a * pii_t * il_bj;
              } else {
                vall += sig * il_bj + g5_sign_a * pii * il_bj;
              }
            }
          }

          // Singlet (s + p γ_5) on (a, α, i).
          quda::complex<double> iu_a_alpha_i = iu_a_DR[alpha * Nc + i];
          quda::complex<double> il_a_alpha_i = il_a_DR[alpha * Nc + i];
          valu += s_val * iu_a_alpha_i + g5_sign_a * p_val * iu_a_alpha_i;
          vall += s_val * il_a_alpha_i + g5_sign_a * p_val * il_a_alpha_i;

          // ===== Pass B: d + n cross =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            auto *iu_b_DR = (b == 0) ? spinor_iu0_DR : spinor_iu1_DR;
            auto *il_b_DR = (b == 0) ? spinor_il0_DR : spinor_il1_DR;

            for (int j = 0; j < Nc; ++j) {
              quda::complex<double> dd(
                  d_d[MatIdxNative(our_eo, a, b, i, j, 0)],
                  d_d[MatIdxNative(our_eo, a, b, i, j, 1)]);
              quda::complex<double> nn(
                  n_d[MatIdxNative(our_eo, a, b, i, j, 0)],
                  n_d[MatIdxNative(our_eo, a, b, i, j, 1)]);

              quda::complex<double> iu_bj = iu_b_DR[alpha * Nc + j];
              quda::complex<double> il_bj = il_b_DR[alpha * Nc + j];

              // Upper gets cross from lower: d·γ5·in_l + n·in_l
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

    // ---------- Update outputs ----------
    // The kernel ACCUMULATES onto whatever's already in out_*.  Since the
    // existing output is in UKQCD storage:
    //   out_UKQCD_new = out_UKQCD_old + T_NR/2 · aux_contribution_DR
    // because if out_DR_new = out_DR_old + aux_contribution_DR, then
    // out_UKQCD_new = T_NR/2 · out_DR_new
    //               = T_NR/2 · out_DR_old + T_NR/2 · aux_contribution_DR
    //               = out_UKQCD_old + T_NR/2 · aux_contribution_DR.
    // Convert aux contribution DR → UKQCD via normalized T_NR/√2.
    quda::complex<double> ou0_inc_UK[12], ou1_inc_UK[12];
    quda::complex<double> ol0_inc_UK[12], ol1_inc_UK[12];
    toNonRelHalfInline(spinor_ou0_DR_inc, ou0_inc_UK);
    toNonRelHalfInline(spinor_ou1_DR_inc, ou1_inc_UK);
    toNonRelHalfInline(spinor_ol0_DR_inc, ol0_inc_UK);
    toNonRelHalfInline(spinor_ol1_DR_inc, ol1_inc_UK);

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

// Runtime dispatcher → one of 4 templated specialisations.
inline void ApplyAuxKernelNative(
    const DtxqcdQudaAuxKernelDevice::DeviceAuxCache &aux,
    const int *perm_d,
    quda::ColorSpinorField &in_u0, quda::ColorSpinorField &in_u1,
    quda::ColorSpinorField &in_l0, quda::ColorSpinorField &in_l1,
    quda::ColorSpinorField &out_u0, quda::ColorSpinorField &out_u1,
    quda::ColorSpinorField &out_l0, quda::ColorSpinorField &out_l1,
    bool transpose_aux = true, bool use_dn_conj = true) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelNativeImpl<true, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (transpose_aux)
    ApplyAuxKernelNativeImpl<true, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else if (use_dn_conj)
    ApplyAuxKernelNativeImpl<false, true>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
  else
    ApplyAuxKernelNativeImpl<false, false>(aux, perm_d,
        in_u0, in_u1, in_l0, in_l1, out_u0, out_u1, out_l0, out_l1);
}

}  // namespace DtxqcdQudaAuxKernelNativeV2
NAMESPACE_END(Grid);
