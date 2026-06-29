#pragma once
// M-wrap.5b.2 Option β: device-resident DTXQCD aux contraction operating on
// the flat 24·V EO buffer layout used by DoubledStateBuf.
//
// The Grid SIMD aux kernel `ApplyFusedDtxqcdAuxKernel` (Grid layout) is
// validated bit-exact at machine eps; this header reproduces the same per-site
// arithmetic on the flat EO buffer layout, so M_device can skip its
// download → Grid aux → upload roundtrip (Option α).
//
// Layout:
//   Spinor:  buf[24·site + 6·spin + 2·color + (re|im)]   (Grid::Quda::fermion_to_eo_buffer)
//   σ,π,d,n: 72 doubles per site, packed as
//              flat[site·72 + (((a·Nf + b)·Nc + i)·Nc + j)·2 + (re|im)]
//            (the same scalar_object layout used by aux_to_eo_buffer below)
//   s,p:     2 doubles per site (re, im); only re is used.
//
// γ_5 in DR (variant 0): negates spin 0, 1; keeps spin 2, 3.  We compute
// γ_5·in components on the fly per-site (a few sign flips on 12 doubles).
//
// Aux contraction (per-site, accumulating into out):
//   Pass A (σ + π + s + p):
//     valu_{aαi} = Σ_{b,j} [σ_{ab}^{ij} · in_u_{bα,j} + π_{ab}^{ij} · γ5_in_u_{bα,j}]
//                + s·in_u_{aα,i} + p·γ5_in_u_{aα,i}
//     vall_{aαi} = Σ_{b,j} [σ_{ba}^{ji}* (TransposeAux ? ) · in_l_{bα,j}
//                            + π_{ba}^{ji}* · γ5_in_l_{bα,j}]
//                + s·in_l_{aα,i} + p·γ5_in_l_{aα,i}
//   Pass B (d + n cross, scale = DtxqcdOffdiagFactor() = 1.0):
//     valu_{aαi} += Σ_{b,j} [d_{ab}^{ij} · γ5_in_l_{bα,j} + n_{ab}^{ij} · in_l_{bα,j}]
//     vall_{aαi} += Σ_{b,j} [d_{ab}^{ij}*(UseDnConj ? conj : id) · γ5_in_u_{bα,j}
//                            + n_{ab}^{ij}*(UseDnConj ? conj : id) · in_u_{bα,j}]
//
// All math mirrors detail::PassFlavorMix + detail::PassCrossDn in
// dtxqcd_quda_aux_kernel.h.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>   // DtxqcdOffdiagFactor
#include <Grid/util/QudaFieldConvert.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdQudaAuxKernelDevice {

// ----------------------------------------------------------------------------
// Per-site aux scalar sizes
// ----------------------------------------------------------------------------
static constexpr std::size_t SigmaDoublesPerSite =
    DtxqcdNf * DtxqcdNf * Nc * Nc * 2;    // 2*2*3*3*2 = 72
static constexpr std::size_t ScalarDoublesPerSite = 2;   // (re, im); only re used

// ----------------------------------------------------------------------------
// CFMatrix / Scalar → EO host buffer (mirrors fermion_to_eo_buffer pattern).
//
// scalar_object for `iScalar<iMatrix<iMatrix<vComplex,Nc>,DtxqcdNf>>` is laid
// out exactly as the per-site complex array we want — memcpy-compatible.
// Same for the singlet `iScalar<iScalar<iScalar<vComplex>>>` → 2 doubles/site.
// ----------------------------------------------------------------------------
template <class AuxField>
inline void aux_to_eo_buffer(const AuxField &aux_grid, double *buf_eo) {
  using SiteAux = typename AuxField::scalar_object;
  using vobj    = typename AuxField::vector_object;
  constexpr std::size_t per_site_doubles = sizeof(SiteAux) / sizeof(double);

  GridBase *grid = aux_grid.Grid();
  Coordinate lc = grid->LocalDimensions();
  assert((lc[0] & 1) == 0);
  int V = Quda::local_volume(grid);
  int V_eo = V / 2;
  const int Nsimd = vobj::vector_type::Nsimd();
  const int ndim = grid->Nd();

  std::vector<Coordinate> icoor_table(Nsimd);
  for (int lane = 0; lane < Nsimd; ++lane) {
    icoor_table[lane].resize(ndim);
    grid->iCoorFromIindex(icoor_table[lane], lane);
  }

  autoView(in_v, aux_grid, CpuRead);
  thread_for(oidx, grid->oSites(), {
    ExtractPointerArray<SiteAux> ptrs(Nsimd);
    Coordinate ocoor(ndim);
    grid->oCoorFromOindex(ocoor, oidx);
    for (int lane = 0; lane < Nsimd; ++lane) {
      int x = ocoor[0] + grid->_rdimensions[0] * icoor_table[lane][0];
      int y = ocoor[1] + grid->_rdimensions[1] * icoor_table[lane][1];
      int z = ocoor[2] + grid->_rdimensions[2] * icoor_table[lane][2];
      int t = ocoor[3] + grid->_rdimensions[3] * icoor_table[lane][3];
      int lex = x + lc[0] * (y + lc[1] * (z + lc[2] * t));
      int parity = (x + y + z + t) & 1;
      int eo_idx = parity * V_eo + (lex >> 1);
      ptrs[lane] = reinterpret_cast<SiteAux *>(
          &buf_eo[(std::size_t)eo_idx * per_site_doubles]);
    }
    extract(in_v[oidx], ptrs, 0);
  });
}

// ----------------------------------------------------------------------------
// Device-side aux cache: 6 device buffers (σ, π, d, n, s, p) in EO layout
// matching the spinor buffer.  Aux fields are constant across all M_device
// calls inside a single multishift CG, so we pack once + reuse.
// ----------------------------------------------------------------------------
struct DeviceAuxCache {
  double *sigma_d = nullptr;   // SigmaDoublesPerSite · V doubles
  double *pi_d    = nullptr;
  double *d_d     = nullptr;
  double *n_d     = nullptr;
  double *s_d     = nullptr;   // ScalarDoublesPerSite · V doubles
  double *p_d     = nullptr;
  std::size_t V = 0;

  bool allocated() const { return sigma_d != nullptr; }
};

inline void allocate_aux_cache(DeviceAuxCache &c, std::size_t V_local) {
  std::size_t mat_bytes    = SigmaDoublesPerSite  * V_local * sizeof(double);
  std::size_t scalar_bytes = ScalarDoublesPerSite * V_local * sizeof(double);
  c.sigma_d = (double *)acceleratorAllocDevice(mat_bytes);
  c.pi_d    = (double *)acceleratorAllocDevice(mat_bytes);
  c.d_d     = (double *)acceleratorAllocDevice(mat_bytes);
  c.n_d     = (double *)acceleratorAllocDevice(mat_bytes);
  c.s_d     = (double *)acceleratorAllocDevice(scalar_bytes);
  c.p_d     = (double *)acceleratorAllocDevice(scalar_bytes);
  c.V = V_local;
}

inline void free_aux_cache(DeviceAuxCache &c) {
  if (c.sigma_d) acceleratorFreeDevice(c.sigma_d);
  if (c.pi_d)    acceleratorFreeDevice(c.pi_d);
  if (c.d_d)     acceleratorFreeDevice(c.d_d);
  if (c.n_d)     acceleratorFreeDevice(c.n_d);
  if (c.s_d)     acceleratorFreeDevice(c.s_d);
  if (c.p_d)     acceleratorFreeDevice(c.p_d);
  c = DeviceAuxCache{};
}

// Pack Grid SIMD aux fields → device EO buffers (one-time per CG; ~1-5 ms).
inline void pack_aux_to_device(const LatticeDtxqcdSigma &sigma,
                               const LatticeDtxqcdPi    &pi,
                               const LatticeDtxqcdD     &d,
                               const LatticeDtxqcdN     &n,
                               const LatticeDtxqcdS     &s,
                               const LatticeDtxqcdP     &p,
                               DeviceAuxCache &c) {
  std::size_t V = c.V;
  std::vector<double> host_mat(SigmaDoublesPerSite * V);
  std::vector<double> host_sc (ScalarDoublesPerSite * V);

  aux_to_eo_buffer(sigma, host_mat.data());
  acceleratorCopyToDevice(host_mat.data(), c.sigma_d,
                          SigmaDoublesPerSite * V * sizeof(double));
  aux_to_eo_buffer(pi, host_mat.data());
  acceleratorCopyToDevice(host_mat.data(), c.pi_d,
                          SigmaDoublesPerSite * V * sizeof(double));
  aux_to_eo_buffer(d, host_mat.data());
  acceleratorCopyToDevice(host_mat.data(), c.d_d,
                          SigmaDoublesPerSite * V * sizeof(double));
  aux_to_eo_buffer(n, host_mat.data());
  acceleratorCopyToDevice(host_mat.data(), c.n_d,
                          SigmaDoublesPerSite * V * sizeof(double));
  aux_to_eo_buffer(s, host_sc.data());
  acceleratorCopyToDevice(host_sc.data(), c.s_d,
                          ScalarDoublesPerSite * V * sizeof(double));
  aux_to_eo_buffer(p, host_sc.data());
  acceleratorCopyToDevice(host_sc.data(), c.p_d,
                          ScalarDoublesPerSite * V * sizeof(double));
}

// ----------------------------------------------------------------------------
// Per-site helpers used inside the device kernel.
// ----------------------------------------------------------------------------

// In-buffer helpers: per-site spinor index of (spin α, color i, re/im).
// site is the EO index (0..V-1).
accelerator_inline std::size_t SpinorIdx(std::size_t site, int spin, int color,
                                         int ri) {
  return site * 24 + 6 * spin + 2 * color + ri;
}

// Aux (matrix) index for σ_{ab}^{ij}, in (re|im).
accelerator_inline std::size_t MatIdx(std::size_t site, int a, int b, int i,
                                      int j, int ri) {
  // Layout: ((a · Nf + b) · Nc + i) · Nc + j → 36-element complex array.
  std::size_t lin = ((a * DtxqcdNf + b) * Nc + i) * Nc + j;
  return site * SigmaDoublesPerSite + lin * 2 + ri;
}

// Grid's γ_5 (per Gamma.h:90 multGamma5): diag(+1, +1, -1, -1) — negate spin 2, 3.
//  Returns the sign multiplier for a given spin: +1 for α<2, -1 for α≥2.
accelerator_inline double Gamma5Sign(int spin) {
  return (spin < 2) ? 1.0 : -1.0;
}

// Read a complex spinor component (re, im) from a flat spinor buffer.
accelerator_inline void ReadSpinor(const double *buf, std::size_t site,
                                   int spin, int color,
                                   double &re, double &im) {
  re = buf[SpinorIdx(site, spin, color, 0)];
  im = buf[SpinorIdx(site, spin, color, 1)];
}

// ----------------------------------------------------------------------------
// Apply the fused aux contraction.  Accumulates onto existing values in out.
// ----------------------------------------------------------------------------
template <bool TransposeAux, bool UseDnConj>
inline void ApplyAuxKernelImpl(
    const DeviceAuxCache &aux,
    double *in_u0, double *in_u1, double *in_l0, double *in_l1,
    double *out_u0, double *out_u1, double *out_l0, double *out_l1,
    std::size_t V) {
  const double *sigma_d = aux.sigma_d;
  const double *pi_d    = aux.pi_d;
  const double *d_d     = aux.d_d;
  const double *n_d     = aux.n_d;
  const double *s_d     = aux.s_d;
  const double *p_d     = aux.p_d;
  const double scale_dn = DtxqcdOffdiagFactor();   // 1.0 in production

  accelerator_for(site, V, 1, {
    // Per-site singlet scalars (real parts only; imag held at zero by
    // RealProjectInPlace).
    double s_val = s_d[site * 2];   // real part
    double p_val = p_d[site * 2];

    // For each output (flavor a, spin α, color i):
    for (int a = 0; a < DtxqcdNf; ++a) {
      double *ou_a = (a == 0) ? out_u0 : out_u1;
      double *ol_a = (a == 0) ? out_l0 : out_l1;
      // Singlet pieces read in_u_a, in_l_a at the SAME (a, α, i):
      double *iu_a = (a == 0) ? in_u0 : in_u1;
      double *il_a = (a == 0) ? in_l0 : in_l1;

      for (int alpha = 0; alpha < 4; ++alpha) {
        double g5_sign_a = Gamma5Sign(alpha);

        for (int i = 0; i < Nc; ++i) {
          double valu_re = 0.0, valu_im = 0.0;
          double vall_re = 0.0, vall_im = 0.0;

          // ===== Pass A: σ + π flavor-mixing color-diagonal =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            double *iu_b = (b == 0) ? in_u0 : in_u1;
            double *il_b = (b == 0) ? in_l0 : in_l1;

            for (int j = 0; j < Nc; ++j) {
              // Read σ_{ab}^{ij} and π_{ab}^{ij}.
              double sig_re = sigma_d[MatIdx(site, a, b, i, j, 0)];
              double sig_im = sigma_d[MatIdx(site, a, b, i, j, 1)];
              double  pi_re =    pi_d[MatIdx(site, a, b, i, j, 0)];
              double  pi_im =    pi_d[MatIdx(site, a, b, i, j, 1)];

              double iu_re = iu_b[SpinorIdx(site, alpha, j, 0)];
              double iu_im = iu_b[SpinorIdx(site, alpha, j, 1)];
              double il_re = il_b[SpinorIdx(site, alpha, j, 0)];
              double il_im = il_b[SpinorIdx(site, alpha, j, 1)];
              // γ_5 · in_b is just the in_b reads scaled by Gamma5Sign(alpha).

              // Upper accumulator: σ_{ab}^{ij} · in_u_b + π_{ab}^{ij} · γ5·in_u_b
              valu_re += sig_re * iu_re - sig_im * iu_im;
              valu_im += sig_re * iu_im + sig_im * iu_re;
              valu_re += g5_sign_a * (pi_re * iu_re - pi_im * iu_im);
              valu_im += g5_sign_a * (pi_re * iu_im + pi_im * iu_re);

              if constexpr (TransposeAux) {
                // Lower reads σ_{ba}^{ji} (transposed combined index) → swap a↔b, i↔j.
                double sig_t_re = sigma_d[MatIdx(site, b, a, j, i, 0)];
                double sig_t_im = sigma_d[MatIdx(site, b, a, j, i, 1)];
                double  pi_t_re =    pi_d[MatIdx(site, b, a, j, i, 0)];
                double  pi_t_im =    pi_d[MatIdx(site, b, a, j, i, 1)];
                vall_re += sig_t_re * il_re - sig_t_im * il_im;
                vall_im += sig_t_re * il_im + sig_t_im * il_re;
                vall_re += g5_sign_a * (pi_t_re * il_re - pi_t_im * il_im);
                vall_im += g5_sign_a * (pi_t_re * il_im + pi_t_im * il_re);
              } else {
                vall_re += sig_re * il_re - sig_im * il_im;
                vall_im += sig_re * il_im + sig_im * il_re;
                vall_re += g5_sign_a * (pi_re * il_re - pi_im * il_im);
                vall_im += g5_sign_a * (pi_re * il_im + pi_im * il_re);
              }
            }
          }

          // Singlet pieces (s + p·γ5) on (a, α, i).
          double iu_a_re = iu_a[SpinorIdx(site, alpha, i, 0)];
          double iu_a_im = iu_a[SpinorIdx(site, alpha, i, 1)];
          double il_a_re = il_a[SpinorIdx(site, alpha, i, 0)];
          double il_a_im = il_a[SpinorIdx(site, alpha, i, 1)];
          valu_re += s_val * iu_a_re + g5_sign_a * p_val * iu_a_re;
          valu_im += s_val * iu_a_im + g5_sign_a * p_val * iu_a_im;
          vall_re += s_val * il_a_re + g5_sign_a * p_val * il_a_re;
          vall_im += s_val * il_a_im + g5_sign_a * p_val * il_a_im;

          // ===== Pass B: d + n cross (color-mixing, scaled by DtxqcdOffdiagFactor) =====
          for (int b = 0; b < DtxqcdNf; ++b) {
            double *iu_b = (b == 0) ? in_u0 : in_u1;
            double *il_b = (b == 0) ? in_l0 : in_l1;

            for (int j = 0; j < Nc; ++j) {
              double d_re = d_d[MatIdx(site, a, b, i, j, 0)];
              double d_im = d_d[MatIdx(site, a, b, i, j, 1)];
              double n_re = n_d[MatIdx(site, a, b, i, j, 0)];
              double n_im = n_d[MatIdx(site, a, b, i, j, 1)];

              double il_re = il_b[SpinorIdx(site, alpha, j, 0)];
              double il_im = il_b[SpinorIdx(site, alpha, j, 1)];
              double iu_re = iu_b[SpinorIdx(site, alpha, j, 0)];
              double iu_im = iu_b[SpinorIdx(site, alpha, j, 1)];

              // Upper gets cross from lower: d·γ5·in_l + n·in_l, no conj.
              valu_re += scale_dn * (g5_sign_a * (d_re * il_re - d_im * il_im));
              valu_im += scale_dn * (g5_sign_a * (d_re * il_im + d_im * il_re));
              valu_re += scale_dn * (n_re * il_re - n_im * il_im);
              valu_im += scale_dn * (n_re * il_im + n_im * il_re);

              if constexpr (UseDnConj) {
                // Conj on (d, n) for the upper→lower direction.
                // conj(d) · γ5·in_u + conj(n) · in_u.  conj flips sign of imag.
                vall_re += scale_dn * (g5_sign_a * (d_re * iu_re + d_im * iu_im));
                vall_im += scale_dn * (g5_sign_a * (d_re * iu_im - d_im * iu_re));
                vall_re += scale_dn * (n_re * iu_re + n_im * iu_im);
                vall_im += scale_dn * (n_re * iu_im - n_im * iu_re);
              } else {
                vall_re += scale_dn * (g5_sign_a * (d_re * iu_re - d_im * iu_im));
                vall_im += scale_dn * (g5_sign_a * (d_re * iu_im + d_im * iu_re));
                vall_re += scale_dn * (n_re * iu_re - n_im * iu_im);
                vall_im += scale_dn * (n_re * iu_im + n_im * iu_re);
              }
            }
          }

          // Accumulate.
          std::size_t ou_idx_re = SpinorIdx(site, alpha, i, 0);
          std::size_t ou_idx_im = SpinorIdx(site, alpha, i, 1);
          ou_a[ou_idx_re] += valu_re;
          ou_a[ou_idx_im] += valu_im;
          ol_a[ou_idx_re] += vall_re;
          ol_a[ou_idx_im] += vall_im;
        }
      }
    }
  });
}

// Runtime-flag dispatcher → one of 4 templated specialisations.
inline void ApplyAuxKernel(
    const DeviceAuxCache &aux,
    double *in_u0, double *in_u1, double *in_l0, double *in_l1,
    double *out_u0, double *out_u1, double *out_l0, double *out_l1,
    std::size_t V,
    bool transpose_aux = true, bool use_dn_conj = true) {
  if (transpose_aux && use_dn_conj)
    ApplyAuxKernelImpl<true,  true >(aux, in_u0, in_u1, in_l0, in_l1,
                                     out_u0, out_u1, out_l0, out_l1, V);
  else if (transpose_aux)
    ApplyAuxKernelImpl<true,  false>(aux, in_u0, in_u1, in_l0, in_l1,
                                     out_u0, out_u1, out_l0, out_l1, V);
  else if (use_dn_conj)
    ApplyAuxKernelImpl<false, true >(aux, in_u0, in_u1, in_l0, in_l1,
                                     out_u0, out_u1, out_l0, out_l1, V);
  else
    ApplyAuxKernelImpl<false, false>(aux, in_u0, in_u1, in_l0, in_l1,
                                     out_u0, out_u1, out_l0, out_l1, V);
}

}  // namespace DtxqcdQudaAuxKernelDevice
NAMESPACE_END(Grid);
