// dtxqcd_quda_hopcorr_native_decl.h — native (CSF-resident) t-hop correction
// kernel for the eta-deformed operator M~ (see production/dtxqcd_deform_quda.h
// for the operator identity and the dagger derivation).
//
//   out(x) += cf[t] * (-1/2)(1 -+ g_t) W3(x)  psi(x + that)
//           + cb[t] * (-1/2)(1 +- g_t) W7(x)  psi(x - that)
//
// on QUDA-native DP ColorSpinorFields (FloatNOrder<double,4,3,2>), converting
// UKQCD<->DR per site exactly like the pre/post native kernels.  W3 carries
// the APBC phase (baked at t = T-1), so neighbor indexing is purely periodic.
// SINGLE-RANK ONLY (no halo exchange): the deform measurements run 1.1.1.1;
// callers must assert one rank before enabling the native path.
//
// Strong definitions live in dtxqcd_quda_hopcorr_native_impl.cc (trove
// isolation, same pattern as the gamma5/pre/post TUs).

#pragma once

#include <cstddef>

namespace quda { class ColorSpinorField; }

namespace Grid {
namespace DtxqcdQudaHopCorrNative {

// Device copies of the phased t-links, parity-blocked cb order (site index
// parity*volCB + x_cb), 9 quda::complex<double> row-major per site.
struct HopGauge {
  void *w3_d = nullptr;
  void *w7_d = nullptr;
  int X[4] = {0, 0, 0, 0};
  std::size_t vol = 0;
};

// Upload from host staging arrays (parity-blocked cb order, 18 doubles/site
// row-major re/im interleaved).  Frees any previous upload.
void UploadHopGauge(const double *w3_host, const double *w7_host,
                    const int X[4], HopGauge &g);
void FreeHopGauge(HopGauge &g);

// Upload a length-T real coefficient array; returns device pointer (caller
// frees with FreeCoeffs).
double *UploadCoeffs(const double *host, int T);
void FreeCoeffs(double *d);

// out += hop-correction(in).  conj_gauge: use conj(W) (the lower block);
// flip_proj: swap the (1 -+ g_t) projectors (the dagger).
void ApplyHopCorrNative(quda::ColorSpinorField &in_csf,
                        quda::ColorSpinorField &out_csf, const HopGauge &g,
                        const double *cf_d, const double *cb_d,
                        bool conj_gauge, bool flip_proj);

// Parity-crossing variant for the EO Schur chain (M~eo/M~oe legs): in and
// out are SINGLE-PARITY CSFs of opposite parity.  out_parity names out's
// parity (0=even, 1=odd); the t+-1 neighbors read from `in` are then on
// in's parity automatically.  Same hop math, gauge indexed at the OUT site.
void ApplyHopCorrNativeEO(quda::ColorSpinorField &in_csf,
                          quda::ColorSpinorField &out_csf, const HopGauge &g,
                          const double *cf_d, const double *cb_d,
                          bool conj_gauge, bool flip_proj, int out_parity);

}  // namespace DtxqcdQudaHopCorrNative
}  // namespace Grid
