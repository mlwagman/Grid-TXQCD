#pragma once
// M-wrap.4-v2: charge-conjugation matrix C = γ_2·γ_4 (DR basis) applied per
// site on a flat double Wilson-spinor buffer (24 doubles/site).
//
// Replaces the broken γ_2·conj sandwich from M-wrap.4 first try.  The actual
// identity that works for Wilson-clover (per DTXQCDDeltaCloverOp.h:8-27 and
// the dtxqcd.tex doubled-action derivation):
//
//   M_22 · ψ_lower  =  C^T · M_QCD^T · C · ψ_lower
//                   =  C   · M_QCD^T · C · ψ_lower      (since C^T = -C, the
//                                                        two minus signs
//                                                        cancel inside the
//                                                        sandwich)
//
// And M_QCD^T · w = conj(M_QCD^† · conj(w)) = conj(MatQuda(conj(w), dagger=YES)).
// So the lower-block protocol per (M·v) iter is:
//
//   v_lower  → [apply_C]    → tmp1                       (this kernel)
//            → conj_inplace                              (sign-flip on imag)
//            → MatQuda(dagger=YES, U gauge) → tmp2       (gives M_QCD^† · ...)
//            → conj_inplace                              (turns M^† into M^T)
//            → [apply_C]    → out_lower                  (same kernel again)
//
// The clover sign flip (-csw/2·F·σ in upper, +csw/2·F^T·σ in lower) and
// F-color-transpose that distinguish the lower block from the upper drop out
// of the M_QCD^T transposition + C-sandwich automatically.  No conjugation
// in C itself.
//
// QUDA DR basis (enum_quda.h:375):
//   γ_4 = (( 0, 1) , ( 1, 0))      — block-swap (no σ inside)
//   γ_2 = (( 0, -iσ_2) , ( iσ_2, 0))
//   with σ_2 = ((0,-i),(i,0)) so -iσ_2 = ((0,-1),(1,0)).
//
// Action on the 4-component spinor:
//   γ_4 (v_0, v_1, v_2, v_3)^T = (v_2, v_3, v_0, v_1)^T
//   γ_2 (v_0, v_1, v_2, v_3)^T = (-v_3, +v_2, +v_1, -v_0)^T   (from
//                                  dtxqcd_quda_gamma2_conj.h:20)
//
// Composition C = γ_2·γ_4:
//   C (v_0, v_1, v_2, v_3)^T = γ_2 (v_2, v_3, v_0, v_1)
//                            = (-v_1, +v_0, +v_3, -v_2)
//
// Purely a permutation + sign in spinor space, color-diagonal, no conj.
//
// As doubles (re/im interleaved, color-inside-spin ordering as in
// fermion_to_eo_buffer):
//   out[0,c]_re = -in[1,c]_re ;  out[0,c]_im = -in[1,c]_im
//   out[1,c]_re = +in[0,c]_re ;  out[1,c]_im = +in[0,c]_im
//   out[2,c]_re = +in[3,c]_re ;  out[2,c]_im = +in[3,c]_im
//   out[3,c]_re = -in[2,c]_re ;  out[3,c]_im = -in[2,c]_im

#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);
namespace Quda {

// In-place safe (out == in): NO — α reordering means distinct buffers required.
//
// V_sites = total number of fermion sites covered by the buffer.  Per-site
// local, so lex vs EO outer ordering is irrelevant.
inline void ApplyCKernel(double *out_d, const double *in_d, int V_sites) {
  constexpr int Nc_ = Nc;   // 3
  constexpr int Ns_ = Ns;   // 4
  static_assert(Nc_ == 3, "Wilson spinor expects Nc=3");
  static_assert(Ns_ == 4, "Wilson spinor expects Ns=4");

  accelerator_for(s, V_sites, 1, {
    const double *in  = &in_d [s * 24];
    double       *out = &out_d[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      // α=0:  out_0 = -v_1
      out[(0 * Nc_ + c) * 2 + 0] = -in[(1 * Nc_ + c) * 2 + 0];
      out[(0 * Nc_ + c) * 2 + 1] = -in[(1 * Nc_ + c) * 2 + 1];
      // α=1:  out_1 = +v_0
      out[(1 * Nc_ + c) * 2 + 0] = +in[(0 * Nc_ + c) * 2 + 0];
      out[(1 * Nc_ + c) * 2 + 1] = +in[(0 * Nc_ + c) * 2 + 1];
      // α=2:  out_2 = +v_3
      out[(2 * Nc_ + c) * 2 + 0] = +in[(3 * Nc_ + c) * 2 + 0];
      out[(2 * Nc_ + c) * 2 + 1] = +in[(3 * Nc_ + c) * 2 + 1];
      // α=3:  out_3 = -v_2
      out[(3 * Nc_ + c) * 2 + 0] = -in[(2 * Nc_ + c) * 2 + 0];
      out[(3 * Nc_ + c) * 2 + 1] = -in[(2 * Nc_ + c) * 2 + 1];
    }
  });
}

// Host-side equivalent.  Used by DTXQCDMpcOpQUDA (M-wrap.4-v2) where the QUDA
// Mat is in CPU_FIELD_LOCATION mode so the buffers are host-resident.
inline void ApplyCKernelHost(double *out_h, const double *in_h, int V_sites) {
  constexpr int Nc_ = Nc;
  constexpr int Ns_ = Ns;
  static_assert(Nc_ == 3 && Ns_ == 4, "Wilson spinor expects Ns=4, Nc=3");
  for (int s = 0; s < V_sites; ++s) {
    const double *in  = &in_h [s * 24];
    double       *out = &out_h[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      out[(0 * Nc_ + c) * 2 + 0] = -in[(1 * Nc_ + c) * 2 + 0];
      out[(0 * Nc_ + c) * 2 + 1] = -in[(1 * Nc_ + c) * 2 + 1];
      out[(1 * Nc_ + c) * 2 + 0] = +in[(0 * Nc_ + c) * 2 + 0];
      out[(1 * Nc_ + c) * 2 + 1] = +in[(0 * Nc_ + c) * 2 + 1];
      out[(2 * Nc_ + c) * 2 + 0] = +in[(3 * Nc_ + c) * 2 + 0];
      out[(2 * Nc_ + c) * 2 + 1] = +in[(3 * Nc_ + c) * 2 + 1];
      out[(3 * Nc_ + c) * 2 + 0] = -in[(2 * Nc_ + c) * 2 + 0];
      out[(3 * Nc_ + c) * 2 + 1] = -in[(2 * Nc_ + c) * 2 + 1];
    }
  }
}

// In-place conjugation (free op): flip sign of every imag entry.
//
// V_sites = total number of fermion sites; per-site local.
inline void ConjInplaceHost(double *buf, int V_sites) {
  for (int s = 0; s < V_sites; ++s) {
    double *b = &buf[s * 24];
    for (int k = 0; k < 24; k += 2) {
      b[k + 1] = -b[k + 1];  // flip imag
    }
  }
}

// M-wrap.5a: device versions for QUDA_CUDA_FIELD_LOCATION mode (M_device).
// In-place conjugation on a device-resident buffer.
inline void ConjInplaceKernel(double *buf_d, int V_sites) {
  accelerator_for(s, V_sites, 1, {
    double *b = &buf_d[s * 24];
    for (int k = 0; k < 24; k += 2) {
      b[k + 1] = -b[k + 1];  // flip imag
    }
  });
}

// Per-site negation of the entire flat double buffer (used after the
// post-MatQuda apply_C step to realize the explicit "-" from C^T = -C).
inline void NegateInplaceKernel(double *buf_d, int V_sites) {
  accelerator_for(s, V_sites, 1, {
    double *b = &buf_d[s * 24];
    for (int k = 0; k < 24; ++k) b[k] = -b[k];
  });
}

// M-wrap.5a fused pre-MatQuda lower-block: scratch_out = conj(C · in_lower).
// Combines ApplyC + ConjInplace into one device kernel, avoiding a separate
// pass.  Per-site math (composing C and conj on spin α):
//   scratch[0,c] = conj(-in[1,c]) = (-in[1,c]_re, +in[1,c]_im)
//   scratch[1,c] = conj(+in[0,c]) = (+in[0,c]_re, -in[0,c]_im)
//   scratch[2,c] = conj(+in[3,c]) = (+in[3,c]_re, -in[3,c]_im)
//   scratch[3,c] = conj(-in[2,c]) = (-in[2,c]_re, +in[2,c]_im)
inline void PreMatLowerKernel(double *scratch_d, const double *in_d, int V_sites) {
  constexpr int Nc_ = Nc;
  accelerator_for(s, V_sites, 1, {
    const double *in  = &in_d[s * 24];
    double *out = &scratch_d[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      // α=0: -in[1,c] with conj → (-re, +im)
      out[(0 * Nc_ + c) * 2 + 0] = -in[(1 * Nc_ + c) * 2 + 0];
      out[(0 * Nc_ + c) * 2 + 1] = +in[(1 * Nc_ + c) * 2 + 1];
      // α=1: +in[0,c] with conj → (+re, -im)
      out[(1 * Nc_ + c) * 2 + 0] = +in[(0 * Nc_ + c) * 2 + 0];
      out[(1 * Nc_ + c) * 2 + 1] = -in[(0 * Nc_ + c) * 2 + 1];
      // α=2: +in[3,c] with conj
      out[(2 * Nc_ + c) * 2 + 0] = +in[(3 * Nc_ + c) * 2 + 0];
      out[(2 * Nc_ + c) * 2 + 1] = -in[(3 * Nc_ + c) * 2 + 1];
      // α=3: -in[2,c] with conj
      out[(3 * Nc_ + c) * 2 + 0] = -in[(2 * Nc_ + c) * 2 + 0];
      out[(3 * Nc_ + c) * 2 + 1] = +in[(2 * Nc_ + c) * 2 + 1];
    }
  });
}

// M-wrap.5a fused post-MatQuda lower-block: out_lower = -C · conj(out_lower).
// In-place safe (read into local registers then write back).  Per-site:
//   new[0,c] = +conj(old[1,c]) = (+old[1,c]_re, -old[1,c]_im)
//   new[1,c] = -conj(old[0,c]) = (-old[0,c]_re, +old[0,c]_im)
//   new[2,c] = -conj(old[3,c]) = (-old[3,c]_re, +old[3,c]_im)
//   new[3,c] = +conj(old[2,c]) = (+old[2,c]_re, -old[2,c]_im)
inline void PostMatLowerKernel(double *out_d, int V_sites) {
  constexpr int Nc_ = Nc;
  accelerator_for(s, V_sites, 1, {
    double *out = &out_d[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      double r0 = out[(0 * Nc_ + c) * 2 + 0], i0 = out[(0 * Nc_ + c) * 2 + 1];
      double r1 = out[(1 * Nc_ + c) * 2 + 0], i1 = out[(1 * Nc_ + c) * 2 + 1];
      double r2 = out[(2 * Nc_ + c) * 2 + 0], i2 = out[(2 * Nc_ + c) * 2 + 1];
      double r3 = out[(3 * Nc_ + c) * 2 + 0], i3 = out[(3 * Nc_ + c) * 2 + 1];
      out[(0 * Nc_ + c) * 2 + 0] = +r1;  out[(0 * Nc_ + c) * 2 + 1] = -i1;
      out[(1 * Nc_ + c) * 2 + 0] = -r0;  out[(1 * Nc_ + c) * 2 + 1] = +i0;
      out[(2 * Nc_ + c) * 2 + 0] = -r3;  out[(2 * Nc_ + c) * 2 + 1] = +i3;
      out[(3 * Nc_ + c) * 2 + 0] = +r2;  out[(3 * Nc_ + c) * 2 + 1] = -i2;
    }
  });
}

}  // namespace Quda
NAMESPACE_END(Grid);
