#pragma once
// M-wrap.2: γ_2·conj() basis-correction kernel on a flat per-site Wilson-spinor
// double buffer (24 doubles/site = Ns·Nc·(re,im)).
//
// Identity:
//   M[U*] · ψ_lower  =  conj( M[U] · γ_2·conj(ψ_lower) )   (modulo basis)
// applied to bridge QUDA's single-gauge MatQuda into the DTXQCD doubled
// operator that has upper sees U / lower sees conj(U).  Per-iter cost
// budget ~0.1 ms at 16³×48 (memory-bandwidth bound, 4·V doubles read +
// 4·V doubles written = ~9 MB at 4⁴, ~45 MB at 16³×48).
//
// γ_2 in QUDA's default DEGRAND_ROSSI basis (enum_quda.h:375):
//   γ_2 = (( 0  ,  -i·σ_2) , ( i·σ_2 , 0 ))
// with σ_2 = ((0,-i),(i,0)).
//
//   -i·σ_2 = (( 0, -1), ( 1,  0))
//    i·σ_2 = (( 0,  1), (-1,  0))
//
// So γ_2 acts on spin α ∈ {0,1,2,3} as:
//   γ_2 (v_0, v_1, v_2, v_3)^T = (-v_3, +v_2, +v_1, -v_0)^T
//
// Then γ_2·conj(v) gives (per color c):
//   out_0 = -conj(v_3)        out_2 = +conj(v_1)
//   out_1 = +conj(v_2)        out_3 = -conj(v_0)
//
// As doubles (re/im interleaved):
//   out[(0)*Nc+c]_re = -in[(3)*Nc+c]_re ;  out[(0)*Nc+c]_im = +in[(3)*Nc+c]_im
//   out[(1)*Nc+c]_re = +in[(2)*Nc+c]_re ;  out[(1)*Nc+c]_im = -in[(2)*Nc+c]_im
//   out[(2)*Nc+c]_re = +in[(1)*Nc+c]_re ;  out[(2)*Nc+c]_im = -in[(1)*Nc+c]_im
//   out[(3)*Nc+c]_re = -in[(0)*Nc+c]_re ;  out[(3)*Nc+c]_im = +in[(0)*Nc+c]_im
//
// Layout matches the per-site memcpy convention used by Grid's QudaFieldConvert
// (`fermion_to_lex_buffer` / `fermion_to_eo_buffer`): per-site 24 doubles =
// [α=0..3][c=0..2][re,im] (color-inside-spin), independent of whether the
// outer buffer ordering is lex or EO (the kernel is per-site-local so the
// outer ordering is irrelevant).

#include <Grid/GridCore.h>

NAMESPACE_BEGIN(Grid);
namespace Quda {

// In-place safe (out == in OK?): NO — the kernel reads in_α and writes out_α
// where α reorders, so out_d and in_d MUST be distinct buffers.
//
// V_sites = total number of fermion sites (lex or EO ordered) covered by the
// buffer; the kernel is per-site-local and doesn't care about the outer order.
inline void Gamma2ConjKernel(double *out_d, const double *in_d, int V_sites) {
  constexpr int Nc_ = Nc;   // 3
  constexpr int Ns_ = Ns;   // 4
  static_assert(Nc_ == 3, "Wilson spinor expects Nc=3");
  static_assert(Ns_ == 4, "Wilson spinor expects Ns=4");

  accelerator_for(s, V_sites, 1, {
    const double *in  = &in_d [s * 24];
    double       *out = &out_d[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      // α=0:  out = -conj(in[3])
      out[(0 * Nc_ + c) * 2 + 0] = -in[(3 * Nc_ + c) * 2 + 0];
      out[(0 * Nc_ + c) * 2 + 1] = +in[(3 * Nc_ + c) * 2 + 1];
      // α=1:  out = +conj(in[2])
      out[(1 * Nc_ + c) * 2 + 0] = +in[(2 * Nc_ + c) * 2 + 0];
      out[(1 * Nc_ + c) * 2 + 1] = -in[(2 * Nc_ + c) * 2 + 1];
      // α=2:  out = +conj(in[1])
      out[(2 * Nc_ + c) * 2 + 0] = +in[(1 * Nc_ + c) * 2 + 0];
      out[(2 * Nc_ + c) * 2 + 1] = -in[(1 * Nc_ + c) * 2 + 1];
      // α=3:  out = -conj(in[0])
      out[(3 * Nc_ + c) * 2 + 0] = -in[(0 * Nc_ + c) * 2 + 0];
      out[(3 * Nc_ + c) * 2 + 1] = +in[(0 * Nc_ + c) * 2 + 1];
    }
  });
}

// Host-side equivalent of Gamma2ConjKernel: same per-site math, no
// accelerator_for.  Used by DTXQCDMpcOpQUDA (M-wrap.4) where the QUDA Mat is
// in CPU_FIELD_LOCATION mode so the buffers are host-resident.  M-wrap.5
// switches the whole inner loop to device-resident and uses Gamma2ConjKernel
// directly on device pointers.
inline void Gamma2ConjKernelHost(double *out_h, const double *in_h, int V_sites) {
  constexpr int Nc_ = Nc;
  constexpr int Ns_ = Ns;
  static_assert(Nc_ == 3 && Ns_ == 4, "Wilson spinor expects Ns=4, Nc=3");
  for (int s = 0; s < V_sites; ++s) {
    const double *in  = &in_h [s * 24];
    double       *out = &out_h[s * 24];
    for (int c = 0; c < Nc_; ++c) {
      out[(0 * Nc_ + c) * 2 + 0] = -in[(3 * Nc_ + c) * 2 + 0];
      out[(0 * Nc_ + c) * 2 + 1] = +in[(3 * Nc_ + c) * 2 + 1];
      out[(1 * Nc_ + c) * 2 + 0] = +in[(2 * Nc_ + c) * 2 + 0];
      out[(1 * Nc_ + c) * 2 + 1] = -in[(2 * Nc_ + c) * 2 + 1];
      out[(2 * Nc_ + c) * 2 + 0] = +in[(1 * Nc_ + c) * 2 + 0];
      out[(2 * Nc_ + c) * 2 + 1] = -in[(1 * Nc_ + c) * 2 + 1];
      out[(3 * Nc_ + c) * 2 + 0] = -in[(0 * Nc_ + c) * 2 + 0];
      out[(3 * Nc_ + c) * 2 + 1] = +in[(0 * Nc_ + c) * 2 + 1];
    }
  }
}

}  // namespace Quda
NAMESPACE_END(Grid);
