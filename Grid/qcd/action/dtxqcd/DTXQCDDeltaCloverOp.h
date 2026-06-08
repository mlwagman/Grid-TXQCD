#pragma once
// DTXQCD clover-term application on the doubled M_ee's upper / lower
// diagonal blocks.
//
// Upper block (standard QCD Wilson-Clover convention):
//   D_clover_upper v[a] = -(csw/2) sum_{mu<nu} F_{mu,nu} (sigma_{mu,nu}_Grid v[a])
//
// Lower block — dtxqcd.tex Eq. (18) gives "C D_qcd C^T" but the correct
// Cstar / doubled-action construction is M_22 = C^T D_qcd^T C (with C^T
// = -C in the paper's convention).  Setting up the doubled form
// S_doubled = (1/2) Ψ̄ M Ψ with Ψ = (q, q^C)^T and equating to S = q̄ D q,
// the lower-right block of M reproduces the original action via
// q̄^C M_22 q^C ↦ q̄ D q.  Mass-term sanity: M_22_mass = m C^T C = m I
// (no sign flip), and the doubled action gives m q̄ q correctly.
//
// For the clover term then:
//   D_clover_lower = C^T D_clover^T C
//                  = -(csw/2) C^T (F sigma)^T C
//                  = -(csw/2) C^T sigma^T F^T C       (transpose reverses order)
//                  = -(csw/2) (C^T sigma^T C) F^T     (F commutes with C)
//                  = -(csw/2) (-sigma) F^T            (using C^T sigma^T C
//                                                      = -C sigma^T C = -sigma)
//                  = +(csw/2) F^T sigma_{mu,nu}
//
// where C sigma^T C = sigma is obtained from sigma^T = C sigma C
// (universal identity from C gamma_mu C = gamma_mu^T with C^2 = -I) so
//   C sigma^T C = C(C sigma C)C = C^2 sigma C^2 = (-I) sigma (-I) = sigma.
//
// Net effect on the implementation: SIGN FLIP relative to upper block,
// SAME sigma_{mu,nu}_Grid, but F is color-transposed (F^T).  Both blocks
// are independently Hermitian when F is anti-Hermitian (F^T is also anti-
// Hermitian when F is — transpose of anti-Hermitian is anti-Hermitian).

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

NAMESPACE_BEGIN(Grid);

namespace dtxqcd_detail {

// Lex index for the 6 antisymmetric (mu<nu) pairs in the order
//   (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).  Matches SigmaMuNuAlgebra.
inline int CloverPairIdx(int mu, int nu) {
  return (mu == 0 && nu == 1) ? 0
       : (mu == 0 && nu == 2) ? 1
       : (mu == 0 && nu == 3) ? 2
       : (mu == 1 && nu == 2) ? 3
       : (mu == 1 && nu == 3) ? 4 : 5;
}

}  // namespace dtxqcd_detail

// Generic Σ_{μ<ν} F σ apply with an explicit signed prefactor.  Used by both
// upper and lower clover wrappers; caller passes F or F^T as needed.
//   out[a] = prefactor * sum_{mu<nu} F_{mu,nu} (sigma_{mu,nu}_Grid v[a])
inline void DtxqcdApplyCloverGeneric(
    RealD prefactor,
    const std::vector<LatticeColourMatrix> &FS,
    const DTXQCDFermionNf &in, DTXQCDFermionNf &out) {
  int cb = in.f[0].Checkerboard();
  GridBase *grid = in.Grid();
  if (prefactor == 0.0) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.f[a] = Zero();
      out.f[a].Checkerboard() = cb;
    }
    return;
  }
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = Zero();
    acc.Checkerboard() = cb;
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        int k = dtxqcd_detail::CloverPairIdx(mu, nu);
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        acc = acc + FS[k] * (smn * in.f[a]);
      }
    }
    out.f[a] = prefactor * acc;
    out.f[a].Checkerboard() = cb;
  }
}

// Build F^T_{mu,nu} (color-transpose, per pair).  transpose() on a
// LatticeColourMatrix swaps the two color indices without conjugating
// (whereas adj() = transpose + conj).
inline std::vector<LatticeColourMatrix>
DtxqcdTransposeFS(const std::vector<LatticeColourMatrix> &FS) {
  std::vector<LatticeColourMatrix> FT;
  FT.reserve(FS.size());
  for (const auto &F : FS) {
    LatticeColourMatrix FT_k(F.Grid());
    FT_k = transpose(F);
    FT.push_back(std::move(FT_k));
  }
  return FT;
}

// Upper-block clover:  out[a] = -(csw/2) sum_{mu<nu} F_{mu,nu} (sigma_{mu,nu} v[a])
inline void DtxqcdApplyCloverUpper(
    RealD csw,
    const std::vector<LatticeColourMatrix> &FS,
    const DTXQCDFermionNf &in, DTXQCDFermionNf &out) {
  DtxqcdApplyCloverGeneric(-0.5 * csw, FS, in, out);
}

// Lower-block clover (corrected Cstar construction M_22 = C^T D^T C):
//   C^T D_clover^T C = -(csw/2) (C^T sigma^T C) F^T = -(csw/2)(-sigma) F^T
//                    = +(csw/2) F^T sigma_{mu,nu}
// Sign FLIPPED relative to upper, F replaced by its color transpose.
// Transposes FS on the fly; for repeated applies, pre-transpose with
// DtxqcdTransposeFS and call DtxqcdApplyCloverGeneric directly.
inline void DtxqcdApplyCloverLower(
    RealD csw,
    const std::vector<LatticeColourMatrix> &FS,
    const DTXQCDFermionNf &in, DTXQCDFermionNf &out) {
  std::vector<LatticeColourMatrix> FT = DtxqcdTransposeFS(FS);
  DtxqcdApplyCloverGeneric(+0.5 * csw, FT, in, out);
}

NAMESPACE_END(Grid);
