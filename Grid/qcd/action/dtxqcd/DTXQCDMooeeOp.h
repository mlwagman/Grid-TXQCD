#pragma once
// Fermion-level application of the v2 DTXQCD M_ee (site-local) operator on
// a doubled fermion (upper, lower) pair of DTXQCDFermionNf.
//
// M_ee | upper   = | ((mass + 4) * I + X) upper + (d gamma5 + n) lower |
//      | lower     | (d gamma5 + n) upper + ((mass + 4) * I - X) lower |
//
// X^{ij}_{ab} = sigma + s delta delta + (pi + p delta delta) gamma5  -- see
// DTXQCDDeltaOp.h.  v2 drops the v1 tensor term (Fierz-eliminated) and the
// v1 factor of 2 on d, n.

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>

NAMESPACE_BEGIN(Grid);

// Full doubled M_ee application:
//   out_upper = (mass + 4) * in_upper + X * in_upper + (d g5 + n) in_lower
//                + (csw && FS ? -(csw/2) F sigma in_upper : 0)
//   out_lower = (mass + 4) * in_lower - X * in_lower + (d g5 + n) in_upper
//                + (csw && FS ? +(csw/2) F^T sigma in_lower : 0)
//
// Clover is optional: pass csw = 0 (default) or FS = nullptr to skip.
inline void DtxqcdApplyMooeeDoubled(
    double mass,
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi    &pi,
    const LatticeDtxqcdD     &d,
    const LatticeDtxqcdN     &n,
    const LatticeDtxqcdS     &s,
    const LatticeDtxqcdP     &p,
    const DTXQCDFermionNf    &in_upper,
    const DTXQCDFermionNf    &in_lower,
    DTXQCDFermionNf          &out_upper,
    DTXQCDFermionNf          &out_lower,
    double                    csw = 0.0,
    const std::vector<LatticeColourMatrix> *FS = nullptr) {
  GridBase *grid = in_upper.Grid();
  int cb = in_upper.f[0].Checkerboard();

  // Diagonal: X piece for upper (+X) and lower (-X).
  DTXQCDFermionNf diag_upper(grid), diag_lower(grid);
  DtxqcdApplyDeltaDiag     (sigma, pi, s, p, in_upper, diag_upper);
  DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, in_lower, diag_lower);

  // Off-diagonal: (d g5 + n) cross-term.  No factor of 2 in v2.
  DTXQCDFermionNf off_upper(grid), off_lower(grid);
  DtxqcdApplyDnCross(d, n, in_lower, off_upper, /*apply_conj=*/false);  // M_UR
  DtxqcdApplyDnCross(d, n, in_upper, off_lower, /*apply_conj=*/true);   // M_LL = conj(M_UR) under DN_COMPLEX_SYMMETRIC

  // Assemble out = (mass + 4) * in + diag + off.
  const double mooee_diag = mass + 4.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_upper.f[a] = ComplexD(mooee_diag, 0.0) * in_upper.f[a]
                   + diag_upper.f[a]
                   + off_upper.f[a];
    out_upper.f[a].Checkerboard() = cb;
    out_lower.f[a] = ComplexD(mooee_diag, 0.0) * in_lower.f[a]
                   + diag_lower.f[a]
                   + off_lower.f[a];
    out_lower.f[a].Checkerboard() = cb;
  }

  // Optional clover contribution: upper -(csw/2) F sigma, lower +(csw/2) F^T sigma.
  if (csw != 0.0 && FS != nullptr) {
    DTXQCDFermionNf clov_upper(grid), clov_lower(grid);
    DtxqcdApplyCloverUpper(csw, *FS, in_upper, clov_upper);
    DtxqcdApplyCloverLower(csw, *FS, in_lower, clov_lower);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_upper.f[a] = out_upper.f[a] + clov_upper.f[a];
      out_lower.f[a] = out_lower.f[a] + clov_lower.f[a];
    }
  }
}

NAMESPACE_END(Grid);
