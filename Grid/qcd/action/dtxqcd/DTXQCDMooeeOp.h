#pragma once
// Fermion-level application of the DTXQCD M_ee (site-local) operator on a
// doubled fermion (upper, lower) pair of DTXQCDFermionNf.
//
// M_ee | upper   = | (mass * I + Delta_diag) upper + (2 d gamma5 + 2 n) lower |
//      | lower     | (2 d gamma5 + 2 n) upper + (mass * I + Delta_diag) lower |
//
// Delta_diag(x) = (1/sqrt 2) sigma^A tau^A
//               + (1/sqrt 2) pi^A    tau^A gamma5
//               + sum_{mu<nu} i t^A_{mu,nu} tau^A Grid_Sigma_{mu,nu}
// applied per upper/lower component (DtxqcdApplyDeltaDiag from DTXQCDDeltaOp.h).
//
// Off-diagonal d, n cross-term (DtxqcdApplyDnCross): identity in flavor;
// color matrix d, n; gamma5 in spin (for d) or identity in spin (for n).
//
// Lower-block Delta_diag follows the Cstar M_22 = C^T X^T C construction:
// sigma and pi pieces unchanged (C^T (.)^T C is identity on Hermitian
// spinless and gamma_5 spin structures), tensor piece sign-flipped
// (C^T sigma^T C = -sigma).  Mass and d, n cross-coupling are identical
// upper/lower.  Clover contribution (where upper -> -(csw/2) F sigma and
// lower -> +(csw/2) F^T sigma) is added when DTXQCDDeltaCloverOp is wired
// into the Mooee path with a per-site F_{mu,nu}.

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>

NAMESPACE_BEGIN(Grid);

// Off-diagonal cross-term: out[a] = 2 * d * (gamma5 * in[a]) + 2 * n * in[a]
// for each flavor a (identity in flavor).  d, n are LatticeDtxqcdD / N whose
// site-tensor type is iScalar<iScalar<iMatrix<vComplex, Nc>>>, layout-identical
// to LatticeColourMatrix — reinterpret_cast lets us use LatticeColourMatrix *
// LatticeFermion arithmetic without copying.
inline void DtxqcdApplyDnCross(const LatticeDtxqcdD &d,
                                const LatticeDtxqcdN &n,
                                const DTXQCDFermionNf &in,
                                DTXQCDFermionNf &out) {
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  int cb = in.f[0].Checkerboard();

  const LatticeColourMatrix &d_cm =
      reinterpret_cast<const LatticeColourMatrix &>(d);
  const LatticeColourMatrix &n_cm =
      reinterpret_cast<const LatticeColourMatrix &>(n);

  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion g5_in(grid);
    g5_in.Checkerboard() = cb;
    g5_in = g5 * in.f[a];
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = ComplexD(2.0, 0.0) * (d_cm * g5_in)
        + ComplexD(2.0, 0.0) * (n_cm * in.f[a]);
    out.f[a] = acc;
    out.f[a].Checkerboard() = cb;
  }
}

// Full doubled M_ee application:
//   out_upper = mass * in_upper + Delta_diag(in_upper) + (2 d g5 + 2 n) in_lower
//                + (csw && FS ? -(csw/2) F sigma in_upper : 0)
//   out_lower = mass * in_lower + Delta_diag_lower(in_lower) + (2 d g5 + 2 n) in_upper
//                + (csw && FS ? +(csw/2) F^T sigma in_lower : 0)
//
// Clover is optional: pass csw = 0 (default) or FS = nullptr to skip the
// clover contribution.  When csw != 0 && FS != nullptr, FS is a 6-entry
// vector of LatticeColourMatrix holding the (mu<nu) anti-Hermitian field
// strength in Grid/TXQCD convention.
inline void DtxqcdApplyMooeeDoubled(
    double mass,
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi &pi,
    const LatticeDtxqcdT &t,
    const LatticeDtxqcdD &d,
    const LatticeDtxqcdN &n,
    const DTXQCDFermionNf &in_upper,
    const DTXQCDFermionNf &in_lower,
    DTXQCDFermionNf &out_upper,
    DTXQCDFermionNf &out_lower,
    double csw = 0.0,
    const std::vector<LatticeColourMatrix> *FS = nullptr) {
  GridBase *grid = in_upper.Grid();
  int cb = in_upper.f[0].Checkerboard();

  // Diagonal: Delta_diag piece for upper and lower.  Lower-block uses
  // DtxqcdApplyDeltaDiagLower (Cstar M_22 = C^T X^T C: sigma and pi pieces
  // unchanged, tensor piece sign-flipped).
  DTXQCDFermionNf diag_upper(grid), diag_lower(grid);
  DtxqcdApplyDeltaDiag(sigma, pi, t, in_upper, diag_upper);
  DtxqcdApplyDeltaDiagLower(sigma, pi, t, in_lower, diag_lower);

  // Off-diagonal: (2 d g5 + 2 n) cross-term.
  DTXQCDFermionNf off_upper(grid), off_lower(grid);
  DtxqcdApplyDnCross(d, n, in_lower, off_upper);  // off-upper from in_lower
  DtxqcdApplyDnCross(d, n, in_upper, off_lower);  // off-lower from in_upper

  // Assemble out = (mass + 4) * in + diag + off.  The "+ 4" is the Wilson
  // hopping-normalization constant: Grid's WilsonFermion convention has
  // M_W = (4 + mass)*I - hopping, so the EO Mooee block diagonal is
  // (4 + mass)*I, not just mass*I.  Matching TXQCD's diag_mass_[a] =
  // 4.0 + mass_[a].  Pre-fix, the missing +4 left Mooee = mass*I with
  // mass=0.3, making Mee^{-1} ~ 1/0.3 = 3.3, which amplified Meo by ~10x
  // in Mpc and pushed lambda_max(Mpc^dag Mpc) to ~2800 on cold gauge
  // (vs. ~75 with the fix).
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
