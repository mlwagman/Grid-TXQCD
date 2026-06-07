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
// v1 caveat: M_lower's QCD diagonal is m * I_color (same as upper); will
// diverge once Delta_clover is added in C clover C^T form.  Until then the
// site-local QCD pieces are identical and we apply Delta_diag to both
// upper and lower with the same code path.

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

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
//   out_lower = mass * in_lower + Delta_diag(in_lower) + (2 d g5 + 2 n) in_upper
inline void DtxqcdApplyMooeeDoubled(double mass,
                                    const LatticeDtxqcdSigma &sigma,
                                    const LatticeDtxqcdPi &pi,
                                    const LatticeDtxqcdT &t,
                                    const LatticeDtxqcdD &d,
                                    const LatticeDtxqcdN &n,
                                    const DTXQCDFermionNf &in_upper,
                                    const DTXQCDFermionNf &in_lower,
                                    DTXQCDFermionNf &out_upper,
                                    DTXQCDFermionNf &out_lower) {
  GridBase *grid = in_upper.Grid();
  int cb = in_upper.f[0].Checkerboard();

  // Diagonal: Delta_diag piece for upper and lower.
  DTXQCDFermionNf diag_upper(grid), diag_lower(grid);
  DtxqcdApplyDeltaDiag(sigma, pi, t, in_upper, diag_upper);
  DtxqcdApplyDeltaDiag(sigma, pi, t, in_lower, diag_lower);

  // Off-diagonal: (2 d g5 + 2 n) cross-term.
  DTXQCDFermionNf off_upper(grid), off_lower(grid);
  DtxqcdApplyDnCross(d, n, in_lower, off_upper);  // off-upper from in_lower
  DtxqcdApplyDnCross(d, n, in_upper, off_lower);  // off-lower from in_upper

  // Assemble out = mass * in + diag + off.
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_upper.f[a] = ComplexD(mass, 0.0) * in_upper.f[a]
                   + diag_upper.f[a]
                   + off_upper.f[a];
    out_upper.f[a].Checkerboard() = cb;
    out_lower.f[a] = ComplexD(mass, 0.0) * in_lower.f[a]
                   + diag_lower.f[a]
                   + off_lower.f[a];
    out_lower.f[a].Checkerboard() = cb;
  }
}

NAMESPACE_END(Grid);
