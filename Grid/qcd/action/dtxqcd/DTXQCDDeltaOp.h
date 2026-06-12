#pragma once
// DTXQCD v2 diagonal-block aux-field insertion X^{ij}_{ab}.
//
// Acts on each diagonal block of the doubled Dirac operator (dtxqcd_v2.tex
// Eq. 297-308):
//
//   X^{ij}_{ab} = sigma^{ij}_{ab}                              (scalar)
//               + s delta^{ij} delta_{ab}                      (scalar singlet)
//               + (pi^{ij}_{ab} + p delta^{ij} delta_{ab}) gamma5  (pseudoscalar)
//
// Upper block contributes +X^{ij}_{ab}; lower (C-conjugated) block
// contributes -X^{ij}_{ab}.  Sigma and pi are 6x6 color-flavor Hermitian
// traceless matrices, with indices (a,b) for flavor and (i,j) for color.
// In the v1 (incorrect) version sigma and pi were Pauli triplets in flavor
// only and the diagonal block also carried a t-tensor piece; v2 drops both
// in favour of a unified color-flavor matrix structure.
//
// The off-diagonal d, n contributions (d gamma5 + n) live in
// DtxqcdApplyDnCross below.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

// Nf-flavor doublet fermion.  Same struct as v1; unchanged in v2 since it
// only depends on Nf, not the aux roster.
struct DTXQCDFermionNf {
  std::array<LatticeFermion, DtxqcdNf> f;

  template <std::size_t... Is>
  static std::array<LatticeFermion, DtxqcdNf> MakeArray(
      GridBase *grid, std::index_sequence<Is...>) {
    return std::array<LatticeFermion, DtxqcdNf>{
        {(static_cast<void>(Is), LatticeFermion(grid))...}};
  }

  DTXQCDFermionNf(GridBase *grid)
      : f(MakeArray(grid, std::make_index_sequence<DtxqcdNf>{})) {}

  GridBase *Grid() const { return f[0].Grid(); }

  DTXQCDFermionNf &operator=(const Zero &) {
    for (auto &ff : f) ff = Zero();
    return *this;
  }
  DTXQCDFermionNf &operator=(const DTXQCDFermionNf &rhs) {
    for (int a = 0; a < DtxqcdNf; ++a) f[a] = rhs.f[a];
    return *this;
  }
};

inline ComplexD innerProduct(const DTXQCDFermionNf &x,
                             const DTXQCDFermionNf &y) {
  ComplexD acc = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) acc += innerProduct(x.f[a], y.f[a]);
  return acc;
}
inline RealD norm2(const DTXQCDFermionNf &x) {
  RealD n = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) n += norm2(x.f[a]);
  return n;
}

// Apply the diagonal X^{ij}_{ab} contribution to a fermion (OVERWRITE).
//   out[a]_alpha(i) = sum_{b,j}  sigma^{ij}_{ab}            in[b]_alpha(j)
//                   + sum_{b,j} (pi^{ij}_{ab})  (gamma5 in[b])_alpha(j)
//                   + s in[a]_alpha(i) + p (gamma5 in[a])_alpha(i)
//
// block_sign multiplies X (+1 for the upper block, -1 for the lower block --
// the doubled construction with X^T = X gives X_upper = +X, X_lower = -X).
//
// Implementation: PeekIndex<1>(sigma, a, b) returns the inner-color matrix
// field sigma^{:,:}_{ab} as a Lattice<iScalar<iScalar<iMatrix<vComplex, Nc>>>>,
// which IS a LatticeColourMatrix.  LatticeColourMatrix * LatticeFermion is
// the standard Grid color-rotation on the fermion's color index.
inline void DtxqcdApplyX(const LatticeDtxqcdSigma &sigma,
                         const LatticeDtxqcdPi    &pi,
                         const LatticeDtxqcdS     &s,
                         const LatticeDtxqcdP     &p,
                         const DTXQCDFermionNf    &in,
                         DTXQCDFermionNf          &out,
                         double                    block_sign = +1.0) {
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  int cb = in.f[0].Checkerboard();

  // Pre-rotate gamma5 in[b] once per flavor.
  std::array<LatticeFermion, DtxqcdNf> g5_in =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  for (int b = 0; b < DtxqcdNf; ++b) g5_in[b] = g5 * in.f[b];

  // PeekIndex<2>(s, 0) collapses iScalar<iScalar<iScalar<...>>> to a
  // LatticeComplex.  Same for p.  (Singlet has only one entry, so the
  // peek index is essentially a type-shape change.)
  // Actually for iScalar<iScalar<iScalar<vComplex>>>, the field is
  // already at depth-3 scalar; we treat it as a LatticeComplex equivalent.
  // Grid's tensor algebra allows direct multiplication of LatticeComplex *
  // LatticeFermion.
  // (No explicit peek needed -- assign-cast via the depth=3 scalar.)
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = Zero();
    acc.Checkerboard() = cb;

    // Color-flavor matrix contributions.
    for (int b = 0; b < DtxqcdNf; ++b) {
      LatticeColourMatrix sig_ab = PeekIndex<1>(sigma, a, b);
      LatticeColourMatrix pi_ab  = PeekIndex<1>(pi,    a, b);
      acc = acc + sig_ab * in.f[b];
      acc = acc + pi_ab  * g5_in[b];
    }

    // Singlet contributions: s delta_{ab} delta^{ij}, p delta_{ab} delta^{ij} gamma5.
    acc = acc + s * in.f[a];
    acc = acc + p * g5_in[a];

    out.f[a] = block_sign * acc;
    out.f[a].Checkerboard() = cb;
  }
}

// Apply +X (upper block).
inline void DtxqcdApplyDeltaDiag(const LatticeDtxqcdSigma &sigma,
                                 const LatticeDtxqcdPi    &pi,
                                 const LatticeDtxqcdS     &s,
                                 const LatticeDtxqcdP     &p,
                                 const DTXQCDFermionNf    &in,
                                 DTXQCDFermionNf          &out) {
  DtxqcdApplyX(sigma, pi, s, p, in, out, +1.0);
}

// Apply +X (lower block).  v2 corrected (2026-06-12): the lower block now
// has +X (same sign as upper), matching M_lower = -C D^T C + X.  Old impl
// applied -X following the superseded M_lower = C D^T C - X form.
inline void DtxqcdApplyDeltaDiagLower(const LatticeDtxqcdSigma &sigma,
                                      const LatticeDtxqcdPi    &pi,
                                      const LatticeDtxqcdS     &s,
                                      const LatticeDtxqcdP     &p,
                                      const DTXQCDFermionNf    &in,
                                      DTXQCDFermionNf          &out) {
  DtxqcdApplyX(sigma, pi, s, p, in, out, +1.0);
}

// Apply the off-diagonal sqrt(2) * (d gamma5 + n) insertion (OVERWRITE).
// Same color-flavor matrix structure as X; entered without the singlet pieces:
//
//   out[a]_alpha(i) = sqrt(2) * sum_{b,j} (d^{ij}_{ab})  (gamma5 in[b])_alpha(j)
//                   + sqrt(2) * sum_{b,j} (n^{ij}_{ab})           in[b]_alpha(j)
//
// v2 corrected (2026-06-12): factor sqrt(2) per dtxqcd_v2.tex Eq 22-25.
// (v1 had factor 2; the earlier "v2 absorbs it" comment was based on an
// incorrect derivation, since corrected.)
inline void DtxqcdApplyDnCross(const LatticeDtxqcdD     &d,
                                const LatticeDtxqcdN     &n,
                                const DTXQCDFermionNf    &in,
                                DTXQCDFermionNf          &out) {
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  int cb = in.f[0].Checkerboard();
  const RealD sqrt2 = std::sqrt(2.0);

  std::array<LatticeFermion, DtxqcdNf> g5_in =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  for (int b = 0; b < DtxqcdNf; ++b) g5_in[b] = g5 * in.f[b];

  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = Zero();
    acc.Checkerboard() = cb;
    for (int b = 0; b < DtxqcdNf; ++b) {
      LatticeColourMatrix d_ab = PeekIndex<1>(d, a, b);
      LatticeColourMatrix n_ab = PeekIndex<1>(n, a, b);
      acc = acc + d_ab * g5_in[b];
      acc = acc + n_ab * in.f[b];
    }
    out.f[a] = sqrt2 * acc;
    out.f[a].Checkerboard() = cb;
  }
}

NAMESPACE_END(Grid);
