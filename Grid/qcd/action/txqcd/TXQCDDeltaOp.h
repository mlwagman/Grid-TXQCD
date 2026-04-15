#pragma once
// Local auxiliary-field insertion Delta(x) for the TXQCD Wilson Dirac operator.
//
// Notes Eq. (8):
//   Delta(x) = sigma_{ab}(x) (x) I_color (x) I_spin
//            + pi_{ab}(x)    (x) I_color (x) gamma5
//            + (1/sqrt 2) s^{ij}(x) (x) I_spin (x) I_flavor
//            + (1/sqrt 2) p^{ij}(x) (x) gamma5 (x) I_flavor
//            + (1/2) t^{ij}_{mu,nu}(x) sigma_{mu,nu} (x) I_flavor
//
// For Nf=2 flavors we represent the TXQCD fermion as a fixed-size array of
// Nf standard LatticeFermion (spin-color vector) fields. This side-steps the
// need for a fresh Grid fermion type while keeping all spin/color contractions
// available through Grid's stock arithmetic.
//
// This header implements only the sigma/pi (flavor+spin) sector for Phase 4c
// bring-up. The color sector (s, p) and the tensor sector (t_{mu,nu}) will
// follow once the flavor-sector tests pass.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

// Container for the Nf flavors of a TXQCD fermion. Arithmetic is defined
// componentwise so the caller can write Delta = Delta + A*x etc. cleanly.
struct TXQCDFermionNf {
  std::array<LatticeFermion, TxqcdNf> f;

  TXQCDFermionNf(GridBase *grid)
      : f{{LatticeFermion(grid), LatticeFermion(grid)}} {
    static_assert(TxqcdNf == 2,
                  "TXQCDFermionNf currently hardcoded for Nf=2");
  }

  GridBase *Grid() const { return f[0].Grid(); }

  TXQCDFermionNf &operator=(const Zero &) {
    for (auto &ff : f) ff = Zero();
    return *this;
  }
};

// Apply Delta_{sigma,pi}: result[a] = sum_b (sigma_{a,b} in[b] + pi_{a,b} g5 in[b])
// where sigma and pi are Nf x Nf Hermitian flavor-matrix site lattices.
inline void ApplyDeltaSigmaPi(const LatticeSigmaField &sigma,
                              const LatticePiField &pi,
                              const TXQCDFermionNf &in, TXQCDFermionNf &out) {
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  std::array<LatticeFermion, TxqcdNf> g5_in{{LatticeFermion(grid),
                                             LatticeFermion(grid)}};
  for (int b = 0; b < TxqcdNf; ++b) g5_in[b] = g5 * in.f[b];

  for (int a = 0; a < TxqcdNf; ++a) {
    out.f[a] = Zero();
    for (int b = 0; b < TxqcdNf; ++b) {
      // Storage is iScalar<iScalar<iMatrix<vComplex,Nf>>>; the matrix is at
      // tensor level 2 (outer iScalar=0, inner iScalar=1).
      auto s_ab = PeekIndex<2>(sigma, a, b);
      auto p_ab = PeekIndex<2>(pi, a, b);
      out.f[a] = out.f[a] + s_ab * in.f[b] + p_ab * g5_in[b];
    }
  }
}

NAMESPACE_END(Grid);
