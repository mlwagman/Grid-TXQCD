#pragma once
// DTXQCD diagonal-block aux-field insertion Delta_diag(x).
//
// Acts on each diagonal block of the doubled Pfaffian operator
// (dtxqcd.tex Eq. 297, 299):
//
//   Delta_diag(x)_{a,b} = (1/sqrt 2) sigma^A(x) tau^A_{a,b} (x) I_spin (x) I_color
//                       + (1/sqrt 2) pi^A(x)    tau^A_{a,b} (x) gamma5 (x) I_color
//                       + (1/2) sum_{mu<nu} t^A_{mu,nu}(x) tau^A_{a,b} (x) sigma_{mu,nu} (x) I_color
//
// where A = 1..3 indexes the SU(2) flavor Pauli generators tau^A, and the
// fields (sigma^A, pi^A, t^A_{mu,nu}) live in DTXQCDAuxFieldTypes.h.
//
// The off-diagonal d, n contributions are NOT in this operator — they enter
// as off-diagonal blocks of the doubled Dirac operator and are applied by
// DTXQCDPfaffianOp.h.
//
// Hermiticity of Delta_diag in (flavor x spin x color x site):
//   - sigma^A, pi^A, t^A are real (imag = 0 in storage).
//   - tau^A Hermitian => sigma_{a,b} = sigma^A tau^A_{a,b} is flavor-Hermitian.
//   - gamma5 Hermitian, Grid's Sigma_{mu,nu} Hermitian.
//   => Delta_diag is Hermitian.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>  // reuse SigmaMuNuAlgebra

NAMESPACE_BEGIN(Grid);

// Nf-flavor doublet fermion. Structurally a sibling of TXQCDFermionNf; kept
// as a distinct type so DTXQCD code can evolve independently.
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

// Pauli matrices tau^A, A = 0..2 corresponding to tau^1, tau^2, tau^3 of the
// paper.  Defined for Nf=2; higher Nf would need a different generator set
// (DTXQCD currently only supports Nf=2).
struct DtxqcdPauli {
  static constexpr int N = DtxqcdNTriplet;  // 3 for Nf=2

  // tau[A][a][b] = (tau^A)_{a,b}, A in {0,1,2} -> {tau1, tau2, tau3}.
  static ComplexD tau(int A, int a, int b) {
    // Real and imag tables for the three Pauli matrices.
    constexpr double re[3][2][2] = {
        {{0, 1}, {1, 0}},    // tau^1
        {{0, 0}, {0, 0}},    // tau^2 (real part)
        {{1, 0}, {0, -1}}    // tau^3
    };
    constexpr double im[3][2][2] = {
        {{0, 0}, {0, 0}},
        {{0, -1}, {1, 0}},   // tau^2 (imag part)
        {{0, 0}, {0, 0}}
    };
    return ComplexD(re[A][a][b], im[A][a][b]);
  }
};

// Apply the (sigma^A, pi^A) piece of Delta_diag:
//   out[a] = sum_{A,b} (1/sqrt 2) sigma^A tau^A_{a,b} in[b]
//          + sum_{A,b} (1/sqrt 2) pi^A    tau^A_{a,b} gamma5 in[b]
//
// LatticeComplex path: extract each sigma^A, pi^A as a LatticeComplex slice
// (depth-2 PeekIndex on the iVector), multiply by the Pauli scalar tau^A_{a,b},
// accumulate.  Correctness first; SIMD-fused accelerator_for is a later
// optimization once the test framework is in place.
inline void DtxqcdApplyDeltaSigmaPi(const LatticeDtxqcdSigma &sigma,
                                    const LatticeDtxqcdPi    &pi,
                                    const DTXQCDFermionNf &in,
                                    DTXQCDFermionNf &out) {
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);

  int cb = in.f[0].Checkerboard();

  // Pre-rotate gamma5 in[b] once per flavor.
  std::array<LatticeFermion, DtxqcdNf> g5_in =
      DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{});
  for (int b = 0; b < DtxqcdNf; ++b) g5_in[b] = g5 * in.f[b];

  // Extract scalar (depth-2) Pauli components as LatticeComplex slices once.
  std::array<LatticeComplex, DtxqcdNTriplet> sigA{LatticeComplex(grid),
                                                  LatticeComplex(grid),
                                                  LatticeComplex(grid)};
  std::array<LatticeComplex, DtxqcdNTriplet> piA{LatticeComplex(grid),
                                                 LatticeComplex(grid),
                                                 LatticeComplex(grid)};
  for (int A = 0; A < DtxqcdNTriplet; ++A) {
    sigA[A] = PeekIndex<2>(sigma, A);
    piA[A]  = PeekIndex<2>(pi,    A);
  }

  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = Zero();
    acc.Checkerboard() = cb;
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int A = 0; A < DtxqcdNTriplet; ++A) {
        ComplexD tau_ab = DtxqcdPauli::tau(A, a, b);
        if (tau_ab == ComplexD(0.0, 0.0)) continue;
        ComplexD coef = inv_sqrt2 * tau_ab;
        acc = acc + coef * (sigA[A] * in.f[b])
                  + coef * (piA[A]  * g5_in[b]);
      }
    }
    out.f[a] = acc;
    out.f[a].Checkerboard() = cb;
  }
}

// Apply the t^A_{mu,nu} tensor piece of Delta_diag (OVERWRITE semantics):
//   out[a] = sum_{A,b,mu<nu} t^A_{mu,nu} tau^A_{a,b} sigma_{mu,nu} in[b]
//
// The (1/2) prefactor on the formula sum_{mu,nu} t^A sigma_{mu,nu} becomes 1
// when restricted to mu<nu via the joint antisymmetry of t and sigma.
//
// Grid's Gamma::Algebra::Sigma{XY,XZ,XT,YZ,YT,ZT} is (i/2)[gamma_mu, gamma_nu]
// and Hermitian.  Pre-rotate sigma_{mu,nu} in[b] for the 6 (mu<nu) pairs;
// extract each (mu,nu, A) scalar as a LatticeComplex slice; accumulate.
inline void DtxqcdApplyDeltaTensor(const LatticeDtxqcdT &t,
                                   const DTXQCDFermionNf &in,
                                   DTXQCDFermionNf &out) {
  GridBase *grid = in.Grid();
  int cb = in.f[0].Checkerboard();

  // Pre-rotate sigma_{mu,nu} in[b] for the 6 (mu<nu) pairs (flat-indexed).
  constexpr int Npairs = 6;
  std::vector<std::array<LatticeFermion, DtxqcdNf>> smn_in;
  smn_in.reserve(Npairs);
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu) {
      Gamma smn(SigmaMuNuAlgebra(mu, nu));
      smn_in.push_back(
          DTXQCDFermionNf::MakeArray(grid, std::make_index_sequence<DtxqcdNf>{}));
      for (int b = 0; b < DtxqcdNf; ++b) smn_in.back()[b] = smn * in.f[b];
    }
  }

  // Build the per-Pauli LatticeComplex slices for each (mu, nu) pair.
  //   t is iScalar<iMatrix<iVector<vComplex,3>, Nd>>.
  //     PeekIndex<1>(t, mu, nu)  -> Lattice<iScalar<iScalar<iVector<...,3>>>>
  //       (the inner iScalar is added by Grid's tensor-peek to preserve depth)
  //     PeekIndex<2>(t_munu, A)  -> LatticeComplex
  std::vector<std::array<LatticeComplex, DtxqcdNTriplet>> tA_munu;
  tA_munu.reserve(Npairs);
  int pair_idx = 0;
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu) {
      auto t_munu = PeekIndex<1>(t, mu, nu);
      std::array<LatticeComplex, DtxqcdNTriplet> slice{LatticeComplex(grid),
                                                       LatticeComplex(grid),
                                                       LatticeComplex(grid)};
      for (int A = 0; A < DtxqcdNTriplet; ++A) {
        slice[A] = PeekIndex<2>(t_munu, A);
      }
      tA_munu.push_back(std::move(slice));
      ++pair_idx;
    }
  }

  // Grid's Gamma::Algebra::Sigma{XY,..} is (1/2)[gamma_mu, gamma_nu] — without
  // the conventional i factor — and is anti-Hermitian.  Multiplying by i
  // gives the Hermitian sigma_{mu,nu} = (i/2)[gamma_mu, gamma_nu] the formula
  // calls for.  (TXQCDDeltaOp does the same; see Test_dtxqcd_delta_herm.)
  const ComplexD ci(0.0, 1.0);
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion acc(grid);
    acc.Checkerboard() = cb;
    acc = Zero();
    acc.Checkerboard() = cb;
    int k = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        for (int b = 0; b < DtxqcdNf; ++b) {
          for (int A = 0; A < DtxqcdNTriplet; ++A) {
            ComplexD tau_ab = DtxqcdPauli::tau(A, a, b);
            if (tau_ab == ComplexD(0.0, 0.0)) continue;
            acc = acc + (ci * tau_ab) * (tA_munu[k][A] * smn_in[k][b]);
          }
        }
        ++k;
      }
    }
    out.f[a] = acc;
    out.f[a].Checkerboard() = cb;
  }
}

// Apply the full Delta_diag = (sigma^A, pi^A) + tensor pieces.  out is
// overwritten with the sum.
inline void DtxqcdApplyDeltaDiag(const LatticeDtxqcdSigma &sigma,
                                 const LatticeDtxqcdPi    &pi,
                                 const LatticeDtxqcdT     &t,
                                 const DTXQCDFermionNf &in,
                                 DTXQCDFermionNf &out) {
  DTXQCDFermionNf tmp(in.Grid());
  DtxqcdApplyDeltaSigmaPi(sigma, pi, in, out);
  DtxqcdApplyDeltaTensor(t, in, tmp);
  for (int a = 0; a < DtxqcdNf; ++a) out.f[a] = out.f[a] + tmp.f[a];
}

NAMESPACE_END(Grid);
