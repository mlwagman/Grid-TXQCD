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
// All five pieces of Delta are Hermitian individually, so the combined
// Delta is Hermitian in the (flavor x spin x color x site) inner product.
// The Wilson hopping is the only gamma5-non-Hermitian bit, and satisfies
// gamma5 D_W gamma5 = D_W^dag; since Delta is Hermitian and commutes with
// gamma5 in structure only through pi and p (both of which carry gamma5
// explicitly), gamma5 M gamma5 = M^dag holds for M = D_W + Delta + mass.

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

  TXQCDFermionNf &operator=(const TXQCDFermionNf &rhs) {
    for (int a = 0; a < TxqcdNf; ++a) f[a] = rhs.f[a];
    return *this;
  }
};

// Componentwise inner product (sum over flavors, full spin/color/site).
inline ComplexD innerProduct(const TXQCDFermionNf &x, const TXQCDFermionNf &y) {
  ComplexD acc = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) acc += innerProduct(x.f[a], y.f[a]);
  return acc;
}

inline RealD norm2(const TXQCDFermionNf &x) {
  RealD n = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) n += norm2(x.f[a]);
  return n;
}

// axpy: y = a*x + y
inline void axpy(TXQCDFermionNf &y, const ComplexD &a,
                 const TXQCDFermionNf &x) {
  for (int aa = 0; aa < TxqcdNf; ++aa) y.f[aa] = a * x.f[aa] + y.f[aa];
}

// Apply Delta_{sigma,pi}: result[a] = sum_b (sigma_{a,b} in[b] + pi_{a,b} g5 in[b])
// where sigma and pi are Nf x Nf Hermitian flavor-matrix site lattices.
//
// Hardcoded Nf=2 unroll.  Runs as accelerator_for on GPU builds (and as
// SIMD-vectorized OpenMP on CPU builds) by reading lattice views with the
// canonical view(s) form and writing via coalescedWrite — eliminates the
// CPU-only thread_for that previously dominated TXQCD MultiShift CG cost.
inline void ApplyDeltaSigmaPi(const LatticeSigmaField &sigma,
                              const LatticePiField &pi,
                              const TXQCDFermionNf &in, TXQCDFermionNf &out) {
  static_assert(TxqcdNf == 2, "ApplyDeltaSigmaPi unroll assumes Nf=2");
  GridBase *grid = in.Grid();
  Gamma g5(Gamma::Algebra::Gamma5);
  std::array<LatticeFermion, TxqcdNf> g5_in{{LatticeFermion(grid),
                                             LatticeFermion(grid)}};
  for (int b = 0; b < TxqcdNf; ++b) g5_in[b] = g5 * in.f[b];
  out.f[0].Checkerboard() = in.f[0].Checkerboard();
  out.f[1].Checkerboard() = in.f[0].Checkerboard();

  autoView(sigmav, sigma, AcceleratorRead);
  autoView(piv,    pi,    AcceleratorRead);
  autoView(in0v,   in.f[0], AcceleratorRead);
  autoView(in1v,   in.f[1], AcceleratorRead);
  autoView(g0v,    g5_in[0], AcceleratorRead);
  autoView(g1v,    g5_in[1], AcceleratorRead);
  autoView(out0v,  out.f[0], AcceleratorWrite);
  autoView(out1v,  out.f[1], AcceleratorWrite);

  const int Nsimd = LatticeFermion::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto sigma_lane = sigmav(ss);
    auto pi_lane    = piv(ss);
    auto in0_lane   = in0v(ss);
    auto in1_lane   = in1v(ss);
    auto g0_lane    = g0v(ss);
    auto g1_lane    = g1v(ss);
    // Same per-lane type as input fermion view; only declare it, don't read.
    typedef decltype(in0_lane) FermSitePerLane;
    FermSitePerLane out0_acc, out1_acc;
    for (int a = 0; a < TxqcdNf; ++a) {
      auto sa0 = sigma_lane()()(a, 0);
      auto sa1 = sigma_lane()()(a, 1);
      auto pa0 = pi_lane()()(a, 0);
      auto pa1 = pi_lane()()(a, 1);
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          auto val = sa0 * in0_lane()(alpha)(i) + sa1 * in1_lane()(alpha)(i)
                   + pa0 *  g0_lane()(alpha)(i) + pa1 *  g1_lane()(alpha)(i);
          if (a == 0) out0_acc()(alpha)(i) = val;
          else        out1_acc()(alpha)(i) = val;
        }
      }
    }
    coalescedWrite(out0v[ss], out0_acc);
    coalescedWrite(out1v[ss], out1_acc);
  });
}

// Map the 6 antisymmetric (mu<nu) pairs to Grid's sigma_{mu,nu} generators.
// Index convention: mu/nu are (0,1,2,3) = (X,Y,Z,T) and Grid's Sigma<MN>
// is defined as sigma_{M,N} = (i/2)[gamma_M, gamma_N].
inline Gamma::Algebra SigmaMuNuAlgebra(int mu, int nu) {
  // Only called for mu < nu.
  if (mu == 0 && nu == 1) return Gamma::Algebra::SigmaXY;
  if (mu == 0 && nu == 2) return Gamma::Algebra::SigmaXZ;
  if (mu == 0 && nu == 3) return Gamma::Algebra::SigmaXT;
  if (mu == 1 && nu == 2) return Gamma::Algebra::SigmaYZ;
  if (mu == 1 && nu == 3) return Gamma::Algebra::SigmaYT;
  if (mu == 2 && nu == 3) return Gamma::Algebra::SigmaZT;
  GRID_ASSERT(0 && "SigmaMuNuAlgebra: invalid (mu,nu)");
  return Gamma::Algebra::Identity;
}

// Color sector:
//   result[a] += (1/sqrt 2) s * in[a]
//              + (1/sqrt 2) p * (gamma5 in[a])
//              + sum_{mu<nu} t_{mu,nu} * (sigma_{mu,nu} in[a])
// where s, p are LatticeColourMatrix (the site-level type of LatticeSFieldC /
// LatticePFieldC is identical to iColourMatrix up to typedef) and t_{mu,nu}
// is the (mu,nu) block of the antisym-tensor field, also a LatticeColourMatrix.
//
// All three pieces are Hermitian: s, p are Hermitian color matrices; p gamma5
// is Hermitian because [p_color, gamma5_spin] = 0; and the tensor term is
// Hermitian because sigma_{mu,nu}^dag = sigma_{mu,nu} and t_{mu,nu}^dag =
// t_{mu,nu}. The result combines as Delta = Delta^dag, so we test Hermiticity
// in the composite (flavor x spin x color x site) inner product.
inline void ApplyDeltaColor(const LatticeSFieldC &s,
                            const LatticePFieldC &p,
                            const LatticeTField &t,
                            const TXQCDFermionNf &in, TXQCDFermionNf &out) {
  const RealD inv_sqrt2 = 1.0 / std::sqrt(2.0);
  Gamma g5(Gamma::Algebra::Gamma5);

  for (int a = 0; a < TxqcdNf; ++a) {
    // s term: (1/sqrt 2) * s * v[a].
    out.f[a] = inv_sqrt2 * (s * in.f[a]);
    // p gamma5 term.
    LatticeFermion g5v(in.Grid());
    g5v = g5 * in.f[a];
    out.f[a] = out.f[a] + inv_sqrt2 * (p * g5v);
    // Tensor term: sum over (mu < nu) of t_{mu,nu} (sigma_{mu,nu} v[a]).
    // Grid's Gamma::SigmaMN is the anti-Hermitian (1/2)[gamma_mu, gamma_nu];
    // the TXQCD notes use the Hermitian sigma_{mu,nu} = (i/2)[gamma_mu,
    // gamma_nu], which is i * Grid's version. We therefore multiply by i
    // explicitly so t_{mu,nu} * sigma_{mu,nu} is Hermitian.
    const ComplexD ci(0.0, 1.0);
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        LatticeFermion smn_v(in.Grid());
        smn_v = smn * in.f[a];
        auto t_mn = PeekIndex<1>(t, mu, nu);
        out.f[a] = out.f[a] + ci * (t_mn * smn_v);
      }
    }
  }
}

// Apply the full Delta: sigma + pi + s + p + t pieces. Result overwrites out.
inline void ApplyDelta(const LatticeSigmaField &sigma,
                       const LatticePiField &pi,
                       const LatticeSFieldC &s,
                       const LatticePFieldC &p,
                       const LatticeTField &t,
                       const TXQCDFermionNf &in, TXQCDFermionNf &out) {
  TXQCDFermionNf tmp(in.Grid());
  ApplyDeltaSigmaPi(sigma, pi, in, out);
  ApplyDeltaColor(s, p, t, in, tmp);
  for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + tmp.f[a];
}

NAMESPACE_END(Grid);
