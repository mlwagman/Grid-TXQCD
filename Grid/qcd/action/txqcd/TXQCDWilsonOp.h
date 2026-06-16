#pragma once
// TXQCD Wilson Dirac operator: M = D_W (per flavor) + Delta(aux fields).
//
// The hopping piece is plain Wilson with the same gauge link applied
// independently to each flavor of TXQCDFermionNf. The local Delta insertion
// (sigma + pi + s + p + t) is added on top.
//
// gamma5-Hermiticity (notes Eq. 9): Delta is composite-Hermitian and
// commutes with gamma5 in structure (the gamma5 factors in pi and p sit
// inside Hermitian objects), so gamma5 Delta gamma5 = Delta. Wilson by
// itself satisfies gamma5 D_W gamma5 = D_W^dag. Therefore
//   gamma5 M gamma5 = D_W^dag + Delta = M^dag,
// i.e. (gamma5 * M) is Hermitian. The test exploits exactly that:
// <w, gamma5 M v> = conj(<v, gamma5 M w>).

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>

NAMESPACE_BEGIN(Grid);

// Holds one WilsonFermion engine plus references to the five aux fields.
// The aux fields are owned by the caller (typically the composite TXQCDField
// inside HMC); this class only borrows them.
class TXQCDWilsonOp {
 public:
  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;
  typedef typename Impl::GaugeField GaugeField;

  // Default ImplParams: antiperiodic time, periodic spatial — the chroma
  // physics convention used throughout TXQCD.  Matches
  // TXQCDWilsonCloverFermionEO::DefaultImplParams() so the rational PF
  // sampler and the EO measurement operator agree on the gauge field.
  static typename Impl::ImplParams DefaultImplParams() {
    typename Impl::ImplParams p;
    p.boundary_phases.resize(Nd, 1.0);
    p.boundary_phases[Nd - 1] = -1.0;
    return p;
  }

  TXQCDWilsonOp(GaugeField &Umu, GridCartesian &grid,
                GridRedBlackCartesian &rbgrid, RealD mass,
                const LatticeSigmaField &sigma, const LatticePiField &pi,
                const LatticeSFieldC &s, const LatticePFieldC &p,
                const LatticeTField &t,
                typename Impl::ImplParams impl_p = DefaultImplParams())
      : Dw(Umu, grid, rbgrid, mass, impl_p),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t) {}

  // Apply M = D_W + Delta. Per flavor: out.f[a] = D_W in.f[a]; then add Delta.
  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dw.M(in.f[a], out.f[a]);
    TXQCDFermionNf d(in.Grid());
    ApplyDelta(sigma_, pi_, s_, p_, t_, in, d);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + d.f[a];
  }

  // Mdag via gamma5 M gamma5 (cheaper than wiring a separate Wilson.Mdag,
  // and we already rely on this identity for the action).
  void Mdag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf g5in(in.Grid());
    for (int a = 0; a < TxqcdNf; ++a) g5in.f[a] = g5 * in.f[a];
    M(g5in, out);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = g5 * out.f[a];
  }

  WilsonOp &Wilson() { return Dw; }

 private:
  WilsonOp Dw;
  const LatticeSigmaField &sigma_;
  const LatticePiField    &pi_;
  const LatticeSFieldC    &s_;
  const LatticePFieldC    &p_;
  const LatticeTField     &t_;
};

NAMESPACE_END(Grid);
