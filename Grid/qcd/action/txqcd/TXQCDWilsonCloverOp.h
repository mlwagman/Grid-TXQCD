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
#include <Grid/qcd/action/txqcd/TXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>

NAMESPACE_BEGIN(Grid);

// Holds one WilsonFermion engine plus references to the five aux fields.
// The aux fields are owned by the caller (typically the composite TXQCDField
// inside HMC); this class only borrows them.
class TXQCDWilsonCloverOp {
 public:
  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;
  typedef typename Impl::GaugeField GaugeField;

  // Default impl params: antiperiodic time BC for fermions (matches chroma).
  static typename Impl::ImplParams DefaultImplParams() {
    typename Impl::ImplParams p;
    p.boundary_phases.resize(Nd, 1.0);
    p.boundary_phases[Nd - 1] = -1.0;
    return p;
  }

  // Per-flavor mass constructor.  Inner Dw is built with mass=0; M() adds
  // mass_[a]*in.f[a] for each flavor (so non-degenerate Nf>2 setups work).
  // mu rescales the Delta term: M_TXQCD = M_QCD + mu * Delta.  Default 1.
  TXQCDWilsonCloverOp(GaugeField &Umu, GridCartesian &grid,
                GridRedBlackCartesian &rbgrid,
                const std::array<RealD, TxqcdNf> &mass,
                const LatticeSigmaField &sigma, const LatticePiField &pi,
                const LatticeSFieldC &s, const LatticePFieldC &p,
                const LatticeTField &t, RealD csw = 0.0,
                typename Impl::ImplParams impl_p = DefaultImplParams(),
                RealD mu = 1.0)
      : Dw(Umu, grid, rbgrid, 0.0, impl_p), mass_(mass), csw_(csw), Umu_(Umu),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t), mu_(mu) {
    if (csw_ != 0.0) {
      for (int mu_lor = 0; mu_lor < Nd; ++mu_lor)
        for (int nu = mu_lor + 1; nu < Nd; ++nu) {
          FS_.emplace_back(&grid);
          WilsonLoops<Impl>::FieldStrength(FS_.back(), Umu, mu_lor, nu);
        }
    }
  }

  // Backward-compat: degenerate scalar mass.
  TXQCDWilsonCloverOp(GaugeField &Umu, GridCartesian &grid,
                GridRedBlackCartesian &rbgrid, RealD mass,
                const LatticeSigmaField &sigma, const LatticePiField &pi,
                const LatticeSFieldC &s, const LatticePFieldC &p,
                const LatticeTField &t, RealD csw = 0.0,
                typename Impl::ImplParams impl_p = DefaultImplParams(),
                RealD mu = 1.0)
      : TXQCDWilsonCloverOp(Umu, grid, rbgrid, MakeMass(mass), sigma, pi, s, p,
                            t, csw, impl_p, mu) {}

  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    // Dw has mass=0, so Dw.M gives 4*I + Wilson_hop per flavor; add mass_[a].
    for (int a = 0; a < TxqcdNf; ++a) {
      Dw.M(in.f[a], out.f[a]);
      out.f[a] = out.f[a] + mass_[a] * in.f[a];
    }
    TXQCDFermionNf d(in.Grid());
    ApplyDelta(sigma_, pi_, s_, p_, t_, in, d);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + mu_ * d.f[a];
    if (csw_ != 0.0) {
      TXQCDFermionNf cl(in.Grid());
      ApplyClover(csw_, FS_, in, cl);
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + cl.f[a];
    }
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
  const std::array<RealD, TxqcdNf> &Mass() const { return mass_; }
  RealD Mu() const { return mu_; }

 private:
  static std::array<RealD, TxqcdNf> MakeMass(RealD m) {
    std::array<RealD, TxqcdNf> a;
    a.fill(m);
    return a;
  }

  WilsonOp Dw;
  std::array<RealD, TxqcdNf> mass_;
  RealD csw_;
  GaugeField &Umu_;
  const LatticeSigmaField &sigma_;
  const LatticePiField    &pi_;
  const LatticeSFieldC    &s_;
  const LatticePFieldC    &p_;
  const LatticeTField     &t_;
  std::vector<LatticeColourMatrix> FS_;
  RealD mu_;
};

NAMESPACE_END(Grid);
