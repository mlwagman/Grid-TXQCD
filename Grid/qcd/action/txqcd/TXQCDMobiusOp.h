#pragma once
// TXQCD Möbius DWF Dirac operator: M = M_Möbius + (b+c) * Delta(aux), with
// Delta inserted at every 5th-dim slice.  See txqcd_clover_dwf_aux.tex Sec 3.
//
// 5D fermion convention.  TXQCDFermionNf is reused; each f[a] is a 5D
// LatticeFermion on FGrid (Ls × 4D).  The aux fields (sigma, pi, s, p, t)
// remain 4D on UGrid; ApplyDelta is called slice-by-slice and accumulated.
//
// gamma5-hermiticity. For Möbius DWF the appropriate hermiticity is
//   R_5 gamma_5 M = M^dag R_5 gamma_5,
// where R_5 reverses the 5th-dim coordinate.  We don't override Mdag here:
// Grid's MobiusFermion already exposes Mdag, and the Delta piece commutes
// with R_5 gamma_5 because Delta is built from gamma_5-Hermitian local
// matrices and Δ acts diagonally in s.  So
//   Mdag = MobiusFermion::Mdag + (b+c) * Delta^dag,
// and Delta^dag = Delta on the relevant subspace (Δ is composite-hermitian).
//
// First cut.  Implements only M, Mdag, and a γ5R5 hermiticity helper.  No
// EO preconditioning, no force.  The Delta application uses ExtractSlice /
// InsertSlice (correct, but ~5–10× slower than a 5D-native kernel).  Once the
// FD-force test validates correctness we can add a 5D-native ApplyDelta5D.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/fermion/MobiusFermion.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>

NAMESPACE_BEGIN(Grid);

class TXQCDMobiusOp {
 public:
  typedef WilsonImplR Impl;
  typedef MobiusFermion<Impl> MobiusOp;
  typedef typename Impl::GaugeField   GaugeField;
  typedef typename Impl::FermionField FermionField;  // 5D when constructed on FGrid

  TXQCDMobiusOp(GaugeField &Umu,
                GridCartesian &FGrid, GridRedBlackCartesian &FrbGrid,
                GridCartesian &UGrid, GridRedBlackCartesian &UrbGrid,
                RealD mass, RealD M5, RealD b, RealD c,
                const LatticeSigmaField &sigma, const LatticePiField &pi,
                const LatticeSFieldC &s, const LatticePFieldC &p,
                const LatticeTField &t)
      : Dmob_(Umu, FGrid, FrbGrid, UGrid, UrbGrid, mass, M5, b, c),
        FGrid_(&FGrid), UGrid_(&UGrid), Ls_(FGrid.GlobalDimensions()[0]),
        bplusc_(b + c),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t) {}

  // Apply M = M_Möbius + (b+c) Δ.  Per flavor: out.f[a] = M_Möbius in.f[a],
  // then add (b+c) · slice-broadcast Δ in.f[a].
  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.M(in.f[a], out.f[a]);
    AddDelta_(in, out, /*dag=*/false);
  }

  // Mdag: same prefactor structure but use MobiusFermion::Mdag and Δ^dag = Δ.
  void Mdag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.Mdag(in.f[a], out.f[a]);
    AddDelta_(in, out, /*dag=*/true);
  }

  MobiusOp &Mobius() { return Dmob_; }
  GridCartesian *FGrid() { return FGrid_; }
  GridCartesian *UGrid() { return UGrid_; }
  int Ls() const { return Ls_; }
  RealD bplusc() const { return bplusc_; }

 private:
  // out.f[a] += (b+c) * Δ(in.f[a])  [or Δ^dag if dag=true; Δ is hermitian so identical]
  // Slice-by-slice: extract 4D slice of each flavor, build a 4D
  // TXQCDFermionNf, call ApplyDelta, then insert back.
  void AddDelta_(const TXQCDFermionNf &in, TXQCDFermionNf &out, bool /*dag*/) {
    // Single 4D scratch for in/out per slice.
    TXQCDFermionNf in4d(UGrid_), out4d(UGrid_);
    FermionField slice4d(UGrid_), tmp4d(UGrid_);
    for (int s = 0; s < Ls_; ++s) {
      for (int a = 0; a < TxqcdNf; ++a) {
        ExtractSlice(slice4d, const_cast<FermionField &>(in.f[a]), s, /*orthog=*/0);
        in4d.f[a] = slice4d;
      }
      ApplyDelta(sigma_, pi_, s_, p_, t_, in4d, out4d);
      for (int a = 0; a < TxqcdNf; ++a) {
        // out.f[a] @ slice s += (b+c) * out4d.f[a]
        ExtractSlice(slice4d, out.f[a], s, 0);
        slice4d = slice4d + bplusc_ * out4d.f[a];
        InsertSlice(slice4d, out.f[a], s, 0);
      }
    }
  }

  MobiusOp Dmob_;
  GridCartesian *FGrid_;
  GridCartesian *UGrid_;
  int Ls_;
  RealD bplusc_;
  const LatticeSigmaField &sigma_;
  const LatticePiField    &pi_;
  const LatticeSFieldC    &s_;
  const LatticePFieldC    &p_;
  const LatticeTField     &t_;
};

NAMESPACE_END(Grid);
