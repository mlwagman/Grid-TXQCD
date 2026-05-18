#pragma once
// Even-odd preconditioned TXQCD Möbius DWF operator.
//
// M(x,s; y,s') = M_Möbius(x,s; y,s') + (b+c) Δ(x) δ_{ss'} δ_{xy}
//
// Δ is 4D, broadcast across all 5th-dim slices, hermitian.  In the 5D EO
// decomposition the same-checkerboard piece (Mooee/Meeoo) is the only
// component that sees Δ — the cross-checkerboard hop Meooe is pure 4D
// Wilson, so unchanged.
//
//   Mee_TX = Mee_QCD + (b+c) Δ_e [broadcast s]
//   Moo_TX = Moo_QCD + (b+c) Δ_o [broadcast s]
//   Meo_TX = Meo_QCD             (Δ doesn't cross checkerboard)
//   Moe_TX = Moe_QCD
//
// MooeeInv_TX is implemented as inner CG on the normal equations
//   (Mooee_TX^† Mooee_TX) x = Mooee_TX^† b.
// (Tried fixed-point refinement against stock MooeeInv_QCD as a
// preconditioner; the spectral radius of (b+c) Δ MooeeInv_QCD turns out
// to be > 1 for typical Möbius parameters, so the iteration diverges.
// Inner CG always converges; we use stock MooeeInv_QCD as the initial
// guess to give a head start.)
//
// γ5R5-hermiticity carries over: stock M_Möbius is γ5R5-hermitian and
// Δ is γ5-hermitian + s-diagonal, so M_TX is γ5R5-hermitian as well.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/fermion/MobiusFermion.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>

NAMESPACE_BEGIN(Grid);

class TXQCDMobiusFermionEO {
 public:
  typedef WilsonImplR Impl;
  typedef MobiusFermion<Impl> MobiusOp;
  typedef typename Impl::GaugeField   GaugeField;
  typedef typename Impl::FermionField FermionField;

  TXQCDMobiusFermionEO(GaugeField &Umu,
                       GridCartesian &FGrid, GridRedBlackCartesian &FrbGrid,
                       GridCartesian &UGrid, GridRedBlackCartesian &UrbGrid,
                       RealD mass, RealD M5, RealD b, RealD c,
                       const LatticeSigmaField &sigma, const LatticePiField &pi,
                       const LatticeSFieldC &s, const LatticePFieldC &p,
                       const LatticeTField &t,
                       int mooee_inv_iter = 1000,
                       RealD mooee_inv_tol = 1e-10)
      : Dmob_(Umu, FGrid, FrbGrid, UGrid, UrbGrid, mass, M5, b, c),
        FGrid_(&FGrid), FrbGrid_(&FrbGrid),
        UGrid_(&UGrid), UrbGrid_(&UrbGrid),
        Ls_(FGrid.GlobalDimensions()[0]), bplusc_(b + c),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t),
        sigma_e_(&UrbGrid), sigma_o_(&UrbGrid),
        pi_e_(&UrbGrid),    pi_o_(&UrbGrid),
        s_e_(&UrbGrid),     s_o_(&UrbGrid),
        p_e_(&UrbGrid),     p_o_(&UrbGrid),
        t_e_(&UrbGrid),     t_o_(&UrbGrid),
        mooee_inv_iter_(mooee_inv_iter),
        mooee_inv_tol_(mooee_inv_tol) {
    ImportAux();
  }

  // Re-extract aux checkerboards.  Call after the aux fields change (e.g.
  // after each MD step).
  void ImportAux() {
    pickCheckerboard(Even, sigma_e_, sigma_); pickCheckerboard(Odd, sigma_o_, sigma_);
    pickCheckerboard(Even, pi_e_,    pi_);    pickCheckerboard(Odd, pi_o_,    pi_);
    pickCheckerboard(Even, s_e_,     s_);     pickCheckerboard(Odd, s_o_,     s_);
    pickCheckerboard(Even, p_e_,     p_);     pickCheckerboard(Odd, p_o_,     p_);
    pickCheckerboard(Even, t_e_,     t_);     pickCheckerboard(Odd, t_o_,     t_);
  }

  // ---- full operator (matches TXQCDMobiusOp::M, used for tests) ----
  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.M(in.f[a], out.f[a]);
    AddDeltaFull_(in, out);
  }
  void Mdag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.Mdag(in.f[a], out.f[a]);
    AddDeltaFull_(in, out);
  }

  // ---- EO: Mooee = stock + (b+c)·Δ at each slice on the in's checkerboard ----
  void Mooee(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.Mooee(in.f[a], out.f[a]);
    AddDeltaCB_(in, out);
  }
  void MooeeDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.MooeeDag(in.f[a], out.f[a]);
    AddDeltaCB_(in, out);
  }
  // ---- EO: Meooe = stock (Δ doesn't cross checkerboard) ----
  void Meooe(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.Meooe(in.f[a], out.f[a]);
  }
  void MeooeDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dmob_.MeooeDag(in.f[a], out.f[a]);
  }

  // ---- MooeeInv via inner CG on the normal equations ----
  // Solve (Mooee_TX^† Mooee_TX) x = Mooee_TX^† in, i.e.
  // x = MooeeInv_TX in.  Initial guess from stock MooeeInv_QCD.
  void MooeeInv(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    MooeeInvCG_(in, out, /*dag=*/false);
  }
  void MooeeInvDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    MooeeInvCG_(in, out, /*dag=*/true);
  }

  MobiusOp &Mobius() { return Dmob_; }
  GridCartesian *FGrid() { return FGrid_; }
  GridRedBlackCartesian *FrbGrid() { return FrbGrid_; }
  int Ls() const { return Ls_; }
  RealD bplusc() const { return bplusc_; }

 private:
  // Slice-by-slice Δ application on full 5D FGrid fields (out += (b+c) Δ in).
  void AddDeltaFull_(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    TXQCDFermionNf in4d(UGrid_), out4d(UGrid_);
    FermionField sl(UGrid_);
    for (int s = 0; s < Ls_; ++s) {
      for (int a = 0; a < TxqcdNf; ++a) {
        ExtractSlice(sl, const_cast<FermionField &>(in.f[a]), s, 0);
        in4d.f[a] = sl;
      }
      ApplyDelta(sigma_, pi_, s_, p_, t_, in4d, out4d);
      for (int a = 0; a < TxqcdNf; ++a) {
        ExtractSlice(sl, out.f[a], s, 0);
        sl = sl + bplusc_ * out4d.f[a];
        InsertSlice(sl, out.f[a], s, 0);
      }
    }
  }

  // Slice-by-slice Δ on a single-checkerboard 5D field; uses the matching
  // 4D aux-field checkerboard to keep operations local in 4D.
  void AddDeltaCB_(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    int cb = in.f[0].Checkerboard();
    const auto &sigma_cb = (cb == Even) ? sigma_e_ : sigma_o_;
    const auto &pi_cb    = (cb == Even) ? pi_e_    : pi_o_;
    const auto &s_cb     = (cb == Even) ? s_e_     : s_o_;
    const auto &p_cb     = (cb == Even) ? p_e_     : p_o_;
    const auto &t_cb     = (cb == Even) ? t_e_     : t_o_;
    TXQCDFermionNf in4d(UrbGrid_), out4d(UrbGrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      in4d.f[a].Checkerboard()  = cb;
      out4d.f[a].Checkerboard() = cb;
    }
    FermionField sl(UrbGrid_); sl.Checkerboard() = cb;
    for (int s = 0; s < Ls_; ++s) {
      for (int a = 0; a < TxqcdNf; ++a) {
        ExtractSlice(sl, const_cast<FermionField &>(in.f[a]), s, 0);
        in4d.f[a] = sl;
      }
      ApplyDelta(sigma_cb, pi_cb, s_cb, p_cb, t_cb, in4d, out4d);
      for (int a = 0; a < TxqcdNf; ++a) {
        ExtractSlice(sl, out.f[a], s, 0);
        sl = sl + bplusc_ * out4d.f[a];
        InsertSlice(sl, out.f[a], s, 0);
      }
    }
  }

  // Preconditioned CG on the normal equations:
  //   solve A x = b  with  A = Mooee_TX^† Mooee_TX,  b = Mooee_TX^† in
  //   preconditioner P ≈ A^{-1} ≈ MooeeInv_QCD MooeeInvDag_QCD.
  // Since (Mooee_TX) ≈ (Mooee_QCD) for small Δ, P A ≈ I and PCG converges in
  // O(1) iterations independently of cond(A).  Iter count stays small even
  // at large aux strength (where unpreconditioned CG would be expensive).
  void MooeeInvCG_(const TXQCDFermionNf &in, TXQCDFermionNf &out, bool dag) {
    int cb = in.f[0].Checkerboard();
    // Initial guess
    for (int a = 0; a < TxqcdNf; ++a) {
      if (dag) Dmob_.MooeeInvDag(in.f[a], out.f[a]);
      else     Dmob_.MooeeInv(in.f[a], out.f[a]);
    }
    // RHS for normal equations
    TXQCDFermionNf b(FrbGrid_);
    for (int a = 0; a < TxqcdNf; ++a) b.f[a].Checkerboard() = cb;
    if (dag) Mooee(in, b);    else     MooeeDag(in, b);

    TXQCDFermionNf Ax(FrbGrid_), tmp(FrbGrid_), r(FrbGrid_),
                   z(FrbGrid_),  p(FrbGrid_),   Ap(FrbGrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      Ax.f[a].Checkerboard()  = cb; tmp.f[a].Checkerboard() = cb;
      r.f[a].Checkerboard()   = cb; z.f[a].Checkerboard()   = cb;
      p.f[a].Checkerboard()   = cb; Ap.f[a].Checkerboard()  = cb;
    }
    auto applyA = [&](const TXQCDFermionNf &x, TXQCDFermionNf &Ax_out) {
      if (dag) { MooeeDag(x, tmp); Mooee(tmp, Ax_out);   }
      else     { Mooee(x, tmp);    MooeeDag(tmp, Ax_out); }
    };
    auto applyP = [&](const TXQCDFermionNf &v, TXQCDFermionNf &Pv) {
      // P = MooeeInv_QCD MooeeInvDag_QCD   (or the dagger swap, doesn't matter
      // for self-adjoint preconditioner of self-adjoint A)
      for (int a = 0; a < TxqcdNf; ++a) {
        Dmob_.MooeeInvDag(v.f[a],   tmp.f[a]);
        Dmob_.MooeeInv(   tmp.f[a], Pv.f[a]);
      }
    };

    applyA(out, Ax);
    for (int a = 0; a < TxqcdNf; ++a) r.f[a] = b.f[a] - Ax.f[a];
    applyP(r, z);
    p = z;
    ComplexD rz = innerProduct(r, z);
    RealD rsq  = norm2(r);
    RealD bsq  = std::max(norm2(b), 1e-30);
    RealD tol2 = mooee_inv_tol_ * mooee_inv_tol_ * bsq;
    int it;
    for (it = 0; it < mooee_inv_iter_; ++it) {
      if (rsq < tol2) break;
      applyA(p, Ap);
      ComplexD pAp = innerProduct(p, Ap);
      ComplexD alpha = rz / pAp;
      axpy(out,  alpha, p);
      axpy(r,   -alpha, Ap);
      rsq = norm2(r);
      if (rsq < tol2) break;
      applyP(r, z);
      ComplexD rz_new = innerProduct(r, z);
      ComplexD beta = rz_new / rz;
      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = z.f[a] + beta * p.f[a];
      rz = rz_new;
    }
    std::cout << GridLogMessage << "[TXQCDMobius MooeeInv"
              << (dag ? "Dag" : "") << " PCG] iter=" << it
              << " final ||r||^2=" << rsq << " tol2=" << tol2 << std::endl;
  }

  MobiusOp Dmob_;
  GridCartesian *FGrid_;
  GridRedBlackCartesian *FrbGrid_;
  GridCartesian *UGrid_;
  GridRedBlackCartesian *UrbGrid_;
  int Ls_;
  RealD bplusc_;
  // Full-grid aux refs (M, Mdag use these).
  const LatticeSigmaField &sigma_;
  const LatticePiField    &pi_;
  const LatticeSFieldC    &s_;
  const LatticePFieldC    &p_;
  const LatticeTField     &t_;
  // RB-extracted aux (Mooee uses these).
  LatticeSigmaField sigma_e_, sigma_o_;
  LatticePiField    pi_e_,    pi_o_;
  LatticeSFieldC    s_e_,     s_o_;
  LatticePFieldC    p_e_,     p_o_;
  LatticeTField     t_e_,     t_o_;
  int mooee_inv_iter_;
  RealD mooee_inv_tol_;
};

NAMESPACE_END(Grid);
