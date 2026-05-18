#pragma once
// Two-flavor pseudofermion action for the TXQCD Möbius Dirac operator
//
//   S = phi^dag (M^dag M)^{-1} phi,   M = M_Möbius + (b+c) Δ_5d(aux),
//
// implemented as Action<TXQCDField>.  Mirrors TXQCDWilsonPseudoFermionAction;
// the differences are:
//   - Fermion fields live on the 5D FGrid (Ls × 4D).
//   - The aux-force builders sum over the 5th-dim slices and scale by (b+c).
//   - The gauge force is delegated to MobiusFermion::MDeriv (5D-aware).
//
// Refresh / S / heatbath are unchanged in form: φ = M† χ ⇒ S(U) = ‖χ‖².
//
// We reuse the existing 4D site-local builders (FlavorBilinear,
// ColorBilinearSpinOp, HermitianFlavorForce, HermitianColorForce, ISigmaMatrix,
// Gamma5Matrix, IdentitySpinMatrix, SpinTable) from
// TXQCDWilsonPseudoFermionAction.h via a slice-by-slice accumulation.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

NAMESPACE_BEGIN(Grid);

// ---------------- 5D → 4D aux-force reductions ----------------

// G_{a,b}(x) = Σ_s ⟨Y_a(x,s) | (Op) X_b(x,s)⟩_(spin,color).
// Reuses the 4D FlavorBilinear by extracting per-slice 4D fermion fields.
inline LatticeSigmaField FlavorBilinear5D(const TXQCDFermionNf &Y5,
                                          const TXQCDFermionNf &X5,
                                          GridBase *UGrid, int Ls) {
  LatticeSigmaField G(UGrid); G = Zero();
  TXQCDFermionNf Y4(UGrid), X4(UGrid);
  LatticeFermion sl(UGrid);
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
    }
    G = G + FlavorBilinear(Y4, X4);
  }
  return G;
}

// Same as above but with γ5 applied to X first (used for π-channel).
inline LatticeSigmaField FlavorBilinearG5_5D(const TXQCDFermionNf &Y5,
                                              const TXQCDFermionNf &X5,
                                              GridBase *UGrid, int Ls) {
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeSigmaField G(UGrid); G = Zero();
  TXQCDFermionNf Y4(UGrid), X4(UGrid), g5X4(UGrid);
  LatticeFermion sl(UGrid);
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
      g5X4.f[a] = g5 * X4.f[a];
    }
    G = G + FlavorBilinear(Y4, g5X4);
  }
  return G;
}

// G(x)_{i,j} = Σ_s Σ_(a,α,β) conj(Y_a(x,s)_{α,i}) Op_{α,β} X_a(x,s)_{β,j}.
template <class Spin4Op>
inline LatticeSFieldC ColorBilinearSpinOp5D(const TXQCDFermionNf &Y5,
                                             const TXQCDFermionNf &X5,
                                             const Spin4Op &Op,
                                             GridBase *UGrid, int Ls) {
  LatticeSFieldC G(UGrid); G = Zero();
  TXQCDFermionNf Y4(UGrid), X4(UGrid);
  LatticeFermion sl(UGrid);
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
    }
    G = G + ColorBilinearSpinOp(Y4, X4, Op);
  }
  return G;
}

// ---------------- the action ----------------

class TXQCDMobiusPseudoFermionAction : public Action<TXQCDField> {
 public:
  TXQCDMobiusPseudoFermionAction(GridCartesian &FGrid, GridRedBlackCartesian &FrbGrid,
                                 GridCartesian &UGrid, GridRedBlackCartesian &UrbGrid,
                                 RealD mass, RealD M5, RealD b, RealD c,
                                 RealD cg_tol = 1e-12, int cg_maxiter = 10000)
      : FGrid_(FGrid), FrbGrid_(FrbGrid), UGrid_(UGrid), UrbGrid_(UrbGrid),
        mass_(mass), M5_(M5), b_(b), c_(c), bplusc_(b + c),
        Ls_(FGrid.GlobalDimensions()[0]),
        cg_tol_(cg_tol), cg_maxiter_(cg_maxiter), Phi(&FGrid) {}

  std::string action_name() override {
    return "TXQCDMobiusPseudoFermionAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " M5=" << M5_ << " b=" << b_ << " c=" << c_ << " Ls=" << Ls_
       << " cg_tol=" << cg_tol_ << " cg_maxiter=" << cg_maxiter_ << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    TXQCDFermionNf chi(&FGrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, chi.f[a]);
      chi.f[a] = scale * chi.f[a];
    }
    TXQCDMobiusOp Mop = MakeOp(U);
    Mop.Mdag(chi, Phi);
  }

  RealD S(const TXQCDField &U) override {
    TXQCDMobiusOp Mop = MakeOp(U);
    TXQCDFermionNf X(&FGrid_); X = Zero();
    SolveMdagM(Mop, Phi, X);
    TXQCDFermionNf Y(&FGrid_);
    Mop.M(X, Y);
    return norm2(Y);
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    TXQCDMobiusOp Mop = MakeOp(U);
    TXQCDFermionNf X(&FGrid_); X = Zero();
    SolveMdagM(Mop, Phi, X);
    TXQCDFermionNf Y(&FGrid_);
    Mop.M(X, Y);

    // ----- aux forces (×(b+c)) -----
    auto Gsig = FlavorBilinear5D(Y, X, &UGrid_, Ls_);
    dSdU.sigma = bplusc_ * HermitianFlavorForce(Gsig);

    auto Gpi  = FlavorBilinearG5_5D(Y, X, &UGrid_, Ls_);
    dSdU.pi   = bplusc_ * HermitianFlavorForce(Gpi);

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};
    auto Gs = ColorBilinearSpinOp5D(Y, X, Id, &UGrid_, Ls_); Gs = inv_sqrt2 * Gs;
    dSdU.s  = bplusc_ * HermitianColorForce(Gs);
    auto Gp = ColorBilinearSpinOp5D(Y, X, G5, &UGrid_, Ls_); Gp = inv_sqrt2 * Gp;
    dSdU.p  = bplusc_ * HermitianColorForce(Gp);

    dSdU.t = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        SpinTable iSig{ISigmaMatrix(mu, nu)};
        auto Gt = ColorBilinearSpinOp5D(Y, X, iSig, &UGrid_, Ls_);
        auto Ft = HermitianColorForce(Gt);
        autoView(dst, dSdU.t, CpuWrite);
        autoView(src, Ft,     CpuRead);
        thread_for(ss, UGrid_.oSites(), {
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              dst[ss]()(mu, nu)(i, j) =  bplusc_ * src[ss]()()(i, j);
              dst[ss]()(nu, mu)(i, j) = -bplusc_ * src[ss]()()(i, j);
            }
          }
        });
      }
    }

    // ----- gauge force per flavor (Möbius MDeriv handles 5D natively) -----
    LatticeGaugeField gforce(&UGrid_); gforce = Zero();
    LatticeGaugeField tmp(&UGrid_);
    Mop.Mobius().ImportGauge(U.U);
    for (int a = 0; a < TxqcdNf; ++a) {
      Mop.Mobius().MDeriv(tmp, Y.f[a], X.f[a], DaggerNo);
      gforce = gforce + tmp;
      Mop.Mobius().MDeriv(tmp, X.f[a], Y.f[a], DaggerYes);
      gforce = gforce + tmp;
    }
    dSdU.U = gforce;
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDMobiusOp MakeOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDMobiusOp(Unc.U, FGrid_, FrbGrid_, UGrid_, UrbGrid_,
                         mass_, M5_, b_, c_,
                         Unc.sigma, Unc.pi, Unc.s, Unc.p, Unc.t);
  }

  void SolveMdagM(TXQCDMobiusOp &Mop, const TXQCDFermionNf &b_in,
                  TXQCDFermionNf &x) {
    TXQCDFermionNf r(&FGrid_), p(&FGrid_), Mp(&FGrid_), MdMp(&FGrid_);
    r = b_in;
    p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(b_in), 1e-30);
    RealD tol2 = cg_tol_ * cg_tol_ * bsq;
    int it;
    for (it = 0; it < cg_maxiter_; ++it) {
      Mop.M(p, Mp);
      Mop.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      axpy(x,  alpha, p);
      axpy(r, -alpha, MdMp);
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) { rsq = rsq_new; break; }
      RealD beta = rsq_new / rsq;
      for (int aa = 0; aa < TxqcdNf; ++aa) p.f[aa] = r.f[aa] + beta * p.f[aa];
      rsq = rsq_new;
    }
    std::cout << GridLogMessage << "[TXQCDMobiusPF CG] iter=" << it
              << " rsq=" << rsq << " tol2=" << tol2 << std::endl;
  }

  GridCartesian &FGrid_;
  GridRedBlackCartesian &FrbGrid_;
  GridCartesian &UGrid_;
  GridRedBlackCartesian &UrbGrid_;
  RealD mass_, M5_, b_, c_, bplusc_;
  int Ls_;
  RealD cg_tol_;
  int cg_maxiter_;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
