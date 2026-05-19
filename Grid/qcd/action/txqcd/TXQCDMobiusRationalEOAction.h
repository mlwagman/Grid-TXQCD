#pragma once
// EO-preconditioned RHMC pseudofermion action for TXQCD Möbius:
//
//   S = Phi^dag f(Mpc^dag Mpc) Phi,  Phi on odd 5D sublattice (half volume).
//
// Mpc = Moo - Moe Mee^{-1} Meo is the Schur complement of the TXQCD Möbius
// operator (TXQCDMobiusFermionEO).  Mirrors TXQCDWilsonRationalEOAction; the
// differences are:
//   - 5D rb fermion fields; aux fields are 4D.
//   - Aux-force builders slice-sum the 5D bilinear to a 4D rb field
//     (checkerboard-aware) and scale by (b+c).
//   - Gauge force via stock MobiusFermion::MoeDeriv / MeoDeriv (5D-aware).
//
// Force per rational pole (same three-piece SchurDifferentiableOperator
// structure as Wilson):
//   X_k = (Mpc†Mpc + σ_k)^{-1} Phi (odd),  Y_k = Mpc X_k (odd)
//   W_e = Mee^{-1} Meo X_k (even),  Z_e = Mee^{-1}† Moe† Y_k (even)
//   aux force = (b+c) [ bilinear5d(Y_k,X_k)|odd + bilinear5d(Z_e,W_e)|even ]
//   gauge     = MoeDeriv/MeoDeriv per flavor.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusPseudoFermionAction.h>

NAMESPACE_BEGIN(Grid);

// ---- checkerboard-aware 5D→4D aux-force reductions ----
// G_{a,b}(x) = Σ_s ⟨Y_a(x,s)|(Op) X_b(x,s)⟩ on the 4D rb grid (checkerboard cb).

inline LatticeSigmaField FlavorBilinear5D_cb(const TXQCDFermionNf &Y5,
                                             const TXQCDFermionNf &X5,
                                             GridRedBlackCartesian *UrbGrid,
                                             int Ls, int cb) {
  LatticeSigmaField G(UrbGrid); G = Zero(); G.Checkerboard() = cb;
  TXQCDFermionNf Y4(UrbGrid), X4(UrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    Y4.f[a].Checkerboard() = cb; X4.f[a].Checkerboard() = cb;
  }
  LatticeFermion sl(UrbGrid); sl.Checkerboard() = cb;
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
    }
    auto Gs = FlavorBilinear(Y4, X4);
    Gs.Checkerboard() = cb;
    G = G + Gs;
  }
  return G;
}

inline LatticeSigmaField FlavorBilinearG5_5D_cb(const TXQCDFermionNf &Y5,
                                                const TXQCDFermionNf &X5,
                                                GridRedBlackCartesian *UrbGrid,
                                                int Ls, int cb) {
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeSigmaField G(UrbGrid); G = Zero(); G.Checkerboard() = cb;
  TXQCDFermionNf Y4(UrbGrid), X4(UrbGrid), g5X4(UrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    Y4.f[a].Checkerboard() = cb; X4.f[a].Checkerboard() = cb;
    g5X4.f[a].Checkerboard() = cb;
  }
  LatticeFermion sl(UrbGrid); sl.Checkerboard() = cb;
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
      g5X4.f[a] = g5 * X4.f[a];
    }
    auto Gs = FlavorBilinear(Y4, g5X4);
    Gs.Checkerboard() = cb;
    G = G + Gs;
  }
  return G;
}

template <class Spin4Op>
inline LatticeSFieldC ColorBilinearSpinOp5D_cb(const TXQCDFermionNf &Y5,
                                                const TXQCDFermionNf &X5,
                                                const Spin4Op &Op,
                                                GridRedBlackCartesian *UrbGrid,
                                                int Ls, int cb) {
  LatticeSFieldC G(UrbGrid); G = Zero(); G.Checkerboard() = cb;
  TXQCDFermionNf Y4(UrbGrid), X4(UrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    Y4.f[a].Checkerboard() = cb; X4.f[a].Checkerboard() = cb;
  }
  LatticeFermion sl(UrbGrid); sl.Checkerboard() = cb;
  for (int s = 0; s < Ls; ++s) {
    for (int a = 0; a < TxqcdNf; ++a) {
      ExtractSlice(sl, const_cast<LatticeFermion &>(Y5.f[a]), s, 0); Y4.f[a] = sl;
      ExtractSlice(sl, const_cast<LatticeFermion &>(X5.f[a]), s, 0); X4.f[a] = sl;
    }
    auto Gs = ColorBilinearSpinOp(Y4, X4, Op);
    Gs.Checkerboard() = cb;
    G = G + Gs;
  }
  return G;
}

class TXQCDMobiusRationalEOAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  TXQCDMobiusRationalEOAction(GridCartesian &FGrid, GridRedBlackCartesian &FrbGrid,
                              GridCartesian &UGrid, GridRedBlackCartesian &UrbGrid,
                              RealD mass, RealD M5, RealD b, RealD c, Params &p)
      : FGrid_(FGrid), FrbGrid_(FrbGrid), UGrid_(UGrid), UrbGrid_(UrbGrid),
        mass_(mass), M5_(M5), b_(b), c_(c), bplusc_(b + c),
        Ls_(FGrid.GlobalDimensions()[0]), param(p), Phi(&FrbGrid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage << "[TXQCDMobiusRationalEO] degree "
              << param.degree << " rational for x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage << "[TXQCDMobiusRationalEO] degree "
              << param.degree << " rational for x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  std::string action_name() override { return "TXQCDMobiusRationalEOAction"; }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " M5=" << M5_ << " b=" << b_ << " c=" << c_ << " Ls=" << Ls_
       << " lo=" << param.lo << " hi=" << param.hi
       << " degree=" << param.degree << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    // pRNG is the 4D composite-field RNG; drawing directly on a 5D rb field
    // would trip pickCheckerboard's dimension assert.  Draw on the full 5D
    // grid (no checkerboard pick — same path the non-EO Möbius PF uses) and
    // then extract the odd checkerboard.
    TXQCDFermionNf eta_full(&FGrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, eta_full.f[a]);
      eta_full.f[a] = scale * eta_full.f[a];
    }
    TXQCDFermionNf eta(&FrbGrid_);
    for (int a = 0; a < TxqcdNf; ++a) {
      eta.f[a].Checkerboard() = Odd;
      pickCheckerboard(Odd, eta.f[a], eta_full.f[a]);
    }
    auto EOp = MakeEOp(U);
    TXQCDMobiusSchurOp SchurOp(EOp);
    ApplyRational(SchurOp, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    auto EOp = MakeEOp(U);
    TXQCDMobiusSchurOp SchurOp(EOp);
    TXQCDFermionNf Y(&FrbGrid_);
    ApplyRational(SchurOp, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto EOp = MakeEOp(U);
    TXQCDMobiusSchurOp SchurOp(EOp);
    const int Npole = static_cast<int>(PowerNegHalf.poles.size());

    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&FrbGrid_);
    std::vector<RealD> md_tol(Npole, param.mdtolerance);

    TXQCDMultiShiftCGSchur MSCG(param.MaxIter);
    MSCG(SchurOp, PowerNegHalf.poles, md_tol, Phi, Xk);

    dSdU.sigma = Zero(); dSdU.pi = Zero(); dSdU.s = Zero();
    dSdU.p = Zero();     dSdU.t  = Zero(); dSdU.U = Zero();

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};

    LatticeGaugeField gforce(&UGrid_), gtmp(&UGrid_);
    GridRedBlackCartesian *forcecb = new GridRedBlackCartesian(&UGrid_);
    LatticeGaugeField ForceO(forcecb), ForceE(forcecb);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];                 // odd
      TXQCDFermionNf Y(&FrbGrid_);
      for (int a = 0; a < TxqcdNf; ++a) Y.f[a].Checkerboard() = Odd;
      SchurOp.Mpc(X, Y);

      TXQCDFermionNf W_e(&FrbGrid_), Z_e(&FrbGrid_), tmp_e(&FrbGrid_);
      for (int a = 0; a < TxqcdNf; ++a) {
        W_e.f[a].Checkerboard()  = Even; Z_e.f[a].Checkerboard()  = Even;
        tmp_e.f[a].Checkerboard() = Even;
      }
      EOp.Meooe(X, tmp_e);          // Meo X_k: odd → even
      EOp.MooeeInv(tmp_e, W_e);     // Mee^{-1} Meo X_k
      EOp.MeooeDag(Y, tmp_e);       // Moe† Y_k: odd → even
      EOp.MooeeInvDag(tmp_e, Z_e);  // Mee^{-1}† Moe† Y_k

      // ---- aux force: odd (Y,X) + even (Z_e,W_e), scaled by (b+c) ----
      AccumulateAuxForce(dSdU, ak, Y,   X,   Odd,  inv_sqrt2, Id, G5);
      AccumulateAuxForce(dSdU, ak, Z_e, W_e, Even, inv_sqrt2, Id, G5);

      // ---- gauge force (Möbius MoeDeriv/MeoDeriv per flavor) ----
      gforce = Zero();
      for (int a = 0; a < TxqcdNf; ++a) {
        EOp.Mobius().MoeDeriv(ForceO, Y.f[a],   W_e.f[a], DaggerNo);
        EOp.Mobius().MeoDeriv(ForceE, Z_e.f[a], X.f[a],   DaggerNo);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;

        EOp.Mobius().MoeDeriv(ForceO, X.f[a],   Z_e.f[a], DaggerYes);
        EOp.Mobius().MeoDeriv(ForceE, W_e.f[a], Y.f[a],   DaggerYes);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;
      }
      dSdU.U = dSdU.U + ak * gforce;
    }
    delete forcecb;
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDMobiusFermionEO MakeEOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    TXQCDMobiusFermionEO EOp(Unc.U, FGrid_, FrbGrid_, UGrid_, UrbGrid_,
                             mass_, M5_, b_, c_,
                             Unc.sigma, Unc.pi, Unc.s, Unc.p, Unc.t);
    if (TXQCDMobiusFermionEO::LUEnabled()) EOp.BuildLU();
    return EOp;
  }

  void ApplyRational(TXQCDMobiusSchurOp &SchurOp, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&FrbGrid_);
    TXQCDMultiShiftCGSchur MSCG(param.MaxIter);
    MSCG(SchurOp, rat.poles, rat.tolerances, in, xk);

    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = rat.norm * in.f[a];
    for (int k = 0; k < nshift; ++k) {
      RealD ck = rat.residues[k];
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + ck * xk[k].f[a];
    }
  }

  // Slice-summed 5D aux force on the cb sublattice, scaled by (b+c)·ak,
  // accumulated into the full-grid dSdU aux slots.
  void AccumulateAuxForce(TXQCDField &dSdU, RealD ak,
                          const TXQCDFermionNf &Y, const TXQCDFermionNf &X,
                          int cb, ComplexD inv_sqrt2,
                          const SpinTable &Id, const SpinTable &G5) {
    const RealD w = bplusc_ * ak;
    // sigma
    {
      auto G = FlavorBilinear5D_cb(Y, X, &UrbGrid_, Ls_, cb);
      LatticeSigmaField F = HermitianFlavorForce(G); F.Checkerboard() = cb;
      LatticeSigmaField tmp(&UGrid_); tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.sigma = dSdU.sigma + w * tmp;
    }
    // pi
    {
      auto G = FlavorBilinearG5_5D_cb(Y, X, &UrbGrid_, Ls_, cb);
      LatticePiField F = HermitianFlavorForce(G); F.Checkerboard() = cb;
      LatticePiField tmp(&UGrid_); tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.pi = dSdU.pi + w * tmp;
    }
    // s
    {
      auto G = ColorBilinearSpinOp5D_cb(Y, X, Id, &UrbGrid_, Ls_, cb);
      G = inv_sqrt2 * G; G.Checkerboard() = cb;
      LatticeSFieldC F = HermitianColorForce(G); F.Checkerboard() = cb;
      LatticeSFieldC tmp(&UGrid_); tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.s = dSdU.s + w * tmp;
    }
    // p
    {
      auto G = ColorBilinearSpinOp5D_cb(Y, X, G5, &UrbGrid_, Ls_, cb);
      G = inv_sqrt2 * G; G.Checkerboard() = cb;
      LatticePFieldC F = HermitianColorForce(G); F.Checkerboard() = cb;
      LatticePFieldC tmp(&UGrid_); tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.p = dSdU.p + w * tmp;
    }
    // t_{mu,nu}
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        SpinTable iSig{ISigmaMatrix(mu, nu)};
        auto Gt = ColorBilinearSpinOp5D_cb(Y, X, iSig, &UrbGrid_, Ls_, cb);
        Gt.Checkerboard() = cb;
        LatticeSFieldC Ft = HermitianColorForce(Gt); Ft.Checkerboard() = cb;
        LatticeSFieldC Ftfull(&UGrid_); Ftfull = Zero();
        setCheckerboard(Ftfull, Ft);
        autoView(dst, dSdU.t, CpuWrite);
        autoView(src, Ftfull, CpuRead);
        thread_for(ss, UGrid_.oSites(), {
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              dst[ss]()(mu, nu)(i, j) =
                  dst[ss]()(mu, nu)(i, j) + w * src[ss]()()(i, j);
              dst[ss]()(nu, mu)(i, j) =
                  dst[ss]()(nu, mu)(i, j) - w * src[ss]()()(i, j);
            }
          }
        });
      }
    }
  }

  GridCartesian &FGrid_;
  GridRedBlackCartesian &FrbGrid_;
  GridCartesian &UGrid_;
  GridRedBlackCartesian &UrbGrid_;
  RealD mass_, M5_, b_, c_, bplusc_;
  int Ls_;
  Params &param;
  MultiShiftFunction PowerHalf;
  MultiShiftFunction PowerNegHalf;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
