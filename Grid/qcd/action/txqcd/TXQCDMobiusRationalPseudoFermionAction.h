#pragma once
// One-flavor rational pseudofermion action for TXQCD Möbius:
//
//   S = phi^dag (M^dag M)^{-1/2} phi,    M = M_Möbius + (b+c) Δ_5d(aux),
//
// giving weight |det M| per PF.  Two such PFs on the same TXQCDField yield
// |det M_TX|^2; with the internal Nf_tx=2 flavor block this gives
// (det M_W_DWF)^{2 Nf_tx} = (det M_W_DWF)^4 at aux=0, matching Nf=4 DWF.
//
// Mirrors TXQCDWilsonRationalPseudoFermionAction.  Differences from the
// Wilson version:
//   * 5D fermion fields on FGrid.
//   * Aux-force builders sum over the 5th-dim slices and scale by (b+c)
//     (same FlavorBilinear5D / ColorBilinearSpinOp5D used by
//     TXQCDMobiusPseudoFermionAction).
//   * Gauge force via stock MobiusFermion::MDeriv.
//
// Math:
//   refresh: eta ~ N(0, 1/2);  Phi = (M^dag M)^{1/4} eta.
//   S(U):    Y = (M^dag M)^{-1/4} Phi;  S = ||Y||^2.
//   deriv:   dS = sum_k a_k [ MDeriv(Y_k, X_k, DaggerNo)
//                            + MDeriv(X_k, Y_k, DaggerYes) ]   (gauge)
//            + (b+c) * sum_k a_k * (4D-aux Hermitian-force builders on Y_k, X_k)
//            with X_k = (M^dag M + poles_k)^{-1} Phi, Y_k = M X_k.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMultiShiftCG.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusPseudoFermionAction.h>

NAMESPACE_BEGIN(Grid);

class TXQCDMobiusRationalPseudoFermionAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  TXQCDMobiusRationalPseudoFermionAction(GridCartesian &FGrid,
                                          GridRedBlackCartesian &FrbGrid,
                                          GridCartesian &UGrid,
                                          GridRedBlackCartesian &UrbGrid,
                                          RealD mass, RealD M5,
                                          RealD b, RealD c, Params &p)
      : FGrid_(FGrid), FrbGrid_(FrbGrid), UGrid_(UGrid), UrbGrid_(UrbGrid),
        mass_(mass), M5_(M5), b_(b), c_(c), bplusc_(b + c),
        Ls_(FGrid.GlobalDimensions()[0]),
        param(p), Phi(&FGrid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage
              << "[TXQCDMobiusRational] generating degree " << param.degree
              << " rational approximation to x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage
              << "[TXQCDMobiusRational] generating degree " << param.degree
              << " rational approximation to x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  std::string action_name() override {
    return "TXQCDMobiusRationalPseudoFermionAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " M5=" << M5_ << " b=" << b_ << " c=" << c_ << " Ls=" << Ls_
       << " lo=" << param.lo << " hi=" << param.hi
       << " degree=" << param.degree << " tol=" << param.tolerance
       << " MaxIter=" << param.MaxIter << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    TXQCDFermionNf eta(&FGrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, eta.f[a]);
      eta.f[a] = scale * eta.f[a];
    }
    TXQCDMobiusOp Mop = MakeOp(U);
    // Phi = (M^dag M)^{1/4} eta
    ApplyRational(Mop, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    TXQCDMobiusOp Mop = MakeOp(U);
    TXQCDFermionNf Y(&FGrid_);
    // Y = (M^dag M)^{-1/4} Phi ;  S = ||Y||^2
    ApplyRational(Mop, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    TXQCDMobiusOp Mop = MakeOp(U);
    const int Npole = static_cast<int>(PowerNegHalf.poles.size());

    // X_k = (M^dag M + poles_k)^{-1} Phi
    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&FGrid_);
    std::vector<RealD> md_tol(Npole, param.mdtolerance);
    TXQCDMultiShiftCG(Mop, PowerNegHalf.poles, md_tol, Phi, Xk, param.MaxIter);

    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    dSdU.U     = Zero();

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};

    Mop.Mobius().ImportGauge(U.U);
    LatticeGaugeField gtmp(&UGrid_);
    LatticeGaugeField gforce(&UGrid_);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];
      TXQCDFermionNf Y(&FGrid_);
      Mop.M(X, Y);

      // ----- aux forces (×(b+c) ak), 5D-summed builders -----
      // sigma
      {
        auto G = FlavorBilinear5D(Y, X, &UGrid_, Ls_);
        LatticeSigmaField F = HermitianFlavorForce(G);
        dSdU.sigma = dSdU.sigma + (bplusc_ * ak) * F;
      }
      // pi
      {
        auto G = FlavorBilinearG5_5D(Y, X, &UGrid_, Ls_);
        LatticePiField F = HermitianFlavorForce(G);
        dSdU.pi = dSdU.pi + (bplusc_ * ak) * F;
      }
      // s
      {
        auto G = ColorBilinearSpinOp5D(Y, X, Id, &UGrid_, Ls_);
        G = inv_sqrt2 * G;
        LatticeSFieldC F = HermitianColorForce(G);
        dSdU.s = dSdU.s + (bplusc_ * ak) * F;
      }
      // p
      {
        auto G = ColorBilinearSpinOp5D(Y, X, G5, &UGrid_, Ls_);
        G = inv_sqrt2 * G;
        LatticePFieldC F = HermitianColorForce(G);
        dSdU.p = dSdU.p + (bplusc_ * ak) * F;
      }
      // t_{mu,nu}
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          SpinTable iSig{ISigmaMatrix(mu, nu)};
          auto Gt = ColorBilinearSpinOp5D(Y, X, iSig, &UGrid_, Ls_);
          LatticeSFieldC Ft = HermitianColorForce(Gt);
          autoView(dst, dSdU.t, CpuWrite);
          autoView(src, Ft, CpuRead);
          const RealD wt = bplusc_ * ak;
          thread_for(ss, UGrid_.oSites(), {
            for (int i = 0; i < Nc; ++i) {
              for (int j = 0; j < Nc; ++j) {
                dst[ss]()(mu, nu)(i, j) =
                    dst[ss]()(mu, nu)(i, j) + wt * src[ss]()()(i, j);
                dst[ss]()(nu, mu)(i, j) =
                    dst[ss]()(nu, mu)(i, j) - wt * src[ss]()()(i, j);
              }
            }
          });
        }
      }

      // ----- gauge force -----
      gforce = Zero();
      for (int a = 0; a < TxqcdNf; ++a) {
        Mop.Mobius().MDeriv(gtmp, Y.f[a], X.f[a], DaggerNo);
        gforce = gforce + gtmp;
        Mop.Mobius().MDeriv(gtmp, X.f[a], Y.f[a], DaggerYes);
        gforce = gforce + gtmp;
      }
      dSdU.U = dSdU.U + ak * gforce;
    }
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDMobiusOp MakeOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDMobiusOp(Unc.U, FGrid_, FrbGrid_, UGrid_, UrbGrid_,
                         mass_, M5_, b_, c_,
                         Unc.sigma, Unc.pi, Unc.s, Unc.p, Unc.t);
  }

  // out = rat.norm * in + sum_k rat.residues_k * (M^dag M + rat.poles_k)^{-1} in
  void ApplyRational(TXQCDMobiusOp &Mop, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&FGrid_);
    TXQCDMultiShiftCG(Mop, rat.poles, rat.tolerances, in, xk, param.MaxIter);

    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = rat.norm * in.f[a];
    for (int k = 0; k < nshift; ++k) {
      RealD ck = rat.residues[k];
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + ck * xk[k].f[a];
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
