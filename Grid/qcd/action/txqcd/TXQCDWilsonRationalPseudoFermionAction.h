#pragma once
// One-flavor rational pseudofermion action for TXQCD Wilson:
//
//   S = phi^dag (M^dag M)^{-1/2} phi,    M = D_W + Delta(aux),
//
// giving weight |det M| per PF (one physical flavor per PF). Contrast with
// TXQCDWilsonPseudoFermionAction which implements S = phi^dag (M^dag M)^{-1}
// phi and therefore carries |det M|^2 per PF. For a clean Nf=2 comparison we
// instantiate TWO TXQCDWilsonRationalPseudoFermionActions on one TXQCDField,
// giving |det M_TX|^2; since M_TX has an internal Nf_tx=2 flavor block the
// effective fermion determinant is (det M_W)^{2 Nf_tx} = (det M_W)^4 at aux=0,
// matching Nf=4 Wilson. Two such HMC levels running independent aux would
// correspond to "two light quarks" in the TXQCD design; here we keep a single
// shared aux + two PFs as the direct Nf=2_TXQCD vs. Nf=2 Wilson comparison.
//
// Structure mirrors Grid's OneFlavourRationalPseudoFermionAction
// (pseudofermion/OneFlavourRational.h) but sits on Action<TXQCDField> and
// uses the hand-rolled TXQCDMultiShiftCG instead of the stock
// ConjugateGradientMultiShift (we would otherwise need to wrap TXQCDFermionNf
// in a LinearOperatorBase).
//
// Math:
//
//   (M^dag M)^{-1/2} = norm + sum_k residues_k / (M^dag M + poles_k)
//
//   refresh: eta ~ N(0, 1/2); Phi = (M^dag M)^{1/4} eta, so that
//            <Phi (M^dag M)^{-1/2} Phi> = <eta eta> -> e^{-|eta|^2} sampling.
//
//   S(U):    Y = (M^dag M)^{-1/4} Phi ; S = |Y|^2.
//
//   deriv:   dS = sum_k a_k [MDeriv(Y_k, X_k, DaggerNo) +
//                            MDeriv(X_k, Y_k, DaggerYes)]
//            where a_k = residues_k of (M^dag M)^{-1/2},
//            X_k       = (M^dag M + poles_k)^{-1} Phi,
//            Y_k       = M X_k,
//            and the aux-force contraction builders are reused from
//            TXQCDWilsonPseudoFermionAction.h.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMultiShiftCG.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonRationalPseudoFermionAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  TXQCDWilsonRationalPseudoFermionAction(GridCartesian &grid,
                                         GridRedBlackCartesian &rbgrid,
                                         RealD mass, Params &p)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), param(p), Phi(&grid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage
              << "[TXQCDWilsonRational] generating degree " << param.degree
              << " rational approximation to x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage
              << "[TXQCDWilsonRational] generating degree " << param.degree
              << " rational approximation to x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  std::string action_name() override {
    return "TXQCDWilsonRationalPseudoFermionAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " lo=" << param.lo << " hi=" << param.hi
       << " degree=" << param.degree << " tol=" << param.tolerance
       << " MaxIter=" << param.MaxIter << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    TXQCDFermionNf eta(&grid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, eta.f[a]);
      eta.f[a] = scale * eta.f[a];
    }
    TXQCDWilsonOp Mop = MakeOp(U);
    // Phi = (M^dag M)^{1/4} eta
    ApplyRational(Mop, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    TXQCDWilsonOp Mop = MakeOp(U);
    TXQCDFermionNf Y(&grid_);
    // Y = (M^dag M)^{-1/4} Phi ;  S = Phi^dag (M^dag M)^{-1/2} Phi = |Y|^2
    ApplyRational(Mop, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    TXQCDWilsonOp Mop = MakeOp(U);
    const int Npole = static_cast<int>(PowerNegHalf.poles.size());

    // X_k = (M^dag M + poles_k)^{-1} Phi. Uses param.mdtolerance (default 1e-6)
    // for the MD force solve, NOT the tighter param.tolerance used in refresh/S.
    // Matches stock OneFlavourRationalRatio's MD-tolerance idiom.
    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&grid_);
    std::vector<RealD> md_tol(Npole, param.mdtolerance);
    TXQCDMultiShiftCG(Mop, PowerNegHalf.poles, md_tol, Phi, Xk, param.MaxIter);

    // Zero all force slots before accumulation
    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    dSdU.U     = Zero();

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};
    Gamma g5(Gamma::Algebra::Gamma5);

    Mop.Wilson().ImportGauge(U.U);
    LatticeGaugeField gtmp(&grid_);
    LatticeGaugeField gforce(&grid_);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];
      TXQCDFermionNf Y(&grid_);
      Mop.M(X, Y);

      // sigma
      {
        auto G = FlavorBilinear(Y, X);
        LatticeSigmaField F = HermitianFlavorForce(G);
        dSdU.sigma = dSdU.sigma + ak * F;
      }
      // pi
      {
        TXQCDFermionNf g5X(&grid_);
        for (int a = 0; a < TxqcdNf; ++a) g5X.f[a] = g5 * X.f[a];
        auto G = FlavorBilinear(Y, g5X);
        LatticePiField F = HermitianFlavorForce(G);
        dSdU.pi = dSdU.pi + ak * F;
      }
      // s
      {
        auto G = ColorBilinearSpinOp(Y, X, Id);
        G = inv_sqrt2 * G;
        LatticeSFieldC F = HermitianColorForce(G);
        dSdU.s = dSdU.s + ak * F;
      }
      // p
      {
        auto G = ColorBilinearSpinOp(Y, X, G5);
        G = inv_sqrt2 * G;
        LatticePFieldC F = HermitianColorForce(G);
        dSdU.p = dSdU.p + ak * F;
      }
      // t_{mu,nu}
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          SpinTable iSig{ISigmaMatrix(mu, nu)};
          auto Gt = ColorBilinearSpinOp(Y, X, iSig);
          LatticeSFieldC Ft = HermitianColorForce(Gt);
          autoView(dst, dSdU.t, CpuWrite);
          autoView(src, Ft, CpuRead);
          thread_for(ss, grid_.oSites(), {
            for (int i = 0; i < Nc; ++i) {
              for (int j = 0; j < Nc; ++j) {
                dst[ss]()(mu, nu)(i, j) =
                    dst[ss]()(mu, nu)(i, j) + ak * src[ss]()()(i, j);
                dst[ss]()(nu, mu)(i, j) =
                    dst[ss]()(nu, mu)(i, j) - ak * src[ss]()()(i, j);
              }
            }
          });
        }
      }

      // gauge force per flavor: ak * sum_a [MDeriv(Y_a, X_a, DaggerNo) +
      //                                     MDeriv(X_a, Y_a, DaggerYes)]
      gforce = Zero();
      for (int a = 0; a < TxqcdNf; ++a) {
        Mop.Wilson().MDeriv(gtmp, Y.f[a], X.f[a], DaggerNo);
        gforce = gforce + gtmp;
        Mop.Wilson().MDeriv(gtmp, X.f[a], Y.f[a], DaggerYes);
        gforce = gforce + gtmp;
      }
      dSdU.U = dSdU.U + ak * gforce;
    }
  }

  // Expose for tests (e.g. force-consistency checks).
  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDWilsonOp MakeOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonOp(Unc.U, grid_, rbgrid_, mass_, Unc.sigma, Unc.pi,
                         Unc.s, Unc.p, Unc.t);
  }

  // Apply a rational function f(M^dag M) given by a MultiShiftFunction:
  //   out = rat.norm * in + sum_k rat.residues_k * (M^dag M + rat.poles_k)^{-1} in
  void ApplyRational(TXQCDWilsonOp &Mop, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&grid_);
    TXQCDMultiShiftCG(Mop, rat.poles, rat.tolerances, in, xk, param.MaxIter);

    // out = rat.norm * in
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = rat.norm * in.f[a];
    for (int k = 0; k < nshift; ++k) {
      RealD c = rat.residues[k];
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + c * xk[k].f[a];
    }
  }

  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  Params &param;
  MultiShiftFunction PowerHalf;
  MultiShiftFunction PowerNegHalf;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
