#pragma once
// EO-preconditioned RHMC pseudofermion action for TXQCD Wilson:
//
//   S = Phi^dag f(Mpc^dag Mpc) Phi,  Phi on odd sublattice (half volume).
//
// Mpc = Moo - Moe Mee^{-1} Meo is the Schur complement of the TXQCD Wilson
// operator (TXQCDWilsonFermionEO). CG and multi-shift CG operate on the
// half-volume odd sublattice, halving the linear system size vs full-grid.
//
// Force has three pieces per rational pole:
//   1. Aux force on odd sites: bilinear(Y_k, X_k) [same as full-grid]
//   2. Aux force on even sites: bilinear(Z_e, W_e) [chain rule through Mee^{-1}]
//   3. Gauge force: MoeDeriv/MeoDeriv per flavor [SchurDifferentiableOperator pattern]
//
// where X_k = (Mpc†Mpc + σ_k)^{-1} Phi, Y_k = Mpc X_k,
//       W_e = Mee^{-1} Meo X_k, Z_e = Mee^{-1}† Moe† Y_k.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonRationalEOAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  // mu rescales Delta in the inner TXQCDWilsonFermionEO operator and the
  // aux-field forces.  Default mu=1 reproduces the original action.
  TXQCDWilsonRationalEOAction(GridCartesian &grid,
                              GridRedBlackCartesian &rbgrid,
                              RealD mass, Params &p, RealD mu = 1.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), mu_(mu),
        param(p), Phi(&rbgrid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage
              << "[TXQCDWilsonRationalEO] degree " << param.degree
              << " rational for x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage
              << "[TXQCDWilsonRationalEO] degree " << param.degree
              << " rational for x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  std::string action_name() override {
    return "TXQCDWilsonRationalEOAction";
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
    TXQCDFermionNf eta(&rbgrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, eta.f[a]);
      eta.f[a] = scale * eta.f[a];
      eta.f[a].Checkerboard() = Odd;
    }
    auto EOp = MakeEOp(U);
    TXQCDSchurOp SchurOp(EOp);
    ApplyRational(SchurOp, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    auto EOp = MakeEOp(U);
    TXQCDSchurOp SchurOp(EOp);
    TXQCDFermionNf Y(&rbgrid_);
    ApplyRational(SchurOp, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto EOp = MakeEOp(U);
    TXQCDSchurOp SchurOp(EOp);
    const int Npole = static_cast<int>(PowerNegHalf.poles.size());

    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&rbgrid_);
    std::vector<RealD> md_tol(Npole, param.mdtolerance);

    TXQCDMultiShiftCGSchur MSCG(param.MaxIter);
    MSCG(SchurOp, PowerNegHalf.poles, md_tol, Phi, Xk);

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

    LatticeGaugeField gforce(&grid_);
    LatticeGaugeField gtmp(&grid_);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];
      TXQCDFermionNf Y(&rbgrid_);
      SchurOp.Mpc(X, Y);

      // Intermediates for even-site aux force and gauge force.
      TXQCDFermionNf W_e(&rbgrid_), Z_e(&rbgrid_);
      TXQCDFermionNf tmp_e(&rbgrid_);
      EOp.Meooe(X, tmp_e);        // Meo X_k: odd → even
      EOp.MooeeInv(tmp_e, W_e);   // Mee^{-1} Meo X_k
      EOp.MeooeDag(Y, tmp_e);     // Moe† Y_k: odd → even
      EOp.MooeeInvDag(tmp_e, Z_e); // Mee^{-1}† Moe† Y_k

      // ---- Aux-field forces ----

      // Chain-rule mu factor on aux forces: dM/daux = mu * dDelta/daux.
      const RealD ak_aux = ak * mu_;

      // Odd-site aux force: bilinear(Y, X) on odd grid.
      AccumulateAuxForce(dSdU, ak_aux, Y, X, g5, inv_sqrt2, Id, G5);

      // Even-site aux force: bilinear(Z_e, W_e) on even grid.
      AccumulateAuxForce(dSdU, ak_aux, Z_e, W_e, g5, inv_sqrt2, Id, G5);

      // ---- Gauge force ----
      // MpcDeriv(Y, X) + MpcDagDeriv(X, Y), per flavor.
      // Each returns -(ForceO + ForceE), accumulated with factor ak.
      gforce = Zero();

      GridRedBlackCartesian *forcecb =
          new GridRedBlackCartesian(&grid_);
      LatticeGaugeField ForceO(forcecb), ForceE(forcecb);

      for (int a = 0; a < TxqcdNf; ++a) {
        // MpcDeriv(Y.f[a], X.f[a]):
        //   ForceO = MoeDeriv(Y, W_e, No)  [odd links]
        //   ForceE = MeoDeriv(Z_e, X, No)  [even links]
        EOp.Wilson().MoeDeriv(ForceO, Y.f[a], W_e.f[a], DaggerNo);
        EOp.Wilson().MeoDeriv(ForceE, Z_e.f[a], X.f[a], DaggerNo);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;

        // MpcDagDeriv(X.f[a], Y.f[a]):
        //   ForceO = MoeDeriv(X, Z_e, Yes) [odd links]
        //   ForceE = MeoDeriv(W_e, Y, Yes) [even links]
        EOp.Wilson().MoeDeriv(ForceO, X.f[a], Z_e.f[a], DaggerYes);
        EOp.Wilson().MeoDeriv(ForceE, W_e.f[a], Y.f[a], DaggerYes);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;
      }
      dSdU.U = dSdU.U + ak * gforce;
      delete forcecb;
    }
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDWilsonFermionEO MakeEOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonFermionEO(
        Unc.U, grid_, rbgrid_, mass_, Unc.sigma, Unc.pi, Unc.s, Unc.p, Unc.t,
        TXQCDWilsonFermionEO::DefaultImplParams(), mu_);
  }

  void ApplyRational(TXQCDSchurOp &SchurOp, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&rbgrid_);
    TXQCDMultiShiftCGSchur MSCG(param.MaxIter);
    MSCG(SchurOp, rat.poles, rat.tolerances, in, xk);

    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = rat.norm * in.f[a];
    for (int k = 0; k < nshift; ++k) {
      RealD c = rat.residues[k];
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + c * xk[k].f[a];
    }
  }

  // Accumulate aux-field force from a bilinear pair (Y, X) on any CB grid.
  void AccumulateAuxForce(TXQCDField &dSdU, RealD ak,
                          const TXQCDFermionNf &Y, const TXQCDFermionNf &X,
                          const Gamma &g5, ComplexD inv_sqrt2,
                          const SpinTable &Id, const SpinTable &G5) {
    GridBase *grid = Y.Grid();
    int cb = Y.f[0].Checkerboard();

    // sigma
    {
      auto G = FlavorBilinear(Y, X);
      G.Checkerboard() = cb;
      LatticeSigmaField F = HermitianFlavorForce(G);
      F.Checkerboard() = cb;
      LatticeSigmaField tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.sigma = dSdU.sigma + ak * tmp;
    }
    // pi
    {
      TXQCDFermionNf g5X(grid);
      for (int a = 0; a < TxqcdNf; ++a) {
        g5X.f[a] = g5 * X.f[a];
        g5X.f[a].Checkerboard() = cb;
      }
      auto G = FlavorBilinear(Y, g5X);
      G.Checkerboard() = cb;
      LatticePiField F = HermitianFlavorForce(G);
      F.Checkerboard() = cb;
      LatticePiField tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.pi = dSdU.pi + ak * tmp;
    }
    // s
    {
      auto G = ColorBilinearSpinOp(Y, X, Id);
      G.Checkerboard() = cb;
      G = inv_sqrt2 * G;
      LatticeSFieldC F = HermitianColorForce(G);
      F.Checkerboard() = cb;
      LatticeSFieldC tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.s = dSdU.s + ak * tmp;
    }
    // p
    {
      auto G = ColorBilinearSpinOp(Y, X, G5);
      G.Checkerboard() = cb;
      G = inv_sqrt2 * G;
      LatticePFieldC F = HermitianColorForce(G);
      F.Checkerboard() = cb;
      LatticePFieldC tmp(&grid_);
      tmp = Zero();
      setCheckerboard(tmp, F);
      dSdU.p = dSdU.p + ak * tmp;
    }
    // t_{mu,nu}
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        SpinTable iSig{ISigmaMatrix(mu, nu)};
        auto Gt = ColorBilinearSpinOp(Y, X, iSig);
        Gt.Checkerboard() = cb;
        LatticeSFieldC Ft = HermitianColorForce(Gt);
        Ft.Checkerboard() = cb;
        LatticeSFieldC Ftfull(&grid_);
        Ftfull = Zero();
        setCheckerboard(Ftfull, Ft);
        autoView(dst, dSdU.t, CpuWrite);
        autoView(src, Ftfull, CpuRead);
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
  }

  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD mu_;
  Params &param;
  MultiShiftFunction PowerHalf;
  MultiShiftFunction PowerNegHalf;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
