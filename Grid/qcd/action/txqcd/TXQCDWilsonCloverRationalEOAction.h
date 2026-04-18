#pragma once
// EO-preconditioned RHMC pseudofermion action for TXQCD Wilson:
//
//   S = Phi^dag f(Mpc^dag Mpc) Phi,  Phi on odd sublattice (half volume).
//
// Mpc = Moo - Moe Mee^{-1} Meo is the Schur complement of the TXQCD Wilson
// operator (TXQCDWilsonCloverFermionEO). CG and multi-shift CG operate on the
// half-volume odd sublattice, halving the linear system size vs full-grid.
//
// Force has three pieces per rational pole:
//   1. Aux force on odd sites: bilinear(Y_k, X_k) [same as full-grid]
//   2. Aux force on even sites: bilinear(Z_e, W_e) [chain rule through Mee^{-1}]
//   3. Gauge force: hopping (MoeDeriv/MeoDeriv) + clover (Cmunu staples, csw != 0)
//
// where X_k = (Mpc†Mpc + σ_k)^{-1} Phi, Y_k = Mpc X_k,
//       W_e = Mee^{-1} Meo X_k, Z_e = Mee^{-1}† Moe† Y_k.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonCloverRationalEOAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  TXQCDWilsonCloverRationalEOAction(GridCartesian &grid,
                              GridRedBlackCartesian &rbgrid,
                              RealD mass, Params &p, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
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
    return "TXQCDWilsonCloverRationalEOAction";
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
    TXQCDCloverSchurOp SchurOp(EOp);
    ApplyRational(SchurOp, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    auto EOp = MakeEOp(U);
    TXQCDCloverSchurOp SchurOp(EOp);
    TXQCDFermionNf Y(&rbgrid_);
    ApplyRational(SchurOp, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto EOp = MakeEOp(U);
    TXQCDCloverSchurOp SchurOp(EOp);
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

      // Odd-site aux force: bilinear(Y, X) on odd grid.
      AccumulateAuxForce(dSdU, ak, Y, X, g5, inv_sqrt2, Id, G5);

      // Even-site aux force: bilinear(Z_e, W_e) on even grid.
      AccumulateAuxForce(dSdU, ak, Z_e, W_e, g5, inv_sqrt2, Id, G5);

      // ---- Gauge force (hopping) ----
      gforce = Zero();

      LatticeGaugeField ForceO(&rbgrid_), ForceE(&rbgrid_);

      for (int a = 0; a < TxqcdNf; ++a) {
        EOp.Wilson().MoeDeriv(ForceO, Y.f[a], W_e.f[a], DaggerNo);
        EOp.Wilson().MeoDeriv(ForceE, Z_e.f[a], X.f[a], DaggerNo);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;

        EOp.Wilson().MoeDeriv(ForceO, X.f[a], Z_e.f[a], DaggerYes);
        EOp.Wilson().MeoDeriv(ForceE, W_e.f[a], Y.f[a], DaggerYes);
        setCheckerboard(gtmp, ForceO);
        setCheckerboard(gtmp, ForceE);
        gforce = gforce - gtmp;
      }

      // ---- Clover gauge force ----
      if (csw_ != 0.0) {
        typedef TXQCDSiteMatrixUtil SMU;
        typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
        typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
        SMU::SpinMatrices sm;

        // Unvectorize fermion fields to per-site scalar objects
        std::array<std::vector<Fsobj>, TxqcdNf> Xv, Yv, Wv, Zv;
        for (int a = 0; a < TxqcdNf; ++a) {
          unvectorizeToLexOrdArray(Xv[a], X.f[a]);
          unvectorizeToLexOrdArray(Yv[a], Y.f[a]);
          unvectorizeToLexOrdArray(Wv[a], W_e.f[a]);
          unvectorizeToLexOrdArray(Zv[a], Z_e.f[a]);
        }
        uint64_t nsites_odd = Xv[0].size();
        uint64_t nsites_even = Wv[0].size();

        std::vector<LatticeColourMatrix> Sigma_full;
        int fk = 0;
        for (int rho = 0; rho < Nd; ++rho) {
          for (int sig = rho + 1; sig < Nd; ++sig) {
            // Odd-site sigma: sigma_1†-sigma_1 from 2Re[Y†(dM/dF)X]
            // = -(i*csw/2) Σ isigma(α,β)[Y*(α,j)X(β,i)+X*(α,j)Y(β,i)]
            std::vector<CMsobj> sig_odd(nsites_odd);
            for (uint64_t x = 0; x < nsites_odd; ++x) {
              for (int i = 0; i < Nc; ++i)
                for (int j = 0; j < Nc; ++j) {
                  std::complex<double> val(0,0);
                  for (int a = 0; a < TxqcdNf; ++a)
                    for (int alpha = 0; alpha < Ns; ++alpha)
                      for (int beta = 0; beta < Ns; ++beta) {
                        auto isig = sm.isigma[rho][sig](alpha, beta);
                        if (isig == std::complex<double>(0,0)) continue;
                        std::complex<double> Yaj(
                            Yv[a][x]()(alpha)(j).real(),
                            Yv[a][x]()(alpha)(j).imag());
                        std::complex<double> Xbi(
                            Xv[a][x]()(beta)(i).real(),
                            Xv[a][x]()(beta)(i).imag());
                        std::complex<double> Xaj(
                            Xv[a][x]()(alpha)(j).real(),
                            Xv[a][x]()(alpha)(j).imag());
                        std::complex<double> Ybi(
                            Yv[a][x]()(beta)(i).real(),
                            Yv[a][x]()(beta)(i).imag());
                        val += isig * (std::conj(Yaj)*Xbi + std::conj(Xaj)*Ybi);
                      }
                  std::complex<double> cv(0.0, -0.5*csw_);
                  std::complex<double> cval = cv * val;
                  sig_odd[x]()()(i,j) = ComplexD(cval.real(), cval.imag());
                }
            }

            // Even-site sigma (same structure with Z,W)
            std::vector<CMsobj> sig_even(nsites_even);
            for (uint64_t x = 0; x < nsites_even; ++x) {
              for (int i = 0; i < Nc; ++i)
                for (int j = 0; j < Nc; ++j) {
                  std::complex<double> val(0,0);
                  for (int a = 0; a < TxqcdNf; ++a)
                    for (int alpha = 0; alpha < Ns; ++alpha)
                      for (int beta = 0; beta < Ns; ++beta) {
                        auto isig = sm.isigma[rho][sig](alpha, beta);
                        if (isig == std::complex<double>(0,0)) continue;
                        std::complex<double> Zaj(
                            Zv[a][x]()(alpha)(j).real(),
                            Zv[a][x]()(alpha)(j).imag());
                        std::complex<double> Wbi(
                            Wv[a][x]()(beta)(i).real(),
                            Wv[a][x]()(beta)(i).imag());
                        std::complex<double> Waj(
                            Wv[a][x]()(alpha)(j).real(),
                            Wv[a][x]()(alpha)(j).imag());
                        std::complex<double> Zbi(
                            Zv[a][x]()(beta)(i).real(),
                            Zv[a][x]()(beta)(i).imag());
                        val += isig * (std::conj(Zaj)*Wbi + std::conj(Waj)*Zbi);
                      }
                  std::complex<double> cv(0.0, -0.5*csw_);
                  std::complex<double> cval = cv * val;
                  sig_even[x]()()(i,j) = ComplexD(cval.real(), cval.imag());
                }
            }

            LatticeColourMatrix lam_odd(&rbgrid_);
            vectorizeFromLexOrdArray(sig_odd, lam_odd);
            lam_odd.Checkerboard() = Odd;
            LatticeColourMatrix lam_even(&rbgrid_);
            vectorizeFromLexOrdArray(sig_even, lam_even);
            lam_even.Checkerboard() = Even;

            Sigma_full.emplace_back(&grid_);
            Sigma_full.back() = Zero();
            setCheckerboard(Sigma_full[fk], lam_odd);
            setCheckerboard(Sigma_full[fk], lam_even);
            ++fk;
          }
        }

        std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
        for (int mu = 0; mu < Nd; ++mu)
          Ulinks[mu] = PeekIndex<LorentzIndex>(EOp.Gauge(), mu);

        LatticeGaugeField clover_gforce(&grid_);
        clover_gforce = Zero();
        for (int mu = 0; mu < Nd; ++mu) {
          LatticeColourMatrix force_mu(&grid_);
          force_mu = Zero();
          for (int nu = 0; nu < Nd; ++nu) {
            if (mu == nu) continue;
            int mn = (mu < nu) ? SMU::FmnIndex(mu, nu)
                               : SMU::FmnIndex(nu, mu);
            LatticeColourMatrix lam = (mu < nu)
                ? Sigma_full[mn] : (-1.0) * Sigma_full[mn];
            force_mu = force_mu + 0.25 *
                WilsonCloverHelpers<WilsonImplR>::Cmunu(
                    Ulinks, lam, mu, nu);
          }
          pokeLorentz(clover_gforce, Ulinks[mu] * force_mu, mu);
        }
        // Cmunu-based force is Convention B; hopping force from MoeDeriv is
        // Convention A. Convert clover to Convention A by multiplying by -1/2.
        gforce = gforce + (-0.5) * clover_gforce;
      }

      dSdU.U = dSdU.U + ak * gforce;
    }
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  TXQCDWilsonCloverFermionEO MakeEOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonCloverFermionEO(Unc.U, grid_, rbgrid_, mass_, Unc.sigma,
                                Unc.pi, Unc.s, Unc.p, Unc.t, csw_);
  }

  void ApplyRational(TXQCDCloverSchurOp &SchurOp, const MultiShiftFunction &rat,
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
  RealD csw_;
  Params &param;
  MultiShiftFunction PowerHalf;
  MultiShiftFunction PowerNegHalf;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
