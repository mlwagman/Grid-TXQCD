#pragma once
// Non-EO RHMC pseudofermion action for TXQCD Wilson-clover:
//
//   S = Phi^dag f(M^dag M) Phi,  Phi on the FULL grid (no Schur preconditioning).
//
// Motivation: the EO-Schur variant (TXQCDWilsonCloverRationalEOAction) suffers
// from a low-lambda cliff that is a preconditioning artifact, not a physical
// instability — see appendix_eo_cliff.tex.  As lambda decreases the random
// tensor insertion drives some M_ee eigenvalues toward zero, which forces a
// compensating divergence of M_pc = M_oo - M_oe M_ee^{-1} M_eo.  The product
// det M = det M_ee . det M_pc is smooth, so the full operator stays
// well-conditioned across the entire production lambda range.  EO solvers only
// see M_pc and report a catastrophe; this action avoids the Schur factorisation
// entirely.
//
// Structure mirrors TXQCDWilsonRationalPseudoFermionAction (non-EO Wilson,
// no clover) and TXQCDWilsonCloverRationalEOAction (EO clover) — combining
// the full-grid PF layout of the former with the clover gauge-force kernel of
// the latter, with the EO-specific Mee^{-1} chain-rule plumbing removed.
//
// Per rational pole k:
//   X_k = (M^dag M + sigma_k)^{-1} Phi              [full-volume multishift CG]
//   Y_k = M X_k
//   dS  += a_k * [ aux-bilinear(Y_k, X_k)
//                  + Wilson MDeriv per flavor
//                  + (csw != 0) clover Cmunu force ]
//
// No even-site W_e = Mee^{-1} Meo X_k / Z_e = Mee^{-1,dag} Moe^dag Y_k
// completions: both Y and X live on the full grid in a single pass.

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMultiShiftCG.h>
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonCloverRationalAction : public Action<TXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;

  // Per-flavor mass constructor.
  TXQCDWilsonCloverRationalAction(GridCartesian &grid,
                                  GridRedBlackCartesian &rbgrid,
                                  const std::array<RealD, TxqcdNf> &mass,
                                  Params &p, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
        param(p), Phi(&grid) {
    AlgRemez remez(param.lo, param.hi, param.precision);
    std::cout << GridLogMessage
              << "[TXQCDWilsonCloverRational] degree " << param.degree
              << " rational for x^(1/2)" << std::endl;
    remez.generateApprox(param.degree, 1, 2);
    PowerHalf.Init(remez, param.tolerance, false);
    PowerNegHalf.Init(remez, param.tolerance, true);
    std::cout << GridLogMessage
              << "[TXQCDWilsonCloverRational] degree " << param.degree
              << " rational for x^(1/4)" << std::endl;
    remez.generateApprox(param.degree, 1, 4);
    PowerQuarter.Init(remez, param.tolerance, false);
    PowerNegQuarter.Init(remez, param.tolerance, true);
  }

  // Backward-compat: degenerate scalar mass.
  TXQCDWilsonCloverRationalAction(GridCartesian &grid,
                                  GridRedBlackCartesian &rbgrid,
                                  RealD mass, Params &p, RealD csw = 0.0)
      : TXQCDWilsonCloverRationalAction(grid, rbgrid,
            TXQCDSiteMatrixUtil::MassArray(mass), p, csw) {}

  std::string action_name() override {
    return "TXQCDWilsonCloverRationalAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=";
    for (int a = 0; a < TxqcdNf; ++a)
      os << (a ? "," : "{") << mass_[a];
    os << "} lo=" << param.lo << " hi=" << param.hi
       << " degree=" << param.degree << " tol=" << param.tolerance
       << " MaxIter=" << param.MaxIter << " csw=" << csw_ << std::endl;
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
    auto Mop = MakeOp(U);
    // Phi = (M^dag M)^{1/4} eta
    ApplyRational(Mop, PowerQuarter, eta, Phi);
  }

  RealD S(const TXQCDField &U) override {
    auto Mop = MakeOp(U);
    TXQCDFermionNf Y(&grid_);
    // Y = (M^dag M)^{-1/4} Phi ;  S = Phi^dag (M^dag M)^{-1/2} Phi = |Y|^2
    ApplyRational(Mop, PowerNegQuarter, Phi, Y);
    RealD action = norm2(Y);
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto Mop = MakeOp(U);
    const int Npole = static_cast<int>(PowerNegHalf.poles.size());

    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&grid_);
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
    Gamma g5(Gamma::Algebra::Gamma5);

    Mop.Wilson().ImportGauge(U.U);
    LatticeGaugeField gtmp(&grid_);
    LatticeGaugeField gforce(&grid_);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];
      TXQCDFermionNf Y(&grid_);
      Mop.M(X, Y);

      // ---- Aux-field forces (full-grid bilinear, no Schur completion) ----
      AccumulateAuxForce(dSdU, ak, Y, X, g5, inv_sqrt2, Id, G5);

      // ---- Gauge hopping force (per-flavor Wilson MDeriv) ----
      gforce = Zero();
      for (int a = 0; a < TxqcdNf; ++a) {
        Mop.Wilson().MDeriv(gtmp, Y.f[a], X.f[a], DaggerNo);
        gforce = gforce + gtmp;
        Mop.Wilson().MDeriv(gtmp, X.f[a], Y.f[a], DaggerYes);
        gforce = gforce + gtmp;
      }

      // ---- Clover gauge force (csw != 0) ----
      if (csw_ != 0.0) {
        AccumulateCloverGaugeForce(gforce, U.U, Y, X);
      }

      dSdU.U = dSdU.U + ak * gforce;
    }
  }

  TXQCDFermionNf &PseudoFermion() { return Phi; }

 protected:
  TXQCDWilsonCloverOp MakeOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonCloverOp(Unc.U, grid_, rbgrid_, mass_, Unc.sigma, Unc.pi,
                               Unc.s, Unc.p, Unc.t, csw_);
  }

  // Apply a rational function f(M^dag M) given by a MultiShiftFunction:
  //   out = rat.norm * in + sum_k rat.residues_k * (M^dag M + rat.poles_k)^{-1} in
  void ApplyRational(TXQCDWilsonCloverOp &Mop, const MultiShiftFunction &rat,
                     const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<TXQCDFermionNf> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&grid_);
    TXQCDMultiShiftCG(Mop, rat.poles, rat.tolerances, in, xk, param.MaxIter);

    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = rat.norm * in.f[a];
    for (int k = 0; k < nshift; ++k) {
      RealD c = rat.residues[k];
      for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + c * xk[k].f[a];
    }
  }

  // Accumulate aux-field force from a bilinear pair (Y, X) on the FULL grid.
  // No checkerboard handling needed; both Y and X are full-volume fields.
  // Mirrors TXQCDWilsonRationalPseudoFermionAction:142-189 but with the
  // pre-existing FlavorBilinear/ColorBilinearSpinOp helpers.  Direct lattice
  // arithmetic (lazy expression templates ⇒ accelerator_for under the hood)
  // for sigma/pi/s/p; per-pair thread_for for the antisymmetric t channel.
  // Step 2 QUDA subclass can override deriv() to replace the t loop.
  void AccumulateAuxForce(TXQCDField &dSdU, RealD ak,
                          const TXQCDFermionNf &Y, const TXQCDFermionNf &X,
                          const Gamma &g5, ComplexD inv_sqrt2,
                          const SpinTable &Id, const SpinTable &G5) {
    GridBase *grid = Y.f[0].Grid();
    // sigma
    {
      auto G = FlavorBilinear(Y, X);
      LatticeSigmaField F = HermitianFlavorForce(G);
      dSdU.sigma = dSdU.sigma + ak * F;
    }
    // pi
    {
      TXQCDFermionNf g5X(grid);
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
    // t_{mu,nu}: 6 (mu < nu) pairs.  Per-pair thread_for; antisymmetric write.
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        SpinTable iSig{ISigmaMatrix(mu, nu)};
        auto Gt = ColorBilinearSpinOp(Y, X, iSig);
        LatticeSFieldC Ft = HermitianColorForce(Gt);
        autoView(dst, dSdU.t, CpuWrite);
        autoView(src, Ft, CpuRead);
        thread_for(ss, grid->oSites(), {
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

  // Clover gauge force on the FULL grid (csw != 0), single-pass.  Builds the
  // 6 Sigma_full[fk] = -(i csw / 2) sum_{a,alpha,beta} isigma_{rho,sig}(alpha,beta) *
  //     (conj(Y_a(alpha,j)) X_a(beta,i) + conj(X_a(alpha,j)) Y_a(beta,i))
  // matrices, then contracts with the Cmunu staple per the EO version.
  // Result is Convention B; converted to Convention A by -1/2 (see
  // TXQCDWilsonCloverRationalEOAction:292-294).  Accumulates into gforce_full.
  void AccumulateCloverGaugeForce(LatticeGaugeField &gforce_full,
                                  const typename TXQCDWilsonCloverOp::GaugeField &Ufull,
                                  const TXQCDFermionNf &Y,
                                  const TXQCDFermionNf &X) {
    typedef TXQCDSiteMatrixUtil SMU;
    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    SMU::SpinMatrices sm;

    std::array<std::vector<Fsobj>, TxqcdNf> Xv, Yv;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(Xv[a], X.f[a]);
      unvectorizeToLexOrdArray(Yv[a], Y.f[a]);
    }
    uint64_t nsites = Xv[0].size();

    std::vector<LatticeColourMatrix> Sigma_full;
    int fk = 0;
    for (int rho = 0; rho < Nd; ++rho) {
      for (int sig = rho + 1; sig < Nd; ++sig) {
        std::vector<CMsobj> sig_lex(nsites);
        thread_for(x, nsites, {
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
              sig_lex[x]()()(i,j) = ComplexD(cval.real(), cval.imag());
            }
        });

        Sigma_full.emplace_back(&grid_);
        vectorizeFromLexOrdArray(sig_lex, Sigma_full.back());
        ++fk;
      }
    }

    std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
    for (int mu = 0; mu < Nd; ++mu)
      Ulinks[mu] = PeekIndex<LorentzIndex>(Ufull, mu);

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
            WilsonCloverHelpers<WilsonImplR>::Cmunu(Ulinks, lam, mu, nu);
      }
      pokeLorentz(clover_gforce, Ulinks[mu] * force_mu, mu);
    }
    // Cmunu-based force is Convention B; hopping force from MDeriv is
    // Convention A.  Convert clover to Convention A by multiplying by -1/2.
    gforce_full = gforce_full + (-0.5) * clover_gforce;
  }

  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  std::array<RealD, TxqcdNf> mass_;
  RealD csw_;
  Params &param;
  MultiShiftFunction PowerHalf;
  MultiShiftFunction PowerNegHalf;
  MultiShiftFunction PowerQuarter;
  MultiShiftFunction PowerNegQuarter;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
