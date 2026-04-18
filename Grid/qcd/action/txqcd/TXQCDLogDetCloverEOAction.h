#pragma once
// Log-determinant action for the even-site diagonal block in the
// EO-preconditioned TXQCD Wilson operator.
//
// det(M) = det(Mee) * det(Mpc), so when using Mpc for the pseudofermion
// action we need S_logdet = -ln det(Mee) as a separate action term.
//
// Since Mee = (4+m)I + Δ_even is site-diagonal with a 24×24 matrix per site,
// the determinant factorises:
//   ln det(Mee) = Σ_{x∈even} ln det(M_site(x))
//
// The force is:
//   dS/daux_even(x) = -Tr(M_site(x)^{-1} dM_site/daux(x))
//
// Only even-site aux fields contribute; odd-site force is zero.
// Gauge force is zero for csw=0; for csw!=0 the clover term couples
// Mee to the gauge links through F_{μν}, producing a gauge force via Cmunu.

#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>

NAMESPACE_BEGIN(Grid);

class TXQCDLogDetCloverEOAction : public Action<TXQCDField> {
 public:
  typedef TXQCDSiteMatrixUtil SMU;
  static constexpr int kDim = SMU::kDim;

  TXQCDLogDetCloverEOAction(GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                      RealD mass, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), diag_mass_(4.0 + mass),
        csw_(csw) {}

  std::string action_name() override { return "TXQCDLogDetCloverEOAction"; }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {}

  RealD S(const TXQCDField &U) override {
    auto aux = GetEvenAux(U);
    auto cl = GetEvenClover(U);
    uint64_t nsites = aux.sig.size();

    SMU::SiteMatrix M;
    RealD logdet = 0.0;
    for (uint64_t x = 0; x < nsites; ++x) {
      std::array<SMU::FmnSobj, 6> fmn_site;
      const std::array<SMU::FmnSobj, 6> *fmn_ptr = nullptr;
      if (csw_ != 0.0) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = cl.fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x],
                          csw_, fmn_ptr, M);
      auto lu = M.partialPivLu();
      auto d = lu.determinant();
      logdet += std::log(std::abs(d));
    }
    grid_.GlobalSum(logdet);
    RealD action = -logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto aux = GetEvenAux(U);
    auto cl = GetEvenClover(U);
    uint64_t nsites = aux.sig.size();

    SMU::SiteMatrix M, Inv;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double neg_csw_half = -0.5 * csw_;

    std::vector<SMU::SigSobj> sig_force(nsites);
    std::vector<SMU::PiSobj> pi_force(nsites);
    std::vector<SMU::SSobj> s_force(nsites);
    std::vector<SMU::PSobj> p_force(nsites);
    std::vector<SMU::TSobj> t_force(nsites);

    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;
    std::array<std::vector<CMsobj>, 6> clover_sigma;
    if (csw_ != 0.0)
      for (int k = 0; k < 6; ++k) clover_sigma[k].resize(nsites);

    for (uint64_t x = 0; x < nsites; ++x) {
      std::array<SMU::FmnSobj, 6> fmn_site;
      const std::array<SMU::FmnSobj, 6> *fmn_ptr = nullptr;
      if (csw_ != 0.0) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = cl.fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x],
                          csw_, fmn_ptr, M);
      Inv = M.inverse();

      // sigma force: F_{ab} = -Σ_{α,i} Inv_{(a,α,i),(b,α,i)}
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int b = 0; b < TxqcdNf; ++b) {
          std::complex<double> val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha)
            for (int i = 0; i < Nc; ++i) {
              int ra = a * Ns * Nc + alpha * Nc + i;
              int rb = b * Ns * Nc + alpha * Nc + i;
              val += Inv(ra, rb);
            }
          sig_force[x]()()(a, b) = ComplexD(-val.real(), -val.imag());
        }
      }

      // pi force: F_{ab} = -Σ_{α,β,i} γ₅(β,α) Inv_{(a,α,i),(b,β,i)}
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int b = 0; b < TxqcdNf; ++b) {
          std::complex<double> val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha)
            for (int beta = 0; beta < Ns; ++beta) {
              auto g5 = sm_.gamma5(beta, alpha);
              if (g5 == std::complex<double>(0, 0)) continue;
              for (int i = 0; i < Nc; ++i) {
                int ra = a * Ns * Nc + alpha * Nc + i;
                int rb = b * Ns * Nc + beta * Nc + i;
                val += g5 * Inv(ra, rb);
              }
            }
          pi_force[x]()()(a, b) = ComplexD(-val.real(), -val.imag());
        }
      }

      // s force: F_{ij} = -(1/√2) Σ_{a,α} Inv_{(a,α,i),(a,α,j)}
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> val(0, 0);
          for (int a = 0; a < TxqcdNf; ++a)
            for (int alpha = 0; alpha < Ns; ++alpha) {
              int ri = a * Ns * Nc + alpha * Nc + i;
              int rj = a * Ns * Nc + alpha * Nc + j;
              val += Inv(ri, rj);
            }
          s_force[x]()()(i, j) =
              ComplexD(-inv_sqrt2 * val.real(), -inv_sqrt2 * val.imag());
        }
      }

      // p force: F_{ij} = -(1/√2) Σ_{a,α,β} γ₅(β,α) Inv_{(a,α,i),(a,β,j)}
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> val(0, 0);
          for (int a = 0; a < TxqcdNf; ++a)
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                auto g5 = sm_.gamma5(beta, alpha);
                if (g5 == std::complex<double>(0, 0)) continue;
                int ri = a * Ns * Nc + alpha * Nc + i;
                int rj = a * Ns * Nc + beta * Nc + j;
                val += g5 * Inv(ri, rj);
              }
          p_force[x]()()(i, j) =
              ComplexD(-inv_sqrt2 * val.real(), -inv_sqrt2 * val.imag());
        }
      }

      // t force: F_{μν,ij} = -Σ_{a,α,β} (iσ_{μν})(β,α) Inv_{(a,α,i),(a,β,j)}
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = 0; nu < Nd; ++nu)
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j)
              t_force[x]()(mu, nu)(i, j) = ComplexD(0, 0);

      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              std::complex<double> val(0, 0);
              for (int a = 0; a < TxqcdNf; ++a)
                for (int alpha = 0; alpha < Ns; ++alpha)
                  for (int beta = 0; beta < Ns; ++beta) {
                    auto isig = sm_.isigma[mu][nu](beta, alpha);
                    if (isig == std::complex<double>(0, 0)) continue;
                    int ri = a * Ns * Nc + alpha * Nc + i;
                    int rj = a * Ns * Nc + beta * Nc + j;
                    val += isig * Inv(ri, rj);
                  }
              t_force[x]()(mu, nu)(i, j) =
                  ComplexD(-val.real(), -val.imag());
              t_force[x]()(nu, mu)(i, j) =
                  ComplexD(val.real(), val.imag());

              if (csw_ != 0.0) {
                // dS/dF_{mu,nu}^{ij} = -Tr(M^{-1} dM/dF)
                // dM/dF = i*(csw/2) * isigma => dS/dF = -i*(csw/2) * val
                int k = SMU::FmnIndex(mu, nu);
                std::complex<double> cv(0.0, -0.5 * csw_);
                std::complex<double> cval = cv * val;
                clover_sigma[k][x]()()(i, j) =
                    ComplexD(cval.real(), cval.imag());
              }
            }
          }
        }
      }
    }

    LatticeSigmaField F_sig_e(&rbgrid_);
    vectorizeFromLexOrdArray(sig_force, F_sig_e);
    F_sig_e.Checkerboard() = Even;

    LatticePiField F_pi_e(&rbgrid_);
    vectorizeFromLexOrdArray(pi_force, F_pi_e);
    F_pi_e.Checkerboard() = Even;

    LatticeSFieldC F_s_e(&rbgrid_);
    vectorizeFromLexOrdArray(s_force, F_s_e);
    F_s_e.Checkerboard() = Even;

    LatticePFieldC F_p_e(&rbgrid_);
    vectorizeFromLexOrdArray(p_force, F_p_e);
    F_p_e.Checkerboard() = Even;

    LatticeTField F_t_e(&rbgrid_);
    vectorizeFromLexOrdArray(t_force, F_t_e);
    F_t_e.Checkerboard() = Even;

    dSdU.sigma = Zero();
    dSdU.pi = Zero();
    dSdU.s = Zero();
    dSdU.p = Zero();
    dSdU.t = Zero();
    dSdU.U = Zero();

    setCheckerboard(dSdU.sigma, F_sig_e);
    setCheckerboard(dSdU.pi, F_pi_e);
    setCheckerboard(dSdU.s, F_s_e);
    setCheckerboard(dSdU.p, F_p_e);
    setCheckerboard(dSdU.t, F_t_e);

    if (csw_ != 0.0) {
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Sigma_full;
      for (int k = 0; k < 6; ++k) {
        LatticeColourMatrix Sigma_e(&rbgrid_);
        vectorizeFromLexOrdArray(clover_sigma[k], Sigma_e);
        Sigma_e.Checkerboard() = Even;
        Sigma_full.emplace_back(&grid_);
        Sigma_full.back() = Zero();
        setCheckerboard(Sigma_full[k], Sigma_e);
      }

      std::vector<LatticeColourMatrix> Ulinks(Nd, &grid_);
      for (int mu = 0; mu < Nd; ++mu)
        Ulinks[mu] = PeekIndex<LorentzIndex>(U.U, mu);

      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          int mn = (mu < nu) ? SMU::FmnIndex(mu, nu) : SMU::FmnIndex(nu, mu);
          LatticeColourMatrix lambda = (mu < nu) ? Sigma_full[mn]
                                                 : (-1.0) * Sigma_full[mn];
          force_mu += 0.25 *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, lambda, mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      // Cmunu-based force is "Convention B" (full gradient: dS/dh = Tr(E*F)).
      // The HMC integrator multiplies gauge forces by HMC_MOMENTUM_DENOMINATOR=2,
      // so deriv() must return "Convention A" (half gradient): multiply by -1/2.
      dSdU.U = (-0.5) * clover_force;
    }
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD diag_mass_;
  RealD csw_;
  SMU::SpinMatrices sm_;
  std::vector<LatticeColourMatrix> FS_;

  SMU::AuxSiteArrays GetEvenAux(const TXQCDField &U) {
    LatticeSigmaField sigma_e(&rbgrid_);
    LatticePiField pi_e(&rbgrid_);
    LatticeSFieldC s_e(&rbgrid_);
    LatticePFieldC p_e(&rbgrid_);
    LatticeTField t_e(&rbgrid_);
    pickCheckerboard(Even, sigma_e, U.sigma);
    pickCheckerboard(Even, pi_e, U.pi);
    pickCheckerboard(Even, s_e, U.s);
    pickCheckerboard(Even, p_e, U.p);
    pickCheckerboard(Even, t_e, U.t);
    return SMU::UnvectorizeAux(sigma_e, pi_e, s_e, p_e, t_e);
  }

  SMU::CloverSiteArrays GetEvenClover(const TXQCDField &U) {
    if (csw_ == 0.0) return SMU::CloverSiteArrays();
    FS_.clear();
    std::vector<LatticeColourMatrix> FS_e;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        FS_.emplace_back(&grid_);
        WilsonLoops<WilsonImplR>::FieldStrength(FS_.back(), U.U, mu, nu);
        FS_e.emplace_back(&rbgrid_);
        pickCheckerboard(Even, FS_e.back(), FS_.back());
      }
    return SMU::UnvectorizeClover(FS_e);
  }
};

NAMESPACE_END(Grid);
