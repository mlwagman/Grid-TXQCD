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
// Only even-site aux fields contribute; odd-site and gauge forces are zero.

#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>

NAMESPACE_BEGIN(Grid);

class TXQCDLogDetEOAction : public Action<TXQCDField> {
 public:
  typedef TXQCDSiteMatrixUtil SMU;
  static constexpr int kDim = SMU::kDim;

  TXQCDLogDetEOAction(GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                      RealD mass)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), diag_mass_(4.0 + mass) {}

  std::string action_name() override { return "TXQCDLogDetEOAction"; }
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
    uint64_t nsites = aux.sig.size();

    SMU::SiteMatrix M;
    RealD logdet = 0.0;
    for (uint64_t x = 0; x < nsites; ++x) {
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x], M);
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
    uint64_t nsites = aux.sig.size();

    SMU::SiteMatrix M, Inv;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    std::vector<SMU::SigSobj> sig_force(nsites);
    std::vector<SMU::PiSobj> pi_force(nsites);
    std::vector<SMU::SSobj> s_force(nsites);
    std::vector<SMU::PSobj> p_force(nsites);
    std::vector<SMU::TSobj> t_force(nsites);

    for (uint64_t x = 0; x < nsites; ++x) {
      SMU::BuildSiteMatrix(sm_, diag_mass_, aux.sig[x], aux.pi[x],
                          aux.s[x], aux.p[x], aux.t[x], M);
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
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD diag_mass_;
  SMU::SpinMatrices sm_;

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
};

NAMESPACE_END(Grid);
