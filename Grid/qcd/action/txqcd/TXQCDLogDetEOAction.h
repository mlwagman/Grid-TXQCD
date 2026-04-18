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

#include <Grid/qcd/action/txqcd/TXQCDWilsonFermionEO.h>

NAMESPACE_BEGIN(Grid);

class TXQCDLogDetEOAction : public Action<TXQCDField> {
 public:
  static constexpr int kDim = TxqcdNf * Ns * Nc;  // 24

  TXQCDLogDetEOAction(GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                      RealD mass)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), diag_mass_(4.0 + mass) {
    PrecomputeSpinMatrices();
  }

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
    auto sites = UnvectorizeEvenAux(U);
    uint64_t nsites = std::get<0>(sites).size();

    Eigen::Matrix<std::complex<double>, kDim, kDim> M;
    RealD logdet = 0.0;
    for (uint64_t x = 0; x < nsites; ++x) {
      BuildSiteMatrix(std::get<0>(sites)[x], std::get<1>(sites)[x],
                      std::get<2>(sites)[x], std::get<3>(sites)[x],
                      std::get<4>(sites)[x], M);
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
    auto sites = UnvectorizeEvenAux(U);
    uint64_t nsites = std::get<0>(sites).size();

    Eigen::Matrix<std::complex<double>, kDim, kDim> M, Inv;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

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

    std::vector<SigSobj> sig_force(nsites);
    std::vector<PiSobj> pi_force(nsites);
    std::vector<SSobj> s_force(nsites);
    std::vector<PSobj> p_force(nsites);
    std::vector<TSobj> t_force(nsites);

    for (uint64_t x = 0; x < nsites; ++x) {
      BuildSiteMatrix(std::get<0>(sites)[x], std::get<1>(sites)[x],
                      std::get<2>(sites)[x], std::get<3>(sites)[x],
                      std::get<4>(sites)[x], M);
      Inv = M.inverse();

      // sigma force: F_{ab} = -Σ_{α,i} Inv_{(a,α,i),(b,α,i)}
      // (note: stored as F_{ab} so that Tr(E*F) = dS/dE for Hermitian E)
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
              auto g5 = gamma5_mat_(beta, alpha);
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
                auto g5 = gamma5_mat_(beta, alpha);
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
      // and F_{νμ,ij} = -F_{μν,ij} (antisymmetry)
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
                    auto isig = isigma_mat_[mu][nu](beta, alpha);
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

    // Vectorize forces back to RB even fields and promote to full grid.
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

  Eigen::Matrix<std::complex<double>, Ns, Ns> gamma5_mat_;
  std::array<std::array<Eigen::Matrix<std::complex<double>, Ns, Ns>, Nd>, Nd>
      isigma_mat_;

  void PrecomputeSpinMatrices() {
    gamma5_mat_ = Eigen::Matrix<std::complex<double>, Ns, Ns>::Zero();
    Gamma g5(Gamma::Algebra::Gamma5);
    for (int b = 0; b < Ns; ++b) {
      SpinVector e; e = Zero(); e()(b) = ComplexD(1.0, 0.0);
      SpinVector r = g5 * e;
      for (int a = 0; a < Ns; ++a)
        gamma5_mat_(a, b) = std::complex<double>(
            TensorRemove(r()(a)).real(), TensorRemove(r()(a)).imag());
    }
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        isigma_mat_[mu][nu] =
            Eigen::Matrix<std::complex<double>, Ns, Ns>::Zero();
        for (int b = 0; b < Ns; ++b) {
          SpinVector e; e = Zero(); e()(b) = ComplexD(1.0, 0.0);
          SpinVector r = smn * e;
          for (int a = 0; a < Ns; ++a) {
            std::complex<double> val(TensorRemove(r()(a)).real(),
                                     TensorRemove(r()(a)).imag());
            isigma_mat_[mu][nu](a, b) = std::complex<double>(0, 1) * val;
          }
        }
      }
  }

  typedef typename LatticeSigmaField::vector_object::scalar_object SigSobj;
  typedef typename LatticePiField::vector_object::scalar_object PiSobj;
  typedef typename LatticeSFieldC::vector_object::scalar_object SSobj;
  typedef typename LatticePFieldC::vector_object::scalar_object PSobj;
  typedef typename LatticeTField::vector_object::scalar_object TSobj;

  typedef std::tuple<std::vector<SigSobj>, std::vector<PiSobj>,
                     std::vector<SSobj>, std::vector<PSobj>,
                     std::vector<TSobj>> AuxSiteArrays;

  AuxSiteArrays UnvectorizeEvenAux(const TXQCDField &U) {

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

    std::vector<SigSobj> sig_s;
    unvectorizeToLexOrdArray(sig_s, sigma_e);
    std::vector<PiSobj> pi_s;
    unvectorizeToLexOrdArray(pi_s, pi_e);
    std::vector<SSobj> s_s;
    unvectorizeToLexOrdArray(s_s, s_e);
    std::vector<PSobj> p_s;
    unvectorizeToLexOrdArray(p_s, p_e);
    std::vector<TSobj> t_s;
    unvectorizeToLexOrdArray(t_s, t_e);

    return std::make_tuple(std::move(sig_s), std::move(pi_s),
                           std::move(s_s), std::move(p_s), std::move(t_s));
  }

  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  void BuildSiteMatrix(
      const SigSobj &sig_site, const PiSobj &pi_site,
      const SSobj &s_site, const PSobj &p_site, const TSobj &t_site,
      Eigen::Matrix<std::complex<double>, kDim, kDim> &M) const {
    M = Eigen::Matrix<std::complex<double>, kDim, kDim>::Zero();
    for (int r = 0; r < kDim; ++r) M(r, r) = diag_mass_;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int a = 0; a < TxqcdNf; ++a) {
      for (int b = 0; b < TxqcdNf; ++b) {
        std::complex<double> sig_ab(sig_site()()(a, b).real(),
                                    sig_site()()(a, b).imag());
        std::complex<double> pi_ab(pi_site()()(a, b).real(),
                                   pi_site()()(a, b).imag());
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int beta = 0; beta < Ns; ++beta) {
            auto g5 = gamma5_mat_(alpha, beta);
            for (int i = 0; i < Nc; ++i) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = b * Ns * Nc + beta * Nc + i;
              if (alpha == beta) M(r, c) += sig_ab;
              M(r, c) += pi_ab * g5;
            }
          }
      }
    }

    for (int a = 0; a < TxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i)
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> s_ij(s_site()()(i, j).real(),
                                    s_site()()(i, j).imag());
          std::complex<double> p_ij(p_site()()(i, j).real(),
                                    p_site()()(i, j).imag());
          for (int alpha = 0; alpha < Ns; ++alpha)
            for (int beta = 0; beta < Ns; ++beta) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = a * Ns * Nc + beta * Nc + j;
              if (alpha == beta) M(r, c) += inv_sqrt2 * s_ij;
              M(r, c) += inv_sqrt2 * p_ij * gamma5_mat_(alpha, beta);
              for (int mu = 0; mu < Nd; ++mu)
                for (int nu = mu + 1; nu < Nd; ++nu) {
                  std::complex<double> t_ij(
                      t_site()(mu, nu)(i, j).real(),
                      t_site()(mu, nu)(i, j).imag());
                  M(r, c) += t_ij * isigma_mat_[mu][nu](alpha, beta);
                }
            }
        }
    }
  }
};

NAMESPACE_END(Grid);
