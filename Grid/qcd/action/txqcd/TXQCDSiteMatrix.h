#pragma once
// Shared utilities for building and inverting the 24×24 site-diagonal matrix
// M_site = (4+m)I + Δ(x) in the TXQCD Wilson operator.
//
// Used by TXQCDWilsonFermionEO (Mooee/MooeeInv) and TXQCDLogDetEOAction
// (log-det and its force). Factored here to avoid duplication.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

struct TXQCDSiteMatrixUtil {
  static constexpr int kDim = TxqcdNf * Ns * Nc;  // 24

  typedef Eigen::Matrix<std::complex<double>, kDim, kDim> SiteMatrix;
  typedef Eigen::Matrix<std::complex<double>, kDim, 1>    SiteVector;

  typedef Eigen::Matrix<std::complex<double>, Ns, Ns> SpinMatrix4;

  struct SpinMatrices {
    SpinMatrix4 gamma5;
    std::array<std::array<SpinMatrix4, Nd>, Nd> isigma;

    SpinMatrices() {
      gamma5 = SpinMatrix4::Zero();
      Gamma g5(Gamma::Algebra::Gamma5);
      for (int b = 0; b < Ns; ++b) {
        SpinVector e;
        e = Zero();
        e()(b) = ComplexD(1.0, 0.0);
        SpinVector r = g5 * e;
        for (int a = 0; a < Ns; ++a)
          gamma5(a, b) = std::complex<double>(TensorRemove(r()(a)).real(),
                                              TensorRemove(r()(a)).imag());
      }
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = mu + 1; nu < Nd; ++nu) {
          Gamma smn(SigmaMuNuAlgebra(mu, nu));
          isigma[mu][nu] = SpinMatrix4::Zero();
          for (int b = 0; b < Ns; ++b) {
            SpinVector e;
            e = Zero();
            e()(b) = ComplexD(1.0, 0.0);
            SpinVector r = smn * e;
            for (int a = 0; a < Ns; ++a) {
              std::complex<double> val(TensorRemove(r()(a)).real(),
                                       TensorRemove(r()(a)).imag());
              isigma[mu][nu](a, b) = std::complex<double>(0, 1) * val;
            }
          }
        }
    }
  };

  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  static void BuildSiteMatrix(const SpinMatrices &sm, RealD diag_mass,
                               const SigSobj &sig_site, const PiSobj &pi_site,
                               const SSobj &s_site, const PSobj &p_site,
                               const TSobj &t_site, SiteMatrix &M) {
    M = SiteMatrix::Zero();
    for (int r = 0; r < kDim; ++r) M(r, r) = diag_mass;

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int a = 0; a < TxqcdNf; ++a) {
      for (int b = 0; b < TxqcdNf; ++b) {
        std::complex<double> sig_ab(sig_site()()(a, b).real(),
                                    sig_site()()(a, b).imag());
        std::complex<double> pi_ab(pi_site()()(a, b).real(),
                                   pi_site()()(a, b).imag());
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int beta = 0; beta < Ns; ++beta) {
            std::complex<double> g5 = sm.gamma5(alpha, beta);
            for (int i = 0; i < Nc; ++i) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = b * Ns * Nc + beta * Nc + i;
              if (alpha == beta) M(r, c) += sig_ab;
              M(r, c) += pi_ab * g5;
            }
          }
        }
      }
    }

    for (int a = 0; a < TxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> s_ij(s_site()()(i, j).real(),
                                    s_site()()(i, j).imag());
          std::complex<double> p_ij(p_site()()(i, j).real(),
                                    p_site()()(i, j).imag());
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = a * Ns * Nc + beta * Nc + j;
              if (alpha == beta) M(r, c) += inv_sqrt2 * s_ij;
              M(r, c) += inv_sqrt2 * p_ij * sm.gamma5(alpha, beta);
              for (int mu = 0; mu < Nd; ++mu)
                for (int nu = mu + 1; nu < Nd; ++nu) {
                  std::complex<double> t_ij(t_site()(mu, nu)(i, j).real(),
                                            t_site()(mu, nu)(i, j).imag());
                  M(r, c) += t_ij * sm.isigma[mu][nu](alpha, beta);
                }
            }
          }
        }
      }
    }
  }

  typedef typename LatticeSigmaField::vector_object::scalar_object SigSobj;
  typedef typename LatticePiField::vector_object::scalar_object    PiSobj;
  typedef typename LatticeSFieldC::vector_object::scalar_object    SSobj;
  typedef typename LatticePFieldC::vector_object::scalar_object    PSobj;
  typedef typename LatticeTField::vector_object::scalar_object     TSobj;

  struct AuxSiteArrays {
    std::vector<SigSobj> sig;
    std::vector<PiSobj>  pi;
    std::vector<SSobj>   s;
    std::vector<PSobj>   p;
    std::vector<TSobj>   t;
  };

  static AuxSiteArrays UnvectorizeAux(const LatticeSigmaField &sigma,
                                       const LatticePiField &pi,
                                       const LatticeSFieldC &s,
                                       const LatticePFieldC &p,
                                       const LatticeTField &t) {
    AuxSiteArrays out;
    unvectorizeToLexOrdArray(out.sig, sigma);
    unvectorizeToLexOrdArray(out.pi, pi);
    unvectorizeToLexOrdArray(out.s, s);
    unvectorizeToLexOrdArray(out.p, p);
    unvectorizeToLexOrdArray(out.t, t);
    return out;
  }

  static void PrecomputeInverses(const SpinMatrices &sm, RealD diag_mass,
                                  const AuxSiteArrays &aux,
                                  std::vector<SiteMatrix> &inv) {
    uint64_t nsites = aux.sig.size();
    inv.resize(nsites);
    SiteMatrix M;
    for (uint64_t x = 0; x < nsites; ++x) {
      BuildSiteMatrix(sm, diag_mass, aux.sig[x], aux.pi[x], aux.s[x],
                      aux.p[x], aux.t[x], M);
      inv[x] = M.inverse();
    }
  }
};

NAMESPACE_END(Grid);
