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

  typedef typename LatticeColourMatrix::vector_object::scalar_object FmnSobj;

  struct CloverSiteArrays {
    std::array<std::vector<FmnSobj>, 6> fs;
  };

  static CloverSiteArrays UnvectorizeClover(
      const std::vector<LatticeColourMatrix> &FS) {
    CloverSiteArrays out;
    for (int k = 0; k < 6; ++k) unvectorizeToLexOrdArray(out.fs[k], FS[k]);
    return out;
  }

  static int FmnIndex(int mu, int nu) {
    if (mu == 0 && nu == 1) return 0;
    if (mu == 0 && nu == 2) return 1;
    if (mu == 0 && nu == 3) return 2;
    if (mu == 1 && nu == 2) return 3;
    if (mu == 1 && nu == 3) return 4;
    if (mu == 2 && nu == 3) return 5;
    return -1;
  }

  // Per-flavor diagonal mass: diag_mass[a] = 4 + m_a goes on the rows
  // (a, alpha, i) for that flavor.  Required for non-degenerate Nf=3 setups
  // (e.g. mass = diag(m_l, m_l, m_s)) where each flavor has its own mass.
  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  static void BuildSiteMatrix(const SpinMatrices &sm,
                               const std::array<RealD, TxqcdNf> &diag_mass,
                               const SigSobj &sig_site, const PiSobj &pi_site,
                               const SSobj &s_site, const PSobj &p_site,
                               const TSobj &t_site,
                               RealD csw,
                               const std::array<FmnSobj, 6> *fmn_site,
                               SiteMatrix &M) {
    M = SiteMatrix::Zero();
    for (int a = 0; a < TxqcdNf; ++a)
      for (int alpha = 0; alpha < Ns; ++alpha)
        for (int i = 0; i < Nc; ++i) {
          int r = a * Ns * Nc + alpha * Nc + i;
          M(r, r) = diag_mass[a];
        }

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // Mode-dependent pre-factors (compile-time selected via TXQCD_T_FLAVOR):
    //   mode A (color-t):  sigma, pi -> 1            ; s, p -> 1/sqrt(2)
    //   mode B (flavor-t): sigma, pi -> 1/sqrt(2)    ; s, p -> 1
    constexpr double sig_pi_factor = TxqcdTIsFlavor ? (1.0 / 1.4142135623730951) : 1.0;
    constexpr double s_p_factor    = TxqcdTIsFlavor ? 1.0 : (1.0 / 1.4142135623730951);

    // ---- sigma, pi contributions: flavor matrix, color-diagonal ----
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
              if (alpha == beta) M(r, c) += sig_pi_factor * sig_ab;
              M(r, c) += sig_pi_factor * pi_ab * g5;
            }
          }
        }
      }
    }

    const std::complex<double> clover_coeff(0.0, 0.5 * csw);

    // ---- s, p contributions: color matrix, flavor-diagonal ----
    // ---- clover F_{mu,nu} contribution: always color, flavor-diagonal ----
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
              if (alpha == beta) M(r, c) += s_p_factor * s_ij;
              M(r, c) += s_p_factor * p_ij * sm.gamma5(alpha, beta);
              if (fmn_site != nullptr) {
                for (int mu = 0; mu < Nd; ++mu)
                  for (int nu = mu + 1; nu < Nd; ++nu) {
                    int k = FmnIndex(mu, nu);
                    std::complex<double> fs_ij(
                        (*fmn_site)[k]()()(i, j).real(),
                        (*fmn_site)[k]()()(i, j).imag());
                    M(r, c) += clover_coeff * fs_ij *
                               sm.isigma[mu][nu](alpha, beta);
                  }
              }
            }
          }
        }
      }
    }

    // ---- t_{mu,nu} contribution: mode-dependent index structure ----
    // mode A (color-t): t indexed by (i,j) color, flavor-diagonal.
    // mode B (flavor-t): t indexed by (a,b) flavor (upper Nf x Nf block of
    //   the Nc x Nc storage), color-diagonal.  Inactive color slots are 0
    //   by invariant maintained at initialization.
    if constexpr (TxqcdTIsFlavor) {
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int b = 0; b < TxqcdNf; ++b) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              for (int i = 0; i < Nc; ++i) {
                int r = a * Ns * Nc + alpha * Nc + i;
                int c = b * Ns * Nc + beta  * Nc + i;
                for (int mu = 0; mu < Nd; ++mu)
                  for (int nu = mu + 1; nu < Nd; ++nu) {
                    std::complex<double> t_ab(t_site()(mu, nu)(a, b).real(),
                                              t_site()(mu, nu)(a, b).imag());
                    M(r, c) += t_ab * sm.isigma[mu][nu](alpha, beta);
                  }
              }
            }
          }
        }
      }
    } else {
      for (int a = 0; a < TxqcdNf; ++a) {
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int beta = 0; beta < Ns; ++beta) {
                int r = a * Ns * Nc + alpha * Nc + i;
                int c = a * Ns * Nc + beta  * Nc + j;
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

  // Per-flavor diagonal mass (preferred entry point).
  // If `fwd` is non-null, also populate it with the pre-inversion matrix M
  // (used by the cuBLAS forward Mooee path).  Backwards compatible — existing
  // callers passing 6 args get the original behaviour.
  static void PrecomputeInverses(const SpinMatrices &sm,
                                  const std::array<RealD, TxqcdNf> &diag_mass,
                                  const AuxSiteArrays &aux,
                                  RealD csw,
                                  const CloverSiteArrays *clover,
                                  std::vector<SiteMatrix> &inv,
                                  std::vector<SiteMatrix> *fwd = nullptr) {
    uint64_t nsites = aux.sig.size();
    inv.resize(nsites);
    if (fwd) fwd->resize(nsites);
    // Per-site BuildSiteMatrix + 24×24 Eigen.inverse() are completely
    // independent across sites; parallelize across CPU cores.  On 16 OMP
    // threads this is ~16× faster than the previous serial loop on the
    // production lattice (98k sites/board).
    thread_for(x, nsites, {
      std::array<FmnSobj, 6> fmn_site;
      const std::array<FmnSobj, 6> *fmn_ptr = nullptr;
      if (clover != nullptr) {
        for (int k = 0; k < 6; ++k) fmn_site[k] = clover->fs[k][x];
        fmn_ptr = &fmn_site;
      }
      SiteMatrix M;
      BuildSiteMatrix(sm, diag_mass, aux.sig[x], aux.pi[x], aux.s[x],
                      aux.p[x], aux.t[x], csw, fmn_ptr, M);
      if (fwd) (*fwd)[x] = M;
      inv[x] = M.inverse();
    });
  }

  // Backward-compat scalar overloads: replicate one mass across all flavors.
  static std::array<RealD, TxqcdNf> MassArray(RealD m) {
    std::array<RealD, TxqcdNf> arr;
    arr.fill(m);
    return arr;
  }

  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  static void BuildSiteMatrix(const SpinMatrices &sm, RealD diag_mass,
                               const SigSobj &sig_site, const PiSobj &pi_site,
                               const SSobj &s_site, const PSobj &p_site,
                               const TSobj &t_site,
                               RealD csw,
                               const std::array<FmnSobj, 6> *fmn_site,
                               SiteMatrix &M) {
    BuildSiteMatrix(sm, MassArray(diag_mass), sig_site, pi_site, s_site,
                    p_site, t_site, csw, fmn_site, M);
  }

  static void PrecomputeInverses(const SpinMatrices &sm, RealD diag_mass,
                                  const AuxSiteArrays &aux,
                                  RealD csw,
                                  const CloverSiteArrays *clover,
                                  std::vector<SiteMatrix> &inv) {
    PrecomputeInverses(sm, MassArray(diag_mass), aux, csw, clover, inv);
  }
};

NAMESPACE_END(Grid);
