#pragma once
// Per-site dense site matrices for the v2 DTXQCD doubled Wilson-Clover
// operator.
//
// Layout: each diagonal block is 24x24 = DtxqcdNf * Ns * Nc.  The full
// doubled site matrix is 48x48 = 2 * 24, with off-diagonal 24x24 blocks
// from the d, n auxiliary fields:
//
//   M48(x) = [ M_upper(x)   M_offdiag(x) ]
//            [ M_offdiag(x) M_lower(x)   ]
//
//   M_upper(x) = (mass + 4) I_24 + X^{ij}_{ab}
//   M_lower(x) = (mass + 4) I_24 - X^{ij}_{ab}
//   X^{ij}_{ab} = sigma^{ij}_{ab}                              (scalar in spin)
//               + s delta^{ij} delta_{ab}                      (scalar singlet)
//               + (pi^{ij}_{ab} + p delta^{ij} delta_{ab}) gamma5
//   M_offdiag(x) = d^{ij}_{ab} gamma5 + n^{ij}_{ab}
//
// Lower block additionally includes the Cstar clover term sign flip
// (DtxqcdAddCloverToDiagBlock24 with lower_block=true).  Cstar mapping
// on X gives M_lower = ... - X due to X^T = X (gamma5^T = gamma5 in
// Grid basis); the entire X term sign-flips between upper and lower.
//
// Eigen 4x4 / 3x3 / scalar Eigen matrices for sub-blocks; per-site 48x48
// matrices use MatrixXcd (heap) for simplicity.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>  // SigmaMuNuAlgebra
#include <Grid/Eigen/Dense>
#include <array>

NAMESPACE_BEGIN(Grid);

// Per-site dense block dimensions.
static constexpr int kDtxqcdSiteDim24 = DtxqcdNf * Ns * Nc;     // 24
static constexpr int kDtxqcdSiteDim48 = 2 * kDtxqcdSiteDim24;   // 48

// Pack (flavor a, spin alpha, color i) -> linear index in [0, 24).
inline int DtxqcdSiteIdx24(int a, int alpha, int color) {
  return a * (Ns * Nc) + alpha * Nc + color;
}

// Portable complex helpers (DtxqcdToStd / DtxqcdConj / DtxqcdAbs / DtxqcdArg)
// live in DTXQCDAuxFieldTypes.h (included above) so every DTXQCD TU sees them.

// ---------- gamma5 + sigma_{mu,nu} 4x4 spin matrices ----------

// Cached gamma5 and sigma_{mu,nu} (mu<nu) as 4x4 Eigen matrices in Grid's
// internal spin basis.  Probed at construction by applying the Gamma to a
// single-site canonical-basis spinor and reading out the result, so the
// stored matrices are guaranteed consistent with what DtxqcdApplyX (and any
// other Gamma-using code) compute.
class DtxqcdSpinMatrices {
 public:
  Eigen::Matrix4cd gamma5;
  std::array<Eigen::Matrix4cd, 6> sigma_munu;  // (mu,nu) lex with mu<nu

  explicit DtxqcdSpinMatrices(GridCartesian& g) {
    gamma5 = ProbeGamma(g, Gamma::Algebra::Gamma5);
    int k = 0;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu)
        sigma_munu[k++] = ProbeGamma(g, SigmaMuNuAlgebra(mu, nu));
  }

 private:
  static Eigen::Matrix4cd ProbeGamma(GridCartesian& g, Gamma::Algebra alg) {
    Gamma G(alg);
    Eigen::Matrix4cd M = Eigen::Matrix4cd::Zero();
    Coordinate coord(std::vector<int>{0, 0, 0, 0});
    LatticeFermion in(&g), out(&g);
    typedef typename LatticeFermion::vector_object::scalar_object Site;
    for (int beta = 0; beta < Ns; ++beta) {
      in = Zero();
      Site s; s = Zero();
      s()(beta)(0) = ComplexD(1, 0);
      pokeSite(s, in, coord);
      out = G * in;
      Site s_out;
      peekSite(s_out, out, coord);
      for (int alpha = 0; alpha < Ns; ++alpha) {
        M(alpha, beta) = DtxqcdToStd(TensorRemove(s_out()(alpha)(0)));
      }
    }
    return M;
  }
};

// ---------- Per-site aux extraction ----------

// Site-local aux values.  sigma, pi, d, n are 6x6 = (Nf x Nc) x (Nf x Nc)
// color-flavor matrices in dense Eigen form; s, p are real scalars.
struct DtxqcdSiteAux {
  Eigen::Matrix<std::complex<double>, DtxqcdNfNc, DtxqcdNfNc> sigma;
  Eigen::Matrix<std::complex<double>, DtxqcdNfNc, DtxqcdNfNc> pi;
  Eigen::Matrix<std::complex<double>, DtxqcdNfNc, DtxqcdNfNc> d;
  Eigen::Matrix<std::complex<double>, DtxqcdNfNc, DtxqcdNfNc> n;
  RealD s;
  RealD p;

  // Combined-index helpers: row = a*Nc + i, col = b*Nc + j.
  static inline int Kab(int a, int i) { return a * Nc + i; }

  static DtxqcdSiteAux Extract(const LatticeDtxqcdSigma &sigma_L,
                                const LatticeDtxqcdPi    &pi_L,
                                const LatticeDtxqcdD     &d_L,
                                const LatticeDtxqcdN     &n_L,
                                const LatticeDtxqcdS     &s_L,
                                const LatticeDtxqcdP     &p_L,
                                const Coordinate& coord) {
    DtxqcdSiteAux out;
    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SiteCF;
    typedef typename LatticeDtxqcdS::vector_object::scalar_object     SiteSc;

    SiteCF sig_s, pi_s, d_s, n_s;
    SiteSc s_s, p_s;
    peekSite(sig_s, sigma_L, coord);
    peekSite(pi_s,  pi_L,    coord);
    peekSite(d_s,   d_L,     coord);
    peekSite(n_s,   n_L,     coord);
    peekSite(s_s,   s_L,     coord);
    peekSite(p_s,   p_L,     coord);

    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int b = 0; b < DtxqcdNf; ++b) {
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            out.sigma(Kab(a, i), Kab(b, j)) =
                DtxqcdToStd(TensorRemove(sig_s()(a, b)(i, j)));
            out.pi(Kab(a, i), Kab(b, j)) =
                DtxqcdToStd(TensorRemove(pi_s()(a, b)(i, j)));
            out.d(Kab(a, i), Kab(b, j)) =
                DtxqcdToStd(TensorRemove(d_s()(a, b)(i, j)));
            out.n(Kab(a, i), Kab(b, j)) =
                DtxqcdToStd(TensorRemove(n_s()(a, b)(i, j)));
          }
        }
      }
    }
    out.s = DtxqcdToStd(TensorRemove(s_s()()())).real();
    out.p = DtxqcdToStd(TensorRemove(p_s()()())).real();
    return out;
  }

  // Build from already-peeked per-site objects (no peekSite).  Used by the
  // multi-rank-LOCAL, GPU-safe cache build: unvectorizeToLexOrdArray the per-CB
  // aux lattices once, then call this per local lex site.  Templated on the
  // scalar_object types so callers don't repeat the typedefs.
  template <class SigS, class PiS, class DS, class NS, class SS, class PS>
  static DtxqcdSiteAux FromSobjs(const SigS &sig_s, const PiS &pi_s,
                                 const DS &d_s, const NS &n_s,
                                 const SS &s_s, const PS &p_s) {
    DtxqcdSiteAux out;
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            out.sigma(Kab(a, i), Kab(b, j)) = DtxqcdToStd(TensorRemove(sig_s()(a, b)(i, j)));
            out.pi   (Kab(a, i), Kab(b, j)) = DtxqcdToStd(TensorRemove(pi_s()(a, b)(i, j)));
            out.d    (Kab(a, i), Kab(b, j)) = DtxqcdToStd(TensorRemove(d_s()(a, b)(i, j)));
            out.n    (Kab(a, i), Kab(b, j)) = DtxqcdToStd(TensorRemove(n_s()(a, b)(i, j)));
          }
    out.s = DtxqcdToStd(TensorRemove(s_s()()())).real();
    out.p = DtxqcdToStd(TensorRemove(p_s()()())).real();
    return out;
  }
};

// Site-local clover field strength: 6 (mu<nu) color matrices F_{mu,nu}.
struct DtxqcdSiteClover {
  std::array<Eigen::Matrix3cd, 6> F_munu;

  static DtxqcdSiteClover Extract(
      const std::vector<LatticeColourMatrix> &FS,
      const Coordinate &coord) {
    DtxqcdSiteClover out;
    typedef typename LatticeColourMatrix::vector_object::scalar_object SiteCM;
    for (int p = 0; p < 6; ++p) {
      SiteCM f_s;
      peekSite(f_s, FS[p], coord);
      out.F_munu[p] = Eigen::Matrix3cd::Zero();
      for (int i = 0; i < Nc; ++i)
        for (int j = 0; j < Nc; ++j)
          out.F_munu[p](i, j) = DtxqcdToStd(TensorRemove(f_s()()(i, j)));
    }
    return out;
  }

  // Build from already-peeked per-site F_{mu,nu} objects (no peekSite), for the
  // multi-rank-local cache build.
  template <class FmnS>
  static DtxqcdSiteClover FromSobjs(const std::array<FmnS, 6> &f_arr) {
    DtxqcdSiteClover out;
    for (int p = 0; p < 6; ++p) {
      out.F_munu[p] = Eigen::Matrix3cd::Zero();
      for (int i = 0; i < Nc; ++i)
        for (int j = 0; j < Nc; ++j)
          out.F_munu[p](i, j) = DtxqcdToStd(TensorRemove(f_arr[p]()()(i, j)));
    }
    return out;
  }
};

// ---------- Block builders ----------

// Diagonal-block builder: M_diag = (mass + 4) * I_24 + block_sign * X^{ij}_{ab}.
//   block_sign = +1 for the upper diagonal block (standard).
//   block_sign = -1 for the lower diagonal block (Cstar M_22 has -X).
//
// X^{ij}_{ab} = sigma^{ij}_{ab} + s delta^{ij} delta_{ab}
//             + (pi^{ij}_{ab} + p delta^{ij} delta_{ab}) gamma5
//
// All four terms are diagonal in spin index (alpha == beta) for the
// scalar pieces, and convolved with gamma5(alpha, beta) for the
// pseudoscalar pieces.
inline void DtxqcdBuildDiagBlock24(double mass,
                                    const DtxqcdSiteAux& aux,
                                    const DtxqcdSpinMatrices& spin,
                                    Eigen::MatrixXcd& M,
                                    double block_sign = +1.0,
                                    bool transpose_aux = false) {
  // Mooee diagonal = (mass + 4) per Grid's WilsonFermion convention.
  const double mass_diag = mass + 4.0;
  M = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim24, kDtxqcdSiteDim24);
  for (int row = 0; row < kDtxqcdSiteDim24; ++row) M(row, row) = ComplexD(mass_diag, 0);

  const std::complex<double> bs(block_sign, 0.0);
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          int kab1 = DtxqcdSiteAux::Kab(a, i);
          int kab2 = DtxqcdSiteAux::Kab(b, j);
          // Under SIGMA_PI_HERMITIAN_ONLY the lower block reads σ^T and π^T
          // (= -C σ^T C, since σ, π have no spin and C·1·C=1).  This
          // restores Pf antisymmetry while keeping σ, π Hermitian (so the
          // H-S Gaussian integral is over the full Hermitian subspace and
          // the Fierz identity is exact).
          int aab = transpose_aux ? kab2 : kab1;
          int bab = transpose_aux ? kab1 : kab2;
          std::complex<double> sig_ij_ab = aux.sigma(aab, bab);
          std::complex<double> pi_ij_ab  = aux.pi   (aab, bab);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int row = DtxqcdSiteIdx24(a, alpha, i);
            // Scalar sigma: diagonal in spin.
            int col = DtxqcdSiteIdx24(b, alpha, j);
            M(row, col) += bs * sig_ij_ab;
            // Pseudoscalar pi: gamma5 in spin.
            for (int beta = 0; beta < Ns; ++beta) {
              int col2 = DtxqcdSiteIdx24(b, beta, j);
              M(row, col2) += bs * pi_ij_ab * spin.gamma5(alpha, beta);
            }
          }
        }
      }
    }
  }
  // Singlet s, p contributions: a == b, i == j.
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int i = 0; i < Nc; ++i) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        int row = DtxqcdSiteIdx24(a, alpha, i);
        // s I in spin
        M(row, row) += bs * std::complex<double>(aux.s, 0.0);
        // p gamma5
        for (int beta = 0; beta < Ns; ++beta) {
          int col = DtxqcdSiteIdx24(a, beta, i);
          M(row, col) += bs * std::complex<double>(aux.p, 0.0) * spin.gamma5(alpha, beta);
        }
      }
    }
  }
}

// Add the clover contribution to a 24x24 diagonal block:
//   upper:  M += -(csw/2) sum_{mu<nu} F_{mu,nu}     sigma_{mu,nu}_Grid
//   lower:  M += +(csw/2) sum_{mu<nu} F^T_{mu,nu}   sigma_{mu,nu}_Grid
//
// The opposite signs on upper/lower satisfy the Pfaffian-antisymmetry
// requirement M_ll = -K_b·M_uu^T·K_b with K_b = Cγ5, (Cγ5)² = -I:
//   M_ll^clover = -K_b·(-csw/2·F·σ_{μν})^T·K_b
//               = (csw/2)·F^T·(K_b·σ_{μν}^T·K_b)
//               = (csw/2)·F^T·σ_{μν}
// (using K_b·σ_{μν}^T·K_b = σ_{μν} since [C,γ5]=0 and Cσ_{μν}C^{-1} = -σ_{μν}^T,
//  with γ5 σ_{μν} γ5 = +σ_{μν}).
//
// History: 2026-06-12 set both upper and lower to -csw/2 to match the
// "M_lower = -C D^T C + X = D[U*] + X" claim.  That broke the C·K
// Pfaffian antisymmetry test added 2026-06-13 (csw=0 passed, csw≠0 failed
// at rel ~ 0.4).  Restoring opposite signs is required for the doubled
// formulation to represent |det M|^{1/2} as a Pfaffian.
inline void DtxqcdAddCloverToDiagBlock24(
    double csw,
    const DtxqcdSiteClover &clover,
    const DtxqcdSpinMatrices &spin,
    Eigen::MatrixXcd &M,
    bool lower_block = false) {
  if (csw == 0.0) return;
  const double prefactor = (lower_block ? +0.5 : -0.5) * csw;
  for (int p = 0; p < 6; ++p) {
    Eigen::Matrix3cd F = lower_block
                              ? Eigen::Matrix3cd(clover.F_munu[p].transpose())
                              : clover.F_munu[p];
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int beta = 0; beta < Ns; ++beta) {
          std::complex<double> smn_ab = spin.sigma_munu[p](alpha, beta);
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              int row = DtxqcdSiteIdx24(a, alpha, i);
              int col = DtxqcdSiteIdx24(a, beta, j);
              M(row, col) += prefactor * F(i, j) * smn_ab;
            }
          }
        }
      }
    }
  }
}

// Upper / lower diagonal blocks differ in:
//   (a) +X vs -X sign on the aux insertion
//   (b) clover prefactor sign and color-transposed F (Cstar M_22 = C^T D^T C)
inline void DtxqcdBuildUpperBlock24(double mass,
                                    const DtxqcdSiteAux &aux,
                                    const DtxqcdSpinMatrices &spin,
                                    Eigen::MatrixXcd &M,
                                    double csw = 0.0,
                                    const DtxqcdSiteClover *clover = nullptr) {
  DtxqcdBuildDiagBlock24(mass, aux, spin, M, +1.0);
  if (csw != 0.0 && clover != nullptr)
    DtxqcdAddCloverToDiagBlock24(csw, *clover, spin, M, /*lower_block=*/false);
}
inline void DtxqcdBuildLowerBlock24(double mass,
                                    const DtxqcdSiteAux &aux,
                                    const DtxqcdSpinMatrices &spin,
                                    Eigen::MatrixXcd &M,
                                    double csw = 0.0,
                                    const DtxqcdSiteClover *clover = nullptr) {
  // Corrected v2 (2026-06-12): M_lower = -C D^T C + X (with +X, NOT -X).
  // Original Eq 305 had the wrong sign C D^T C - X; the corrected derivation
  // confirms the lower block has +X (matching the upper block).  -C D^T C =
  // D[U*] for the kinetic part (verified for mass + Wilson hopping + clover
  // when the clover prefactor sign is correctly handled).
  //
  // Under SIGMA_PI_HERMITIAN_ONLY mode: M_lower uses σ^T, π^T (the natural
  // -C X^T C with C acting trivially on the spinless σ, π).  Restores Pf
  // antisymmetry with Hermitian (not real-symm) σ, π.
  const bool transpose_aux = DtxqcdSigmaPiHermitianOnly();
  DtxqcdBuildDiagBlock24(mass, aux, spin, M, +1.0, transpose_aux);
  if (csw != 0.0 && clover != nullptr)
    DtxqcdAddCloverToDiagBlock24(csw, *clover, spin, M, /*lower_block=*/true);
}

// Off-diagonal block: sqrt(2) * (d^{ij}_{ab} gamma5 + n^{ij}_{ab}).  v2
// corrected (2026-06-12): the doubled Dirac off-diagonal carries a sqrt(2)
// factor per the corrected dtxqcd_v2.tex Eq 22-25 (the earlier no-factor
// form was wrong; v1 had factor 2, v2 corrected is sqrt(2)).
inline void DtxqcdBuildOffDiagBlock24(const DtxqcdSiteAux& aux,
                                      const DtxqcdSpinMatrices& spin,
                                      Eigen::MatrixXcd& M) {
  const std::complex<double> sqrt2(DtxqcdOffdiagFactor(), 0.0);
  M = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim24, kDtxqcdSiteDim24);
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          int kab1 = DtxqcdSiteAux::Kab(a, i);
          int kab2 = DtxqcdSiteAux::Kab(b, j);
          std::complex<double> d_ij_ab = sqrt2 * aux.d(kab1, kab2);
          std::complex<double> n_ij_ab = sqrt2 * aux.n(kab1, kab2);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int row = DtxqcdSiteIdx24(a, alpha, i);
            // n: diagonal in spin
            int col = DtxqcdSiteIdx24(b, alpha, j);
            M(row, col) += n_ij_ab;
            // d: gamma5 in spin
            for (int beta = 0; beta < Ns; ++beta) {
              int col2 = DtxqcdSiteIdx24(b, beta, j);
              M(row, col2) += d_ij_ab * spin.gamma5(alpha, beta);
            }
          }
        }
      }
    }
  }
}

// Assemble the doubled 48x48 site matrix.  Off-diagonal block goes into both
// upper-right and lower-left positions.
inline void DtxqcdAssembleDoubled48(const Eigen::MatrixXcd& M_upper,
                                    const Eigen::MatrixXcd& M_lower,
                                    const Eigen::MatrixXcd& M_offdiag,
                                    Eigen::MatrixXcd& M48) {
  M48 = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim48, kDtxqcdSiteDim48);
  const int N = kDtxqcdSiteDim24;
  M48.block(0, 0, N, N) = M_upper;
  M48.block(0, N, N, N) = M_offdiag;
  // Under DTXQCD_DN_COMPLEX_SYMMETRIC: M_LL = conj(M_UR), the "d, d*
  // independent" form.  This must match the on-the-fly Apply path
  // (DtxqcdApplyDnCross with apply_conj=true) — both use conj
  // CONSISTENTLY to satisfy Mooee * MooeeInv = I.
  M48.block(N, 0, N, N) = DtxqcdDnComplexSymmetric()
                              ? M_offdiag.conjugate()
                              : M_offdiag;
  M48.block(N, N, N, N) = M_lower;
}

// log |det(M48)| via Eigen partial-pivot LU.
inline RealD DtxqcdLogAbsDet48(const Eigen::MatrixXcd& M48) {
  Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M48);
  return std::log(std::abs(lu.determinant()));
}

// Per-site LU factor and inverse.  Used by the EO operator to precompute
// Mooee^{-1} once per ImportAux.
inline Eigen::PartialPivLU<Eigen::MatrixXcd>
DtxqcdLU48(const Eigen::MatrixXcd& M48) {
  return Eigen::PartialPivLU<Eigen::MatrixXcd>(M48);
}

NAMESPACE_END(Grid);
