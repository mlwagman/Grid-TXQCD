#pragma once
// Per-site dense site matrices for the doubled DTXQCD Wilson-Clover operator.
//
// Layout: each diagonal block is 24x24 = DtxqcdNf * Ns * Nc (one flavor
// doubling).  The full doubled site matrix is 48x48 = 2 * 24, with off-
// diagonal 24x24 blocks from the d, n auxiliary fields:
//
//   M48(x) = [ M_upper(x)   M_offdiag(x) ]
//            [ M_offdiag(x) M_lower(x)   ]
//
//   M_upper(x) = m I_24 + Delta_diag(x)
//   M_lower(x) = m I_24 + Delta_diag(x)            (v1 placeholder; see note)
//   Delta_diag(x) = (1/sqrt 2) sigma^A(x) tau^A I_spin I_color
//                 + (1/sqrt 2) pi^A(x)    tau^A gamma5 I_color
//                 + sum_{mu<nu} i t^A_{mu,nu}(x) tau^A Grid_Sigma_{mu,nu} I_color
//   M_offdiag(x) = 2 d^{ij}(x) gamma5 + 2 n^{ij}(x)        (flavor identity)
//
// v1 simplification: M_lower = M_upper.  At the M_ee (site-local) level with
// no clover or hopping the QCD piece is just m * I_color, and C (m I) C^T =
// m I, so upper and lower agree.  When clover (-c_sw/4 F_{mu,nu} sigma_{mu,nu})
// is added the lower-block clover gets a C ... C^T sandwich and the two
// diverge — to be implemented when DTXQCDDeltaCloverOp is added.
//
// Eigen 4x4 / 2x2 / 3x3 are used for the gamma / Pauli / color sub-blocks.
// Per-site 48x48 matrices use MatrixXcd (heap-allocated) for v1 simplicity;
// switching to fixed-size Matrix<ComplexD, 48, 48> is an optimization once
// the EO operator wraps this in a SIMD-vectorized tensor.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>  // SigmaMuNuAlgebra + DtxqcdPauli
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

// ---------- Pauli + spin matrix caches ----------

// Pauli matrices tau^A in 2x2 Eigen form (Hermitian, traceless).  Singleton
// — built once on first access.
inline const std::array<Eigen::Matrix2cd, DtxqcdNTriplet>& DtxqcdPauliEigen() {
  static const auto T = []() {
    std::array<Eigen::Matrix2cd, DtxqcdNTriplet> arr;
    arr[0] << ComplexD(0, 0), ComplexD(1, 0),
              ComplexD(1, 0), ComplexD(0, 0);   // tau^1
    arr[1] << ComplexD(0, 0), ComplexD(0, -1),
              ComplexD(0, 1), ComplexD(0, 0);   // tau^2
    arr[2] << ComplexD(1, 0), ComplexD(0, 0),
              ComplexD(0, 0), ComplexD(-1, 0);  // tau^3
    return arr;
  }();
  return T;
}

// Cached gamma5 and sigma_{mu,nu} (mu<nu) as 4x4 Eigen matrices in Grid's
// internal spin basis.  Probed at construction by applying the Gamma to a
// single-site canonical-basis spinor and reading out the result, so the
// stored matrices are guaranteed consistent with what
// DtxqcdApplyDeltaSigmaPi / DtxqcdApplyDeltaTensor (and any other Gamma-
// using code) compute.
class DtxqcdSpinMatrices {
 public:
  Eigen::Matrix4cd gamma5;
  std::array<Eigen::Matrix4cd, 6> sigma_munu;  // pair index = (mu,nu) lex with mu<nu

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
        M(alpha, beta) = ComplexD(TensorRemove(s_out()(alpha)(0)));
      }
    }
    return M;
  }
};

// ---------- Per-site aux extraction ----------

// Site-local aux field values (per Pauli triplet component, plus full 3x3
// Hermitian color matrices d, n).  Built by peeking a single site of the
// Lattice fields.
struct DtxqcdSiteAux {
  std::array<ComplexD, DtxqcdNTriplet> sigma;
  std::array<ComplexD, DtxqcdNTriplet> pi;
  std::array<std::array<ComplexD, DtxqcdNTriplet>, 6> t;  // [pair_idx][A]
  Eigen::Matrix3cd d;
  Eigen::Matrix3cd n;

  static DtxqcdSiteAux Extract(const LatticeDtxqcdSigma& sigma_L,
                                const LatticeDtxqcdPi& pi_L,
                                const LatticeDtxqcdT& t_L,
                                const LatticeDtxqcdD& d_L,
                                const LatticeDtxqcdN& n_L,
                                const Coordinate& coord) {
    DtxqcdSiteAux out;
    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SiteSig;
    typedef typename LatticeDtxqcdT::vector_object::scalar_object     SiteT;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     SiteD;

    SiteSig sig_s, pi_s; SiteT t_s; SiteD d_s, n_s;
    peekSite(sig_s, sigma_L, coord);
    peekSite(pi_s,  pi_L,    coord);
    peekSite(t_s,   t_L,     coord);
    peekSite(d_s,   d_L,     coord);
    peekSite(n_s,   n_L,     coord);

    for (int A = 0; A < DtxqcdNTriplet; ++A) {
      out.sigma[A] = ComplexD(TensorRemove(sig_s()()(A)));
      out.pi[A]    = ComplexD(TensorRemove(pi_s()()(A)));
    }
    int p = 0;
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        for (int A = 0; A < DtxqcdNTriplet; ++A)
          out.t[p][A] = ComplexD(TensorRemove(t_s()(mu, nu)(A)));
        ++p;
      }
    out.d = Eigen::Matrix3cd::Zero();
    out.n = Eigen::Matrix3cd::Zero();
    for (int i = 0; i < Nc; ++i)
      for (int j = 0; j < Nc; ++j) {
        out.d(i, j) = ComplexD(TensorRemove(d_s()()(i, j)));
        out.n(i, j) = ComplexD(TensorRemove(n_s()()(i, j)));
      }
    return out;
  }
};

// ---------- Block builders ----------

// Diagonal-block builder: M_diag = mass * I_24 + Delta_diag (sigma^A, pi^A,
// t^A) using the Pauli flavor structure v^A_{a,b} = tau^A_{a,b}.
inline void DtxqcdBuildDiagBlock24(double mass,
                                    const DtxqcdSiteAux& aux,
                                    const DtxqcdSpinMatrices& spin,
                                    Eigen::MatrixXcd& M) {
  M = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim24, kDtxqcdSiteDim24);
  for (int row = 0; row < kDtxqcdSiteDim24; ++row) M(row, row) = ComplexD(mass, 0);

  const auto& tau = DtxqcdPauliEigen();
  const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
  const ComplexD ci(0.0, 1.0);

  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int A = 0; A < DtxqcdNTriplet; ++A) {
        ComplexD tab = tau[A](a, b);
        if (tab == ComplexD(0, 0)) continue;
        ComplexD coef_s = inv_sqrt2 * aux.sigma[A] * tab;
        ComplexD coef_p = inv_sqrt2 * aux.pi[A]    * tab;
        // sigma + pi diagonal-in-color
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int beta = 0; beta < Ns; ++beta) {
            ComplexD g5_ab = spin.gamma5(alpha, beta);
            for (int i = 0; i < Nc; ++i) {
              int row = DtxqcdSiteIdx24(a, alpha, i);
              int col = DtxqcdSiteIdx24(b, beta,  i);
              if (alpha == beta) M(row, col) += coef_s;
              M(row, col) += coef_p * g5_ab;
            }
          }
        }
        // tensor: sum mu<nu of i t^A_{mu,nu} sigma_{mu,nu} diagonal-in-color
        for (int p_idx = 0; p_idx < 6; ++p_idx) {
          ComplexD coef_t = ci * aux.t[p_idx][A] * tab;
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              ComplexD smn_ab = spin.sigma_munu[p_idx](alpha, beta);
              for (int i = 0; i < Nc; ++i) {
                int row = DtxqcdSiteIdx24(a, alpha, i);
                int col = DtxqcdSiteIdx24(b, beta,  i);
                M(row, col) += coef_t * smn_ab;
              }
            }
          }
        }
      }
    }
  }
}

// Upper / lower diagonal blocks.  v1: identical (site-local QCD piece has no
// C-conjugation distinction without clover).  Function names already
// distinguished so DTXQCDDeltaCloverOp can plug into BuildLowerBlock24
// later with C (clover) C^T.
inline void DtxqcdBuildUpperBlock24(double mass,
                                    const DtxqcdSiteAux& aux,
                                    const DtxqcdSpinMatrices& spin,
                                    Eigen::MatrixXcd& M) {
  DtxqcdBuildDiagBlock24(mass, aux, spin, M);
}
inline void DtxqcdBuildLowerBlock24(double mass,
                                    const DtxqcdSiteAux& aux,
                                    const DtxqcdSpinMatrices& spin,
                                    Eigen::MatrixXcd& M) {
  DtxqcdBuildDiagBlock24(mass, aux, spin, M);
}

// Off-diagonal block: 2 d gamma5 + 2 n.  Identity in flavor; color matrix
// d, n; gamma5 vs identity in spin.  Hermitian when d, n are Hermitian color
// matrices (d, n commute with gamma5 because they live on disjoint indices).
inline void DtxqcdBuildOffDiagBlock24(const DtxqcdSiteAux& aux,
                                      const DtxqcdSpinMatrices& spin,
                                      Eigen::MatrixXcd& M) {
  M = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim24, kDtxqcdSiteDim24);
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int alpha = 0; alpha < Ns; ++alpha) {
      for (int beta = 0; beta < Ns; ++beta) {
        ComplexD g5_ab = spin.gamma5(alpha, beta);
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            int row = DtxqcdSiteIdx24(a, alpha, i);
            int col = DtxqcdSiteIdx24(a, beta,  j);
            ComplexD val = ComplexD(2.0, 0) * aux.d(i, j) * g5_ab;
            if (alpha == beta) val += ComplexD(2.0, 0) * aux.n(i, j);
            M(row, col) += val;
          }
        }
      }
    }
  }
}

// Assemble the doubled 48x48 site matrix.  Off-diagonal block goes into both
// the (upper-right) and (lower-left) positions.  The off-diagonal is
// Hermitian individually, so this assignment makes M48 Hermitian when the
// diagonal blocks are.
inline void DtxqcdAssembleDoubled48(const Eigen::MatrixXcd& M_upper,
                                    const Eigen::MatrixXcd& M_lower,
                                    const Eigen::MatrixXcd& M_offdiag,
                                    Eigen::MatrixXcd& M48) {
  M48 = Eigen::MatrixXcd::Zero(kDtxqcdSiteDim48, kDtxqcdSiteDim48);
  const int N = kDtxqcdSiteDim24;
  M48.block(0, 0, N, N) = M_upper;
  M48.block(0, N, N, N) = M_offdiag;
  M48.block(N, 0, N, N) = M_offdiag;
  M48.block(N, N, N, N) = M_lower;
}

// log |det(M48)| via Eigen partial-pivot LU.  Phase / sign tracking is left
// to DTXQCDPfaffianSignDiagnostic (Phase 6).
inline RealD DtxqcdLogAbsDet48(const Eigen::MatrixXcd& M48) {
  Eigen::PartialPivLU<Eigen::MatrixXcd> lu(M48);
  return std::log(std::abs(lu.determinant()));
}

// Per-site LU-factor and inverse.  Used by the EO operator to precompute
// Mooee^{-1} once per ImportAux.  Returns the LU factorization object for
// downstream solves; the caller can also call .inverse() if a dense inverse
// is needed.
inline Eigen::PartialPivLU<Eigen::MatrixXcd>
DtxqcdLU48(const Eigen::MatrixXcd& M48) {
  return Eigen::PartialPivLU<Eigen::MatrixXcd>(M48);
}

NAMESPACE_END(Grid);
