#pragma once
// Per-site v2 DTXQCD aux/gauge-clover force kernels.  Parameterised over a
// "weight-matrix lookup" callable Inv(R, C) -> ComplexD that abstracts what
// stands in for M_ee_48^{-1}(x) (or the per-site rank-1 RHMC bilinear) in the
// trace formula.
//
// Two callers (unchanged from v1):
//   * DTXQCDLogDetCloverEOAction:
//        Inv(R, C) = M^{-1}(x)[R][C].
//        dS_LD/dX(x) = -sum_{R,C} dM_{RC}/dX(x) * Inv(C, R)
//        (note the C<->R swap inside Inv -- that's the trace formula's
//        M^{-1}_{C, R} read off as Inv(R, C) since we work with Inv keyed by
//        the original (row, col) of M; the kernel handles the bookkeeping.)
//
//   * DTXQCDWilsonCloverRationalEOAction:
//        Inv(R, C) = (0.5)*(conj(Y[C])*X[R] + conj(X[C])*Y[R])  -- symmetrised
//        Wirtinger bilinear, after the multishift-CG solutions.
//
// Output kernels write WITHOUT the overall caller-supplied scaling (LogDet -1
// or RHMC -2 alpha_k).  Caller multiplies after.
//
// v1 -> v2 changes:
//   * sigma, pi, d, n force outputs now CF matrices (DtxqcdSiteCFMatrix) per
//     site, not Pauli-triplet vectors.  Output type matches the input field
//     scalar_object.
//   * Drop tensor t force (no t field in v2).
//   * Add singlet s, p force outputs.
//   * +X/-X sign-flip between upper and lower diagonal blocks means the sigma
//     and pi force contributions from the two blocks SUBTRACT instead of
//     summing (v1 had same sign in both blocks).

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdSiteForceKernel {

// Scalar-object aliases pulled from the LatticeXxx types so callers don't
// have to repeat the typedef boilerplate.
using SigSobj = typename LatticeDtxqcdSigma::vector_object::scalar_object;
using PiSobj  = typename LatticeDtxqcdPi::vector_object::scalar_object;
using DSobj   = typename LatticeDtxqcdD::vector_object::scalar_object;
using NSobj   = typename LatticeDtxqcdN::vector_object::scalar_object;
using SSobj   = typename LatticeDtxqcdS::vector_object::scalar_object;
using PSobj   = typename LatticeDtxqcdP::vector_object::scalar_object;
using CMsobj  = typename LatticeColourMatrix::vector_object::scalar_object;

static constexpr int kDim24 = kDtxqcdSiteDim24;

// ----------------------------------------------------------------------
// Aux-field force kernel for the v2 roster (sigma, pi, d, n, s, p).
//
// dS/dX = -sum_{R,C} dM_{R,C}/dX * Inv(R, C)
// (Inv(R, C) is keyed in the original M index convention; see header above.)
//
// Block layout in M48:
//   upper-left  block 0: rows/cols 0..23   -- (mass+4)I + X
//   upper-right block 0,1: cols 24..47    -- d gamma5 + n
//   lower-left  block 1,0:                 -- d gamma5 + n
//   lower-right block 1: rows/cols 24..47  -- (mass+4)I - X
// ----------------------------------------------------------------------
template <class InvLookup>
inline void AuxForceAt(InvLookup Inv,
                       const DtxqcdSpinMatrices &spin,
                       SigSobj &sig_force,
                       PiSobj  &pi_force,
                       DSobj   &d_force,
                       NSobj   &n_force,
                       SSobj   &s_force,
                       PSobj   &p_force) {
  sig_force = Zero();
  pi_force  = Zero();
  d_force   = Zero();
  n_force   = Zero();
  s_force   = Zero();
  p_force   = Zero();

  // ---- sigma^{ij}_{ab} force ---------------------------------------------
  //
  // sigma enters M with +1 on upper diagonal, -1 on lower diagonal, diagonal
  // in spin (delta_{alpha, beta}).  So:
  //
  //   dM_{(a, alpha, i, blk), (b, alpha, j, blk)} / dsigma^{ij}_{ab}
  //     = +1 (blk=0) or -1 (blk=1)
  //
  //   F_sigma^{ij}_{ab} = -sum_alpha [ Inv((a,alpha,i,0), (b,alpha,j,0))
  //                                  - Inv((a,alpha,i,1), (b,alpha,j,1)) ]
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int ra = DtxqcdSiteIdx24(a, alpha, i);
            int cb = DtxqcdSiteIdx24(b, alpha, j);
            val += Inv(ra, cb) - Inv(kDim24 + ra, kDim24 + cb);
          }
          sig_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- pi^{ij}_{ab} force ------------------------------------------------
  //
  // pi enters M with +/- gamma5(alpha, beta), upper/lower diagonal:
  //
  //   F_pi^{ij}_{ab} = -sum_{alpha,beta} gamma5(alpha, beta)
  //                  * [ Inv((a,alpha,i,0), (b,beta,j,0))
  //                    - Inv((a,alpha,i,1), (b,beta,j,1)) ]
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              ComplexD g5 = spin.gamma5(alpha, beta);
              if (g5 == ComplexD(0, 0)) continue;
              int ra = DtxqcdSiteIdx24(a, alpha, i);
              int cb = DtxqcdSiteIdx24(b, beta,  j);
              val += g5 * (Inv(ra, cb) - Inv(kDim24 + ra, kDim24 + cb));
            }
          }
          pi_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- d^{ij}_{ab} force (off-diagonal, +gamma5, both upper-right and lower-left)
  //
  //   dM_{(a,alpha,i,0), (b,beta,j,1)} / dd^{ij}_{ab} = +gamma5(alpha, beta)
  //   dM_{(a,alpha,i,1), (b,beta,j,0)} / dd^{ij}_{ab} = +gamma5(alpha, beta)
  //
  //   F_d^{ij}_{ab} = -sum_{alpha,beta} gamma5(alpha, beta)
  //               * [ Inv((a,alpha,i,0), (b,beta,j,1))
  //                 + Inv((a,alpha,i,1), (b,beta,j,0)) ]
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              ComplexD g5 = spin.gamma5(alpha, beta);
              if (g5 == ComplexD(0, 0)) continue;
              int ra = DtxqcdSiteIdx24(a, alpha, i);
              int cb = DtxqcdSiteIdx24(b, beta,  j);
              val += g5 * (Inv(ra, kDim24 + cb) + Inv(kDim24 + ra, cb));
            }
          }
          d_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- n^{ij}_{ab} force (off-diagonal, +identity in spin) -----------------
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int ra = DtxqcdSiteIdx24(a, alpha, i);
            int cb = DtxqcdSiteIdx24(b, alpha, j);
            val += Inv(ra, kDim24 + cb) + Inv(kDim24 + ra, cb);
          }
          n_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- s singlet force (diagonal in (a==b, i==j), spin scalar, +/-) -------
  //
  //   F_s = -sum_a sum_alpha sum_i [ Inv((a,alpha,i,0), (a,alpha,i,0))
  //                                - Inv((a,alpha,i,1), (a,alpha,i,1)) ]
  {
    ComplexD val(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          int r = DtxqcdSiteIdx24(a, alpha, i);
          val += Inv(r, r) - Inv(kDim24 + r, kDim24 + r);
        }
      }
    }
    s_force()()() = -val;
  }

  // ---- p singlet force (diagonal in (a==b, i==j), gamma5 in spin, +/-) ----
  {
    ComplexD val(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int beta = 0; beta < Ns; ++beta) {
          ComplexD g5 = spin.gamma5(alpha, beta);
          if (g5 == ComplexD(0, 0)) continue;
          for (int i = 0; i < Nc; ++i) {
            int ra = DtxqcdSiteIdx24(a, alpha, i);
            int cb = DtxqcdSiteIdx24(a, beta,  i);
            val += g5 * (Inv(ra, cb) - Inv(kDim24 + ra, kDim24 + cb));
          }
        }
      }
    }
    p_force()()() = -val;
  }
}

// ----------------------------------------------------------------------
// Per-site clover_sigma builder (unchanged from v1 -- clover only acts in
// color and spin, never flavor; structure is identical).
// ----------------------------------------------------------------------
template <class InvLookup>
inline void CloverSigmaAt(InvLookup Inv,
                          const DtxqcdSpinMatrices &spin,
                          RealD csw,
                          std::array<CMsobj, 6> &clover_sigma) {
  for (int p_idx = 0; p_idx < 6; ++p_idx) {
    CMsobj cs;
    cs = Zero();
    for (int i_c = 0; i_c < Nc; ++i_c) {
      for (int j_c = 0; j_c < Nc; ++j_c) {
        ComplexD val(0, 0);
        for (int a = 0; a < DtxqcdNf; ++a) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              ComplexD smn = spin.sigma_munu[p_idx](alpha, beta);
              if (smn == ComplexD(0, 0)) continue;
              int row_u = DtxqcdSiteIdx24(a, beta,  j_c);
              int col_u = DtxqcdSiteIdx24(a, alpha, i_c);
              int row_l = DtxqcdSiteIdx24(a, beta,  i_c);
              int col_l = DtxqcdSiteIdx24(a, alpha, j_c);
              val += smn * Inv(row_u, col_u);
              val -= smn * Inv(kDim24 + row_l, kDim24 + col_l);
            }
          }
        }
        cs()()(i_c, j_c) = ComplexD(-0.5 * csw, 0.0) * std::conj(val);
      }
    }
    clover_sigma[p_idx] = cs;
  }
}

}  // namespace DtxqcdSiteForceKernel

NAMESPACE_END(Grid);
