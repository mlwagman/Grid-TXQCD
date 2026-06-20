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
  // v2 corrected (2026-06-12): sigma enters M with +1 in BOTH upper and lower
  // diagonal (previously +1 / -1; the old -1 in the lower was tied to the
  // superseded M_lower = C D^T C - X form).  Trace formula now:
  //   F_sigma^{ij}_{ab} = -sum_alpha [ Inv((b,alpha,j,0), (a,alpha,i,0))
  //                                  + Inv((b,alpha,j,1), (a,alpha,i,1)) ]
  // Under SIGMA_PI_HERMITIAN_ONLY the lower block uses σ^T (= -C σ^T C),
  // so ∂M_lower/∂σ_{(a,i),(b,j)} hits the TRANSPOSED LL position
  // → trace contribution is Inv(kDim24+ra, kDim24+cb) (rather than the
  // un-transposed Inv(kDim24+cb, kDim24+ra)).
  const bool sigpi_T = DtxqcdSigmaPiHermitianOnly();
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int ra = DtxqcdSiteIdx24(a, alpha, i);
            int cb = DtxqcdSiteIdx24(b, alpha, j);
            val += Inv(cb, ra)
                 + (sigpi_T ? Inv(kDim24 + ra, kDim24 + cb)
                             : Inv(kDim24 + cb, kDim24 + ra));
          }
          sig_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- pi^{ij}_{ab} force ------------------------------------------------
  //
  // v2 corrected: dM/dpi has +gamma5 in BOTH upper and lower diagonal.
  //   F_pi^{ij}_{ab} = -sum_{alpha,beta} gamma5(alpha, beta)
  //                  * [ Inv((b,beta,j,0), (a,alpha,i,0))
  //                    + Inv((b,beta,j,1), (a,alpha,i,1)) ]
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
              val += g5 * (Inv(cb, ra)
                          + (sigpi_T ? Inv(kDim24 + ra, kDim24 + cb)
                                      : Inv(kDim24 + cb, kDim24 + ra)));
            }
          }
          pi_force()(a, b)(i, j) = -val;
        }
      }
    }
  }

  // ---- d^{ij}_{ab} force (off-diagonal, +sqrt(2)*gamma5 in spin) ---------
  //
  // v2 corrected: off-diagonal has +sqrt(2) factor on d (and n) blocks.
  //   F_d^{ij}_{ab} = -sqrt(2) * sum_{alpha,beta} gamma5(alpha, beta)
  //               * [ Inv((b,beta,j,1), (a,alpha,i,0))
  //                 + Inv((b,beta,j,0), (a,alpha,i,1)) ]
  //
  // Under DTXQCD_DN_COMPLEX_SYMMETRIC: M48_LL = conj(M48_UR), so a single
  // complex variable d_{(a,i),(b,j)} embeds at M48_UR_{ra, kDim24+cb} =
  // c*g5*d AND at M48_LL_{kDim24+ra, cb} = c*g5*conj(d).  For real
  // S = -(1/2) log|det M48| the full directional derivative is
  //   dS/dε = 2 Re { c*g5 * [∂S/∂M_UR + conj(∂S/∂M_LL)] * Y }
  // where ∂S/∂M_UR = -(1/4) Inv(kDim24+cb, ra) and ∂S/∂M_LL =
  // -(1/4) Inv(cb, kDim24+ra).  Matching to AuxInnerReal = Re Tr(F * adj Y)
  // = Re Σ F * conj(Y) requires F = conj(holomorphic) + antiholomorphic, i.e.,
  //   F_d ∝ conj(Inv(kDim24+cb, ra)) + Inv(cb, kDim24+ra).
  // (FD vs analytic verified at rel ~ 1e-7 in Test_dtxqcd_logdet_aux_force.)
  const RealD sqrt2 = DtxqcdOffdiagFactor();
  const bool dn_cs = DtxqcdDnComplexSymmetric();
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
              if (dn_cs) {
                // M48_LL = conj(M48_UR): the holomorphic ∂_d (via UR) and
                // anti-holomorphic ∂_{d*} (via LL) contribute; AuxInnerReal
                // uses conj(Y), so the holomorphic term needs a conj to
                // land on the right Re/Im signs.
                val += g5 * (DtxqcdConj(Inv(kDim24 + cb, ra)) + Inv(cb, kDim24 + ra));
              } else {
                // Hermitian d: both UR + LL positions contribute directly.
                val += g5 * (Inv(kDim24 + cb, ra) + Inv(cb, kDim24 + ra));
              }
            }
          }
          d_force()(a, b)(i, j) = -sqrt2 * val;
        }
      }
    }
  }

  // ---- n^{ij}_{ab} force (off-diagonal, +sqrt(2)*identity in spin) -------
  //
  // v2 corrected: same sqrt(2) factor as d.
  //   F_n^{ij}_{ab} = -sqrt(2) * sum_alpha [ Inv((b,alpha,j,1), (a,alpha,i,0))
  //                                        + Inv((b,alpha,j,0), (a,alpha,i,1)) ]
  // (Same UR/LL split logic as d when DTXQCD_DN_COMPLEX_SYMMETRIC is ON.)
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          ComplexD val(0, 0);
          for (int alpha = 0; alpha < Ns; ++alpha) {
            int ra = DtxqcdSiteIdx24(a, alpha, i);
            int cb = DtxqcdSiteIdx24(b, alpha, j);
            if (dn_cs) {
              val += DtxqcdConj(Inv(kDim24 + cb, ra)) + Inv(cb, kDim24 + ra);
            } else {
              val += Inv(kDim24 + cb, ra) + Inv(cb, kDim24 + ra);
            }
          }
          n_force()(a, b)(i, j) = -sqrt2 * val;
        }
      }
    }
  }

  // ---- s singlet force (diagonal in (a==b, i==j), spin scalar) -----------
  //
  // v2 corrected: +s in BOTH upper and lower diagonals (no sign flip on
  // the lower).
  //   F_s = -sum_a sum_alpha sum_i [ Inv((a,alpha,i,0), (a,alpha,i,0))
  //                                + Inv((a,alpha,i,1), (a,alpha,i,1)) ]
  {
    ComplexD val(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a) {
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          int r = DtxqcdSiteIdx24(a, alpha, i);
          val += Inv(r, r) + Inv(kDim24 + r, kDim24 + r);
        }
      }
    }
    s_force()()() = -val;
  }

  // ---- p singlet force (diagonal in (a==b, i==j), gamma5 in spin) --------
  //
  // v2 corrected: +p gamma5 in BOTH upper and lower diagonals.
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
            val += g5 * (Inv(ra, cb) + Inv(kDim24 + ra, kDim24 + cb));
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
  // 2026-06-13 (Pfaffian restore): upper-block clover has -(csw/2) F σ_{μν}
  // and lower-block clover has +(csw/2) F^T σ_{μν} (opposite sign on lower,
  // required by the C·K Pfaffian antisymmetry — see DTXQCDDeltaCloverOp.h).
  // So dS/dF_p has upper minus lower contribution; we accumulate val =
  // (upper_contrib) - (lower_contrib) and multiply by -csw/2 at the end:
  //   cs = -csw/2 · (upper - lower) = -csw/2·upper + csw/2·lower  ✓
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
        cs()()(i_c, j_c) = ComplexD(-0.5 * csw, 0.0) * DtxqcdConj(val);
      }
    }
    clover_sigma[p_idx] = cs;
  }
}

}  // namespace DtxqcdSiteForceKernel

NAMESPACE_END(Grid);
