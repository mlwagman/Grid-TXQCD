#pragma once
// Per-site DTXQCD aux/gauge-clover force kernels, parameterized over a
// "weight-matrix lookup" callable Inv(R, C) -> ComplexD that abstracts what
// stands in for M_ee_48^{-1}(x) in the trace formula.
//
// Two callers:
//   * DTXQCDLogDetCloverEOAction:
//        Inv(R, C) is literally the (R, C) entry of the per-site 48x48 inverse
//        of M_ee_48(x).  Force formula at each even site x:
//             dS/dX(x) = -Tr( M^{-1}(x) * dM(x)/dX )
//                      = -sum_{R,C} dM_{RC}/dX(x) * M^{-1}_{CR}(x)
//        (so the "Inv(row, col)" passed in is the matrix entry M^{-1}[row][col],
//        which appears multiplied by dM[col][row] in the trace expression.)
//
//   * DTXQCDWilsonCloverRationalEOAction (per rational pole k):
//        Inv(R, C) = conj(Y(x)[R]) * X(x)[C]  (a rank-1 bilinear), giving
//             -2 Re < Y, dMpc/dX  X > = -2 Re sum_{R,C} dM_{RC}/dX * conj(Y_R) X_C
//        at each odd site (with the analogous (Z, W) for the even-block term).
//        The factor "-2 Re" is applied by the caller AFTER the kernel runs
//        (kernel returns the unconjugated sum); see comments at end of file.
//
// The kernel updates per-site scalar-object force slots in-place (additive
// in caller; the helpers assume callers have zeroed them or scaled them).
// For LogDet the slots are overwritten (= per-site value), for RHMC they are
// summed across poles with a coefficient.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>

NAMESPACE_BEGIN(Grid);

namespace DtxqcdSiteForceKernel {

// Scalar-object aliases pulled from the LatticeXxx types so callers don't
// have to repeat the typedef boilerplate.
using SigSobj = typename LatticeDtxqcdSigma::vector_object::scalar_object;
using PiSobj  = typename LatticeDtxqcdPi::vector_object::scalar_object;
using TSobj   = typename LatticeDtxqcdT::vector_object::scalar_object;
using DSobj   = typename LatticeDtxqcdD::vector_object::scalar_object;
using NSobj   = typename LatticeDtxqcdN::vector_object::scalar_object;
using CMsobj  = typename LatticeColourMatrix::vector_object::scalar_object;

static constexpr int kDim24 = kDtxqcdSiteDim24;

// ----------------------------------------------------------------------
// Aux-field force kernel.  The factor "-coeff_*" prefactors are determined
// inside the kernel from the dM/dX coefficients (1/sqrt(2), 2, i, ...).
// The kernel writes WITHOUT the overall caller-supplied scaling; the caller
// folds in the LogDet (-1) or RHMC (-2 Re alpha_k) afterwards.
//
// Output convention (LogDet-style): each per-site sobj stores
//     F_X(x) = -sum_{R,C} dM_{RC}/dX(x) * Inv(R, C)
// so for LogDet, dS_LD/dX = sum_x F_X(x); and for RHMC, the caller multiplies
// by 2 alpha_k and takes Re(...) per slot, then sums over poles.
//
// (For RHMC: dS_RHMC/dX = -sum_k alpha_k * 2 Re sum_x [-F_X(x)]; i.e. the kernel
// returns -dS/dX up to those scalar factors, with the bilinear-Inv plugged in.)
// ----------------------------------------------------------------------
template <class InvLookup>
inline void AuxForceAt(InvLookup Inv,
                       const DtxqcdSpinMatrices &spin,
                       SigSobj &sig_force,
                       PiSobj  &pi_force,
                       TSobj   &t_force,
                       DSobj   &d_force,
                       NSobj   &n_force) {
  const auto &tau = DtxqcdPauliEigen();
  const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);

  sig_force = Zero();
  pi_force  = Zero();
  t_force   = Zero();
  d_force   = Zero();
  n_force   = Zero();

  // ---- sigma^A force ----------------------------------------------------
  for (int A = 0; A < DtxqcdNTriplet; ++A) {
    ComplexD val(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b) {
        ComplexD tab = tau[A](a, b);
        if (tab == ComplexD(0, 0)) continue;
        ComplexD s(0, 0);
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            int row_b = DtxqcdSiteIdx24(b, alpha, i);
            int col_a = DtxqcdSiteIdx24(a, alpha, i);
            s += Inv(row_b, col_a)
               + Inv(kDim24 + row_b, kDim24 + col_a);
          }
        val += tab * s;
      }
    sig_force()()(A) = -inv_sqrt2 * val;
  }

  // ---- pi^A force --------------------------------------------------------
  for (int A = 0; A < DtxqcdNTriplet; ++A) {
    ComplexD val(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b) {
        ComplexD tab = tau[A](a, b);
        if (tab == ComplexD(0, 0)) continue;
        ComplexD inner(0, 0);
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int beta = 0; beta < Ns; ++beta) {
            ComplexD g5 = spin.gamma5(alpha, beta);
            if (g5 == ComplexD(0, 0)) continue;
            for (int i = 0; i < Nc; ++i) {
              int row_b_beta  = DtxqcdSiteIdx24(b, beta,  i);
              int col_a_alpha = DtxqcdSiteIdx24(a, alpha, i);
              inner += g5 * (Inv(row_b_beta, col_a_alpha)
                           + Inv(kDim24 + row_b_beta,
                                 kDim24 + col_a_alpha));
            }
          }
        val += tab * inner;
      }
    pi_force()()(A) = -inv_sqrt2 * val;
  }

  // ---- t^A_{mu,nu} force -------------------------------------------------
  const ComplexD ci(0.0, 1.0);
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu) {
      int p_idx = -1;
      {
        int k = 0;
        for (int m2 = 0; m2 < Nd; ++m2)
          for (int n2 = m2 + 1; n2 < Nd; ++n2) {
            if (m2 == mu && n2 == nu) p_idx = k;
            ++k;
          }
      }
      for (int A = 0; A < DtxqcdNTriplet; ++A) {
        ComplexD val(0, 0);
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int b = 0; b < DtxqcdNf; ++b) {
            ComplexD tab = tau[A](a, b);
            if (tab == ComplexD(0, 0)) continue;
            ComplexD inner(0, 0);
            for (int alpha = 0; alpha < Ns; ++alpha)
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD smn = spin.sigma_munu[p_idx](alpha, beta);
                if (smn == ComplexD(0, 0)) continue;
                for (int i = 0; i < Nc; ++i) {
                  int row_b_beta  = DtxqcdSiteIdx24(b, beta,  i);
                  int col_a_alpha = DtxqcdSiteIdx24(a, alpha, i);
                  inner += smn * (Inv(row_b_beta, col_a_alpha)
                                - Inv(kDim24 + row_b_beta,
                                      kDim24 + col_a_alpha));
                }
              }
            val += tab * inner;
          }
        ComplexD t_A = -ci * val;
        t_force()(mu, nu)(A) =  t_A;
        t_force()(nu, mu)(A) = -t_A;
      }
    }
  }

  // ---- d^{ij} force (off-diagonal, +2 gamma5) ---------------------------
  for (int k = 0; k < Nc; ++k) {
    for (int l = 0; l < Nc; ++l) {
      ComplexD val(0, 0);
      for (int a = 0; a < DtxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int beta = 0; beta < Ns; ++beta) {
            ComplexD g5 = spin.gamma5(alpha, beta);
            if (g5 == ComplexD(0, 0)) continue;
            int u_a_alpha_k = DtxqcdSiteIdx24(a, alpha, k);
            int u_a_beta_l  = DtxqcdSiteIdx24(a, beta,  l);
            val += g5 * (Inv(kDim24 + u_a_beta_l, u_a_alpha_k)
                       + Inv(u_a_beta_l, kDim24 + u_a_alpha_k));
          }
      d_force()()(k, l) = ComplexD(-2.0, 0.0) * val;
    }
  }

  // ---- n^{ij} force (off-diagonal, +2 identity) -------------------------
  for (int k = 0; k < Nc; ++k) {
    for (int l = 0; l < Nc; ++l) {
      ComplexD val(0, 0);
      for (int a = 0; a < DtxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha) {
          int u_a_alpha_k = DtxqcdSiteIdx24(a, alpha, k);
          int u_a_alpha_l = DtxqcdSiteIdx24(a, alpha, l);
          val += Inv(kDim24 + u_a_alpha_l, u_a_alpha_k)
               + Inv(u_a_alpha_l, kDim24 + u_a_alpha_k);
        }
      n_force()()(k, l) = ComplexD(-2.0, 0.0) * val;
    }
  }
}

// ----------------------------------------------------------------------
// Per-site clover_sigma builder (dS/dF_{mu,nu, (i, j)} at site x), for the
// Cmunu chain rule.  Same conventions as LogDet: result is
//     cs(i, j) = -0.5 * csw * conj(   sum sigma(a, b)
//                                       * [ Inv(b, j-col-upper, a, i-col-upper)
//                                         - Inv(b, i-col-lower, a, j-col-lower) ] )
// The conj() and -0.5 csw factors arise from Convention-B Cmunu input
// (anti-Hermitian sigma_Grid) and Convention-A HMC integrator.  Caller's
// final force gets an additional -0.5 from the Cmunu chain rule, applied
// AFTER this kernel.
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
