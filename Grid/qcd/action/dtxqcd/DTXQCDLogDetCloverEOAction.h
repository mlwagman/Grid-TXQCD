#pragma once
// Log-determinant action for the even-site diagonal block of the
// EO-preconditioned DTXQCD doubled Wilson-Clover operator.
//
//   det(M) = det(Mee) * det(Mpc), so when using Mpc for the pseudofermion
//   action we need S_logdet = -ln |det(Mee_48)| as a separate action term.
//
// Each per-site M_ee is the 48x48 doubled site matrix from DTXQCDSiteMatrix:
//   upper block:  m I + Delta_diag (sigma^A, pi^A, t^A) + (optional) -(csw/2) F sigma
//   lower block:  m I + Delta_diag_lower (tensor sign-flipped) + (optional) +(csw/2) F^T sigma
//   off-diag:     2 d gamma_5 + 2 n
//
// Force formula at each even site x:
//   dS/d(X)(x) = -Tr(M_ee^{-1}(x) * dM_ee/dX(x))
// where X ranges over the aux fields (sigma^A, pi^A, t^A_{mu,nu}, d^{ij}, n^{ij})
// and (csw != 0 case) the gauge field via the clover term.
//
// v1 implementation:
//   - CPU-only (no GPU acceleration; TXQCD's GPU path can port later).
//   - Aux-field forces:    full analytic deriv (this file).
//   - Gauge clover force:  via dF/dU chain rule, WilsonCloverHelpers::Cmunu,
//                          mirroring TXQCDLogDetCloverEOAction.  Includes
//                          contributions from BOTH upper (-(csw/2) F sigma)
//                          and lower (+(csw/2) F^T sigma) clover terms.
//                          For the lower block, F^T means F's (j, i) entry,
//                          so the M_inv_ll trace formula has swapped (i, j)
//                          indices on the M_inv access.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>
#include <Grid/qcd/utils/WilsonLoops.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDLogDetCloverEOAction : public Action<DTXQCDField> {
 public:
  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDLogDetCloverEOAction(GridCartesian &grid,
                             GridRedBlackCartesian &rbgrid,
                             RealD mass, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw), spin_(grid) {}

  std::string action_name() override { return "DTXQCDLogDetCloverEOAction"; }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " csw=" << csw_ << std::endl;
    return os.str();
  }

  void refresh(const DTXQCDField &U, GridSerialRNG &, GridParallelRNG &) override {}

  // ------------------------------------------------------------------
  //  S(U) = -sum_{x in EVEN} log |det(M_ee_48(x))|
  // ------------------------------------------------------------------
  RealD S(const DTXQCDField &U) override {
    auto FS = BuildFS(U);
    RealD logdet = 0.0;
    Coordinate gd(grid_.GlobalDimensions());
    for (int x = 0; x < gd[0]; ++x)
      for (int y = 0; y < gd[1]; ++y)
        for (int z = 0; z < gd[2]; ++z)
          for (int s = 0; s < gd[3]; ++s) {
            if (((x + y + z + s) & 1) != 0) continue;  // EVEN only
            Coordinate coord(std::vector<int>{x, y, z, s});
            Eigen::MatrixXcd M48;
            BuildSiteMatrix48(U, FS, coord, M48);
            ComplexD det = M48.partialPivLu().determinant();
            logdet += std::log(std::abs(det));
          }
    grid_.GlobalSum(logdet);
    RealD action = -logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  // ------------------------------------------------------------------
  //  deriv(U, dSdU): aux-field forces.  Gauge force is zero (TODO: clover).
  // ------------------------------------------------------------------
  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    dSdU = Zero();
    auto FS = BuildFS(U);
    Coordinate gd(grid_.GlobalDimensions());

    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
    typedef typename LatticeDtxqcdPi::vector_object::scalar_object    PiSobj;
    typedef typename LatticeDtxqcdT::vector_object::scalar_object     TSobj;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSobj;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSobj;
    typedef typename LatticeColourMatrix::vector_object::scalar_object CMsobj;

    const auto &tau = DtxqcdPauliEigen();
    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);

    // clover_sigma_full[mn] holds dS/dF_{mu,nu, (i, j)} per (mu<nu) pair as a
    // LatticeColourMatrix, evaluated only on EVEN sites (zero on odd).  Fed
    // to WilsonCloverHelpers::Cmunu after the per-site loop to chain-rule
    // through dF/dU.  Used only when csw != 0.
    std::vector<LatticeColourMatrix> clover_sigma_full;
    if (csw_ != 0.0) {
      clover_sigma_full.reserve(6);
      for (int k = 0; k < 6; ++k) {
        clover_sigma_full.emplace_back(&grid_);
        clover_sigma_full.back() = Zero();
      }
    }

    for (int x = 0; x < gd[0]; ++x)
      for (int y = 0; y < gd[1]; ++y)
        for (int z = 0; z < gd[2]; ++z)
          for (int s = 0; s < gd[3]; ++s) {
            if (((x + y + z + s) & 1) != 0) continue;  // EVEN only
            Coordinate coord(std::vector<int>{x, y, z, s});

            Eigen::MatrixXcd M48, Inv;
            BuildSiteMatrix48(U, FS, coord, M48);
            Inv = M48.inverse();

            // Aux-field forces at this even site.
            SigSobj sig_force; sig_force = Zero();
            PiSobj  pi_force;  pi_force  = Zero();
            TSobj   t_force;   t_force   = Zero();
            DSobj   d_force;   d_force   = Zero();
            NSobj   n_force;   n_force   = Zero();

            // ---- sigma^A force ----------------------------------------
            // dS/dsigma^A = -Tr(M^{-1} dM/dsigma^A) where the trace sums
            //   Sigma_{R,K} M^{-1}[R,K] dM[K,R]   (Tr(AB) = sum A[i,j] B[j,i])
            // dM/dsigma^A nonzero at (K=(a,alpha,i), R=(b,alpha,i)) with value
            //   (1/sqrt 2) tau^A_{ab} for both upper and lower diagonal blocks
            // (sigma piece unchanged under Cstar), so
            //   dS/dsigma^A = -(1/sqrt 2) sum_{a,b,alpha,i} tau^A_{ab}
            //                   * [Inv[U(b,alpha,i), U(a,alpha,i)]
            //                    + Inv[L(b,alpha,i), L(a,alpha,i)]]
            // Note: M^{-1}[R, K] with R-index from b, K-index from a -- the
            // (a, b) trace-formula transposition is what makes this match the
            // FD for imaginary Pauli (tau^2); a naive Inv[(a,...), (b,...)]
            // would be the Hermitian conjugate and disagree in sign for tau^2.
            for (int A = 0; A < DtxqcdNTriplet; ++A) {
              ComplexD val(0, 0);
              for (int a = 0; a < DtxqcdNf; ++a)
                for (int b = 0; b < DtxqcdNf; ++b) {
                  ComplexD tab = tau[A](a, b);
                  if (tab == ComplexD(0, 0)) continue;
                  ComplexD s(0, 0);
                  for (int alpha = 0; alpha < Ns; ++alpha)
                    for (int i = 0; i < Nc; ++i) {
                      int row_b = DtxqcdSiteIdx24(b, alpha, i);  // M^{-1} row
                      int col_a = DtxqcdSiteIdx24(a, alpha, i);  // M^{-1} col
                      s += Inv(row_b, col_a)
                         + Inv(kDim24 + row_b, kDim24 + col_a);
                    }
                  val += tab * s;
                }
              sig_force()()(A) = -inv_sqrt2 * val;
            }

            // ---- pi^A force ---------------------------------------------
            // dM/dpi^A nonzero at (K=(a,alpha,i), R=(b,beta,i)) with value
            //   (1/sqrt 2) tau^A_{ab} gamma5(alpha, beta).
            // Tr formula: M^{-1}[R=(b,beta,i), K=(a,alpha,i)] * tau^A_{ab} gamma5(alpha,beta).
            //   dS/dpi^A = -(1/sqrt 2) sum tau^A_{ab} gamma5(alpha, beta)
            //                * [Inv[U(b,beta,i), U(a,alpha,i)]
            //                 + Inv[L(b,beta,i), L(a,alpha,i)]]
            for (int A = 0; A < DtxqcdNTriplet; ++A) {
              ComplexD val(0, 0);
              for (int a = 0; a < DtxqcdNf; ++a)
                for (int b = 0; b < DtxqcdNf; ++b) {
                  ComplexD tab = tau[A](a, b);
                  if (tab == ComplexD(0, 0)) continue;
                  ComplexD inner(0, 0);
                  for (int alpha = 0; alpha < Ns; ++alpha)
                    for (int beta = 0; beta < Ns; ++beta) {
                      ComplexD g5 = spin_.gamma5(alpha, beta);
                      if (g5 == ComplexD(0, 0)) continue;
                      for (int i = 0; i < Nc; ++i) {
                        int row_b_beta = DtxqcdSiteIdx24(b, beta,  i);
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

            // ---- t^A_{mu,nu} force --------------------------------------
            // dM/dt^A_{mu,nu} is (+i) tau^A sigma_munu in upper, (-i) in lower.
            //   t_force(mu,nu, A) = -(i) sum tau^A sigma_munu * (Inv_U - Inv_L)
            // Antisymmetrize at the end: t(nu,mu,A) = -t(mu,nu,A).
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
                // dM/dt^A_{mu,nu} = +i tau^A sigma_munu in upper, -i in lower.
                // Tr formula uses M^{-1}[R=(b,beta,i), K=(a,alpha,i)]
                // with coefficient tau^A_{ab} sigma_munu(alpha, beta).
                for (int A = 0; A < DtxqcdNTriplet; ++A) {
                  ComplexD val(0, 0);
                  for (int a = 0; a < DtxqcdNf; ++a)
                    for (int b = 0; b < DtxqcdNf; ++b) {
                      ComplexD tab = tau[A](a, b);
                      if (tab == ComplexD(0, 0)) continue;
                      ComplexD inner(0, 0);
                      for (int alpha = 0; alpha < Ns; ++alpha)
                        for (int beta = 0; beta < Ns; ++beta) {
                          ComplexD smn = spin_.sigma_munu[p_idx](alpha, beta);
                          if (smn == ComplexD(0, 0)) continue;
                          for (int i = 0; i < Nc; ++i) {
                            int row_b_beta = DtxqcdSiteIdx24(b, beta, i);
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

            // ---- d^{ij} force (off-diagonal block, +2 gamma5) -----------
            // dM/dd^{ij}_{kl} = 2 gamma5 in (upper-row=(a,alpha,k), col=(lower,a,beta,l))
            //                  and (lower-row=(a,alpha,k), col=(upper,a,beta,l)).
            //   d_force(k,l) = -2 sum_{a,alpha,beta} gamma5(alpha,beta)
            //                  * [Inv[L(a,beta,l), U(a,alpha,k)]
            //                   + Inv[U(a,beta,l), L(a,alpha,k)]]
            for (int k = 0; k < Nc; ++k) {
              for (int l = 0; l < Nc; ++l) {
                ComplexD val(0, 0);
                for (int a = 0; a < DtxqcdNf; ++a)
                  for (int alpha = 0; alpha < Ns; ++alpha)
                    for (int beta = 0; beta < Ns; ++beta) {
                      ComplexD g5 = spin_.gamma5(alpha, beta);
                      if (g5 == ComplexD(0, 0)) continue;
                      int u_a_alpha_k = DtxqcdSiteIdx24(a, alpha, k);
                      int u_a_beta_l  = DtxqcdSiteIdx24(a, beta,  l);
                      val += g5 * (Inv(kDim24 + u_a_beta_l, u_a_alpha_k)
                                 + Inv(u_a_beta_l, kDim24 + u_a_alpha_k));
                    }
                d_force()()(k, l) = ComplexD(-2.0, 0.0) * val;
              }
            }

            // ---- n^{ij} force (off-diagonal, +2 identity) ---------------
            //   n_force(k,l) = -2 sum_{a,alpha}
            //                  * [Inv[L(a,alpha,l), U(a,alpha,k)]
            //                   + Inv[U(a,alpha,l), L(a,alpha,k)]]
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

            // ---- Clover Sigma per (mu<nu) ------------------------------
            // Build clover_sigma[mn](i, j) = dS/dF_{mu,nu, (i, j)}(x) for the
            // Cmunu chain rule below.  From the trace formula:
            //
            //   upper:   M_clover += -(csw/2) F sigma_{Grid}
            //     dS/dF (upper) = +(csw/2) sum sigma(alpha, beta)
            //                      * M_inv_uu[(a, beta, j), (a, alpha, i)]
            //   lower:   M_clover += +(csw/2) F^T sigma_{Grid}
            //     dS/dF (lower) = -(csw/2) sum sigma(alpha, beta)
            //                      * M_inv_ll[(a, beta, i), (a, alpha, j)]
            //
            // Note: lower contribution uses indices swapped (i <-> j in the
            // M_inv rows/cols) because F^T's (k, l) entry is F's (l, k).
            if (csw_ != 0.0) {
              for (int p_idx = 0; p_idx < 6; ++p_idx) {
                CMsobj cs;
                cs = Zero();
                for (int i_c = 0; i_c < Nc; ++i_c) {
                  for (int j_c = 0; j_c < Nc; ++j_c) {
                    ComplexD val(0, 0);
                    for (int a = 0; a < DtxqcdNf; ++a) {
                      for (int alpha = 0; alpha < Ns; ++alpha) {
                        for (int beta = 0; beta < Ns; ++beta) {
                          ComplexD smn = spin_.sigma_munu[p_idx](alpha, beta);
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
                    // Cmunu expects TXQCD's convention, which for anti-Hermitian
                    // sigma_Grid is -conj(natural-Wirtinger dS/dF).  The minus
                    // comes from sigma_Grid(beta, alpha) = -conj(sigma_Grid(alpha, beta))
                    // for anti-Hermitian sigma_Grid.
                    cs()()(i_c, j_c) = ComplexD(-0.5 * csw_, 0.0) * std::conj(val);
                  }
                }
                pokeSite(cs, clover_sigma_full[p_idx], coord);
              }
            }

            // Poke per-site forces into the full-volume lattice slots.
            pokeSite(sig_force, dSdU.sigma, coord);
            pokeSite(pi_force,  dSdU.pi,    coord);
            pokeSite(t_force,   dSdU.t,     coord);
            pokeSite(d_force,   dSdU.d,     coord);
            pokeSite(n_force,   dSdU.n,     coord);
          }

    // ---- Gauge clover force via Cmunu chain rule ------------------------
    // For each Lorentz mu, sum over nu != mu of (sign * Cmunu(Ulinks,
    // clover_sigma_full[mn], mu, nu)), then poke Ulinks[mu] * force_mu into
    // the dSdU.U Lorentz mu slot.  Cmunu returns "Convention B" (full
    // gradient); apply -0.5 to convert to "Convention A" expected by
    // Grid's HMC integrator.  Mirrors TXQCDLogDetCloverEOAction.
    if (csw_ != 0.0) {
      typedef WilsonImplR Impl;
      std::vector<LatticeColourMatrix> Ulinks;
      Ulinks.reserve(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu(&grid_);
        Umu = PeekIndex<LorentzIndex>(U.U, mu);
        Ulinks.push_back(std::move(Umu));
      }

      LatticeGaugeField clover_force(&grid_);
      clover_force = Zero();

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix force_mu(&grid_);
        force_mu = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (mu == nu) continue;
          // Canonical (m<n) index from (mu, nu) -- our 6-pair ordering.
          int m = std::min(mu, nu);
          int n = std::max(mu, nu);
          int mn = 0;
          {
            int k = 0;
            for (int mm = 0; mm < Nd; ++mm)
              for (int nn = mm + 1; nn < Nd; ++nn) {
                if (mm == m && nn == n) mn = k;
                ++k;
              }
          }
          // sigma_{nu, mu} = -sigma_{mu, nu}: fold antisymmetry sign into a
          // scalar rather than building a redundant -clover_sigma lattice.
          RealD sign = (mu < nu) ? 1.0 : -1.0;
          force_mu += (0.25 * sign) *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, clover_sigma_full[mn],
                                               mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      dSdU.U = ComplexD(-0.5, 0.0) * clover_force;
    }
  }

 private:
  // Build the field-strength F_{mu,nu} (6 pairs) from U via Grid's WilsonLoops.
  std::vector<LatticeColourMatrix> BuildFS(const DTXQCDField &U) const {
    std::vector<LatticeColourMatrix> FS;
    if (csw_ == 0.0) return FS;
    FS.reserve(6);
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(U.U.Grid());
        WilsonLoops<WilsonImplR>::FieldStrength(F, U.U, mu, nu);
        FS.push_back(std::move(F));
      }
    return FS;
  }

  // Build the 48x48 per-site M_ee matrix.
  void BuildSiteMatrix48(const DTXQCDField &U,
                         const std::vector<LatticeColourMatrix> &FS,
                         const Coordinate &coord,
                         Eigen::MatrixXcd &M48) {
    DtxqcdSiteAux aux =
        DtxqcdSiteAux::Extract(U.sigma, U.pi, U.t, U.d, U.n, coord);
    Eigen::MatrixXcd M_upper, M_lower, M_off;
    if (csw_ != 0.0) {
      DtxqcdSiteClover clover = DtxqcdSiteClover::Extract(FS, coord);
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper, csw_, &clover);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower, csw_, &clover);
    } else {
      DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper);
      DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower);
    }
    DtxqcdBuildOffDiagBlock24(aux, spin_, M_off);
    DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);
  }

  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD                  mass_;
  RealD                  csw_;
  DtxqcdSpinMatrices     spin_;
};

NAMESPACE_END(Grid);
