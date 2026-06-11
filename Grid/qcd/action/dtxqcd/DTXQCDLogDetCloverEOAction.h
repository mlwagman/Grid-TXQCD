#pragma once
// Log-determinant action for the even-site diagonal block of the
// EO-preconditioned DTXQCD doubled Wilson-Clover operator.
//
// For the Pfaffian weight  |Pf(D_doubled)| = |det(D_doubled)|^{1/2}
// (paper Eq. 321: Pf(D) = det(D^2)^{1/4}), the total fermion action is
//   S = -log|Pf(D)| = -(1/2) log|det(D)|
//                  = -(1/2) [log|det(M_ee)| + log|det(Mpc)|]
// so this LogDet action contributes
//   S_LD = -(1/2) sum_{x even} log|det(M_ee_48(x))|
// The complementary 1/2 factor goes into the RHMC pseudofermion action on
// Mpc via the x^{-1/4} (action) and x^{+1/8} (heatbath) rational exponents,
// so each block contributes |det|^{1/2} to the path-integral weight.
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
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h>
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
  //  S(U) = -(1/2) sum_{x in EVEN} log |det(M_ee_48(x))|
  //  (1/2 factor for the Pfaffian weight; see header.)
  // ------------------------------------------------------------------
  RealD S(const DTXQCDField &U) override {
    auto FS = BuildFS(U);
    RealD logdet = 0.0;
    RealD sum_arg = 0.0;
    RealD min_log_absdet = 1e300;
    RealD n_small = 0.0;
    const RealD small_threshold = -10.0;  // log|det| < -10 ~ |det| < 5e-5
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
            RealD logabs = std::log(std::abs(det));
            logdet += logabs;
            sum_arg += std::arg(det);
            if (logabs < min_log_absdet) min_log_absdet = logabs;
            if (logabs < small_threshold) n_small += 1.0;
          }
    grid_.GlobalSum(logdet);
    grid_.GlobalSum(sum_arg);
    grid_.GlobalSum(n_small);
    RealD global_min;
    {
      RealD x = min_log_absdet;
      grid_.GlobalSum(x);
      global_min = x;
    }
    RealD action = -0.5 * logdet;
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << "  sum_arg(det) = " << sum_arg
              << "  min_log|det| = " << min_log_absdet
              << "  n_small = " << n_small << std::endl;
    return action;
  }

  // ------------------------------------------------------------------
  //  deriv(U, dSdU): aux-field forces + gauge clover force (csw != 0).
  //  Aux: per-site -Tr(M_ee^{-1} dM_ee/dX) via DTXQCDSiteForceKernel.
  //  Gauge: per-site clover_sigma fed to WilsonCloverHelpers::Cmunu, both
  //  upper (-(csw/2) F sigma) and lower (+(csw/2) F^T sigma) contributions.
  // ------------------------------------------------------------------
  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    dSdU = Zero();
    auto FS = BuildFS(U);
    Coordinate gd(grid_.GlobalDimensions());

    using DtxqcdSiteForceKernel::SigSobj;
    using DtxqcdSiteForceKernel::PiSobj;
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::SSobj;
    using DtxqcdSiteForceKernel::PSobj;

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

            // Aux-field forces at this even site (kernel writes per-site sobj
            // values; the LogDet trace formula is folded into the kernel's
            // -(1/sqrt 2) / -2 / -i prefactors).  See DTXQCDSiteForceKernel.h.
            SigSobj sig_force;
            PiSobj  pi_force;
            DSobj   d_force;
            NSobj   n_force;
            SSobj   s_force;
            PSobj   p_force;
            auto InvLookup =
                [&Inv](int r, int c) -> ComplexD { return Inv(r, c); };
            DtxqcdSiteForceKernel::AuxForceAt(InvLookup, spin_, sig_force,
                                              pi_force, d_force, n_force,
                                              s_force, p_force);

            // ---- Clover Sigma per (mu<nu) -------------------------------
            // dS/dF_{mu,nu, (i, j)}(x) at site x for the Cmunu chain rule.
            // Both upper (-(csw/2) F sigma) and lower (+(csw/2) F^T sigma)
            // contributions are folded in by CloverSigmaAt.
            if (csw_ != 0.0) {
              std::array<DtxqcdSiteForceKernel::CMsobj, 6> cs_arr;
              DtxqcdSiteForceKernel::CloverSigmaAt(InvLookup, spin_, csw_, cs_arr);
              for (int p_idx = 0; p_idx < 6; ++p_idx)
                pokeSite(cs_arr[p_idx], clover_sigma_full[p_idx], coord);
            }

            // Wirtinger -> physical-gradient conversion: for a real-valued
            // S(sigma), the Wirtinger derivative F_W = dS/dsigma_ab^ij and
            // the physical gradient F_phys (the Hamilton-flow force the
            // integrator expects) are related by F_phys = 2 conj(F_W).  For
            // Hermitian F_W (which our force kernel returns), conj(F_W) =
            // transpose(F_W), so F_phys = 2 * transpose(F_W).  The factor of 2
            // is folded into the rational coefficient downstream of this poke
            // (and into the FD-test normalization on the aux side); here we
            // need only swap indices to deliver the physical gradient.
            auto TransposePoke = [&coord](auto &dst, const auto &fv) {
              std::remove_const_t<std::remove_reference_t<decltype(fv)>> ft;
              for (int a = 0; a < DtxqcdNf; ++a)
                for (int b = 0; b < DtxqcdNf; ++b)
                  for (int i = 0; i < Nc; ++i)
                    for (int j = 0; j < Nc; ++j)
                      ft()(a, b)(i, j) = fv()(b, a)(j, i);
              pokeSite(ft, dst, coord);
            };
            TransposePoke(dSdU.sigma, sig_force);
            TransposePoke(dSdU.pi,    pi_force);
            TransposePoke(dSdU.d,     d_force);
            TransposePoke(dSdU.n,     n_force);
            pokeSite(s_force,   dSdU.s,     coord);
            pokeSite(p_force,   dSdU.p,     coord);
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

    // Overall Pfaffian factor: S = -(1/2) log|det(M_ee)|, so scale all
    // contributions (aux + gauge) by 1/2.  See header comment.
    dSdU.sigma = ComplexD(0.5, 0.0) * dSdU.sigma;
    dSdU.pi    = ComplexD(0.5, 0.0) * dSdU.pi;
    dSdU.d     = ComplexD(0.5, 0.0) * dSdU.d;
    dSdU.n     = ComplexD(0.5, 0.0) * dSdU.n;
    dSdU.s     = ComplexD(0.5, 0.0) * dSdU.s;
    dSdU.p     = ComplexD(0.5, 0.0) * dSdU.p;
    dSdU.U     = ComplexD(0.5, 0.0) * dSdU.U;
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
        DtxqcdSiteAux::Extract(U.sigma, U.pi, U.d, U.n, U.s, U.p, coord);
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
