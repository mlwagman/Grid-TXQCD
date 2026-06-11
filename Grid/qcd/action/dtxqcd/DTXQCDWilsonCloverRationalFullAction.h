#pragma once
// Non-EO RHMC pseudofermion action for the DTXQCD doubled Wilson-Clover
// operator, exponent 1/4 (Pfaffian).  Sibling of
// DTXQCDWilsonCloverRationalEOAction but operating on the full doubled
// M instead of the EO Schur complement Mpc.
//
// Motivation (2026-06-09 PSD diagnostic, Test_dtxqcd_psd_check):
//   The EO Schur Mpc on the DTXQCD doubled operator has a sharp cliff
//   above aux_std ~ 0.5 -- max(Mpc^dag Mpc) jumps from O(30) to O(10^4)
//   while the full M^dag M stays mild (O(30) -> O(130) across the same
//   aux range).  The HMC breakdowns we tracked at production lambda=3
//   are entirely EO-side; the full operator is fine.  This action
//   bypasses the Schur cliff at the cost of slower multi-shift CG per
//   trajectory.  The TXQCD non-EO scout at lambda=3 gave 10/10
//   acceptance where the EO scout gave 0/10.
//
// Physics (same as EO sibling, no Schur factorization):
//   |Pf(D)| = det(D^2)^{1/4}
//   ⇒  S_RHMC = Phi^dag (M^dag M)^{-1/4} Phi
//   ⇒  heatbath:  Phi = (M^dag M)^{+1/8} eta,   eta ~ exp(-eta^dag eta)
//
// The non-EO factorization absorbs both the EO LogDet (Tr log M_ee) and
// the EO rational (det Mpc) into a single rational pseudofermion on the
// full operator.  So this action is used standalone -- no LogDet
// companion in the action stack.
//
// Force structure simplifies massively vs. the EO version: no
// batched MooeeInv intermediates (Y_k, W_e_k, Z_e_k), no Schur chain
// rule.  Just:
//   X_k = (M^dag M + sigma_k)^{-1} Phi   (multishift on full-volume M)
//   Y_k = M X_k
//   dS_k/dU contribution = -2 alpha_k * Re[Y_k^dag (dM/dU) X_k]
//
// Aux + clover-sigma forces use the same DTXQCDSiteForceKernel::AuxForceAt
// + CloverSigmaAt path as the EO version, with a per-site bilinear
// computed from (X_k, Y_k) and a coefficient coef = 2 * alpha_k.  The
// per-site loop runs over ALL sites (no CB filter) since X and Y live
// on the full volume.
//
// Hopping force per flavor per block uses Grid's stock
// WilsonFermion<>::DhopDeriv on full-volume X_k.upper/lower vs.
// Y_k.upper/lower -- matches the OneFlavourRational pattern in
// Grid/qcd/action/pseudofermion/OneFlavourRational.h.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/algorithms/approx/Remez.h>
#include <Grid/algorithms/approx/MultiShiftFunction.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDWilsonCloverRationalFullAction : public Action<DTXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;
  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDWilsonCloverRationalFullAction(GridCartesian &grid,
                                       GridRedBlackCartesian &rbgrid,
                                       RealD mass, Params &p,
                                       RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
        param_(p), Phi_(&grid), spin_(grid) {
    AlgRemez remez(param_.lo, param_.hi, param_.precision);
    std::cout << GridLogMessage
              << "[DTXQCDWilsonRationalFull] degree " << param_.degree
              << " rational for x^(-1/4)" << std::endl;
    remez.generateApprox(param_.degree, 1, 4);
    PowerNegQuarter.Init(remez, param_.tolerance, true);
    std::cout << GridLogMessage
              << "[DTXQCDWilsonRationalFull] degree " << param_.degree
              << " rational for x^(+1/8)" << std::endl;
    remez.generateApprox(param_.degree, 1, 8);
    PowerEighth.Init(remez, param_.tolerance, false);
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalFullAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " csw=" << csw_ << " lo=" << param_.lo << " hi=" << param_.hi
       << " degree=" << param_.degree << " tol=" << param_.tolerance
       << " MaxIter=" << param_.MaxIter << std::endl;
    return os.str();
  }

  // ------------------------------------------------------------------
  // refresh: Phi = (M^dag M)^{+1/8} eta,  eta full-volume gaussian.
  // The 1/sqrt(2) scale on eta matches Grid's CPS_MD_TIME convention.
  // ------------------------------------------------------------------
  void refresh(const DTXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    DTXQCDFermionDoubled eta(&grid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, eta.upper.f[a]);
      gaussian(pRNG, eta.lower.f[a]);
      eta.upper.f[a] = scale * eta.upper.f[a];
      eta.lower.f[a] = scale * eta.lower.f[a];
    }
    auto Dw = MakeEOp(U);
    DTXQCDMOp Mop(Dw);
    ApplyRational(Mop, PowerEighth, eta, Phi_);
  }

  // ------------------------------------------------------------------
  // S(U) = Phi^dag g_rat(M^dag M) Phi   where g_rat ≈ x^{-1/4}.
  // Same pole/residue identity argument as the EO sibling: deriv()
  // uses g_rat's poles, so S() must too for FD consistency.
  // ------------------------------------------------------------------
  RealD S(const DTXQCDField &U) override {
    auto Dw = MakeEOp(U);
    DTXQCDMOp Mop(Dw);
    DTXQCDFermionDoubled Y(&grid_);
    ApplyRational(Mop, PowerNegQuarter, Phi_, Y);
    ComplexD ip = innerProduct(Phi_, Y);
    RealD action = ip.real();
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  // ------------------------------------------------------------------
  // deriv: aux + clover gauge force + hopping gauge force on full M.
  //
  // Per pole k with X_k = (M^dag M + sigma_k)^{-1} Phi and Y_k = M X_k:
  //
  //   dS_k/dU contribution = -2 alpha_k Re[Y_k^dag (dM/dU) X_k]
  //
  // implemented as:
  //   - Aux + clover-sigma: DTXQCDSiteForceKernel::AuxForceAt /
  //     CloverSigmaAt with bilinear from (X_k, Y_k) at every site
  //   - Hopping: WilsonFermion<>::DhopDeriv on full-volume per-flavor
  //     per-block (upper, lower); lower block has gauge U_conj, chain
  //     rule maps via entry-wise conjugate()
  //   - Clover Cmunu chain rule from the accumulated clover_sigma
  // ------------------------------------------------------------------
  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    dSdU = Zero();
    auto Dw = MakeEOp(U);
    DTXQCDMOp Mop(Dw);

    // ---- Multi-shift solve into Xk[k] = (M^dag M + sigma_k)^{-1} Phi ----
    const int Npole = static_cast<int>(PowerNegQuarter.poles.size());
    std::vector<DTXQCDFermionDoubled> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&grid_);
    std::vector<RealD> md_tol(Npole, param_.mdtolerance);
    DTXQCDMultiShiftCG(Mop, PowerNegQuarter.poles, md_tol, Phi_, Xk,
                       param_.MaxIter);

    // ---- Y_k = M X_k (one Wilson + Delta + Cross + Clover apply per pole) ----
    std::vector<DTXQCDFermionDoubled> Yk;
    Yk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Yk.emplace_back(&grid_);
    for (int k = 0; k < Npole; ++k) Dw.M(Xk[k], Yk[k]);

    // ---- Build F_{mu,nu} for clover Cmunu chain rule (csw != 0 only) ----
    std::vector<LatticeColourMatrix> FS;
    if (csw_ != 0.0) {
      FS.reserve(6);
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = mu + 1; nu < Nd; ++nu) {
          LatticeColourMatrix F(&grid_);
          WilsonLoops<WilsonImplR>::FieldStrength(F, U.U, mu, nu);
          FS.push_back(std::move(F));
        }
    }

    std::vector<LatticeColourMatrix> clover_sigma_full;
    if (csw_ != 0.0) {
      clover_sigma_full.reserve(6);
      for (int k = 0; k < 6; ++k) {
        clover_sigma_full.emplace_back(&grid_);
        clover_sigma_full.back() = Zero();
      }
    }

    // ---- Per-pole force accumulation -----------------------------------
    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegQuarter.residues[k];
      AccumulateSiteForcesAll(Xk[k], Yk[k], ak, dSdU, clover_sigma_full);
      AccumulateHoppingForce(Xk[k], Yk[k], ak, Dw, dSdU);
    }

    // ---- Gauge clover force via Cmunu chain rule (csw != 0 only) ----
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
          int m = std::min(mu, nu);
          int n = std::max(mu, nu);
          int mn = 0;
          {
            int idx = 0;
            for (int mm = 0; mm < Nd; ++mm)
              for (int nn = mm + 1; nn < Nd; ++nn) {
                if (mm == m && nn == n) mn = idx;
                ++idx;
              }
          }
          RealD sign = (mu < nu) ? 1.0 : -1.0;
          force_mu += (0.25 * sign) *
              WilsonCloverHelpers<Impl>::Cmunu(Ulinks, clover_sigma_full[mn],
                                               mu, nu);
        }
        pokeLorentz(clover_force, Ulinks[mu] * force_mu, mu);
      }
      // Cmunu output is Convention B; multiply by -1/2 to get the integrator's
      // Convention A (mirrors LogDet + EO RHMC).
      dSdU.U = dSdU.U + ComplexD(-0.5, 0.0) * clover_force;
    }
  }

  DTXQCDFermionDoubled &PseudoFermion() { return Phi_; }

 protected:
  DTXQCDWilsonCloverFermionEO MakeEOp(const DTXQCDField &U) {
    DTXQCDField &Unc = const_cast<DTXQCDField &>(U);
    return DTXQCDWilsonCloverFermionEO(Unc.U, grid_, rbgrid_, mass_, csw_,
                                       Unc.sigma, Unc.pi, Unc.d, Unc.n,
                                       Unc.s,     Unc.p);
  }

  // Multishift-CG + linear combination: out = norm * in + sum_k residues[k] * xk[k]
  // where xk[k] = (M^dag M + poles[k])^{-1} in (multi-shift solve).
  // Identical structure to the EO sibling, but on full-volume grid.
  void ApplyRational(DTXQCDMOp &Mop, const MultiShiftFunction &rat,
                     const DTXQCDFermionDoubled &in,
                     DTXQCDFermionDoubled &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<DTXQCDFermionDoubled> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&grid_);
    DTXQCDMultiShiftCG(Mop, rat.poles, rat.tolerances, in, xk,
                       param_.MaxIter);

    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = rat.norm * in.upper.f[a];
      out.lower.f[a] = rat.norm * in.lower.f[a];
    }
    for (int k = 0; k < nshift; ++k) {
      RealD c = rat.residues[k];
      for (int a = 0; a < DtxqcdNf; ++a) {
        out.upper.f[a] = out.upper.f[a] + c * xk[k].upper.f[a];
        out.lower.f[a] = out.lower.f[a] + c * xk[k].lower.f[a];
      }
    }
  }

  // Extract a 48-component complex vector at one site of a doubled fermion.
  // Same layout as the EO sibling.
  static void ExtractSiteVec48(const DTXQCDFermionDoubled &F,
                                const Coordinate &c,
                                std::array<ComplexD, kDtxqcdSiteDim48> &v) {
    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    for (int a = 0; a < DtxqcdNf; ++a) {
      Fsobj sU, sL;
      peekSite(sU, F.upper.f[a], c);
      peekSite(sL, F.lower.f[a], c);
      for (int alpha = 0; alpha < Ns; ++alpha)
        for (int i = 0; i < Nc; ++i) {
          int idx = DtxqcdSiteIdx24(a, alpha, i);
          v[idx]            = ComplexD(sU()(alpha)(i).real(),
                                       sU()(alpha)(i).imag());
          v[kDim24 + idx]   = ComplexD(sL()(alpha)(i).real(),
                                       sL()(alpha)(i).imag());
        }
    }
  }

  // Per-pole hopping gauge force on the full operator.
  //
  // Per flavor a, per block (upper, lower):
  //   DhopDeriv(force, Y, X, DaggerNo) + DhopDeriv(force, X, Y, DaggerYes)
  // accumulated with the stock OneFlavourRational sign convention:
  // dSdU = dSdU + ak * tmp.  (The EO sibling's minus signs trace to the
  // Schur chain-rule structure; here the non-EO chain gives the direct
  // positive sign.  Validated by Test_dtxqcd_rational_full_force.)
  // Lower block uses Dw_lower (Wilson on U_conj) -> dS/dU_conj;
  // chain-ruled by conjugate() to dS/dU.
  void AccumulateHoppingForce(const DTXQCDFermionDoubled &X,
                              const DTXQCDFermionDoubled &Y,
                              RealD ak,
                              DTXQCDWilsonCloverFermionEO &Dw,
                              DTXQCDField &dSdU) {
    auto &meooe = Dw.MeooeEngine();
    auto &Wu    = meooe.UpperWilson();
    auto &Wl    = meooe.LowerWilson();

    LatticeGaugeField gforce_upper(&grid_);  gforce_upper = Zero();
    LatticeGaugeField gforce_lower(&grid_);  gforce_lower = Zero();
    LatticeGaugeField gtmp(&grid_);

    for (int a = 0; a < DtxqcdNf; ++a) {
      // Upper block (gauge = U).
      Wu.DhopDeriv(gtmp, Y.upper.f[a], X.upper.f[a], DaggerNo);
      gforce_upper = gforce_upper + gtmp;
      Wu.DhopDeriv(gtmp, X.upper.f[a], Y.upper.f[a], DaggerYes);
      gforce_upper = gforce_upper + gtmp;

      // Lower block (gauge = U_conj).
      Wl.DhopDeriv(gtmp, Y.lower.f[a], X.lower.f[a], DaggerNo);
      gforce_lower = gforce_lower + gtmp;
      Wl.DhopDeriv(gtmp, X.lower.f[a], Y.lower.f[a], DaggerYes);
      gforce_lower = gforce_lower + gtmp;
    }

    // Map dS/dU_conj -> dS/dU via entry-wise conjugation.
    gforce_lower = conjugate(gforce_lower);

    dSdU.U = dSdU.U + ak * (gforce_upper + gforce_lower);
  }

  // Per-pole aux + clover-sigma site loop on ALL sites (no CB filter).
  // Bilinear and coefficient identical to the EO sibling's
  // AccumulateSiteForces; only the CB filter is removed.
  void AccumulateSiteForcesAll(const DTXQCDFermionDoubled &X,
                               const DTXQCDFermionDoubled &Y,
                               RealD ak,
                               DTXQCDField &dSdU,
                               std::vector<LatticeColourMatrix> &clover_sigma_full) {
    using DtxqcdSiteForceKernel::SigSobj;
    using DtxqcdSiteForceKernel::PiSobj;
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::SSobj;
    using DtxqcdSiteForceKernel::PSobj;
    using DtxqcdSiteForceKernel::CMsobj;

    Coordinate gd(grid_.GlobalDimensions());
    const ComplexD coef(2.0 * ak, 0.0);

    for (int x = 0; x < gd[0]; ++x)
      for (int y = 0; y < gd[1]; ++y)
        for (int z = 0; z < gd[2]; ++z)
          for (int s = 0; s < gd[3]; ++s) {
            Coordinate coord(std::vector<int>{x, y, z, s});

            std::array<ComplexD, kDim48> X_x, Y_x;
            ExtractSiteVec48(X, coord, X_x);
            ExtractSiteVec48(Y, coord, Y_x);

            // Symmetrized bilinear (same Wirtinger argument as EO).
            auto Bil = [&X_x, &Y_x](int R, int C) -> ComplexD {
              return ComplexD(0.5, 0.0) *
                  (std::conj(Y_x[C]) * X_x[R]
                 + std::conj(X_x[C]) * Y_x[R]);
            };

            SigSobj sig_force;
            PiSobj  pi_force;
            DSobj   d_force;
            NSobj   n_force;
            SSobj   s_force;
            PSobj   p_force;
            DtxqcdSiteForceKernel::AuxForceAt(Bil, spin_, sig_force,
                                               pi_force, d_force, n_force,
                                               s_force, p_force);

            // Wirtinger -> physical-gradient conversion: transpose(F_W) for
            // Hermitian F_W.  See companion comment in
            // DTXQCDLogDetCloverEOAction.h::deriv() for the derivation.
            auto AddCF = [&](LatticeDtxqcdSigma &dst, const SigSobj &fv) {
              SigSobj cur; peekSite(cur, dst, coord);
              for (int a = 0; a < DtxqcdNf; ++a)
                for (int b = 0; b < DtxqcdNf; ++b)
                  for (int i = 0; i < Nc; ++i)
                    for (int j = 0; j < Nc; ++j)
                      cur()(a, b)(i, j) =
                          cur()(a, b)(i, j) + coef * fv()(b, a)(j, i);
              pokeSite(cur, dst, coord);
            };
            AddCF(dSdU.sigma, sig_force);
            AddCF(dSdU.pi,    pi_force);
            AddCF(dSdU.d,     d_force);
            AddCF(dSdU.n,     n_force);

            {
              SSobj cur; peekSite(cur, dSdU.s, coord);
              cur()()() = cur()()() + coef * s_force()()();
              pokeSite(cur, dSdU.s, coord);
            }
            {
              PSobj cur; peekSite(cur, dSdU.p, coord);
              cur()()() = cur()()() + coef * p_force()()();
              pokeSite(cur, dSdU.p, coord);
            }

            if (csw_ != 0.0) {
              std::array<CMsobj, 6> cs_arr;
              DtxqcdSiteForceKernel::CloverSigmaAt(Bil, spin_, csw_, cs_arr);
              for (int p = 0; p < 6; ++p) {
                CMsobj cur; peekSite(cur, clover_sigma_full[p], coord);
                for (int i_c = 0; i_c < Nc; ++i_c)
                  for (int j_c = 0; j_c < Nc; ++j_c)
                    cur()()(i_c, j_c) = cur()()(i_c, j_c)
                                      + coef * cs_arr[p]()()(i_c, j_c);
                pokeSite(cur, clover_sigma_full[p], coord);
              }
            }
          }
  }

  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD                  mass_;
  RealD                  csw_;
  Params                &param_;
  MultiShiftFunction     PowerNegQuarter;   // x^{-1/4}: S() + deriv() multishift
  MultiShiftFunction     PowerEighth;       // x^{+1/8}: refresh()
  DTXQCDFermionDoubled   Phi_;
  DtxqcdSpinMatrices     spin_;
};

NAMESPACE_END(Grid);
