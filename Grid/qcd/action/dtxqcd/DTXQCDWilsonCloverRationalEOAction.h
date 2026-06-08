#pragma once
// EO-preconditioned RHMC pseudofermion action for the DTXQCD doubled
// Wilson-Clover operator, exponent 1/4 (Pfaffian).
//
// Physics (paper Eq. 321):
//   |Pf(D)| = det(D^2)^{1/4}  and  det(D) = det(M_ee) * det(Mpc)
// so the path-integral weight |Pf(D)| factorizes:
//   |Pf(D)| = |det(M_ee)|^{1/2} * |det(Mpc)|^{1/2}
// The DTXQCDLogDetCloverEOAction handles the first factor (with its own
// 1/2 normalization).  This action handles the second:
//   exp(-S_RHMC) ~ det(Mpc^dag Mpc)^{1/4}
//   ⇒  S_RHMC = Phi^dag (Mpc^dag Mpc)^{-1/4} Phi
//   ⇒  heatbath:  Phi = (Mpc^dag Mpc)^{+1/8} eta,   eta ~ exp(-eta^dag eta)
//
// Rational approximations needed:
//   x^{1/4}  (PowerQuarter)    -- not used directly, kept for completeness
//   x^{-1/4} (PowerNegQuarter) -- used by deriv() multishift CG (poles σ_k)
//   x^{+1/8} (PowerEighth)     -- used by refresh() to generate Phi
//   x^{-1/8} (PowerNegEighth)  -- used by S() to compute ||Y||^2 = S
//
// Force structure mirrors TXQCDWilsonCloverRationalEOAction with two changes:
//   1. fermion field is DTXQCDFermionDoubled (upper, lower) × per-flavor;
//      Mpc operator is DTXQCDMpcOp wrapping DTXQCDWilsonCloverFermionEO.
//   2. per-site dM/dX kernels are the doubled DTXQCD ones already factored
//      into DTXQCDSiteForceKernel.h, fed a bilinear lookup
//        Bil(R, C) = conj(Y[C]) * X[R]
//      which gives -Y^dag (dM_ee/dX) X (with the kernel's intrinsic sign).
//
// At each pole k with rational residue alpha_k and per-site bilinear (Y_k, X_k)
// on odd CB and (Z_e, W_e) on even CB:
//   dS_k/dX_aux contribution at site x = +2 * alpha_k * (kernel-output at x)
// summed over both CBs.  The +2 (vs the LogDet -tr formula's -1) absorbs the
// "-2 Re" of the RHMC chain rule together with the kernel's intrinsic -1 sign
// (kernel returns -Y^dag dM X for the RHMC case; -2 alpha_k * (-1) = +2 alpha_k).
//
// v1 implementation:
//   - Aux + clover gauge forces analytic via per-site kernel.
//   - Hopping gauge force deferred (TODO; csw=0 gauge FD will fail until
//     hopping force is added).  Aux FD on csw=0 and csw!=0, plus clover gauge
//     FD on csw!=0 (aux-only U fixed), are the v1 validation targets.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/algorithms/approx/Remez.h>
#include <Grid/algorithms/approx/MultiShiftFunction.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDWilsonCloverRationalEOAction : public Action<DTXQCDField> {
 public:
  typedef OneFlavourRationalParams Params;
  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDWilsonCloverRationalEOAction(GridCartesian &grid,
                                     GridRedBlackCartesian &rbgrid,
                                     RealD mass, Params &p, RealD csw = 0.0)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw),
        param_(p), Phi_(&rbgrid), spin_(grid) {
    AlgRemez remez(param_.lo, param_.hi, param_.precision);
    std::cout << GridLogMessage
              << "[DTXQCDWilsonRationalEO] degree " << param_.degree
              << " rational for x^(1/4)" << std::endl;
    remez.generateApprox(param_.degree, 1, 4);
    PowerQuarter.Init(remez, param_.tolerance, false);
    PowerNegQuarter.Init(remez, param_.tolerance, true);
    std::cout << GridLogMessage
              << "[DTXQCDWilsonRationalEO] degree " << param_.degree
              << " rational for x^(1/8)" << std::endl;
    remez.generateApprox(param_.degree, 1, 8);
    PowerEighth.Init(remez, param_.tolerance, false);
    PowerNegEighth.Init(remez, param_.tolerance, true);
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalEOAction";
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
  // refresh: Phi = (Mpc^dag Mpc)^{+1/8} eta,  eta gaussian on odd CB.
  // The 1/sqrt(2) scale on eta matches Grid's CPS_MD_TIME convention
  // (Var(eta_re) = Var(eta_im) = 1/2, so |eta|^2 has unit variance).
  // ------------------------------------------------------------------
  void refresh(const DTXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    DTXQCDFermionDoubled eta(&rbgrid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, eta.upper.f[a]);
      gaussian(pRNG, eta.lower.f[a]);
      eta.upper.f[a] = scale * eta.upper.f[a];
      eta.lower.f[a] = scale * eta.lower.f[a];
      eta.upper.f[a].Checkerboard() = Odd;
      eta.lower.f[a].Checkerboard() = Odd;
    }
    auto Dw = MakeEOp(U);
    DTXQCDMpcOp Mop(Dw);
    ApplyRational(Mop, PowerEighth, eta, Phi_);
  }

  // ------------------------------------------------------------------
  // S(U) = Phi^dag g_rat(Mpc^dag Mpc) Phi   where g_rat ≈ x^{-1/4}.
  //
  // NOT computed as ||f_rat(M)Phi||^2 with f_rat ≈ x^{-1/8}, because
  // (f_rat)^2 is the squared 1/8-rational and is a *different* rational
  // function from the 1/4-rational g_rat = PowerNegQuarter.  The deriv()
  // multi-shift uses g_rat's poles + residues, so S() must too — otherwise
  // FD on S won't match the analytic chain-rule formula.
  // ------------------------------------------------------------------
  RealD S(const DTXQCDField &U) override {
    auto Dw = MakeEOp(U);
    DTXQCDMpcOp Mop(Dw);
    DTXQCDFermionDoubled Y(&rbgrid_);
    ApplyRational(Mop, PowerNegQuarter, Phi_, Y);  // Y = (M^dag M)^{-1/4} Phi
    // S = Re(Phi^dag Y).  Im part is zero up to CG noise (operator Hermitian).
    ComplexD ip = innerProduct(Phi_, Y);
    RealD action = ip.real();
    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  // ------------------------------------------------------------------
  // deriv: aux + clover gauge force + hopping gauge force.
  //
  // Hopping force structure (per pole k):
  //   dS_k/dU contribution = +alpha_k * 2 Re[Y^dag (dM_oe/dU) W_e + Z_e^dag (dM_eo/dU) X]
  //     = +alpha_k * (Wilson hopping derivative of Y^dag M_oe W_e + Z_e^dag M_eo X
  //                   summed with its dagger)
  // Implemented as 4 MoeDeriv/MeoDeriv calls per flavor per pole, separately for
  // the upper and lower fermion blocks.  Upper block uses Dw_upper (= Wilson
  // on gauge field U); lower block uses Dw_lower (= Wilson on U_conj).  Force
  // returned by Dw_lower is dS/dU_conj; chain-ruling through U_conj = conj(U)
  // gives dS/dU = conjugate(dS/dU_conj) entry-wise (anti-Hermitian preserved).
  // ------------------------------------------------------------------
  void deriv(const DTXQCDField &U, DTXQCDField &dSdU) override {
    dSdU = Zero();
    auto Dw = MakeEOp(U);
    DTXQCDMpcOp Mop(Dw);

    // ---- Multi-shift solve into Xk[k] = (M^dag M + sigma_k)^{-1} Phi ----
    const int Npole = static_cast<int>(PowerNegQuarter.poles.size());
    std::vector<DTXQCDFermionDoubled> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&rbgrid_);
    std::vector<RealD> md_tol(Npole, param_.mdtolerance);
    DTXQCDMultiShiftCG(Mop, PowerNegQuarter.poles, md_tol, Phi_, Xk,
                       param_.MaxIter);

    // ---- Build F_{mu,nu} field strength (for clover, csw != 0 only) ----
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

    // ---- clover_sigma_full[mn] accumulator across (poles, sites, CBs) ----
    std::vector<LatticeColourMatrix> clover_sigma_full;
    if (csw_ != 0.0) {
      clover_sigma_full.reserve(6);
      for (int k = 0; k < 6; ++k) {
        clover_sigma_full.emplace_back(&grid_);
        clover_sigma_full.back() = Zero();
      }
    }

    // ---- Per-pole intermediates: batched MooeeInv / MooeeInvDag --------
    //
    // The Schur complement Mpc applied to each X_k decomposes as
    //   Y_k   = M_oo X_k - M_oe (M_ee^{-1} M_eo X_k)
    //         = M_oo X_k - M_oe  W_e_k
    //   Z_e_k = (M_ee^{-1})^dag M_oe^dag Y_k
    //
    // Two reusable identities cut the post-CG work:
    //   1. M_oe (W_e_k) is the same Meooe that Mpc.M used internally; computing
    //      it once and reusing it for Y_k and AccumulateHoppingForce avoids
    //      the duplicate Mpc + Dw.Meooe(X_k) chain in v1.
    //   2. The two CB-Even inverse applies (one for W_e, one for Z_e) batch
    //      across all Npole rational poles -- MooeeInvN does one Eigen
    //      48 x Npole gemm per site instead of Npole separate gemvs, so the
    //      cached per-site inverse is loaded once per site for all shifts.
    //
    // The hopping + aux + clover-sigma accumulators stay per-pole (each pole
    // has its own residue alpha_k), so the second pass below sums them
    // sequentially.
    std::vector<DTXQCDFermionDoubled> Yk;       Yk.reserve(Npole);
    std::vector<DTXQCDFermionDoubled> Wek;      Wek.reserve(Npole);
    std::vector<DTXQCDFermionDoubled> Zek;      Zek.reserve(Npole);
    std::vector<DTXQCDFermionDoubled> Meo_Xk;   Meo_Xk.reserve(Npole);
    std::vector<DTXQCDFermionDoubled> Moeh_Yk;  Moeh_Yk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) {
      Yk.emplace_back(&rbgrid_);
      Wek.emplace_back(&rbgrid_);
      Zek.emplace_back(&rbgrid_);
      Meo_Xk.emplace_back(&rbgrid_);
      Moeh_Yk.emplace_back(&rbgrid_);
    }

    // Meooe X_k for all k (per-RHS Wilson stencil; no natural batched apply).
    for (int k = 0; k < Npole; ++k) Dw.Meooe(Xk[k], Meo_Xk[k]);

    // W_e_k = M_ee^{-1} M_eo X_k for all k -- single batched MooeeInvN call.
    {
      std::vector<const DTXQCDFermionDoubled *> ins(Npole);
      std::vector<DTXQCDFermionDoubled *>       outs(Npole);
      for (int k = 0; k < Npole; ++k) { ins[k] = &Meo_Xk[k]; outs[k] = &Wek[k]; }
      Dw.MooeeInvN(ins, outs);
    }

    // Y_k = M_oo X_k - M_oe W_e_k.
    DTXQCDFermionDoubled mooee_X(&rbgrid_), moe_W(&rbgrid_);
    for (int k = 0; k < Npole; ++k) {
      Dw.Mooee(Xk[k], mooee_X);    // odd-CB site-local apply (lattice op,
                                   //   forward Mooee is fast via per-CB aux).
      Dw.Meooe(Wek[k], moe_W);     // M_oe W_e_k -> odd CB.
      for (int a = 0; a < DtxqcdNf; ++a) {
        Yk[k].upper.f[a] = mooee_X.upper.f[a] - moe_W.upper.f[a];
        Yk[k].lower.f[a] = mooee_X.lower.f[a] - moe_W.lower.f[a];
      }
    }

    // M_oe^dag Y_k for all k.
    for (int k = 0; k < Npole; ++k) Dw.MeooeDag(Yk[k], Moeh_Yk[k]);

    // Z_e_k = (M_ee^{-1})^dag M_oe^dag Y_k -- single batched MooeeInvDagN call.
    {
      std::vector<const DTXQCDFermionDoubled *> ins(Npole);
      std::vector<DTXQCDFermionDoubled *>       outs(Npole);
      for (int k = 0; k < Npole; ++k) { ins[k] = &Moeh_Yk[k]; outs[k] = &Zek[k]; }
      Dw.MooeeInvDagN(ins, outs);
    }

    // ---- Per-pole force accumulation (sequential -- each has its own ak) ----
    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegQuarter.residues[k];
      AccumulateSiteForces(Xk[k],  Yk[k], /*odd_cb=*/true,  ak,
                           dSdU, clover_sigma_full);
      AccumulateSiteForces(Wek[k], Zek[k], /*odd_cb=*/false, ak,
                           dSdU, clover_sigma_full);
      AccumulateHoppingForce(Xk[k], Yk[k], Wek[k], Zek[k], ak, Dw, dSdU);
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
      // Convention A (mirrors LogDet).  ACCUMULATE (+=) into dSdU.U so the
      // hopping gauge force (already added per pole) is preserved.
      dSdU.U = dSdU.U + ComplexD(-0.5, 0.0) * clover_force;
    }
  }

  DTXQCDFermionDoubled &PseudoFermion() { return Phi_; }

 protected:
  DTXQCDWilsonCloverFermionEO MakeEOp(const DTXQCDField &U) {
    DTXQCDField &Unc = const_cast<DTXQCDField &>(U);
    return DTXQCDWilsonCloverFermionEO(Unc.U, grid_, rbgrid_, mass_, csw_,
                                       Unc.sigma, Unc.pi, Unc.t, Unc.d, Unc.n);
  }

  // Multishift-CG + linear combination: out = norm * in + sum_k residues[k] * xk[k]
  // where xk[k] = (M^dag M + poles[k])^{-1} in (multi-shift solve).
  void ApplyRational(DTXQCDMpcOp &Mop, const MultiShiftFunction &rat,
                     const DTXQCDFermionDoubled &in,
                     DTXQCDFermionDoubled &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<DTXQCDFermionDoubled> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&rbgrid_);
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
  // Index layout:  v[r]            = upper.f[a](x)[alpha][i]   for r = idx24(a,alpha,i)
  //                v[kDim24 + r]   = lower.f[a](x)[alpha][i]
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

  // Per-pole hopping gauge force, accumulated directly into dSdU.U.
  //
  // For each flavor a and each fermion block (upper, lower):
  //   - 2 MoeDeriv calls (DaggerNo and DaggerYes) giving the contributions
  //     from Y^dag M_oe W_e and X^dag M_oe^dag Z_e respectively.
  //   - 2 MeoDeriv calls giving Z_e^dag M_eo X and W_e^dag M_eo^dag Y.
  // Per-block sums are negated (sign convention mirroring TXQCDWilsonClover-
  // RationalEOAction: gforce = -sum(force calls), then dSdU += ak * gforce).
  //
  // Lower-block forces come from Dw_lower (Wilson on U_conj), giving
  // dS/dU_conj.  Chain rule for U_conj = conj(U) gives
  //   dS/dU = conjugate(dS/dU_conj)
  // (entry-wise complex conjugation; preserves anti-Hermiticity).
  void AccumulateHoppingForce(const DTXQCDFermionDoubled &X,
                              const DTXQCDFermionDoubled &Y,
                              const DTXQCDFermionDoubled &W_e,
                              const DTXQCDFermionDoubled &Z_e,
                              RealD ak,
                              DTXQCDWilsonCloverFermionEO &Dw,
                              DTXQCDField &dSdU) {
    auto &meooe = Dw.MeooeEngine();
    auto &Wu    = meooe.UpperWilson();
    auto &Wl    = meooe.LowerWilson();

    LatticeGaugeField gforce_upper(&grid_);  gforce_upper = Zero();
    LatticeGaugeField gforce_lower(&grid_);  gforce_lower = Zero();
    LatticeGaugeField gtmp(&grid_);
    LatticeGaugeField ForceO(&rbgrid_), ForceE(&rbgrid_);

    for (int a = 0; a < DtxqcdNf; ++a) {
      // ---- Upper block (gauge = U) -------------------------------------
      Wu.MoeDeriv(ForceO, Y.upper.f[a],   W_e.upper.f[a], DaggerNo);
      Wu.MeoDeriv(ForceE, Z_e.upper.f[a], X.upper.f[a],   DaggerNo);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce_upper = gforce_upper - gtmp;

      Wu.MoeDeriv(ForceO, X.upper.f[a],   Z_e.upper.f[a], DaggerYes);
      Wu.MeoDeriv(ForceE, W_e.upper.f[a], Y.upper.f[a],   DaggerYes);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce_upper = gforce_upper - gtmp;

      // ---- Lower block (gauge = U_conj) --------------------------------
      Wl.MoeDeriv(ForceO, Y.lower.f[a],   W_e.lower.f[a], DaggerNo);
      Wl.MeoDeriv(ForceE, Z_e.lower.f[a], X.lower.f[a],   DaggerNo);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce_lower = gforce_lower - gtmp;

      Wl.MoeDeriv(ForceO, X.lower.f[a],   Z_e.lower.f[a], DaggerYes);
      Wl.MeoDeriv(ForceE, W_e.lower.f[a], Y.lower.f[a],   DaggerYes);
      setCheckerboard(gtmp, ForceO);
      setCheckerboard(gtmp, ForceE);
      gforce_lower = gforce_lower - gtmp;
    }

    // Map dS/dU_conj → dS/dU via entry-wise conjugation.
    gforce_lower = conjugate(gforce_lower);

    dSdU.U = dSdU.U + ak * (gforce_upper + gforce_lower);
  }

  // Per-pole per-CB site loop: accumulate aux + clover-sigma contributions
  // from the bilinear (Y, X) into dSdU and clover_sigma_full.
  //
  //   coef_factor = 2 * alpha_k
  // multiplied into the kernel-returned per-site values (the kernel's
  // intrinsic sign + this +2 give the action chain-rule "-2 Re Y^dag dM X"
  // identity; see header).
  void AccumulateSiteForces(const DTXQCDFermionDoubled &X,
                            const DTXQCDFermionDoubled &Y,
                            bool odd_cb,
                            RealD ak,
                            DTXQCDField &dSdU,
                            std::vector<LatticeColourMatrix> &clover_sigma_full) {
    using DtxqcdSiteForceKernel::SigSobj;
    using DtxqcdSiteForceKernel::PiSobj;
    using DtxqcdSiteForceKernel::TSobj;
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::CMsobj;

    Coordinate gd(grid_.GlobalDimensions());
    const ComplexD coef(2.0 * ak, 0.0);

    for (int x = 0; x < gd[0]; ++x)
      for (int y = 0; y < gd[1]; ++y)
        for (int z = 0; z < gd[2]; ++z)
          for (int s = 0; s < gd[3]; ++s) {
            bool site_is_odd = (((x + y + z + s) & 1) == 1);
            if (site_is_odd != odd_cb) continue;
            Coordinate coord(std::vector<int>{x, y, z, s});

            std::array<ComplexD, kDim48> X_x, Y_x;
            ExtractSiteVec48(X, coord, X_x);
            ExtractSiteVec48(Y, coord, Y_x);

            // Bil(R, C) = 0.5 * [conj(Y[C]) * X[R] + conj(X[C]) * Y[R]]
            //
            // (Y, X) symmetrization, mirroring TXQCDWilsonCloverRationalEOAction's
            // conj(Y)*X + conj(X)*Y clover-sigma formula.  Without it, the
            // asymmetric conj(Y[C])*X[R] is the natural-Wirtinger bilinear
            // (gives the right Re value when summed against a real direction
            // for the aux force test), but its complex Im part leaks into the
            // clover_sigma_full and pollutes Cmunu's chain rule when the FD
            // gauge-perturbation test extracts Re Tr(E * F_mu).  The
            // symmetrization sets val_sym = Re(val_orig) for the aux force
            // kernels (no change in effective Re-projected output) and gives
            // the proper "cs = -conj(dS_k/dF)" form for the clover kernel.
            //
            // The R↔C swap of indices (vs LogDet's Inv(R, C) = M^{-1}[R][C])
            // is still needed: the kernel evaluates Tr-formula entries
            // A[K, R] * dM[R, K] (K is the column of dM), while we want
            // Y^dag dM X = sum dM[R, C] conj(Y[R]) X[C], so the natural-
            // Wirtinger half stores conj(Y[C])*X[R].  The 0.5*(...) average
            // and the (X, Y)-swap cancel the Wirtinger Im contribution.
            auto Bil = [&X_x, &Y_x](int R, int C) -> ComplexD {
              return ComplexD(0.5, 0.0) *
                  (std::conj(Y_x[C]) * X_x[R]
                 + std::conj(X_x[C]) * Y_x[R]);
            };

            SigSobj sig_force;
            PiSobj  pi_force;
            TSobj   t_force;
            DSobj   d_force;
            NSobj   n_force;
            DtxqcdSiteForceKernel::AuxForceAt(Bil, spin_, sig_force,
                                               pi_force, t_force,
                                               d_force, n_force);

            // Accumulate into dSdU (peek + add scaled + poke).
            {
              SigSobj cur; peekSite(cur, dSdU.sigma, coord);
              for (int A = 0; A < DtxqcdNTriplet; ++A)
                cur()()(A) = cur()()(A) + coef * sig_force()()(A);
              pokeSite(cur, dSdU.sigma, coord);
            }
            {
              PiSobj cur; peekSite(cur, dSdU.pi, coord);
              for (int A = 0; A < DtxqcdNTriplet; ++A)
                cur()()(A) = cur()()(A) + coef * pi_force()()(A);
              pokeSite(cur, dSdU.pi, coord);
            }
            {
              TSobj cur; peekSite(cur, dSdU.t, coord);
              for (int mu = 0; mu < Nd; ++mu)
                for (int nu = 0; nu < Nd; ++nu)
                  for (int A = 0; A < DtxqcdNTriplet; ++A)
                    cur()(mu, nu)(A) =
                        cur()(mu, nu)(A) + coef * t_force()(mu, nu)(A);
              pokeSite(cur, dSdU.t, coord);
            }
            {
              DSobj cur; peekSite(cur, dSdU.d, coord);
              for (int kk = 0; kk < Nc; ++kk)
                for (int ll = 0; ll < Nc; ++ll)
                  cur()()(kk, ll) =
                      cur()()(kk, ll) + coef * d_force()()(kk, ll);
              pokeSite(cur, dSdU.d, coord);
            }
            {
              NSobj cur; peekSite(cur, dSdU.n, coord);
              for (int kk = 0; kk < Nc; ++kk)
                for (int ll = 0; ll < Nc; ++ll)
                  cur()()(kk, ll) =
                      cur()()(kk, ll) + coef * n_force()()(kk, ll);
              pokeSite(cur, dSdU.n, coord);
            }

            // Clover sigma contribution (csw != 0 only).
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
  MultiShiftFunction     PowerQuarter;
  MultiShiftFunction     PowerNegQuarter;
  MultiShiftFunction     PowerEighth;
  MultiShiftFunction     PowerNegEighth;
  DTXQCDFermionDoubled   Phi_;
  DtxqcdSpinMatrices     spin_;
};

NAMESPACE_END(Grid);
