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
//   x^{-1/4} (PowerNegQuarter) -- used by S() AND deriv() multishift CG
//                                  (poles σ_k of the same rational so that
//                                   the FD on S exactly matches deriv()'s
//                                   analytic chain-rule formula).
//   x^{+1/8} (PowerEighth)     -- used by refresh() to generate Phi from eta:
//                                  Phi = (Mpc^dag Mpc)^{+1/8} eta gives
//                                  <Phi^dag (Mpc^dag Mpc)^{-1/4} Phi> = <eta^dag eta>.
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
// Force coverage (all validated by Test_dtxqcd_rational_aux_force FD checks):
//   - Aux forces (sigma^A, pi^A, t^A, d, n) via per-site
//     DTXQCDSiteForceKernel::AuxForceAt.
//   - Gauge hopping force via AccumulateHoppingForce: 4 MoeDeriv/MeoDeriv
//     calls per flavor per fermion block; lower block on Dw_lower (Wilson
//     on U_conj) is chain-ruled to dS/dU by entry-wise conjugate().
//   - Gauge clover force via per-site CloverSigmaAt + WilsonCloverHelpers::Cmunu.

#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpF.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGMixedPrec.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDRationalForceGpuKernel.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDRemezAutoScale.h>
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
        param_(p), Phi_(&rbgrid), spin_(grid),
        auto_(DtxqcdRemezAutoScaleParams::FromEnv(p.lo, p.hi)) {
    // Defer the Remez build to the first refresh() when auto-scale is active:
    // it rebuilds on the Lanczos-derived bounds anyway, so building here on the
    // (possibly doomed) fixed [lo,hi] is wasted -- and at a tiny fixed lo the
    // AlgRemez compute is very slow.  refresh() is the first method the HMC
    // integrator calls (before S()/deriv()), so deferral is safe.
    if (!auto_.enabled) {
      BuildRemez(auto_.current_lo, auto_.current_hi);
    } else {
      std::cout << GridLogMessage
                << "[DTXQCDWilsonRationalEO] Remez build deferred to refresh() "
                   "(auto-scale active)" << std::endl;
    }
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalEOAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " csw=" << csw_ << " lo=" << auto_.current_lo
       << " hi=" << auto_.current_hi
       << (auto_.enabled ? " (auto-scale)" : "")
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
    auto Dw = MakeEOp(U);
    DTXQCDMpcOp Mop(Dw);

    // ---- Auto-scale the Remez bounds from the doubled Mpc^dag Mpc spectrum.
    // Lanczos runs on the SAME Schur operator the multishift CG inverts, on the
    // RB grid with Odd checkerboard.  Gated entirely on RAT_AUTO_HI (default
    // OFF -> no Lanczos, no rebuild, byte-identical to the pre-autoscale path).
    {
      RealD new_lo = auto_.current_lo, new_hi = auto_.current_hi;
      bool auto_rebuild = DtxqcdRefreshAutoScale(auto_, Mop, &rbgrid_, pRNG,
                              /*cb=*/Odd, "DTXQCDWilsonRationalEO", new_lo, new_hi);
      // First refresh after a deferred ctor build: the Remez is empty, so build
      // it now (on the auto-scaled bounds) even if the bounds didn't widen.
      if (auto_rebuild || PowerNegQuarter.poles.size() == 0) {
        std::cout << GridLogMessage << "[" << action_name()
                  << "] building Remez on [" << new_lo << ", " << new_hi
                  << "]" << std::endl;
        BuildRemez(new_lo, new_hi);
      }
    }

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
    ApplyRational(Dw, Mop, PowerEighth, eta, Phi_);
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
    ApplyRational(Dw, Mop, PowerNegQuarter, Phi_, Y);  // Y = (M^dag M)^{-1/4} Phi
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
    GridStopWatch t_cg, t_force;  // perf instrumentation: CG vs force-assembly
    t_cg.Start();
    RunMultiShiftCG(Dw, Mop, PowerNegQuarter.poles, md_tol, Phi_, Xk);
    t_cg.Stop();
    t_force.Start();
    // Finer force-assembly sub-timers (usecond-based accumulators, cheap).
    double us_schur = 0.0, us_site = 0.0, us_hop = 0.0, us_clov = 0.0;

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

    us_schur -= usecond();
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
    us_schur += usecond();

    // ---- Per-pole site/aux force accumulation (each has its own ak) ----
    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegQuarter.residues[k];
      us_site -= usecond();
      AccumulateSiteForces(Xk[k],  Yk[k], /*odd_cb=*/true,  ak,
                           dSdU, clover_sigma_full);
      AccumulateSiteForces(Wek[k], Zek[k], /*odd_cb=*/false, ak,
                           dSdU, clover_sigma_full);
      us_site += usecond();
    }
    // ---- Hopping (Wilson dslash-deriv) gauge force, all poles ----
    // Virtual hook: the QUDA-primitive subclass batches all (flavor, pole) rhs
    // into computeCloverWilsonForceWithSchurFields (one call per doubled block,
    // upper on U + lower on conj(U)); the default loops per-pole over the Grid
    // MoeDeriv/MeoDeriv reference path.
    us_hop -= usecond();
    AccumulateHoppingForceAllPoles(U, Xk, Yk, Wek, Zek, Dw, dSdU);
    us_hop += usecond();

    // ---- Gauge clover force via Cmunu chain rule (csw != 0 only) ----
    // Virtual hook: the QUDA-primitive subclass overrides this to batch every
    // (block, flavor, pole) rhs into one computeCloverSigmaForceWithSchurFields
    // call per doubled block (DTXQCD_QUDA_FULL=1 path).  Default = the Cmunu
    // chain on the already-accumulated clover_sigma_full[] (csw != 0 only).
    us_clov -= usecond();
    AccumulateGaugeCloverForce(U, Xk, Yk, Wek, Zek, Dw,
                                clover_sigma_full, dSdU);
    us_clov += usecond();
    t_force.Stop();
    std::cout << GridLogMessage << "[" << action_name()
              << "] deriv timing: CG=" << t_cg.Elapsed()
              << "  force-assembly=" << t_force.Elapsed() << std::endl;
    std::cout << GridLogMessage << "[" << action_name()
              << "] deriv timing: schur=" << (us_schur * 1e-6)
              << "  siteforce=" << (us_site * 1e-6)
              << "  hopping=" << (us_hop * 1e-6)
              << "  clover=" << (us_clov * 1e-6) << " (s)" << std::endl;
  }

  DTXQCDFermionDoubled &PseudoFermion() { return Phi_; }

 protected:
  // Build/rebuild the two Remez approximations (x^{-1/4}, x^{+1/8}) on [lo, hi]
  // and record the active bounds.  Called once in the constructor and again by
  // refresh() when the auto-scale widens the window.
  void BuildRemez(RealD lo, RealD hi) {
    auto_.current_lo = lo;
    auto_.current_hi = hi;
    DtxqcdRebuildRemez(lo, hi, param_.degree, param_.precision,
                       param_.tolerance, "DTXQCDWilsonRationalEO",
                       PowerNegQuarter, PowerEighth);
  }

  DTXQCDWilsonCloverFermionEO MakeEOp(const DTXQCDField &U) {
    DTXQCDField &Unc = const_cast<DTXQCDField &>(U);
    return DTXQCDWilsonCloverFermionEO(Unc.U, grid_, rbgrid_, mass_, csw_,
                                       Unc.sigma, Unc.pi, Unc.d, Unc.n,
                                       Unc.s,     Unc.p);
  }

  // Multishift-CG + linear combination: out = norm * in + sum_k residues[k] * xk[k]
  // where xk[k] = (M^dag M + poles[k])^{-1} in (multi-shift solve).
  //
  // The solve is DOUBLE precision by default.  DTXQCD_MP_CG>=2 also routes the
  // S()/refresh() rationals through the mixed-precision solver; the default
  // DTXQCD_MP_CG=1 keeps S()/refresh() in DP so they remain the high-accuracy
  // reference that deriv()'s (now-MP) force is validated against.
  void ApplyRational(DTXQCDWilsonCloverFermionEO &Dw, DTXQCDMpcOp &Mop,
                     const MultiShiftFunction &rat,
                     const DTXQCDFermionDoubled &in,
                     DTXQCDFermionDoubled &out) {
    const int nshift = static_cast<int>(rat.poles.size());
    std::vector<DTXQCDFermionDoubled> xk;
    xk.reserve(nshift);
    for (int k = 0; k < nshift; ++k) xk.emplace_back(&rbgrid_);
    if (MpCgEnabled() >= 2) {
      if (!Dw.SinglePrecEnabled()) Dw.EnableSinglePrec();
      DTXQCDMpcOpF Mop_f(Dw);
      DTXQCDMultiShiftCGMixedPrec(Mop, Mop_f, rat.poles, rat.tolerances, in, xk,
                                  param_.MaxIter, MpCgReliableFreq());
    } else {
      DTXQCDMultiShiftCG(Mop, rat.poles, rat.tolerances, in, xk,
                         param_.MaxIter);
    }

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

  // All-poles hopping gauge force.  Virtual so the QUDA-primitive subclass can
  // batch every (flavor, pole) rhs into one computeCloverWilsonForceWithSchur-
  // Fields call per doubled block.  Default = the per-pole Grid reference path
  // (bit-identical to the previous in-loop behaviour; accumulation commutes).
  // U is supplied so a QUDA override can SetGauge(U.U) / SetGauge(conj(U.U)).
  virtual void AccumulateHoppingForceAllPoles(
      const DTXQCDField &U,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      DTXQCDWilsonCloverFermionEO &Dw,
      DTXQCDField &dSdU) {
    (void)U;
    const int Npole = static_cast<int>(Xk.size());
    for (int k = 0; k < Npole; ++k) {
      const RealD ak = PowerNegQuarter.residues[k];
      AccumulateHoppingForce(Xk[k], Yk[k], Wek[k], Zek[k], ak, Dw, dSdU);
    }
  }

  // Gauge clover-sigma force (csw != 0 only).  Virtual so the QUDA-primitive
  // subclass can replace the Cmunu chain with a batched
  // computeCloverSigmaForceWithSchurFields call (DTXQCD_QUDA_FULL=1 path).
  //
  // Default impl = the original deriv() inline Cmunu loop:
  //   For each μ:  Σ_{ν≠μ}  0.25·sign(μ,ν)·Cmunu(U, CS[mn(μ,ν)], μ, ν)
  //   pokeLorentz(F, U_μ · force_μ, μ)
  //   dSdU.U += -0.5 · F   (Convention-A correction)
  //
  // (Xk, Yk, Wek, Zek) carry the Schur-completed Nf fermion blocks for every
  // pole — unused in the default impl but required by overrides that bypass
  // clover_sigma_full and pack directly into QUDA.
  virtual void AccumulateGaugeCloverForce(
      const DTXQCDField &U,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      DTXQCDWilsonCloverFermionEO &Dw,
      std::vector<LatticeColourMatrix> &clover_sigma_full,
      DTXQCDField &dSdU) {
    (void)Xk; (void)Yk; (void)Wek; (void)Zek; (void)Dw;
    if (csw_ == 0.0) return;
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

  // Accumulate (+= coef * arr) a lex-ordered CB host array into the CB sites of
  // a full-grid lattice (odd/even sites of `full`; the other parity untouched).
  // Used by AccumulateSiteForces, which sums per-pole/per-CB into dSdU.
  template <class Field, class Sobj>
  void AddEvenOddToFull(std::vector<Sobj> &arr, Field &full, int cb,
                        ComplexD coef) {
    Field contrib(&rbgrid_);
    contrib.Checkerboard() = cb;
    vectorizeFromLexOrdArray(arr, contrib);
    Field cur(&rbgrid_);
    cur.Checkerboard() = cb;
    pickCheckerboard(cb, cur, full);
    cur = cur + coef * contrib;
    setCheckerboard(full, cur);
  }

  // GPU twin of AddEvenOddToFull: contrib is already a CB-resident Lattice
  // (written by the GPU kernel), so no vectorize roundtrip -- pick the cb
  // parity of `full`, accumulate coef*contrib, set it back.
  template <class Field>
  void AddCBLatticeToFull(const Field &contrib, Field &full, int cb,
                          ComplexD coef) {
    Field cur(&rbgrid_);
    cur.Checkerboard() = cb;
    pickCheckerboard(cb, cur, full);
    cur = cur + coef * contrib;
    setCheckerboard(full, cur);
  }

  // DTXQCD_RATFORCE_GPU: route AccumulateSiteForces through the GPU kernel
  // (DTXQCDRationalForceGpuKernel.h).  DEFAULT OFF everywhere (CPU thread_for
  // is the bit-comparable reference); off-CUDA always CPU.
  //
  // Reads the env on every call so tests can flip it within one process via
  // setenv("DTXQCD_RATFORCE_GPU", "0|1", 1).  This is called per-pole-per-CB
  // (not per-site), so the getenv cost is negligible vs the force eval cost.
  static int RatForceGpuEnabled() {
#ifndef GRID_CUDA
    return 0;
#else
    const char *e = std::getenv("DTXQCD_RATFORCE_GPU");
    return (e && *e) ? std::atoi(e) : 0;
#endif
  }

  // DTXQCD_MP_CG: route the multishift CG in deriv() (and, when
  // DTXQCD_MP_CG>=2, also S()/refresh()) through the reliable-update
  // mixed-precision solver.  DEFAULT OFF -- the all-double DTXQCDMultiShiftCG
  // is the production default and the numerical reference.  Read per call so
  // tests can flip it within one process via setenv.
  static int MpCgEnabled() {
    const char *e = std::getenv("DTXQCD_MP_CG");
    return (e && *e) ? std::atoi(e) : 0;
  }
  static int MpCgReliableFreq() {
    const char *e = std::getenv("DTXQCD_MP_CG_RELIABLE_FREQ");
    return (e && *e) ? std::max(1, std::atoi(e)) : 50;
  }

  // Dispatch the (M^dag M + poles)^{-1} multishift solve: double-precision by
  // default, mixed-precision (SP matvec + DP reliable update) when
  // DTXQCD_MP_CG is set.  Mop is the DP Schur operator over Dw; the SP twin is
  // built from Dw's downcast caches (Dw.EnableSinglePrec() built them).
  void RunMultiShiftCG(DTXQCDWilsonCloverFermionEO &Dw, DTXQCDMpcOp &Mop,
                       const std::vector<RealD> &poles,
                       const std::vector<RealD> &tol,
                       const DTXQCDFermionDoubled &src,
                       std::vector<DTXQCDFermionDoubled> &xk) {
    if (MpCgEnabled()) {
      if (!Dw.SinglePrecEnabled()) Dw.EnableSinglePrec();
      DTXQCDMpcOpF Mop_f(Dw);
      DTXQCDMultiShiftCGMixedPrec(Mop, Mop_f, poles, tol, src, xk,
                                  param_.MaxIter, MpCgReliableFreq());
    } else {
      DTXQCDMultiShiftCG(Mop, poles, tol, src, xk, param_.MaxIter);
    }
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
    using DtxqcdSiteForceKernel::DSobj;
    using DtxqcdSiteForceKernel::NSobj;
    using DtxqcdSiteForceKernel::SSobj;
    using DtxqcdSiteForceKernel::PSobj;
    using DtxqcdSiteForceKernel::CMsobj;

    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    const int cb = odd_cb ? Odd : Even;
    const ComplexD coef(2.0 * ak, 0.0);

    // ---- GPU path (DTXQCD_RATFORCE_GPU): device kernel + CB accumulate ----
    if (RatForceGpuEnabled()) {
      LatticeDtxqcdSigma F_sig(&rbgrid_);
      LatticeDtxqcdPi    F_pi (&rbgrid_);
      LatticeDtxqcdD     F_d  (&rbgrid_);
      LatticeDtxqcdN     F_n  (&rbgrid_);
      LatticeDtxqcdS     F_s  (&rbgrid_);
      LatticeDtxqcdP     F_p  (&rbgrid_);
      std::array<LatticeColourMatrix, 6> F_cs{
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_),
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_),
          LatticeColourMatrix(&rbgrid_), LatticeColourMatrix(&rbgrid_)};
      DtxqcdRatForceGpu::Extract(X, Y, cb, csw_, spin_,
                                 F_sig, F_pi, F_d, F_n, F_s, F_p, F_cs);
      AddCBLatticeToFull(F_sig, dSdU.sigma, cb, coef);
      AddCBLatticeToFull(F_pi,  dSdU.pi,    cb, coef);
      AddCBLatticeToFull(F_d,   dSdU.d,     cb, coef);
      AddCBLatticeToFull(F_n,   dSdU.n,     cb, coef);
      AddCBLatticeToFull(F_s,   dSdU.s,     cb, coef);
      AddCBLatticeToFull(F_p,   dSdU.p,     cb, coef);
      if (csw_ != 0.0)
        for (int k = 0; k < 6; ++k)
          AddCBLatticeToFull(F_cs[k], clover_sigma_full[k], cb, coef);
      return;
    }

    // Unvectorize the per-pole CG solutions on THIS rank's CB sublattice.  The
    // rational force uses ONLY the fermion bilinear Bil(R,C) (no aux peek), so
    // X, Y are all we unvectorize.  Local + parallel; replaces the serial
    // global-coord peekSite loop (the ~69 s/eval bottleneck + multi-rank
    // OOM/over-count).  Accumulation (+=) into dSdU is preserved per CB.
    std::array<std::vector<Fsobj>, DtxqcdNf> Xu, Xl, Yu, Yl;
    for (int a = 0; a < DtxqcdNf; ++a) {
      unvectorizeToLexOrdArray(Xu[a], X.upper.f[a]);
      unvectorizeToLexOrdArray(Xl[a], X.lower.f[a]);
      unvectorizeToLexOrdArray(Yu[a], Y.upper.f[a]);
      unvectorizeToLexOrdArray(Yl[a], Y.lower.f[a]);
    }
    const uint64_t Nsite = Xu[0].size();

    std::vector<SigSobj> fsig(Nsite); std::vector<PiSobj> fpi(Nsite);
    std::vector<DSobj>   fd(Nsite);   std::vector<NSobj>  fn(Nsite);
    std::vector<SSobj>   fss(Nsite);  std::vector<PSobj>  fpp(Nsite);
    std::array<std::vector<CMsobj>, 6> fcs;
    if (csw_ != 0.0) for (int k = 0; k < 6; ++k) fcs[k].resize(Nsite);

    thread_for(idx, Nsite, {
      std::array<ComplexD, kDim48> X_x, Y_x;
      for (int a = 0; a < DtxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            int r = DtxqcdSiteIdx24(a, alpha, i);
            X_x[r]          = ComplexD(Xu[a][idx]()(alpha)(i).real(),
                                       Xu[a][idx]()(alpha)(i).imag());
            X_x[kDim24 + r] = ComplexD(Xl[a][idx]()(alpha)(i).real(),
                                       Xl[a][idx]()(alpha)(i).imag());
            Y_x[r]          = ComplexD(Yu[a][idx]()(alpha)(i).real(),
                                       Yu[a][idx]()(alpha)(i).imag());
            Y_x[kDim24 + r] = ComplexD(Yl[a][idx]()(alpha)(i).real(),
                                       Yl[a][idx]()(alpha)(i).imag());
          }
      // Bil(R,C) = 0.5*[conj(Y[C])X[R] + conj(X[C])Y[R]]  (Y<->X symmetrized
      // Wirtinger bilinear; cancels the Im leak into the clover chain rule).
      auto Bil = [&X_x, &Y_x](int R, int C) -> ComplexD {
        return ComplexD(0.5, 0.0) *
            (DtxqcdConj(Y_x[C]) * X_x[R] + DtxqcdConj(X_x[C]) * Y_x[R]);
      };
      SigSobj sig_force; PiSobj pi_force; DSobj d_force;
      NSobj n_force; SSobj s_force; PSobj p_force;
      DtxqcdSiteForceKernel::AuxForceAt(Bil, spin_, sig_force, pi_force,
                                        d_force, n_force, s_force, p_force);
      if (csw_ != 0.0) {
        std::array<CMsobj, 6> cs_arr;
        DtxqcdSiteForceKernel::CloverSigmaAt(Bil, spin_, csw_, cs_arr);
        for (int k = 0; k < 6; ++k) fcs[k][idx] = cs_arr[k];
      }
      // Wirtinger -> physical: transpose (a<->b, i<->j) on the CF fields.
      auto T = [](const auto &fv, auto &ft) {
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int b = 0; b < DtxqcdNf; ++b)
            for (int i = 0; i < Nc; ++i)
              for (int j = 0; j < Nc; ++j)
                ft()(a, b)(i, j) = fv()(b, a)(j, i);
      };
      T(sig_force, fsig[idx]);
      T(pi_force,  fpi[idx]);
      T(d_force,   fd[idx]);
      T(n_force,   fn[idx]);
      fss[idx] = s_force;
      fpp[idx] = p_force;
    });

    AddEvenOddToFull(fsig, dSdU.sigma, cb, coef);
    AddEvenOddToFull(fpi,  dSdU.pi,    cb, coef);
    AddEvenOddToFull(fd,   dSdU.d,     cb, coef);
    AddEvenOddToFull(fn,   dSdU.n,     cb, coef);
    AddEvenOddToFull(fss,  dSdU.s,     cb, coef);
    AddEvenOddToFull(fpp,  dSdU.p,     cb, coef);
    if (csw_ != 0.0)
      for (int k = 0; k < 6; ++k)
        AddEvenOddToFull(fcs[k], clover_sigma_full[k], cb, coef);
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
  DtxqcdRemezAutoScaleParams auto_;         // RAT_AUTO_HI Remez bound auto-scale
};

NAMESPACE_END(Grid);
