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
#include <Grid/qcd/action/dtxqcd/DTXQCDRationalForceGpuKernel.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDRemezAutoScale.h>
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
        param_(p), Phi_(&grid), spin_(grid),
        auto_(DtxqcdRemezAutoScaleParams::FromEnv(p.lo, p.hi)) {
    // Defer the Remez build to the first refresh() when auto-scale is active
    // (it rebuilds on the Lanczos-derived bounds anyway; building here on the
    // fixed [lo,hi] is wasted, and slow at a tiny lo).  refresh() is the first
    // method the HMC integrator calls (before S()/deriv()), so deferral is safe.
    if (!auto_.enabled) {
      BuildRemez(auto_.current_lo, auto_.current_hi);
    } else {
      std::cout << GridLogMessage
                << "[DTXQCDWilsonRationalFull] Remez build deferred to refresh() "
                   "(auto-scale active)" << std::endl;
    }
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalFullAction";
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
  // refresh: Phi = (M^dag M)^{+1/8} eta,  eta full-volume gaussian.
  // The 1/sqrt(2) scale on eta matches Grid's CPS_MD_TIME convention.
  // ------------------------------------------------------------------
  void refresh(const DTXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    auto Dw = MakeEOp(U);
    DTXQCDMOp Mop(Dw);

    // ---- Auto-scale the Remez bounds from the doubled M^dag M spectrum.
    // Lanczos runs on the SAME full-volume operator the multishift CG inverts
    // (cb<0: no checkerboard).  Gated entirely on RAT_AUTO_HI (default OFF ->
    // no Lanczos, no rebuild, byte-identical to the pre-autoscale path).
    {
      RealD new_lo = auto_.current_lo, new_hi = auto_.current_hi;
      bool auto_rebuild = DtxqcdRefreshAutoScale(auto_, Mop, &grid_, pRNG,
                              /*cb=*/-1, "DTXQCDWilsonRationalFull", new_lo, new_hi);
      // First refresh after a deferred ctor build: the Remez is empty, so build
      // it now (on the auto-scaled bounds) even if the bounds didn't widen.
      if (auto_rebuild || PowerNegQuarter.poles.size() == 0) {
        std::cout << GridLogMessage << "[" << action_name()
                  << "] building Remez on [" << new_lo << ", " << new_hi
                  << "]" << std::endl;
        BuildRemez(new_lo, new_hi);
      }
    }

    DTXQCDFermionDoubled eta(&grid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, eta.upper.f[a]);
      gaussian(pRNG, eta.lower.f[a]);
      eta.upper.f[a] = scale * eta.upper.f[a];
      eta.lower.f[a] = scale * eta.lower.f[a];
    }
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
  // Build/rebuild the two Remez approximations (x^{-1/4}, x^{+1/8}) on [lo, hi]
  // and record the active bounds.  Called once in the constructor and again by
  // refresh() when the auto-scale widens the window.
  void BuildRemez(RealD lo, RealD hi) {
    auto_.current_lo = lo;
    auto_.current_hi = hi;
    DtxqcdRebuildRemez(lo, hi, param_.degree, param_.precision,
                       param_.tolerance, "DTXQCDWilsonRationalFull",
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

  // Accumulate (+= coef * arr) a lex-ordered FULL-grid host array into a
  // full-grid lattice.  Twin of the EO sibling's AddEvenOddToFull, but with
  // no checkerboard: vectorizeFromLexOrdArray packs the per-rank local-lex
  // array back to the SIMD-vectorized full grid, then += coef*contrib.
  template <class Field, class Sobj>
  void AddLexToFull(std::vector<Sobj> &arr, Field &full, ComplexD coef) {
    Field contrib(&grid_);
    vectorizeFromLexOrdArray(arr, contrib);
    full = full + coef * contrib;
  }

  // GPU twin of AddLexToFull: contrib is already a full-grid Lattice (written
  // by the GPU kernel), so no vectorize roundtrip -- just += coef*contrib.
  template <class Field>
  void AddFullLatticeToFull(const Field &contrib, Field &full, ComplexD coef) {
    full = full + coef * contrib;
  }

  // DTXQCD_RATFORCE_GPU: route AccumulateSiteForcesAll through the GPU kernel
  // (DTXQCDRationalForceGpuKernel.h, ExtractAll variant).  DEFAULT OFF
  // everywhere (the CPU thread_for is the bit-comparable reference) until
  // FD-verified; off-CUDA always returns the CPU path.
  static int RatForceGpuEnabled() {
    static int v = []() {
#ifndef GRID_CUDA
      return 0;
#else
      const char *e = std::getenv("DTXQCD_RATFORCE_GPU");
      return (e && *e) ? std::atoi(e) : 0;
#endif
    }();
    return v;
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

  // Per-pole aux + clover-sigma site loop on ALL local sites (no CB filter).
  // Bilinear and coefficient identical to the EO sibling's
  // AccumulateSiteForces; the only difference is no checkerboard (X, Y and
  // the force outputs all live on the full Cartesian grid).
  //
  // Multi-rank correctness: unvectorizeToLexOrdArray gives THIS rank's local
  // sublattice (lSites() entries), the thread_for runs over local indices, and
  // vectorizeFromLexOrdArray packs the per-rank result back -- no global-coord
  // peekSite/pokeSite (which built the whole global lattice on every rank and
  // over-counted under GlobalSum at mpi != 1.1.1.1).
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

    typedef typename LatticeFermion::vector_object::scalar_object Fsobj;
    const ComplexD coef(2.0 * ak, 0.0);

    // ---- GPU path (DTXQCD_RATFORCE_GPU): device kernel + full accumulate ----
    if (RatForceGpuEnabled()) {
      LatticeDtxqcdSigma F_sig(&grid_);
      LatticeDtxqcdPi    F_pi (&grid_);
      LatticeDtxqcdD     F_d  (&grid_);
      LatticeDtxqcdN     F_n  (&grid_);
      LatticeDtxqcdS     F_s  (&grid_);
      LatticeDtxqcdP     F_p  (&grid_);
      std::array<LatticeColourMatrix, 6> F_cs{
          LatticeColourMatrix(&grid_), LatticeColourMatrix(&grid_),
          LatticeColourMatrix(&grid_), LatticeColourMatrix(&grid_),
          LatticeColourMatrix(&grid_), LatticeColourMatrix(&grid_)};
      DtxqcdRatForceGpu::ExtractAll(X, Y, csw_, spin_,
                                    F_sig, F_pi, F_d, F_n, F_s, F_p, F_cs);
      AddFullLatticeToFull(F_sig, dSdU.sigma, coef);
      AddFullLatticeToFull(F_pi,  dSdU.pi,    coef);
      AddFullLatticeToFull(F_d,   dSdU.d,     coef);
      AddFullLatticeToFull(F_n,   dSdU.n,     coef);
      AddFullLatticeToFull(F_s,   dSdU.s,     coef);
      AddFullLatticeToFull(F_p,   dSdU.p,     coef);
      if (csw_ != 0.0)
        for (int k = 0; k < 6; ++k)
          AddFullLatticeToFull(F_cs[k], clover_sigma_full[k], coef);
      return;
    }

    // ---- CPU reference path: unvectorize -> thread_for -> vectorize ----
    // The rational force uses ONLY the fermion bilinear Bil(R,C) (no aux peek),
    // so X, Y are all we unvectorize.
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
      // Symmetrized Wirtinger bilinear (same argument as the EO sibling).
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

    AddLexToFull(fsig, dSdU.sigma, coef);
    AddLexToFull(fpi,  dSdU.pi,    coef);
    AddLexToFull(fd,   dSdU.d,     coef);
    AddLexToFull(fn,   dSdU.n,     coef);
    AddLexToFull(fss,  dSdU.s,     coef);
    AddLexToFull(fpp,  dSdU.p,     coef);
    if (csw_ != 0.0)
      for (int k = 0; k < 6; ++k)
        AddLexToFull(fcs[k], clover_sigma_full[k], coef);
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
