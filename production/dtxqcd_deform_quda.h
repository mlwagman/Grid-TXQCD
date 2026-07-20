// dtxqcd_deform_quda.h — QUDA-backed eta-deformed operator M-tilde and CGNR
// solver for the contour-deformation measurements (HANDOFF_QUDA_DEFORM_
// HOPWEIGHTS.md, 2026-07-13).
//
// Chemical-potential frame: M~ = G^{-1} M_def G has UNDEFORMED aux
// cross-blocks and untouched site-diagonal blocks; the entire deformation is
// per-timeslice scalar weights on the temporal hops of the block Wilson
// dslash (upper block fwd w_f(t) = e^{-eta dp(t)/2}, bwd w_b = 1/w_f; lower
// block reciprocal).  Implementation ("Option C" — no QUDA patch, no scaled
// links):
//
//     M~  =  M_base (untouched StyleC device op, unitary links, clover and
//            aux exactly as validated)  +  4 hop-CORRECTION terms
//
//     corr.upper(x) = (w_f(t)-1)(-1/2)(1-g_t) W3(x) psi(x+t)
//                   + (w_b(t)-1)(-1/2)(1+g_t) W7(x) psi(x-t)
//     corr.lower    = same with W3c/W7c and reciprocal weights
//
// with W3 = APBC-phased fwd t-link, W7(x) = W3(x-t)^dag — the exact
// (DERIVCHECK-certified) hop structure of Test_dtxqcd_eta_stoch_pf's dMhop;
// the correction coefficients are e^{-+eta dp/2} - 1 where dMhop carries the
// eta-derivative weights.
//
// DAGGER (the #1 trap in the handoff): M~^dag = M_base^dag (StyleC Mdag,
// exists) + corr^dag, and working out the adjoint hop terms gives the tidy
// identity
//     corr^dag = corr with UPPER<->LOWER coefficient arrays swapped and both
//                spin projectors flipped ((1-g_t) <-> (1+g_t)),
// i.e. the same kernel with two parameter changes — no gauge operations, no
// reciprocal-link reloads.  Verified by the adjointness gate <u,M~v> =
// <M~^dag u,v> and the dense 4^4 OPCHECK.
//
// W3/W7 are t0-INDEPENDENT: per-source (source-centered tent) profile swaps
// rebuild only 4 scalar LatticeComplex masks — SetProfile is O(T) cheap.
//
// Device residency: base op via DtxqcdPackedFullHermOp::M/MdagDeviceDirect
// (packed device conversion -> M_device_csf -> back; no host roundtrip);
// corrections are Grid device expressions (Cshift + pointwise).  Non-QUDA
// builds fall back to the struct-path reference inside the adapter, so all
// gates run on CPU too.

#pragma once

#include <Grid/Grid.h>
#include <Grid/Grid_Eigen_Dense.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_hopcorr_native_decl.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <vector>

namespace Grid {

class DtxqcdDeformQuda {
 public:
  typedef DTXQCDFermionDoubled F;

  DtxqcdDeformQuda(DtxqcdPackedFullHermOp &adapter, GridCartesian &grid,
                   const LatticeGaugeField &Usm, int T)
      : adapter_(adapter), grid_(grid), T_(T),
        g_t_(Gamma::Algebra::GammaT),
        W3_(&grid), W7_(&grid), W3c_(&grid), W7c_(&grid),
        cfu_(&grid), cbu_(&grid), cfl_(&grid), cbl_(&grid),
        tc_(&grid), pk_in_(&grid), pk_out_(&grid) {
    LatticeCoordinate(tc_, Nd - 1);
    // W3 = APBC-phased forward t-link; W7(x) = W3(x-t)^dag; W3c/W7c = the
    // conjugate-gauge pair for the lower (charge-conjugate) block.  The
    // stoch_pf convention block, verbatim.
    LatticeColourMatrix Ut(&grid);
    Ut = PeekIndex<LorentzIndex>(Usm, Nd - 1);
    LatticeColourMatrix mUt(&grid);
    mUt = ComplexD(-1.0, 0.0) * Ut;  // APBC phase on the wrap link
    W3_ = where(tc_ == Integer(T_ - 1), mUt, Ut);
    W7_ = Cshift(LatticeColourMatrix(adj(W3_)), Nd - 1, -1);
    W3c_ = conjugate(W3_);
    W7c_ = Cshift(LatticeColourMatrix(adj(W3c_)), Nd - 1, -1);

#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    // DTXQCD_DEFORM_NATIVE_CORR: CSF-resident hop-correction kernel.
    // Default ON since 2026-07-13 (qcheck 1e-15 vs the Grid sandwich,
    // 19.1 -> 15.2 ms/leg); =0 restores the Grid Cshift corrections.
    // Single-rank only (no halo exchange) — auto-disables on multi-rank;
    // the deform measurements run 1.1.1.1.  Gauge upload is once per
    // config (t0-independent).
    {
      const char *e = std::getenv("DTXQCD_DEFORM_NATIVE_CORR");
      native_ = !(e && *e) || std::atoi(e) != 0;
      if (native_ && grid_._Nprocessors != 1) {
        std::cout << GridLogMessage
                  << "[deform-native] DISABLED: multi-rank run (kernel has "
                  << "no halo exchange)" << std::endl;
        native_ = false;
      }
      if (native_) {
        Coordinate ldims = grid_.LocalDimensions();
        int X[4] = {(int)ldims[0], (int)ldims[1], (int)ldims[2],
                    (int)ldims[3]};
        const std::size_t vol =
            (std::size_t)X[0] * X[1] * X[2] * X[3];
        std::vector<double> h3(vol * 18), h7(vol * 18);
        Coordinate site(Nd);
        typename LatticeColourMatrix::vector_object::scalar_object m3, m7;
        for (int t = 0; t < X[3]; ++t)
          for (int z = 0; z < X[2]; ++z)
            for (int y = 0; y < X[1]; ++y)
              for (int xx = 0; xx < X[0]; ++xx) {
                site[0] = xx; site[1] = y; site[2] = z; site[3] = t;
                peekLocalSite(m3, W3_, site);
                peekLocalSite(m7, W7_, site);
                const int par = (xx + y + z + t) & 1;
                const std::size_t lex =
                    xx + (std::size_t)X[0] *
                             (y + (std::size_t)X[1] *
                                      (z + (std::size_t)X[2] * t));
                const std::size_t gsite = (std::size_t)par * (vol / 2) +
                                          (lex >> 1);
                for (int c = 0; c < Nc; ++c)
                  for (int b = 0; b < Nc; ++b) {
                    const std::complex<double> v3 = DtxqcdToStd(m3()()(c, b));
                    const std::complex<double> v7 = DtxqcdToStd(m7()()(c, b));
                    h3[gsite * 18 + (c * 3 + b) * 2 + 0] = v3.real();
                    h3[gsite * 18 + (c * 3 + b) * 2 + 1] = v3.imag();
                    h7[gsite * 18 + (c * 3 + b) * 2 + 0] = v7.real();
                    h7[gsite * 18 + (c * 3 + b) * 2 + 1] = v7.imag();
                  }
              }
        DtxqcdQudaHopCorrNative::UploadHopGauge(h3.data(), h7.data(), X,
                                                hop_gauge_);
        std::cout << GridLogMessage
                  << "[deform-native] hop gauge uploaded (" << vol
                  << " sites)" << std::endl;
      }
    }
#endif
  }

  // Tent-profile convenience: remembers (eta, t0) so the deflated EO solve
  // can switch to the eigenbasis frame and back (t0 shifts are lattice
  // translations — the deflation basis is built ONCE per (config, eta) and
  // every solve is conjugated into its frame; see SolveTilEODeflated).
  void SetTentProfile(double eta, int t0) {
    cur_eta_ = eta;
    cur_t0_ = t0;
    SetProfile(eta, [this, t0](int t) {
      const int dt = ((t - t0) % T_ + T_) % T_;
      return (double)std::min(dt, T_ - dt);
    });
  }

  // Rebuild the 4 per-timeslice coefficient masks for profile p(.) at
  // deformation strength eta.  Coefficients are (weight - 1):
  //   cfu(t) = e^{-eta dp(t)/2} - 1        dp(t) = p(t+1 mod T) - p(t)
  //   cbu(t) = e^{+eta dp(t-1)/2} - 1
  //   cfl/cbl = reciprocal (lower block / opposite charge)
  void SetProfile(double eta, const std::function<double(int)> &p) {
    prof_eta_ = eta;
    prof_p_ = p;
    ApplyProfile_(eta, p);
  }
  // Temporarily flip the deformation sign (M~(-eta) recip solves — the
  // deform-MG left-vector recipe via the quasi-g5 identity).  Restore with
  // sign=+1.  Does not touch the SetTentProfile deflation bookkeeping.
  void SetEtaSign(int sign) { ApplyProfile_(sign * prof_eta_, prof_p_); }

 private:
  double prof_eta_ = 0.0;
  std::function<double(int)> prof_p_;
  void ApplyProfile_(double eta, const std::function<double(int)> &p) {
    auto dp = [&](int t) { return p((t + 1) % T_) - p(t); };
    LatticeComplex one(&grid_);
    for (int t = 0; t < T_; ++t) {
      const double dpf = dp(t);
      const double dpb = dp((t - 1 + T_) % T_);
      one = ComplexD(std::exp(-eta * dpf / 2.0) - 1.0, 0.0);
      cfu_ = where(tc_ == Integer(t), one, cfu_);
      one = ComplexD(std::exp(+eta * dpb / 2.0) - 1.0, 0.0);
      cbu_ = where(tc_ == Integer(t), one, cbu_);
      one = ComplexD(std::exp(+eta * dpf / 2.0) - 1.0, 0.0);
      cfl_ = where(tc_ == Integer(t), one, cfl_);
      one = ComplexD(std::exp(-eta * dpb / 2.0) - 1.0, 0.0);
      cbl_ = where(tc_ == Integer(t), one, cbl_);
    }
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (native_) {
      std::vector<double> fu(T_), bu(T_), fl(T_), bl(T_);
      for (int t = 0; t < T_; ++t) {
        const double dpf = dp(t);
        const double dpb = dp((t - 1 + T_) % T_);
        fu[t] = std::exp(-eta * dpf / 2.0) - 1.0;
        bu[t] = std::exp(+eta * dpb / 2.0) - 1.0;
        fl[t] = std::exp(+eta * dpf / 2.0) - 1.0;
        bl[t] = std::exp(-eta * dpb / 2.0) - 1.0;
      }
      using namespace DtxqcdQudaHopCorrNative;
      FreeCoeffs(cfu_d_); FreeCoeffs(cbu_d_);
      FreeCoeffs(cfl_d_); FreeCoeffs(cbl_d_);
      cfu_d_ = UploadCoeffs(fu.data(), T_);
      cbu_d_ = UploadCoeffs(bu.data(), T_);
      cfl_d_ = UploadCoeffs(fl.data(), T_);
      cbl_d_ = UploadCoeffs(bl.data(), T_);
      PushEOWeights_();  // device addresses changed — refresh the EO op
    }
#endif
  }

 public:
  void Mtil(const F &in, F &out) { Leg_(in, out, /*dagger=*/false); }
  void MtilDag(const F &in, F &out) { Leg_(in, out, /*dagger=*/true); }

  void Leg_(const F &in, F &out, bool dagger) {
    // DTXQCD_DEFORM_PROFILE=1: per-phase wallclock, printed every 200 legs.
    static const bool prof = [] {
      const char *e = std::getenv("DTXQCD_DEFORM_PROFILE");
      return e && *e && std::atoi(e) != 0;
    }();
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    if (prof) { accelerator_barrier(); t0 = usecond(); }
    DtxqcdPackDoubled(in, pk_in_);
    if (prof) { accelerator_barrier(); t1 = usecond(); }
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (native_ && adapter_.HasDeviceLeg()) {
      // CSF-resident corrections: out_csf += corr(in_csf) inside the leg,
      // no Grid Cshift pass.  Dagger = swapped arrays + flipped projectors.
      const double *fu = dagger ? cfl_d_ : cfu_d_;
      const double *bu = dagger ? cbl_d_ : cbu_d_;
      const double *fl = dagger ? cfu_d_ : cfl_d_;
      const double *bl = dagger ? cbu_d_ : cbl_d_;
      auto hook = [&](quda::ColorSpinorField *ics[4],
                      quda::ColorSpinorField *ocs[4]) {
        using DtxqcdQudaHopCorrNative::ApplyHopCorrNative;
        for (int a = 0; a < DtxqcdNf; ++a) {
          ApplyHopCorrNative(*ics[a], *ocs[a], hop_gauge_, fu, bu,
                             /*conj_gauge=*/false, /*flip_proj=*/dagger);
          ApplyHopCorrNative(*ics[2 + a], *ocs[2 + a], hop_gauge_, fl, bl,
                             /*conj_gauge=*/true, /*flip_proj=*/dagger);
        }
      };
      adapter_.SingleLegDirectHooked(pk_in_, pk_out_, dagger, hook);
      if (prof) { accelerator_barrier(); t2 = usecond(); }
      DtxqcdUnpackDoubled(pk_out_, out);
      if (prof) {
        accelerator_barrier();
        t3 = usecond();
        p_pack_ += t1 - t0;
        p_base_ += t2 - t1;  // includes the native corr kernels
        p_unpack_ += t3 - t2;
        if (++p_n_ % 200 == 0) {
          const double inv = 1e-3 / p_n_;
          std::cout << GridLogMessage << "[deform-prof:native] " << p_n_
                    << " legs, per-leg ms: pack " << p_pack_ * inv
                    << " + base+corr " << p_base_ * inv << " + unpack "
                    << p_unpack_ * inv << " = total "
                    << (p_pack_ + p_base_ + p_unpack_) * inv << std::endl;
        }
      }
      return;
    }
#endif
    if (dagger)
      adapter_.MdagDeviceDirect(pk_in_, pk_out_);
    else
      adapter_.MDeviceDirect(pk_in_, pk_out_);
    if (prof) { accelerator_barrier(); t2 = usecond(); }
    DtxqcdUnpackDoubled(pk_out_, out);
    if (prof) { accelerator_barrier(); t3 = usecond(); }
    CorrAdd_(in, out, dagger);
    if (prof) {
      accelerator_barrier();
      t4 = usecond();
      p_pack_ += t1 - t0;
      p_base_ += t2 - t1;
      p_unpack_ += t3 - t2;
      p_corr_ += t4 - t3;
      if (++p_n_ % 200 == 0) {
        const double inv = 1e-3 / p_n_;
        std::cout << GridLogMessage << "[deform-prof] " << p_n_
                  << " legs, per-leg ms: pack " << p_pack_ * inv << " + base "
                  << p_base_ * inv << " + unpack " << p_unpack_ * inv
                  << " + corr " << p_corr_ * inv << " = total "
                  << (p_pack_ + p_base_ + p_unpack_ + p_corr_) * inv
                  << std::endl;
      }
    }
  }

#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  // ---- EO Schur path (the lambda=5/6 accelerator; HANDOFF §5b consumer) --
  // Mpc~ = Moo - M~oe Mee^{-1} M~eo with M~ee = Mee EXACTLY (the deformation
  // lives only on the t-hops), so the EO op's cached site inverses carry
  // over untouched; the hop legs gain the same correction kernels
  // (parity-crossing variant), injected inside Mq_eo's M_device_csf.
  // Schur prep/back-substitution run ONCE per solve through the validated
  // full-volume M~ (zero-extend + parity picks) + the host MooeeInv.
  void AttachEO(DtxqcdPackedMpcHermOp &eo_adapter, DTXQCDMpcOpQUDA_EO &mq_eo,
                GridRedBlackCartesian &rbgrid,
                DTXQCDWilsonCloverFermionEO &dw) {
    GRID_ASSERT(native_ &&
                "AttachEO requires the native hop-gauge upload "
                "(DTXQCD_DEFORM_NATIVE_CORR)");
    eo_adapter_ = &eo_adapter;
    mq_eo_ = &mq_eo;
    rbgrid_ = &rbgrid;
    dw_ = &dw;
    pk_eo_in_.reset(new LatticeDtxqcdPacked(&rbgrid));
    pk_eo_out_.reset(new LatticeDtxqcdPacked(&rbgrid));
    PushEOWeights_();
  }
  bool HasEO() const { return mq_eo_ != nullptr; }

  // CGNR on Mpc~^dag Mpc~ for the odd block, with even prep/backsub:
  //   b'_o = b_o - M~oe Mee^{-1} b_e ;  solve Mpc~ x_o = b'_o ;
  //   x_e  = Mee^{-1} (b_e - M~eo x_o).
  int SolveTilEO(const F &src, F &x, RealD tol, int cgmax, RealD &rel_out) {
    GRID_ASSERT(mq_eo_);
    F tmp(&grid_), ze(&grid_);
    F b_e(rbgrid_), b_o(rbgrid_), t_e(rbgrid_), t_o(rbgrid_);
    PickCb_(Even, b_e, src);
    PickCb_(Odd, b_o, src);
    // t_e = Mee^{-1} b_e ; b'_o = b_o - pick_o( M~ (t_e ⊕ 0) )
    dw_->MooeeInv(b_e, t_e);
    ZeroExtend_(t_e, ze);
    Mtil(ze, tmp);
    PickCb_(Odd, t_o, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      b_o.upper.f[a] = b_o.upper.f[a] - t_o.upper.f[a];
      b_o.lower.f[a] = b_o.lower.f[a] - t_o.lower.f[a];
    }
    // Odd-block solve: CGNR by default, or the pluggable solver (the
    // two-level deform MG — dtxqcd_deform_mg.h — installs itself here via
    // SetOddSolver; the Schur prep/backsub sandwich is identical).
    F xo(rbgrid_);
    const int it = odd_solver_
                       ? odd_solver_(b_o, xo, tol, cgmax, rel_out)
                       : CgnrEO_(b_o, xo, tol, cgmax, rel_out);
    // Backsub: x_e = Mee^{-1}( b_e - pick_e( M~ (0 ⊕ x_o) ) ).
    ZeroExtend_(xo, ze);
    Mtil(ze, tmp);
    PickCb_(Even, t_e, tmp);
    PickCb_(Even, b_e, src);
    for (int a = 0; a < DtxqcdNf; ++a) {
      t_e.upper.f[a] = b_e.upper.f[a] - t_e.upper.f[a];
      t_e.lower.f[a] = b_e.lower.f[a] - t_e.lower.f[a];
    }
    F x_e(rbgrid_);
    dw_->MooeeInv(t_e, x_e);
    SetCb_(x, x_e);
    SetCb_(x, xo);
    return it;
  }

  // Single Mpc~ / Mpc~^dag applies on odd struct fields (gates + CGNR).
  void MpcTil(const F &in_o, F &out_o) { EoLeg_(in_o, out_o, false); }
  void MpcTilDag(const F &in_o, F &out_o) { EoLeg_(in_o, out_o, true); }

  // Pluggable odd-block solver (b_o, x_o, tol, maxit, rel_out) -> iters.
  void SetOddSolver(
      std::function<int(const F &, F &, RealD, int, RealD &)> s) {
    odd_solver_ = std::move(s);
  }
  void ClearOddSolver() { odd_solver_ = nullptr; }

  // ---- deflated EO solve -------------------------------------------------
  // Build NEV lowest eigenpairs of Mpc~^dag Mpc~ ONCE per (config, eta) at
  // the CURRENT profile (Chebyshev-IRL on the EO adapter's fused device
  // HermOp — deform corrections included).
  // *** STALE-HOPE WARNING (retracted twice: 2026-07-14 and 2026-07-20)
  // *** The original "sparse spectrum, nev~12-16 -> 60k->1k" estimate was
  // *** an ord-301-era resolution artifact.  The true low spectrum is a
  // *** dense band/tower whose modes pack below the achievable IRL
  // *** residual (Ritz mush; filter-quasi-degenerate at any ord tried):
  // *** small-Nstop bases converge to ARBITRARY sub-slices and deflated
  // *** WALL solves come out SLOWER than plain CG (measured at eta=0.6
  // *** AND 0.8).  Production recipe: wall props = plain EO CG.  This
  // *** path remains for the reweighting estimator's exact-term capture
  // *** (bias-free for any basis via the oblique projector) and for
  // *** explicitly-gated experiments only.
  void BuildEODeflation(int nev, GridParallelRNG &pRNG, int ord,
                        RealD lo_frac, RealD irl_tol, int restarts) {
    GRID_ASSERT(mq_eo_);
    defl_evec_.clear();
    defl_eval_.clear();
    defl_t0_ = cur_t0_;
    defl_frame_t0_ = cur_t0_;
    defl_eta_ = cur_eta_;
    LatticeDtxqcdPacked src(rbgrid_);
    gaussian(pRNG, src);  // content-random; cb is the adapter's concern
    src.Checkerboard() = Odd;
    PowerMethod<LatticeDtxqcdPacked> PM;
    const RealD lmax = PM(*eo_adapter_, src);
    std::cout << GridLogMessage << "[deform-defl] lambda_max(Mpc~dagMpc~) ~ "
              << lmax << std::endl;
    const RealD lo = lo_frac * lmax, hi = 1.5 * lmax;
    const int Nstop = nev, Nk = nev + 16, Nm = Nk + std::max(32, Nk);
    Chebyshev<LatticeDtxqcdPacked> cheb(lo, hi, ord);
    FunctionHermOp<LatticeDtxqcdPacked> OpCheby(cheb, *eo_adapter_);
    PlainHermOp<LatticeDtxqcdPacked> Op(*eo_adapter_);
    ImplicitlyRestartedLanczos<LatticeDtxqcdPacked> IRL(OpCheby, Op, Nstop,
                                                        Nk, Nm, irl_tol,
                                                        restarts);
    std::vector<RealD> eval(Nm);
    std::vector<LatticeDtxqcdPacked> evec(Nm, rbgrid_);
    int Nconv = 0;
    const double t0 = usecond();
    IRL.calc(eval, evec, src, Nconv);
    const int nuse = std::min(Nconv, (int)eval.size());
    defl_eval_.assign(eval.begin(), eval.begin() + nuse);
    for (int i = 0; i < nuse; ++i) defl_evec_.push_back(evec[i]);
    std::cout << GridLogMessage << "[deform-defl] " << nuse
              << " eigenpairs in " << (usecond() - t0) * 1e-6
              << " s; ev[0]=" << (nuse ? defl_eval_[0] : 0)
              << " ev[last]=" << (nuse ? defl_eval_[nuse - 1] : 0)
              << " (t0=" << defl_t0_ << " eta=" << defl_eta_ << ")"
              << std::endl;
    // GALERKIN coarse matrix A_c = V^dag (Mpc~^dag Mpc~) V, factored once.
    // Corrections solve A_c y = V^dag r instead of dividing by the Ritz
    // values: eigenvalue division amplifies eigenvector error by 1/lambda
    // (Ritz residual ~1e-3*lambda_MAX vs lambda ~1e-4 => O(1e3) garbage —
    // the Lambda^-1 guess bought ~1% and the Lambda^-1 projection STALLED
    // the solve, jobs 1301281/89).  Galerkin is variationally optimal in
    // span(V) and never divides by an inaccurate eigenvalue.
    defl_Ac_.resize(nuse, nuse);
    {
      LatticeDtxqcdPacked Av(rbgrid_);
      for (int j = 0; j < nuse; ++j) {
        eo_adapter_->HermOp(defl_evec_[j], Av);
        for (int i = 0; i < nuse; ++i) {
          const ComplexD z = innerProduct(defl_evec_[i], Av);
          defl_Ac_(i, j) =
              std::complex<double>(DtxqcdToStd(z).real(),
                                   DtxqcdToStd(z).imag());
        }
      }
      defl_Ac_lu_.compute(defl_Ac_);
      std::cout << GridLogMessage << "[deform-defl] Galerkin A_c ("
                << nuse << "x" << nuse << ") factored" << std::endl;
    }
  }
  bool HasDeflation() const { return !defl_evec_.empty(); }
  // Read access to the deflation basis (packed RB fields) — the stochastic
  // reweighting estimator builds its oblique-projector trace split from
  // these (Test_dtxqcd_eta_stoch_pf, DTXQCD_STOCHPF_DEFLATE).
  const std::vector<LatticeDtxqcdPacked> &EODeflEvecs() const {
    return defl_evec_;
  }
  const std::vector<RealD> &EODeflEvals() const { return defl_eval_; }

  // DIAG gate: apply the CG-SIDE operator (MpcTilDag . MpcTil, the exact
  // legs CgnrEO_ iterates) to basis vectors and print Rayleigh quotient +
  // residual.  Discriminates "IRL pairs are under-resolved mush" (RQ ~
  // eval, residual >> eval — the 9-decade band vs the lambda_max-relative
  // convergence test) from "eigensolve ran on a DIFFERENT operator than
  // the CG" (RQ wildly off eval).
  void VerifyDeflationBasis() {
    if (defl_evec_.empty()) return;
    F v(rbgrid_), t(rbgrid_), Av(rbgrid_);
    LatticeDtxqcdPacked pAv(rbgrid_);
    const int n = (int)defl_evec_.size();
    const int picks[3] = {0, 1, n - 1};
    for (int pi = 0; pi < 3; ++pi) {
      const int i = picks[pi];
      DtxqcdUnpackDoubled(defl_evec_[i], v);
      for (int a = 0; a < DtxqcdNf; ++a) {
        v.upper.f[a].Checkerboard() = Odd;
        v.lower.f[a].Checkerboard() = Odd;
      }
      MpcTil(v, t);
      MpcTilDag(t, Av);
      DtxqcdPackDoubled(Av, pAv);
      pAv.Checkerboard() = Odd;
      const ComplexD rq = innerProduct(defl_evec_[i], pAv);
      pAv = pAv - rq * defl_evec_[i];
      const RealD rn = std::sqrt(norm2(pAv));
      std::cout << GridLogMessage << "[deform-defl-diag] CGop check v[" << i
                << "]: eval " << defl_eval_[i] << "  RQ "
                << DtxqcdToStd(rq).real() << "  |Av-RQ v| " << rn
                << std::endl;
    }
  }

  // ---- deflation basis cache -------------------------------------------
  // The tight (tol=1e-5) eigensolve is ~20-30 min per (config, eta) and is
  // t0-translatable, so it is shared between the propagator solves and the
  // sign-reweighting estimator (same Mpc~).  Raw binary sidecar, SAME-
  // MACHINE cache only (native-endian lex-ordered payload, single rank).
  // Header carries a gauge fingerprint (norm2 of the phased t-link field)
  // so a stale file can never deflate the wrong config.  Files are ~75 MB
  // x nev — put the cache dir somewhere with quota headroom.
  static std::string EODeflationCacheName(int nev, double tol, int ord,
                                          double eta, int t0,
                                          const Coordinate &gd) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "deform_evecs_L%dx%dx%dx%d_eta%.4f_t0%d_nev%d_tol%.0e_ord%d.evc",
                  (int)gd[0], (int)gd[1], (int)gd[2], (int)gd[3], eta, t0,
                  nev, tol, ord);
    return std::string(buf);
  }
  bool SaveEODeflation(const std::string &fn) {
    if (defl_evec_.empty()) return false;
    if (grid_._Nprocessors != 1) {
      std::cout << GridLogMessage
                << "[deform-defl] cache save skipped (multi-rank)"
                << std::endl;
      return false;
    }
    GRID_ASSERT(defl_frame_t0_ == defl_t0_ &&
                "save the basis in its build frame (before any solve "
                "translates it)");
    const std::string tmp = fn + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;
    const uint64_t magic = 0x4445464C45563031ULL;  // "DEFLEV01"
    const int64_t nev = (int64_t)defl_evec_.size();
    Coordinate gd = grid_.GlobalDimensions();
    int64_t dims[4] = {gd[0], gd[1], gd[2], gd[3]};
    const int64_t t0 = defl_t0_;
    const double eta = defl_eta_;
    const double fp = norm2(W3_);
    f.write((const char *)&magic, 8);
    f.write((const char *)dims, 32);
    f.write((const char *)&nev, 8);
    f.write((const char *)&t0, 8);
    f.write((const char *)&eta, 8);
    f.write((const char *)&fp, 8);
    f.write((const char *)defl_eval_.data(), nev * 8);
    std::vector<std::complex<double>> ac(nev * nev);
    for (int j = 0; j < nev; ++j)
      for (int i = 0; i < nev; ++i) ac[j * nev + i] = defl_Ac_(i, j);
    f.write((const char *)ac.data(), nev * nev * 16);
    std::vector<typename LatticeDtxqcdPacked::scalar_object> buf(
        rbgrid_->lSites());
    for (int64_t i = 0; i < nev; ++i) {
      unvectorizeToLexOrdArray(buf, defl_evec_[i]);
      f.write((const char *)buf.data(),
              buf.size() * sizeof(buf[0]));
    }
    f.close();
    if (!f.good()) { std::remove(tmp.c_str()); return false; }
    std::rename(tmp.c_str(), fn.c_str());
    std::cout << GridLogMessage << "[deform-defl] basis cached to " << fn
              << std::endl;
    return true;
  }
  bool LoadEODeflation(const std::string &fn) {
    if (grid_._Nprocessors != 1) return false;
    std::ifstream f(fn, std::ios::binary);
    if (!f) return false;
    uint64_t magic = 0;
    int64_t dims[4] = {0, 0, 0, 0}, nev = 0, t0 = 0;
    double eta = 0, fp = 0;
    f.read((char *)&magic, 8);
    f.read((char *)dims, 32);
    f.read((char *)&nev, 8);
    f.read((char *)&t0, 8);
    f.read((char *)&eta, 8);
    f.read((char *)&fp, 8);
    Coordinate gd = grid_.GlobalDimensions();
    if (magic != 0x4445464C45563031ULL || nev <= 0 || nev > 1024) return false;
    for (int d = 0; d < 4; ++d)
      if (dims[d] != (int64_t)gd[d]) return false;
    if (std::abs(eta - cur_eta_) > 1e-12) return false;
    const double fp_now = norm2(W3_);
    if (std::abs(fp - fp_now) > 1e-8 * std::max(1.0, std::abs(fp_now))) {
      std::cout << GridLogMessage << "[deform-defl] cache " << fn
                << " REJECTED: gauge fingerprint mismatch (" << fp << " vs "
                << fp_now << ")" << std::endl;
      return false;
    }
    defl_eval_.resize(nev);
    f.read((char *)defl_eval_.data(), nev * 8);
    std::vector<std::complex<double>> ac(nev * nev);
    f.read((char *)ac.data(), nev * nev * 16);
    defl_evec_.clear();
    std::vector<typename LatticeDtxqcdPacked::scalar_object> buf(
        rbgrid_->lSites());
    for (int64_t i = 0; i < nev; ++i) {
      f.read((char *)buf.data(), buf.size() * sizeof(buf[0]));
      if (!f.good()) { defl_evec_.clear(); return false; }
      defl_evec_.emplace_back(rbgrid_);
      vectorizeFromLexOrdArray(buf, defl_evec_.back());
      defl_evec_.back().Checkerboard() = Odd;
    }
    defl_t0_ = (int)t0;
    defl_frame_t0_ = (int)t0;
    defl_eta_ = eta;
    defl_Ac_.resize(nev, nev);
    for (int j = 0; j < nev; ++j)
      for (int i = 0; i < nev; ++i) defl_Ac_(i, j) = ac[j * nev + i];
    defl_Ac_lu_.compute(defl_Ac_);
    std::cout << GridLogMessage << "[deform-defl] basis loaded from " << fn
              << " (nev=" << nev << " t0=" << t0 << " eta=" << eta
              << " ev[0]=" << defl_eval_[0] << ")" << std::endl;
    return true;
  }

  // Deflated solve for the CURRENT tent profile (eta must match the basis;
  // t0 may differ by an EVEN shift).  The solve is ALWAYS on the true
  // operator at cur_t0_; only the deflation BASIS is translated between
  // frames.  Conjugating the solve itself (v1 of this routine) is invalid
  // on a fixed gauge configuration: T^dag M~(t0=0,U) T = M~(dt, T^dag U)
  // — the translation moves the gauge field too, so the conjugated solve
  // returned a propagator on a TRANSLATED background (D1 row-24 at 4.4e-3,
  // unchanged by the APBC sign fix — jobs 1301281/1301298).  A translated
  // basis is only APPROXIMATE deflation of the true operator (right tent
  // alignment, wrong gauge modulation), but with exact residual
  // re-formation that costs iterations, never correctness.
  int SolveTilEODeflated(const F &src, F &x, RealD tol, int cgmax,
                         RealD &rel_out) {
    GRID_ASSERT(HasDeflation());
    GRID_ASSERT(std::abs(cur_eta_ - defl_eta_) < 1e-12 &&
                "deflation basis is per-eta");
    if (cur_t0_ != defl_frame_t0_) TranslateDeflationFrame_(cur_t0_);
    return SolveTilEO(src, x, tol, cgmax, rel_out);
  }
#endif

  // CGNR on M~^dag M~ (same loop/semantics as the Grid-path SolveTil).
  // Returns iterations; rel_out = final |r|/|b| (caller decides on
  // nonconvergence policy).
  int SolveTil(const F &src, F &x, RealD tol, int cgmax, RealD &rel_out) {
    F b(&grid_), r(&grid_), pv(&grid_), t1(&grid_), Ap(&grid_);
    MtilDag(src, b);
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = Zero();
      x.lower.f[a] = Zero();
    }
    r = b;
    pv = b;
    RealD r2 = fnorm2_(r), b2 = std::max(fnorm2_(b), 1e-300);
    RealD tol2 = tol * tol * b2;
    int k = 0;
    for (; k < cgmax; ++k) {
      Mtil(pv, t1);
      MtilDag(t1, Ap);
      RealD pAp = finner_(pv, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * pv.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * pv.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD rn = fnorm2_(r);
      if (rn < tol2) { r2 = rn; ++k; break; }
      RealD beta = rn / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        pv.upper.f[a] = r.upper.f[a] + beta * pv.upper.f[a];
        pv.lower.f[a] = r.lower.f[a] + beta * pv.lower.f[a];
      }
      r2 = rn;
    }
    rel_out = std::sqrt(r2 / b2);
    return k;
  }

  // Struct-field linalg helpers (public: also used by the OPCHECK gate).
  static RealD fnorm2_(const F &v) {
    RealD s = 0;
    for (int a = 0; a < DtxqcdNf; ++a)
      s += norm2(v.upper.f[a]) + norm2(v.lower.f[a]);
    return s;
  }
  static ComplexD finner_(const F &u, const F &v) {
    ComplexD s(0, 0);
    for (int a = 0; a < DtxqcdNf; ++a)
      s += innerProduct(u.upper.f[a], v.upper.f[a]) +
           innerProduct(u.lower.f[a], v.lower.f[a]);
    return s;
  }

 private:
  // The 4 hop-correction terms.  dagger: upper<->lower coefficient arrays
  // swapped + both projectors flipped (= gamma5-hermiticity + charge-swap
  // identity in coefficient form; see header comment).
  void CorrAdd_(const F &in, F &out, bool dagger) {
    const LatticeComplex &au = dagger ? cfl_ : cfu_;
    const LatticeComplex &bu = dagger ? cbl_ : cbu_;
    const LatticeComplex &al = dagger ? cfu_ : cfl_;
    const LatticeComplex &bl = dagger ? cbu_ : cbl_;
    const double ps = dagger ? -1.0 : +1.0;  // +1: fwd(1-g_t); -1: flipped
    for (int a = 0; a < DtxqcdNf; ++a) {
      HopAdd_(in.upper.f[a], out.upper.f[a], W3_, W7_, au, bu, ps);
      HopAdd_(in.lower.f[a], out.lower.f[a], W3c_, W7c_, al, bl, ps);
    }
  }
  void HopAdd_(const LatticeFermion &psi, LatticeFermion &out,
               const LatticeColourMatrix &Wf, const LatticeColourMatrix &Wb,
               const LatticeComplex &af, const LatticeComplex &ab,
               double ps) {
    // ps=+1: fwd (1-g_t), bwd (1+g_t) [the dMhop/base-M convention];
    // ps=-1: both flipped (the dagger).  Written as the two literal
    // (DERIVCHECK-certified) expressions rather than a sign-parametrized
    // one to keep Grid's expression templates on the validated path.
    LatticeFermion pp = Cshift(psi, Nd - 1, +1);
    LatticeFermion tf = Wf * pp;
    LatticeFermion pm = Cshift(psi, Nd - 1, -1);
    LatticeFermion tb = Wb * pm;
    if (ps > 0) {
      tf = ComplexD(-0.5, 0.0) * (tf - g_t_ * tf);  // -1/2 (1 - g_t)
      tb = ComplexD(-0.5, 0.0) * (tb + g_t_ * tb);  // -1/2 (1 + g_t)
    } else {
      tf = ComplexD(-0.5, 0.0) * (tf + g_t_ * tf);  // -1/2 (1 + g_t)
      tb = ComplexD(-0.5, 0.0) * (tb - g_t_ * tb);  // -1/2 (1 - g_t)
    }
    out = out + af * tf + ab * tb;
  }

  DtxqcdPackedFullHermOp &adapter_;
  GridCartesian &grid_;
  int T_;
  // Current tent-profile params (SetTentProfile; used by the deflated
  // solve's frame conjugation).
  double cur_eta_ = 0;
  int cur_t0_ = 0;
  // DTXQCD_DEFORM_PROFILE accumulators (us) — see Leg_.
  double p_pack_ = 0, p_base_ = 0, p_unpack_ = 0, p_corr_ = 0;
  long p_n_ = 0;

#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  // ---- EO members / helpers ----
  DtxqcdPackedMpcHermOp *eo_adapter_ = nullptr;
  std::function<int(const F &, F &, RealD, int, RealD &)> odd_solver_;
  DTXQCDMpcOpQUDA_EO *mq_eo_ = nullptr;
  GridRedBlackCartesian *rbgrid_ = nullptr;
  DTXQCDWilsonCloverFermionEO *dw_ = nullptr;
  std::unique_ptr<LatticeDtxqcdPacked> pk_eo_in_, pk_eo_out_;
  // Deflation state (per config+eta; t0 handled by solve conjugation).
  std::vector<LatticeDtxqcdPacked> defl_evec_;
  std::vector<RealD> defl_eval_;
  int defl_t0_ = -1;        // frame the basis was BUILT in
  int defl_frame_t0_ = -1;  // frame the basis currently LIVES in
  double defl_eta_ = -1e9;
  Eigen::MatrixXcd defl_Ac_;
  Eigen::LDLT<Eigen::MatrixXcd> defl_Ac_lu_;

  // Translate the deflation basis in place to the frame t0_new (even shift
  // only — odd flips EO parity).  APBC signed translation: v'(t)=v(t-dt)
  // via Cshift(v,T,-dt) with a sign flip on the wrapped slab t<dt.  The
  // per-vector GLOBAL sign from composing shifts is irrelevant (Galerkin
  // is basis-sign invariant), so in-place composition across frames is
  // safe.  A_c = V'^dag A V' is rebuilt against the CURRENT operator
  // (caller must have SetTentProfile'd to t0_new already).
  void TranslateDeflationFrame_(int t0_new) {
    GRID_ASSERT(cur_t0_ == t0_new &&
                "set the tent profile to the target frame first");
    const int dt = ((t0_new - defl_frame_t0_) % T_ + T_) % T_;
    if (dt == 0) { defl_frame_t0_ = t0_new; return; }
    GRID_ASSERT(dt % 2 == 0 &&
                "odd t0 shift flips EO parity — use even t_src spacing");
    const double tt0 = usecond();
    LatticeDtxqcdPacked full(&grid_), shifted(&grid_), half(rbgrid_);
    for (size_t i = 0; i < defl_evec_.size(); ++i) {
      full = Zero();
      setCheckerboard(full, defl_evec_[i]);
      shifted = Cshift(full, Nd - 1, -dt);
      shifted = where(tc_ < Integer(dt), LatticeDtxqcdPacked(-shifted),
                      shifted);
      pickCheckerboard(Odd, half, shifted);
      defl_evec_[i] = half;
    }
    defl_frame_t0_ = t0_new;
    const int nuse = (int)defl_evec_.size();
    defl_Ac_.resize(nuse, nuse);
    {
      LatticeDtxqcdPacked Av(rbgrid_);
      for (int j = 0; j < nuse; ++j) {
        eo_adapter_->HermOp(defl_evec_[j], Av);
        for (int i = 0; i < nuse; ++i) {
          const ComplexD z = innerProduct(defl_evec_[i], Av);
          defl_Ac_(i, j) =
              std::complex<double>(DtxqcdToStd(z).real(),
                                   DtxqcdToStd(z).imag());
        }
      }
      defl_Ac_lu_.compute(defl_Ac_);
    }
    std::cout << GridLogMessage << "[deform-defl] basis translated to t0="
              << t0_new << " + A_c rebuilt in "
              << (usecond() - tt0) * 1e-6 << " s" << std::endl;
  }

  // Galerkin coarse correction g = V A_c^-1 V^dag pr (packed odd fields).
  void GalerkinCorrection_(const LatticeDtxqcdPacked &pr,
                           LatticeDtxqcdPacked &g) {
    const int n = (int)defl_evec_.size();
    Eigen::VectorXcd c(n);
    for (int i = 0; i < n; ++i) {
      const std::complex<double> z =
          DtxqcdToStd(ComplexD(innerProduct(defl_evec_[i], pr)));
      c(i) = z;
    }
    const Eigen::VectorXcd y = defl_Ac_lu_.solve(c);
    g = Zero();
    g.Checkerboard() = Odd;
    for (int i = 0; i < n; ++i)
      g = g + ComplexD(y(i).real(), y(i).imag()) * defl_evec_[i];
  }

  void PushEOWeights_() {
    if (mq_eo_)
      mq_eo_->SetDeformHopWeights(&hop_gauge_, cfu_d_, cbu_d_, cfl_d_,
                                  cbl_d_);
  }
  void PickCb_(int cb, F &half, const F &full) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(cb, half.upper.f[a], full.upper.f[a]);
      pickCheckerboard(cb, half.lower.f[a], full.lower.f[a]);
    }
  }
  void SetCb_(F &full, const F &half) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      setCheckerboard(full.upper.f[a], half.upper.f[a]);
      setCheckerboard(full.lower.f[a], half.lower.f[a]);
    }
  }
  void ZeroExtend_(const F &half, F &full) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      full.upper.f[a] = Zero();
      full.lower.f[a] = Zero();
    }
    SetCb_(full, half);
  }
  void EoLeg_(const F &in_o, F &out_o, bool dagger) {
    DtxqcdPackDoubled(in_o, *pk_eo_in_);
    pk_eo_in_->Checkerboard() = Odd;
    if (dagger)
      eo_adapter_->MdagDeviceDirect(*pk_eo_in_, *pk_eo_out_);
    else
      eo_adapter_->MDeviceDirect(*pk_eo_in_, *pk_eo_out_);
    DtxqcdUnpackDoubled(*pk_eo_out_, out_o);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_o.upper.f[a].Checkerboard() = Odd;
      out_o.lower.f[a].Checkerboard() = Odd;
    }
  }
  int CgnrEO_(const F &src_o, F &x_o, RealD tol, int cgmax, RealD &rel_out) {
    F b(rbgrid_), r(rbgrid_), pv(rbgrid_), t1(rbgrid_), Ap(rbgrid_);
    MpcTilDag(src_o, b);
    const RealD b2_orig = std::max(fnorm2_(b), 1e-300);  // pre-deflation RHS
    for (int a = 0; a < DtxqcdNf; ++a) {
      x_o.upper.f[a] = Zero();
      x_o.lower.f[a] = Zero();
      x_o.upper.f[a].Checkerboard() = Odd;
      x_o.lower.f[a].Checkerboard() = Odd;
    }
    // Galerkin initial guess x0 = V A_c^-1 V^dag b (guess-deflation on the
    // normal equations; the basis must live in the current frame —
    // SolveTilEODeflated translates it before dispatching here).
    const bool defl = HasDeflation() && cur_t0_ == defl_frame_t0_ &&
                      std::abs(cur_eta_ - defl_eta_) < 1e-12;
    // DTXQCD_DEFORM_EO_DEFLATE_DIAG=1: per-mode source overlaps + the
    // post-guess residual ratio.  Discriminates "source orthogonal to the
    // basis" (|r0|/|b| ~ 1, guess correctly inert) from "angle / hidden-
    // deeper-mode problem" (|r0|/|b| << 1 yet iters unchanged) — the
    // eta=0.8 tight-basis guess bought +2 iters on 59,931 (job 1301317)
    // and only this ratio says why.
    static const bool defl_diag = [] {
      const char *e = std::getenv("DTXQCD_DEFORM_EO_DEFLATE_DIAG");
      return e && *e && std::atoi(e) != 0;
    }();
    if (defl) {
      DtxqcdPackDoubled(b, *pk_eo_in_);
      pk_eo_in_->Checkerboard() = Odd;
      if (defl_diag) {
        // NB pk_eo_in_ is clobbered by every MpcTil/MpcTilDag — overlaps
        // must be taken HERE, while it still holds the pre-guess RHS.
        for (size_t i = 0; i < defl_evec_.size(); ++i) {
          const ComplexD c = innerProduct(defl_evec_[i], *pk_eo_in_);
          std::cout << GridLogMessage << "[deform-defl-diag] |v[" << i
                    << "]^dag b|/|b| = "
                    << DtxqcdAbs(c) / std::sqrt(b2_orig)
                    << "  (eval " << defl_eval_[i] << ")" << std::endl;
        }
      }
      LatticeDtxqcdPacked g(rbgrid_);
      GalerkinCorrection_(*pk_eo_in_, g);
      DtxqcdUnpackDoubled(g, x_o);
      for (int a = 0; a < DtxqcdNf; ++a) {
        x_o.upper.f[a].Checkerboard() = Odd;
        x_o.lower.f[a].Checkerboard() = Odd;
      }
      MpcTil(x_o, t1);
      MpcTilDag(t1, Ap);
      for (int a = 0; a < DtxqcdNf; ++a) {
        b.upper.f[a] = b.upper.f[a] - Ap.upper.f[a];
        b.lower.f[a] = b.lower.f[a] - Ap.lower.f[a];
      }
      if (defl_diag) {
        const RealD r02 = fnorm2_(b);
        std::cout << GridLogMessage << "[deform-defl-diag] |r0|/|b| = "
                  << std::sqrt(r02 / b2_orig) << std::endl;
      }
    }
    r = b;
    pv = b;
    RealD r2 = fnorm2_(r);
    const RealD b2 = b2_orig;
    RealD tol2 = tol * tol * b2;
    // Periodic coarse re-correction (two-level deflated CG): the guess
    // alone is defeated by Ritz-pair error (1e-3 of lambda_MAX >> the
    // eigenvalues themselves — job 1301281: 16-evec guess bought ~1% at
    // eta=0.6).  Every PROJ_EVERY iterations: x += V L^-1 V^dag r, r
    // recomputed exactly, CG restarted.  ~+2% matvec cost at 50.
    static const int proj_every = [] {
      const char *e = std::getenv("DTXQCD_DEFORM_EO_DEFLATE_PROJ");
      return (e && *e) ? std::atoi(e) : 50;
    }();
    int k = 0;
    int since_proj = 0;
    for (; k < cgmax; ++k) {
      MpcTil(pv, t1);
      MpcTilDag(t1, Ap);
      RealD pAp = finner_(pv, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x_o.upper.f[a] = x_o.upper.f[a] + alpha * pv.upper.f[a];
        x_o.lower.f[a] = x_o.lower.f[a] + alpha * pv.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD rn = fnorm2_(r);
      if (rn < tol2) { r2 = rn; ++k; break; }
      if (defl && proj_every > 0 && ++since_proj >= proj_every) {
        since_proj = 0;
        // Galerkin correction x += V A_c^-1 V^dag r ; r recomputed exactly;
        // CG restarted.
        const RealD rn_pre = rn;  // pre-correction (for the DIAG print)
        DtxqcdPackDoubled(r, *pk_eo_in_);
        pk_eo_in_->Checkerboard() = Odd;
        LatticeDtxqcdPacked g(rbgrid_);
        GalerkinCorrection_(*pk_eo_in_, g);
        F dx(rbgrid_);
        DtxqcdUnpackDoubled(g, dx);
        for (int a = 0; a < DtxqcdNf; ++a) {
          dx.upper.f[a].Checkerboard() = Odd;
          dx.lower.f[a].Checkerboard() = Odd;
          x_o.upper.f[a] = x_o.upper.f[a] + dx.upper.f[a];
          x_o.lower.f[a] = x_o.lower.f[a] + dx.lower.f[a];
        }
        MpcTil(x_o, t1);
        MpcTilDag(t1, Ap);
        MpcTilDag(src_o, b);  // re-form the true RHS
        for (int a = 0; a < DtxqcdNf; ++a) {
          r.upper.f[a] = b.upper.f[a] - Ap.upper.f[a];
          r.lower.f[a] = b.lower.f[a] - Ap.lower.f[a];
        }
        rn = fnorm2_(r);
        // Failure-mode discriminator (needs DIAG=1): a big pre->post drop
        // means the stalled weight WAS in span(V) (cadence/restart story);
        // no drop means it lives outside the basis (need more nev).
        if (defl_diag)
          std::cout << GridLogMessage << "[deform-defl-diag] corr@" << k
                    << " relres " << std::sqrt(rn_pre / b2_orig) << " -> "
                    << std::sqrt(rn / b2_orig) << std::endl;
        if (rn < tol2) { r2 = rn; ++k; break; }
        pv = r;  // restart the Krylov recurrence
        r2 = rn;
        continue;
      }
      RealD beta = rn / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        pv.upper.f[a] = r.upper.f[a] + beta * pv.upper.f[a];
        pv.lower.f[a] = r.lower.f[a] + beta * pv.lower.f[a];
      }
      r2 = rn;
    }
    rel_out = std::sqrt(r2 / b2);
    return k;
  }
#endif
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  // DTXQCD_DEFORM_NATIVE_CORR state (see ctor).
  bool native_ = false;
  DtxqcdQudaHopCorrNative::HopGauge hop_gauge_;
  double *cfu_d_ = nullptr, *cbu_d_ = nullptr;
  double *cfl_d_ = nullptr, *cbl_d_ = nullptr;

 public:
  ~DtxqcdDeformQuda() {
    using namespace DtxqcdQudaHopCorrNative;
    FreeHopGauge(hop_gauge_);
    FreeCoeffs(cfu_d_); FreeCoeffs(cbu_d_);
    FreeCoeffs(cfl_d_); FreeCoeffs(cbl_d_);
  }

 private:
#endif
  Gamma g_t_;
  LatticeColourMatrix W3_, W7_, W3c_, W7c_;
  LatticeComplex cfu_, cbu_, cfl_, cbl_;
  LatticeInteger tc_;
  LatticeDtxqcdPacked pk_in_, pk_out_;
};

}  // namespace Grid
