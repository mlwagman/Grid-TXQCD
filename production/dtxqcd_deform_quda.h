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
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_hopcorr_native_decl.h>
#endif

#include <cmath>
#include <cstdlib>
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

  // Rebuild the 4 per-timeslice coefficient masks for profile p(.) at
  // deformation strength eta.  Coefficients are (weight - 1):
  //   cfu(t) = e^{-eta dp(t)/2} - 1        dp(t) = p(t+1 mod T) - p(t)
  //   cbu(t) = e^{+eta dp(t-1)/2} - 1
  //   cfl/cbl = reciprocal (lower block / opposite charge)
  void SetProfile(double eta, const std::function<double(int)> &p) {
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
    }
#endif
  }

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
    if (native_) {
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
  // DTXQCD_DEFORM_PROFILE accumulators (us) — see Leg_.
  double p_pack_ = 0, p_base_ = 0, p_unpack_ = 0, p_corr_ = 0;
  long p_n_ = 0;
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
