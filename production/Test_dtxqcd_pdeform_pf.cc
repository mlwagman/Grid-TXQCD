// Test_dtxqcd_pdeform_pf: REFERENCE reweighting-factor estimator for the
// p-field (eta' singlet) contour deformation.
//
// Contour: a real per-timeslice rescale of the pseudoscalar singlet,
//     p(x,t) -> ptil(x,t) = e^{-eta prof(t)} p(x,t).
// p is baryon-charge NEUTRAL, so unlike the d/n deformation there is no
// conjugate-block inverse scaling and no chemical-potential frame: the SAME
// factor multiplies p everywhere it appears (upper and lower blocks).  It is
// a genuine aux-field contour, not a hop-weight trick.  Consequences:
//   - the reweighting cost lives wherever the profile has AMPLITUDE (a
//     per-slice modification of the site matrix), not where it has
//     gradients -- the tent-optimality argument for d does NOT carry over;
//   - expected conditioning: mild.  Large eta*prof just interpolates the
//     operator toward its p=0 (parity-symmetric) version.  The per-node CG
//     iteration counts printed here ARE the conditioning measurement.
//
// Weight (graded-down observable convention, eta > 0):
//   w = J * exp(-dS_gauss) * exp(+dlogPf)
//   log J    = -eta * V3 * sum_t prof(t)          (constant: 1 real DOF/site;
//                                                  cancels in <Ow>/<w>, printed
//                                                  for bookkeeping only)
//   dS_gauss = (lambda^2/2) c_sp sum_t (e^{-2 eta prof(t)} - 1) P2(t)  EXACT,
//              P2(t) = sum_x |p(x,t)|^2,  c_sp = 6/7 (TRACELESS_AUX) or 1
//   dlogPf   = logPf(eta) - logPf(0)
//            = int_0^eta (1/2) Tr[ Mdef(e)^{-1} dMdef/de ] de   (stochastic:
//              Gauss-Legendre in e, Gaussian-noise Hutchinson, host CGNR)
// with  Mdef(e) = M48 + X_p[(e^{-e prof} - 1) p]  on BOTH diagonal blocks
// (X_p = the p gamma5 singlet coupling, DtxqcdApplyDeltaDiag{,Lower}), and
//       dMdef/de = X_p[ -prof(t) e^{-e prof(t)} p ].
// The p gamma5 term is Hermitian per site (real p), so Mdag gets the SAME
// additions.  DERIVCHECK certifies the vertex against the finite difference
// of Mdef in eta on a random vector before any statistics are collected.
//
// This is the deliberately-simple correctness reference for the meson-sector
// deformation program (only-p contour targeting the eta' correlator; see the
// dibaryon graded-reweighting notes for why graded-down + reweighting is the
// S/N mechanism).  Optimization (EO Schur, QUDA, dilution, TSRC ratios)
// follows the d-estimator's stack and is the engineering fork's job.
//
// Env: LATT (4.4.4.4), CFG_DIR, TRAJ (250), ETA (0.3), PROFILE (tent|cos),
//      PROF_T0 (0), N_NOISE (16), N_GL (4), MASS (-0.245),
//      CSW (1.24930970916466), STOUT_NSMEAR (1), STOUT_RHO (0.125),
//      LAMBDA_DTXQCD (5), CG_TOL (1e-8), CG_MAX (20000), NOISE_SEED (77),
//      DERIVCHECK (1), PDEF_MODE (phase|rescale, default phase),
//      IMP_WIDTH (1.0, units of the physical p width), IMP_SEED (555 --
//      keep FIXED across eta and reuse in the offline observable grading).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atoi(v) : d;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::vector<int> latt_dims{4, 4, 4, 4};
  if (const char *v = std::getenv("LATT"); v && *v) {
    latt_dims.clear();
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid);
  const int T = latt[Nd - 1];
  RealD V3 = 1.0;
  for (int mu = 0; mu < Nd - 1; ++mu) V3 *= (RealD)latt[mu];

  const int traj      = env_int("TRAJ", 250);
  const double eta    = env_real("ETA", 0.3);
  const int n_noise   = env_int("N_NOISE", 16);
  const int n_gl      = env_int("N_GL", 4);
  const RealD mass    = env_real("MASS", -0.245);
  const RealD csw     = env_real("CSW", 1.24930970916466);
  const int nstout    = env_int("STOUT_NSMEAR", 1);
  const RealD rho     = env_real("STOUT_RHO", 0.125);
  const RealD lam     = env_real("LAMBDA_DTXQCD", 5.0);
  const RealD tol     = env_real("CG_TOL", 1e-8);
  const int cgmax     = env_int("CG_MAX", 20000);
  const int prof_t0   = env_int("PROF_T0", 0);
  const bool prof_cos = [&] {
    const char *v = std::getenv("PROFILE");
    return v && std::string(v) == "cos";
  }();
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam5.0000_4t4_signhop_lam5";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  std::cout << GridLogMessage << "==== Test_dtxqcd_pdeform_pf ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " eta=" << eta << " profile=" << (prof_cos ? "cos" : "tent")
            << " t0=" << prof_t0 << " N_NOISE=" << n_noise
            << " N_GL=" << n_gl << " lambda=" << lam << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  {
    const int s = env_int("NOISE_SEED", 77);
    pRNG.SeedFixedIntegers({s, s + 3, s + 7, s + 13, s + 21});
  }
  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);

  // ---- profile ----------------------------------------------------------
  auto prof = [&](int t) {
    double d = (double)(((t - prof_t0) % T + T) % T);
    if (prof_cos)
      return 0.25 * (double)T * (1.0 - std::cos(2.0 * M_PI * d / (double)T));
    return std::min(d, (double)T - d);   // sharp tent
  };

  // ---- PDEF_MODE=phase (default) | rescale ------------------------------
  // rescale: the REAL change of variables p -> e^{-e prof} p (kept as the
  //   reference/control; weight has the exact extensive Gaussian piece and
  //   provably cannot beat plain sampling -- see the 2026-08-02 spread pilot).
  // phase (mwagman): reinterpret the real Gaussian as |p_c|^2 with
  //   p_c = Re p + i Im p, Im p a FICTITIOUS decoupled Gaussian (drawn here,
  //   IMP_SEED/IMP_WIDTH; ultralocal, so <p_c p_c> = <Re p Re p> for dt>=1),
  //   fermions couple to Re p only.  Deform arg p_c -> arg p_c + i e prof(t):
  //   a true CONTOUR move -- Gaussian |p_c|^2 INVARIANT (no extensive dS_g),
  //   Jacobian 1, observable monomials p_c grade by e^{-e prof} exactly, and
  //   the fermion coupling becomes
  //     Re p -> Re p cosh(e prof) - i Im p sinh(e prof)
  //   so the O(e) weight is a PURE PHASE (-i Im p sinh gamma5 insertion, the
  //   charge-trace analog) with magnitude cost only at O(e^2) via
  //   (cosh-1) Re p.  Weight = Pf ratio ONLY, complex.  The dagger additions
  //   use the CONJUGATE weighted field (gamma5-Hermiticity is broken by
  //   complex p; CGNR is agnostic).
  const bool phase_mode = [&] {
    const char *v = std::getenv("PDEF_MODE");
    return !v || std::string(v) != "rescale";
  }();
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  LatticeDtxqcdP imp(&Grid);
  imp = Zero();
  if (phase_mode) {
    // Im p ~ N(0, (IMP_WIDTH * sigma_p)^2), sigma_p^2 = 1/(lambda^2 c_sp)
    // the physical p width.  IMP_WIDTH = 1 is LOCKED by the construction
    // (mwagman): the arg-shift is a valid contour ONLY if the Gaussian is a
    // function of |p_c|^2 alone, i.e. ISOTROPIC -- Im width == Re width.
    // Any other value breaks the invariance and the Pf-ratio-only weight is
    // missing a Gaussian piece: diagnostic use only, NOT a valid contour.
    // Deterministic per config: IMP_SEED must be FIXED across eta (and
    // reused by the offline observable grading).
    const double c_sp0 = DtxqcdTracelessAux() ? 6.0 / 7.0 : 1.0;
    const double width =
        env_real("IMP_WIDTH", 1.0) / (lam * std::sqrt(c_sp0));
    GridParallelRNG iRNG(&Grid);
    const int is = env_int("IMP_SEED", 555);
    iRNG.SeedFixedIntegers({is, is + 5, is + 11, is + 17, is + 23});
    gaussian(iRNG, imp);
    imp = 0.5 * (imp + conjugate(imp));          // real part, variance 1/2
    imp = ComplexD(width * std::sqrt(2.0), 0.0) * imp;
    std::cout << GridLogMessage << "PDEF phase mode: IMP_WIDTH="
              << env_real("IMP_WIDTH", 1.0) << " (abs " << width
              << ")  IMP_SEED=" << is << std::endl;
  }
  LatticeDtxqcdP p_w(&Grid), p_wc(&Grid), p_d(&Grid);
  auto WeightP = [&](LatticeDtxqcdP &out, const LatticeDtxqcdP &base,
                     auto fac) {
    out = base;
    for (int t = 0; t < T; ++t) {
      auto scaled = base;
      scaled = fac(t) * scaled;
      out = where(tc == Integer(t), scaled, out);
    }
  };
  LatticeDtxqcdP tmpP(&Grid);
  auto BuildWeights = [&](double e) {
    if (!phase_mode) {
      WeightP(p_w, U.p,
              [&](int t) { return ComplexD(std::exp(-e * prof(t)) - 1.0, 0.0); });
      WeightP(p_d, U.p,
              [&](int t) { return ComplexD(-prof(t) * std::exp(-e * prof(t)), 0.0); });
      p_wc = p_w;
      return;
    }
    // insertion  = (cosh(e prof)-1) Re p - i sinh(e prof) Im p
    // d/de       = prof (sinh(e prof) Re p - i cosh(e prof) Im p)
    WeightP(p_w, U.p,
            [&](int t) { return ComplexD(std::cosh(e * prof(t)) - 1.0, 0.0); });
    WeightP(tmpP, imp,
            [&](int t) { return ComplexD(0.0, -std::sinh(e * prof(t))); });
    p_w = p_w + tmpP;
    p_wc = conjugate(p_w);
    WeightP(p_d, U.p,
            [&](int t) { return ComplexD(prof(t) * std::sinh(e * prof(t)), 0.0); });
    WeightP(tmpP, imp,
            [&](int t) { return ComplexD(0.0, -prof(t) * std::cosh(e * prof(t))); });
    p_d = p_d + tmpP;
  };

  // ---- deformed operator + vertex ---------------------------------------
  // X_p additions on BOTH blocks (charge-neutral: same factor, no G-frame).
  // Zero sigma/pi/s spectators for the p-only apply.
  LatticeDtxqcdSigma sig0(&Grid); sig0 = Zero();
  LatticeDtxqcdPi    pi0(&Grid);  pi0  = Zero();
  LatticeDtxqcdS     s0(&Grid);   s0   = Zero();
  typedef DTXQCDFermionDoubled F;
  auto AddPDiag = [&](const LatticeDtxqcdP &pp, const F &in, F &out) {
    DTXQCDFermionNf c(&Grid);
    DtxqcdApplyDeltaDiag(sig0, pi0, s0, pp, in.upper, c);
    for (int a = 0; a < DtxqcdNf; ++a)
      out.upper.f[a] = out.upper.f[a] + c.f[a];
    DtxqcdApplyDeltaDiagLower(sig0, pi0, s0, pp, in.lower, c);
    for (int a = 0; a < DtxqcdNf; ++a)
      out.lower.f[a] = out.lower.f[a] + c.f[a];
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    AddPDiag(p_w, in, out);
  };
  // p gamma5 is Hermitian per site (real p, singlet transpose-trivial), so
  // the dagger gets the SAME additions -- certified by DERIVCHECK + the CGNR
  // residual actually converging (a wrong dagger stalls CGNR immediately).
  auto MdefDag = [&](const F &in, F &out) {
    Mop.Mdag(in, out);
    AddPDiag(p_wc, in, out);      // conjugate field (== p_w in rescale mode)
  };
  auto dMp = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = Zero();
      out.lower.f[a] = Zero();
    }
    AddPDiag(p_d, in, out);
  };

  auto axpyF = [&](F &y, RealD a, const F &x) {
    for (int b = 0; b < DtxqcdNf; ++b) {
      y.upper.f[b] = y.upper.f[b] + a * x.upper.f[b];
      y.lower.f[b] = y.lower.f[b] + a * x.lower.f[b];
    }
  };
  auto norm2F = [&](const F &x) {
    RealD n = 0.0;
    for (int b = 0; b < DtxqcdNf; ++b)
      n += norm2(x.upper.f[b]) + norm2(x.lower.f[b]);
    return n;
  };
  auto innerF = [&](const F &x, const F &y) {
    ComplexD z(0.0, 0.0);
    for (int b = 0; b < DtxqcdNf; ++b)
      z += innerProduct(x.upper.f[b], y.upper.f[b]) +
           innerProduct(x.lower.f[b], y.lower.f[b]);
    return z;
  };

  // ---- DERIVCHECK: dMp == FD of Mdef in eta -----------------------------
  if (env_int("DERIVCHECK", 1) != 0) {
    const double e0 = 0.5 * eta, de = 1e-5;
    F v(&Grid), ya(&Grid), yp(&Grid), ym(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, v.upper.f[a]);
      gaussian(pRNG, v.lower.f[a]);
    }
    BuildWeights(e0);
    dMp(v, ya);
    BuildWeights(e0 + de);
    Mdef(v, yp);
    BuildWeights(e0 - de);
    Mdef(v, ym);
    axpyF(yp, -1.0, ym);                    // yp = M(e+de) v - M(e-de) v
    RealD fd2 = 0.0, dev2 = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion du(&Grid), dl(&Grid);
      du = ComplexD(0.5 / de, 0.0) * yp.upper.f[a] - ya.upper.f[a];
      dl = ComplexD(0.5 / de, 0.0) * yp.lower.f[a] - ya.lower.f[a];
      dev2 += norm2(du) + norm2(dl);
      du = ComplexD(0.5 / de, 0.0) * yp.upper.f[a];
      dl = ComplexD(0.5 / de, 0.0) * yp.lower.f[a];
      fd2 += norm2(du) + norm2(dl);
    }
    const double rel = std::sqrt(dev2 / std::max(fd2, 1e-300));
    std::cout << GridLogMessage << "PDEF-DERIVCHECK |dMp - FD(Mdef)|/|FD| = "
              << rel << (rel < 1e-4 ? "   MATCH" : "   MISMATCH") << std::endl;
    if (rel >= 1e-4) { Grid_finalize(); std::_Exit(2); }
  }

  // ---- exact pieces ------------------------------------------------------
  // dS_gauss from per-slice sum of |p|^2; log J analytic.
  std::vector<RealD> P2(T);
  {
    LatticeComplex pp(&Grid);
    pp = U.p * conjugate(U.p);
    std::vector<typename LatticeComplex::scalar_object> sl;
    sliceSum(pp, sl, Nd - 1);
    for (int t = 0; t < T; ++t) P2[t] = TensorRemove(sl[t]).real();
  }
  const RealD c_sp = DtxqcdTracelessAux() ? 6.0 / 7.0 : 1.0;
  RealD dSg = 0.0, sumprof = 0.0;
  for (int t = 0; t < T; ++t) {
    if (!phase_mode)
      dSg += 0.5 * lam * lam * c_sp *
             (std::exp(-2.0 * eta * prof(t)) - 1.0) * P2[t];
    sumprof += prof(t);
  }
  // phase mode: Gaussian |p_c|^2 invariant, Jacobian 1 -- both identically 0.
  const RealD logJ = phase_mode ? 0.0 : -eta * V3 * sumprof;

  // ---- GL quadrature + Hutchinson ---------------------------------------
  static const double gl_x1[1] = {0.5};
  static const double gl_w1[1] = {1.0};
  static const double gl_x2[2] = {0.21132486540518713, 0.78867513459481290};
  static const double gl_w2[2] = {0.5, 0.5};
  static const double gl_x4[4] = {0.06943184420297371, 0.33000947820757187,
                                  0.66999052179242810, 0.93056815579702629};
  static const double gl_w4[4] = {0.17392742256872693, 0.32607257743127307,
                                  0.32607257743127307, 0.17392742256872693};
  static const double gl_x8[8] = {0.01985507, 0.10166676, 0.23723379, 0.40828268,
                                  0.59171732, 0.76276621, 0.89833324, 0.98014493};
  static const double gl_w8[8] = {0.05061427, 0.11119052, 0.15685332, 0.18134189,
                                  0.18134189, 0.15685332, 0.11119052, 0.05061427};
  GRID_ASSERT(n_gl == 1 || n_gl == 2 || n_gl == 4 || n_gl == 8);
  const double *gl_x = (n_gl == 1) ? gl_x1 : (n_gl == 2) ? gl_x2
                       : (n_gl == 4) ? gl_x4 : gl_x8;
  const double *gl_w = (n_gl == 1) ? gl_w1 : (n_gl == 2) ? gl_w2
                       : (n_gl == 4) ? gl_w4 : gl_w8;

  // Grid gaussian fills each real component ~ N(0,1) => E[xi xi^dag] = 2.
  // Verified at runtime below (a wrong normalization scales dlogPf by 2x).
  const RealD noise_var = 2.0;
  {
    F xi(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, xi.upper.f[a]);
      gaussian(pRNG, xi.lower.f[a]);
    }
    const RealD v = norm2F(xi) / (48.0 * (RealD)Grid.gSites());
    std::cout << GridLogMessage << "PDEF noise-variance check: " << v
              << " (using " << noise_var << ")" << std::endl;
    if (std::abs(v - noise_var) / noise_var > 0.10) {
      std::cout << GridLogMessage << "PDEF noise normalization MISMATCH -- fix "
                << "noise_var before trusting dlogPf" << std::endl;
      Grid_finalize();
      std::_Exit(4);
    }
  }

  ComplexD dlogpf(0.0, 0.0);
  RealD dlogpf_var_re = 0.0, dlogpf_var_im = 0.0;
  for (int q = 0; q < n_gl; ++q) {
    const double e = eta * gl_x[q];
    BuildWeights(e);
    ComplexD acc(0.0, 0.0);
    RealD acc2_re = 0.0, acc2_im = 0.0;
    int last_it = 0;
    for (int s = 0; s < n_noise; ++s) {
      F xi(&Grid), x(&Grid), b(&Grid), r(&Grid), pv(&Grid), t1(&Grid), Ap(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, xi.upper.f[a]);
        gaussian(pRNG, xi.lower.f[a]);
      }
      // host CGNR: (Mdag M) x = Mdag xi
      MdefDag(xi, b);
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = Zero(); x.lower.f[a] = Zero();
      }
      r = b; pv = b;
      RealD r2 = norm2F(r), b2 = std::max(norm2F(b), 1e-300);
      const RealD tol2 = tol * tol * b2;
      int k = 0;
      for (; k < cgmax; ++k) {
        Mdef(pv, t1);
        MdefDag(t1, Ap);
        RealD pAp = innerF(pv, Ap).real();
        RealD alpha = r2 / pAp;
        axpyF(x, +alpha, pv);
        axpyF(r, -alpha, Ap);
        RealD rn = norm2F(r);
        if (rn < tol2) break;
        RealD beta = rn / r2;
        for (int a = 0; a < DtxqcdNf; ++a) {
          pv.upper.f[a] = r.upper.f[a] + beta * pv.upper.f[a];
          pv.lower.f[a] = r.lower.f[a] + beta * pv.lower.f[a];
        }
        r2 = rn;
      }
      const RealD relres = std::sqrt(norm2F(r) / b2);
      if (relres > tol) {
        std::cout << GridLogMessage << "[pdef-CG] NOT CONVERGED at CG_MAX="
                  << cgmax << " (relres " << relres << ") -- aborting: an "
                  << "unconverged solve BIASES the trace" << std::endl;
        Grid_finalize();
        std::_Exit(3);
      }
      last_it = k + 1;
      dMp(x, t1);
      const ComplexD est = 0.5 * innerF(xi, t1) / noise_var;
      acc += est;
      acc2_re += est.real() * est.real();
      acc2_im += est.imag() * est.imag();
    }
    const ComplexD mean = acc / (RealD)n_noise;
    auto serr = [&](RealD a2, RealD m) {
      return std::sqrt(std::max(a2 / n_noise - m * m, 0.0) /
                       std::max(n_noise - 1, 1));
    };
    const RealD err_re = serr(acc2_re, mean.real());
    const RealD err_im = serr(acc2_im, mean.imag());
    std::cout << GridLogMessage << "PDEF-NODE q=" << q << " e=" << e
              << " (1/2)Tr[M^-1 dM/de] = " << mean.real() << " +- " << err_re
              << "  im " << mean.imag() << " +- " << err_im
              << "  iters=" << last_it << std::endl;
    dlogpf += eta * gl_w[q] * mean;
    dlogpf_var_re += (eta * gl_w[q] * err_re) * (eta * gl_w[q] * err_re);
    dlogpf_var_im += (eta * gl_w[q] * err_im) * (eta * gl_w[q] * err_im);
  }
  const RealD dlogpf_err_re = std::sqrt(dlogpf_var_re);
  const RealD dlogpf_err_im = std::sqrt(dlogpf_var_im);

  std::cout << GridLogMessage << "PDEF-RESULT eta=" << eta
            << " dlogPf re = " << dlogpf.real() << " +- " << dlogpf_err_re
            << "  im (phase) = " << dlogpf.imag() << " +- " << dlogpf_err_im
            << "  dS_gauss(exact) = " << dSg
            << "  logJ(const) = " << logJ << std::endl;
  std::cout << GridLogMessage << "PDEF-RESULT log|w| (config-dependent) = "
            << (-dSg + dlogpf.real()) << " +- " << dlogpf_err_re
            << "   arg w = " << dlogpf.imag() << " +- " << dlogpf_err_im
            << std::endl;

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
