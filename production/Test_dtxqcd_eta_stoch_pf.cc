// Test_dtxqcd_eta_stoch_pf: STOCHASTIC estimator of the eta-deformation
// reweighting factor  Delta log det M48(eta)  -- the production-scalable
// replacement for the dense LU of Test_dtxqcd_eta_deform_pf.
//
//   Delta log det(eta) = int_0^eta deta' Tr[ M(eta')^{-1} dM(eta')/deta' ]
//
// with the deformation d,n -> e^{-eta p(t)} (UR), d*,n* -> e^{+eta p(t)} (LL),
// p(t) the profile (tent: min(t, T-t); linear: t).  dM/deta is SITE-LOCAL
// (the -/+ p(t)-weighted scaled cross blocks), so the trace is Hutchinson-
// estimable with gaussian noise + CG solves of the DEFORMED operator:
//
//   M_def = M + C_UR[w-] + C_LL[w+],   w-+ = e^{-+eta p(t)} - 1
//   M_def^dag = M^dag + C_UR-slot[w+] + C_LL-slot[w-]     (weights swap;
//     complex symmetry d^T = d gives d^dag = d*, so the adjoint of each
//     cross branch is the OTHER branch with the same real weight)
//
// realized entirely with the exposed DtxqcdApplyDnCross (coefficient +1 in
// AddDiagAndCrossAndClover; the apply_conj branch conjugates internally and
// real weights pass through conj()).
//
// Quadrature: Gauss-Legendre (N_GL nodes) in eta' -- the integrand is
// analytic.  Noise: Grid gaussian (E<v,Av> = 2 Tr A -> divide by 2),
// N_NOISE vectors per node.  Output: per-node complex traces + the
// integrated d log|det| (Re) and d arg (Im) with jackknife-style noise
// errors.  Validate against the dense ETA_SCAN values on 4^4 / 4^3x8;
// then any geometry (8^4, production 16^3x48) is reachable.
//
// Env: LATT (4.4.4.4), CFG_DIR, TRAJ (250), ETA (0.3), PROFILE (tent|linear),
//      N_NOISE (16), N_GL (8), MASS (-0.245), CSW (1.24930970916466),
//      STOUT_NSMEAR (1), STOUT_RHO (0.125), CG_TOL (1e-8), CG_MAX (20000),
//      NOISE_SEED (77).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#endif
#include "dtxqcd_deform_quda.h"
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
#include "dtxqcd_deform_mg.h"
#endif

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

  RealD mass = env_real("MASS", -0.245);
  RealD csw  = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho  = env_real("STOUT_RHO", 0.125);
  RealD eta  = env_real("ETA", 0.3);
  // ETA_LO: quadrature over the SEGMENT [eta_lo, eta] instead of [0, eta].
  // The eta-target family {0.2, 0.4, 0.6} shares nested segments
  // [0,.2][.2,.4][.4,.6]: each target's DlogPf = the sum of its segments,
  // computed once each -- and the affordability ledger is per-segment
  // (the eta=0.6 premium is ONLY its top segment).
  RealD eta_lo = env_real("ETA_LO", 0.0);
  GRID_ASSERT(eta_lo >= 0.0 && eta_lo < eta);
  int n_noise = env_int("N_NOISE", 16);
  int n_gl    = env_int("N_GL", 8);
  // Sampling-solve tolerance.  Default 1e-4 since the 2026-07-18 ladder
  // gate (job 1302547): means shift 1.5e-4 across 1e-8..1e-4 vs +-1.25
  // statistical -- solve error enters a Hutchinson sample as <w, eps_x>,
  // noise not bias.  x1.8 measured.  Tighten explicitly for non-sampling
  // uses.
  RealD tol   = env_real("CG_TOL", 1e-4);
  int cgmax   = env_int("CG_MAX", 20000);
  int traj    = env_int("TRAJ", 250);
  // PROFILE=tent|linear|tent_rounded|tent_rounded2|cos — the same dispatch
  // (and the same lambda) as meas_baryon_deform_dtxqcd, so the ΔlogPf
  // computed here is on EXACTLY the contour the propagators were solved on.
  const std::string prof_kind = [&] {
    const char *v = std::getenv("PROFILE");
    return v ? std::string(v) : std::string("tent");
  }();
  const bool tent = prof_kind != "linear";
  const bool prof_rounded = prof_kind == "tent_rounded" ||
                            prof_kind == "tent_rounded2";
  const bool prof_round_src = prof_kind == "tent_rounded2";
  const bool prof_cos = prof_kind == "cos";
  const double prof_w = env_real("PROFILE_W", 4.0);
  const double prof_w_src = env_real("PROFILE_W_SRC", prof_w);
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam4.0000_4t4_signhop_lam4";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  std::cout << GridLogMessage << "==== Test_dtxqcd_eta_stoch_pf ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " eta=" << eta << " profile=" << prof_kind
            << " N_NOISE=" << n_noise << " N_GL=" << n_gl << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  // Independent noise seed (decoupled from the checkpoint RNG state).
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

  // ---- fast deformed-solve stack (mirrors meas_baryon_deform_dtxqcd) ----
  // DTXQCD_DEFORM_QUDA=1 (default): quadrature solves go through the
  // DtxqcdDeformQuda M~ (StyleC hop corrections; EO Schur when
  // DTXQCD_DEFORM_EO, default lam >= 5).  Requires CHEMPOT_FRAME=1 (default)
  // — the device M~ IS the chemical-potential-frame operator, so a
  // CHEMPOT_FRAME=0 run falls back to the host CGNR reference path.
  const RealD lam = env_real("LAMBDA_DTXQCD", 4.0);
  const bool use_qdef = env_int("DTXQCD_DEFORM_QUDA", 1) != 0;
  const bool use_eo_def =
      env_int("DTXQCD_DEFORM_EO", (lam >= 5.0) ? 1 : 0) != 0;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<DTXQCDMpcOpQUDA> Mq;
#endif
  std::unique_ptr<DtxqcdPackedFullHermOp> Hop;
  std::unique_ptr<DtxqcdDeformQuda> Qdef;
  if (use_qdef) {
#ifdef GRID_HAVE_QUDA
    Grid::Quda::initialize();  // idempotent
    if (!use_eo_def) {
      Mq.reset(new DTXQCDMpcOpQUDA(Usm, Grid, RBGrid, mass, csw,
                                   U.sigma, U.pi, U.d, U.n, U.s, U.p));
      Mq->SetAuxStatic(true);
      Hop.reset(new DtxqcdPackedFullHermOp(Mop, *Mq, &Grid, Even));
    } else {
      Hop.reset(new DtxqcdPackedFullHermOp(Mop, &Grid, Even));
    }
#else
    Hop.reset(new DtxqcdPackedFullHermOp(Mop, &Grid, Even));
#endif
    Qdef.reset(new DtxqcdDeformQuda(*Hop, Grid, Usm, T));
    std::cout << GridLogMessage << "[stochpf] backend=QUDA-hopcorr"
#ifndef GRID_HAVE_QUDA
              << " (non-QUDA build: struct-path reference fallback)"
#endif
              << std::endl;
  }
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<DTXQCDMpcOpQUDA_EO> Mq_eo;
  std::unique_ptr<DTXQCDMpcOp> Mpc_cpu;
  std::unique_ptr<DtxqcdPackedMpcHermOp> HopEO;
  if (use_eo_def && Qdef) {
    Mq_eo.reset(new DTXQCDMpcOpQUDA_EO(Dw, Usm, Grid, RBGrid, mass, csw,
                                       U.sigma, U.pi, U.d, U.n, U.s, U.p));
    Mpc_cpu.reset(new DTXQCDMpcOp(Dw));
    HopEO.reset(new DtxqcdPackedMpcHermOp(*Mpc_cpu, *Mq_eo, &RBGrid, Odd));
    Qdef->AttachEO(*HopEO, *Mq_eo, RBGrid, Dw);
    std::cout << GridLogMessage << "[stochpf] EO Schur solve ENABLED"
              << std::endl;
  }
#endif

  // t-profile weight applied to a d/n-type aux field: out = f(t) * field.
  // PROF_T0 (default 0): profile offset -- p(t) = tent/linear of (t - t0)
  // mod T.  Matches meas_baryon_deform_dtxqcd's source-centered contours:
  // translation averaging there needs one ΔlogPf per offset, computed here.
  // NON-const: the RHOT0 instrument switches the contour offset mid-run
  // (all weight builders + the prof lambda capture it by reference).
  int prof_t0 = env_int("PROF_T0", 0);
  // CONTINUOUS offset shift (TSRC-ratio quadrature): the profile argument
  // becomes t - prof_t0 - prof_shift with prof_shift real.  0 = the
  // integer-offset behavior, bit-identical.
  double prof_shift = 0.0;
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  auto prof = [&](int t) {
    double d = std::fmod((double)(t - prof_t0) - prof_shift, (double)T);
    if (d < 0) d += (double)T;
    if (!tent) return d;
    if (prof_cos)
      return 0.25 * (double)T * (1.0 - std::cos(2.0 * M_PI * d / (double)T));
    double v = std::min(d, (double)T - d);
    if (!prof_rounded) return v;
    const double a = std::abs(d - 0.5 * T);
    if (a < prof_w)
      v = 0.5 * T - 0.5 * prof_w - a * a / (2.0 * prof_w);
    if (prof_round_src && prof_w_src > 0) {
      const double b = std::min(d, (double)T - d);
      if (b < prof_w_src)
        v = 0.5 * prof_w_src + b * b / (2.0 * prof_w_src);
    }
    return v;
  };
  // p'(t - t0 - shift) for the cos profile (the TSRC-ratio insertion).
  auto profp = [&](int t) {
    double d = std::fmod((double)(t - prof_t0) - prof_shift, (double)T);
    if (d < 0) d += (double)T;
    return 0.5 * M_PI * std::sin(2.0 * M_PI * d / (double)T);
  };
  auto WeightField = [&](const auto &f, auto &out, auto fac) {
    out = f;
    for (int t = 0; t < T; ++t) {
      auto scaled = f;
      scaled = ComplexD(fac(t), 0.0) * scaled;
      out = where(tc == Integer(t), scaled, out);
    }
  };

  // Scaled copies of d, n for the deformed operator and its eta-derivative,
  // rebuilt per quadrature node.
  LatticeDtxqcdD d_wm(&Grid), d_wp(&Grid), d_dm(&Grid), d_dp(&Grid);
  LatticeDtxqcdN n_wm(&Grid), n_wp(&Grid), n_dm(&Grid), n_dp(&Grid);

  auto BuildWeights = [&](double e) {
    WeightField(U.d, d_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.n, n_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.d, d_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
    WeightField(U.n, n_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
    WeightField(U.d, d_dm, [&](int t) { return -prof(t) * std::exp(-e * prof(t)); });
    WeightField(U.n, n_dm, [&](int t) { return -prof(t) * std::exp(-e * prof(t)); });
    WeightField(U.d, d_dp, [&](int t) { return +prof(t) * std::exp(+e * prof(t)); });
    WeightField(U.n, n_dp, [&](int t) { return +prof(t) * std::exp(+e * prof(t)); });
  };

  typedef DTXQCDFermionDoubled F;
  auto AddCross = [&](const LatticeDtxqcdD &dd, const LatticeDtxqcdN &nn,
                      const F &in, F &out, bool lower_to_upper) {
    DTXQCDFermionNf c(&Grid);
    if (lower_to_upper) {          // UR slot: in.lower -> out.upper
      DtxqcdApplyDnCross(dd, nn, in.lower, c, /*apply_conj=*/false);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.upper.f[a] = out.upper.f[a] + c.f[a];
    } else {                       // LL slot: in.upper -> out.lower (conj'd)
      DtxqcdApplyDnCross(dd, nn, in.upper, c, /*apply_conj=*/true);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.lower.f[a] = out.lower.f[a] + c.f[a];
    }
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    AddCross(d_wm, n_wm, in, out, true);    // UR weighted by e^{-} - 1
    AddCross(d_wp, n_wp, in, out, false);   // LL weighted by e^{+} - 1
  };
  auto MdefDag = [&](const F &in, F &out) {
    Mop.Mdag(in, out);
    AddCross(d_wp, n_wp, in, out, true);    // weights SWAP under dagger
    AddCross(d_wm, n_wm, in, out, false);
  };
  auto dM = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = Zero();
      out.lower.f[a] = Zero();
    }
    AddCross(d_dm, n_dm, in, out, true);
    AddCross(d_dp, n_dp, in, out, false);
  };

  // CHEMPOT_FRAME=1 (default): work with M~ = G^{-1} M_def G, the
  // chemical-potential representation (G = diag(e^{-eta p(t)/2} upper,
  // e^{+eta p(t)/2} lower), site-diagonal).  M~ has UNDEFORMED aux blocks and
  // only e^{+-eta/2}-twisted Wilson time hops -> near-undeformed CG
  // conditioning even when e^{eta p} reaches e^{4.8} (production tent), where
  // the aux-frame M_def is pathologically ill-conditioned.  The trace is
  // frame-invariant: Tr[M~^{-1} G^{-1} dM G] = Tr[M_def^{-1} dM]; the
  // G-sandwich of dM also cancels its e^{+-eta p} weight blowup (leaving ~p).
  // Implemented as scale-in / validated-M_def / scale-out sandwiches.
  const bool chempot = env_int("CHEMPOT_FRAME", 1) != 0;
  // G weight fields, PRECOMPUTED once per GL node (BuildGWeights): GApply is
  // called 4x per CG iteration, so it must be two multiplies, not a 48-slice
  // where() loop (the first production smoke died on exactly that churn).
  LatticeComplex wGu(&Grid), wGl(&Grid), wGui(&Grid), wGli(&Grid);
  auto BuildGWeights = [&](double e) {
    wGu = ComplexD(1.0, 0.0);
    wGl = ComplexD(1.0, 0.0);
    LatticeComplex one(&Grid);
    for (int t = 0; t < T; ++t) {
      one = ComplexD(std::exp(-e * prof(t) / 2.0), 0.0);
      wGu = where(tc == Integer(t), one, wGu);
      one = ComplexD(std::exp(+e * prof(t) / 2.0), 0.0);
      wGl = where(tc == Integer(t), one, wGl);
    }
    wGui = ComplexD(1.0, 0.0);
    wGli = ComplexD(1.0, 0.0);
    wGui = wGui / wGu;    // pointwise reciprocals: G^{-1}
    wGli = wGli / wGl;
  };
  auto GApply = [&](F &v, int sign) {
    // sign=+1: apply G; sign=-1: apply G^{-1}.
    const LatticeComplex &fu = (sign > 0) ? wGu : wGui;
    const LatticeComplex &fl = (sign > 0) ? wGl : wGli;
    for (int a = 0; a < DtxqcdNf; ++a) {
      v.upper.f[a] = fu * v.upper.f[a];
      v.lower.f[a] = fl * v.lower.f[a];
    }
  };
  auto Mtil = [&](const F &in, F &out) {
    if (!chempot) { Mdef(in, out); return; }
    F w(&Grid); w = in;
    GApply(w, +1);
    Mdef(w, out);
    GApply(out, -1);
  };
  auto MtilDag = [&](const F &in, F &out) {
    if (!chempot) { MdefDag(in, out); return; }
    F w(&Grid); w = in;
    GApply(w, -1);
    MdefDag(w, out);
    GApply(out, +1);
  };
  auto dMtil_aux = [&](const F &in, F &out) {
    if (!chempot) { dM(in, out); return; }
    F w(&Grid); w = in;
    GApply(w, +1);
    dM(w, out);
    GApply(out, -1);
  };

  // HOP-FORM trace insertion dM~/deta (INSERTION=hop, default): the exact
  // eta-derivative of M~ in the chemical-potential frame, supported ONLY on
  // the Wilson TIME HOPS with weights ~ dp/2 = O(1/2) -- vs the aux-frame
  // insertion's profile weights p(t) <= T/2, whose square is the variance.
  // (The two insertions differ by the traceless commutator [M~, Lambda], so
  // the TRACE is identical; the hop form is the low-variance representation.)
  //
  //   dM~.upper(x) = af_u(t) (-1/2)(1-g_t) W3(x) psi(x+t)
  //                + ab_u(t) (-1/2)(1+g_t) W7(x) psi(x-t)     (lower: W3c/W7c)
  //   af_u(t) = (-dp(t)/2)   e^{-eta dp(t)/2},   dp(t) = p(t+1 mod T) - p(t)
  //   ab_u(t) = (+dp(t-1)/2) e^{+eta dp(t-1)/2}          (lower: signs flip)
  // W3 = APBC-phase-applied forward t-link, W7(x) = W3(x-t)^dag -- the same
  // objects Grid's DoubleStore puts in slots 3 and 7.  Block-diagonal (no
  // cross terms).  All spin/sign conventions are certified by DERIVCHECK
  // against the finite difference of the exact G-sandwich M~.
  const bool hop_insert = [&] {
    const char *v = std::getenv("INSERTION");
    return !v || std::string(v) != "aux";
  }();
  Gamma g_t(Gamma::Algebra::GammaT);
  LatticeColourMatrix W3(&Grid), W7(&Grid), W3c(&Grid), W7c(&Grid);
  {
    LatticeColourMatrix Ut(&Grid);
    Ut = PeekIndex<LorentzIndex>(Usm, Nd - 1);
    LatticeColourMatrix mUt(&Grid);
    mUt = ComplexD(-1.0, 0.0) * Ut;                    // APBC phase on wrap
    W3 = where(tc == Integer(T - 1), mUt, Ut);
    W7 = Cshift(LatticeColourMatrix(adj(W3)), Nd - 1, -1);
    W3c = conjugate(W3);                               // lower block gauge
    W7c = Cshift(LatticeColourMatrix(adj(W3c)), Nd - 1, -1);
  }
  LatticeComplex af_u(&Grid), ab_u(&Grid), af_l(&Grid), ab_l(&Grid);
  auto BuildHopWeights = [&](double e) {
    auto dp = [&](int t) { return prof((t + 1) % T) - prof(t); };
    LatticeComplex one(&Grid);
    for (int t = 0; t < T; ++t) {
      const double dpf = dp(t);
      const double dpb = dp((t - 1 + T) % T);
      one = ComplexD(-0.5 * dpf * std::exp(-e * dpf / 2.0), 0.0);
      af_u = where(tc == Integer(t), one, af_u);
      one = ComplexD(+0.5 * dpb * std::exp(+e * dpb / 2.0), 0.0);
      ab_u = where(tc == Integer(t), one, ab_u);
      one = ComplexD(+0.5 * dpf * std::exp(+e * dpf / 2.0), 0.0);
      af_l = where(tc == Integer(t), one, af_l);
      one = ComplexD(-0.5 * dpb * std::exp(-e * dpb / 2.0), 0.0);
      ab_l = where(tc == Integer(t), one, ab_l);
    }
  };
  // dM~/ds coefficients (contour-OFFSET derivative at fixed eta): the hop
  // weight w_f(t) = e^{-e dp(t)/2} depends on the offset s only through
  // dp, and d(dp)/ds = -[p'(t+1-s) - p'(t-s)] = -dpp, so
  //   d w_f/ds = +(e/2) dpp w_f,   d w_b/ds = -(e/2) dpp(t-1) w_b,
  // lower block reciprocal.  Fills the SAME af/ab fields, so dMhop (and
  // through it dStil) becomes the d/ds insertion -- coefficients are
  // O(e * p'') vs the eta-insertion's O(dp/2): parametrically small.
  auto BuildHopWeightsS = [&](double e) {
    auto dp = [&](int t) { return prof((t + 1) % T) - prof(t); };
    auto dpp = [&](int t) { return profp((t + 1) % T) - profp(t); };
    LatticeComplex one(&Grid);
    for (int t = 0; t < T; ++t) {
      const double dpf = dp(t), dppf = dpp(t);
      const double dpb = dp((t - 1 + T) % T), dppb = dpp((t - 1 + T) % T);
      one = ComplexD(+0.5 * e * dppf * std::exp(-e * dpf / 2.0), 0.0);
      af_u = where(tc == Integer(t), one, af_u);
      one = ComplexD(-0.5 * e * dppb * std::exp(+e * dpb / 2.0), 0.0);
      ab_u = where(tc == Integer(t), one, ab_u);
      one = ComplexD(-0.5 * e * dppf * std::exp(+e * dpf / 2.0), 0.0);
      af_l = where(tc == Integer(t), one, af_l);
      one = ComplexD(+0.5 * e * dppb * std::exp(-e * dpb / 2.0), 0.0);
      ab_l = where(tc == Integer(t), one, ab_l);
    }
  };
  auto dMhopBlock = [&](const LatticeFermion &psi, LatticeFermion &out,
                        const LatticeColourMatrix &Wf,
                        const LatticeColourMatrix &Wb,
                        const LatticeComplex &af, const LatticeComplex &ab) {
    LatticeFermion pp = Cshift(psi, Nd - 1, +1);
    LatticeFermion tf = Wf * pp;
    tf = ComplexD(-0.5, 0.0) * (tf - g_t * tf);        // -1/2 (1 - g_t)
    LatticeFermion pm = Cshift(psi, Nd - 1, -1);
    LatticeFermion tb = Wb * pm;
    tb = ComplexD(-0.5, 0.0) * (tb + g_t * tb);        // -1/2 (1 + g_t)
    out = af * tf + ab * tb;
  };
  auto dMhop = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      dMhopBlock(in.upper.f[a], out.upper.f[a], W3, W7, af_u, ab_u);
      dMhopBlock(in.lower.f[a], out.lower.f[a], W3c, W7c, af_l, ab_l);
    }
  };
  auto dMtil = [&](const F &in, F &out) {
    if (hop_insert) dMhop(in, out);
    else            dMtil_aux(in, out);
  };

  // ---- SCHUR-form estimator (DTXQCD_STOCHPF_SCHUR) ----------------------
  // Tr[M~^-1 dM~] = Tr[S~^-1 dS~] exactly (Mee is eta-independent), with
  // S~ = Moo - M~oe Mee^-1 M~eo the odd Schur complement and
  // dS~ = -(dM_oe Mee^-1 M~eo + M~oe Mee^-1 dM_eo)  [hop insertion only:
  // dM_oo = dM_ee = 0 -- the aux-form insertion has site-local cross
  // blocks and is NOT supported here].  Validated dense at 4^4 vs
  // FD[log det S~] to 1e-7 (Test_dtxqcd_rwf_dense_defl, job 1302016).
  //
  // DTXQCD_STOCHPF_DEFLATE=K adds the OBLIQUE-PROJECTOR split: with any
  // trial basis V (K lowest eigenpairs of Mpc~^dag Mpc~ from the deform
  // deflation stack, loose tol is FINE), P = S~V G^-1 (S~V)^dag with
  // G = (S~V)^dag (S~V):
  //     Tr[S~^-1 dS~] = Tr[W G^-1] + E_w <w, S~^-1 (1-P) dS~ w>,
  //     W_jk = (S~ v_j)^dag dS~ v_k
  // -- the split is EXACT for ANY V (no triplet-tolerance bias: the naive
  // sigma_k^-1 u^dag dS v form amplifies eigenpair error ~850x through
  // the exact/remainder cancellation measured at 4^4).
  // DTXQCD_STOCHPF_DILUTE_EOT=1 splits the remainder noise by t-parity
  // (the T-independent version of the 4^4 t-slice dilution win: the hop
  // insertion connects t<->t+-1, so same-t-parity blocks carry no direct
  // insertion term).
  const int defl_nev = env_int("DTXQCD_STOCHPF_DEFLATE", 0);
  const int dilute_eot = env_int("DTXQCD_STOCHPF_DILUTE_EOT", 0);
  // DTXQCD_STOCHPF_DILUTE_TBLOCK=B: contiguous t-block dilution, D = T/B
  // blocks of B slices (0 = off; overrides EOT).  The scan over B maps the
  // integrated Dt-decay of A = S~^-1 dS~ at production volume -- the 4^4
  // t-slice win (x1.78 equal-budget) came from T ~ the S~^-1 correlation
  // length, so blocks of ~xi slices are the T=48 candidate.
  const int dilute_tb = env_int("DTXQCD_STOCHPF_DILUTE_TBLOCK", 0);
  // DTXQCD_STOCHPF_DILUTE_TSTRIDE=k: interleaved comb dilution, block r =
  // {t : t mod k == r} (k=2 == EOT).  The hop insertion concentrates
  // A's off-diagonals at |Dt|=1, so a k-comb removes all cross terms up
  // to |Dt|=k-1 at D=k cost.
  // Default 4: comb-4 is the measured universal optimum across the
  // production node range (F-map ladder 2026-07-18: equal-budget ratio
  // 0.68/0.70/0.74/0.82 at ep=0.05/0.16/0.32/0.40; the hop insertion's
  // F(Dt) peaks at Dt=1 with xi~2-3).  =0 restores the undiluted
  // estimator (with DTXQCD_STOCHPF_SCHUR=0 for the full legacy path).
  const int dilute_ts = env_int("DTXQCD_STOCHPF_DILUTE_TSTRIDE", 4);
  // DTXQCD_STOCHPF_DILUTE_ZPAR=1: z-parity dilution (D=2) -- the spatial
  // control arm (structural prediction: neutral, like spin/color).
  const int dilute_zp = env_int("DTXQCD_STOCHPF_DILUTE_ZPAR", 0);
  // DTXQCD_STOCHPF_TSLICE=1: log per-timeslice partial estimates s_t
  // (TSLICE lines) each sample.  CAUTION (2026-07-18): with a single
  // probe the conjugated covariance of s_t is Wick-blind to cross-t
  // Frobenius mass (contractions force t=t') -- mode 1 does NOT predict
  // dilution schemes.  Kept for the per-sample decomposition only.
  // DTXQCD_STOCHPF_TSLICE=2: the F-MAP instrument -- per "sample", one
  // probe pair (z, z'); z' is solved once per masked t' slice (T solves)
  // and FMAP lines record |slice_t <z, A z'|_t'>|^2-ready partials: the
  // sample-mean of |f(t,t')|^2 is an unbiased estimate of
  // F(t,t') = sum_{i in t, j in t'} |A_ij|^2, the exact object every
  // t-dilution scheme's variance is a partial sum of.
  const int tslice_log = env_int("DTXQCD_STOCHPF_TSLICE", 0);
  // DTXQCD_STOCHPF_RHOT0=<t0B>: the DIFFERENCE-TELESCOPE correlation
  // instrument.  Per sample, the SAME noise w is pushed through the
  // estimator at contour offsets t0A (=PROF_T0) and t0B; RHOPAIR lines
  // record both estimates.  Offline corr(estA, estB) prices the
  // common-noise difference estimator DlogPf(t0B) - DlogPf(t0A):
  // Var(diff) = (1-rho)-suppressed => one anchor offset + 47 cheap
  // differences instead of 48 full estimates, IF rho is high.
  // Runs with DEFLATE=0 (the basis is t0-dependent; rho of the plain
  // Schur estimator is the relevant number).  -1 = off.
  const int rho_t0b = env_int("DTXQCD_STOCHPF_RHOT0", -1);
  const int use_anti = env_int("DTXQCD_STOCHPF_ANTI", 1);
  const int schur_mode = env_int("DTXQCD_STOCHPF_SCHUR",
                                 (defl_nev > 0 || dilute_eot || dilute_tb ||
                                  dilute_ts || dilute_zp || rho_t0b >= 0)
                                     ? 1 : 0);
  if (rho_t0b >= 0)
    GRID_ASSERT(defl_nev == 0 &&
                "RHOT0 instrument runs undeflated (basis is t0-dependent)");
  if (dilute_tb > 0) GRID_ASSERT(T % dilute_tb == 0);
  if (dilute_ts > 0) GRID_ASSERT(T % dilute_ts == 0);
  GRID_ASSERT((dilute_tb > 0) + (dilute_ts > 0) + (dilute_eot != 0) +
                  (dilute_zp != 0) <= 1 &&
              "one dilution scheme at a time");
  if (schur_mode) {
    GRID_ASSERT(hop_insert && chempot && Qdef &&
                "SCHUR estimator needs INSERTION=hop, CHEMPOT_FRAME=1 and "
                "the Qdef fast path");
#ifdef GRID_HAVE_QUDA
    GRID_ASSERT(Qdef->HasEO() && "SCHUR estimator needs the EO attach");
#endif
  }
  auto MaskKeepCb = [&](F &v, int cb) {
    LatticeFermion h(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(cb, h, v.upper.f[a]);
      v.upper.f[a] = Zero();
      setCheckerboard(v.upper.f[a], h);
      pickCheckerboard(cb, h, v.lower.f[a]);
      v.lower.f[a] = Zero();
      setCheckerboard(v.lower.f[a], h);
    }
  };
  auto MeeInvEven = [&](const F &fin, F &fout) {
    F he(&RBGrid), hi(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(Even, he.upper.f[a], fin.upper.f[a]);
      pickCheckerboard(Even, he.lower.f[a], fin.lower.f[a]);
    }
    Dw.MooeeInv(he, hi);
    for (int a = 0; a < DtxqcdNf; ++a) {
      fout.upper.f[a] = Zero();
      fout.lower.f[a] = Zero();
      setCheckerboard(fout.upper.f[a], hi.upper.f[a]);
      setCheckerboard(fout.lower.f[a], hi.lower.f[a]);
    }
  };
  // dS~ on a full-grid odd-supported field (result odd-supported).
  auto dStil = [&](const F &w, F &out) {
    F t1(&Grid), t2(&Grid), t3(&Grid);
    Mtil(w, t1);
    MeeInvEven(t1, t2);
    dMhop(t2, t1);
    MaskKeepCb(t1, Odd);
    dMhop(w, t2);
    MeeInvEven(t2, t3);
    Mtil(t3, t2);
    MaskKeepCb(t2, Odd);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = ComplexD(-1.0, 0.0) * (t1.upper.f[a] + t2.upper.f[a]);
      out.lower.f[a] = ComplexD(-1.0, 0.0) * (t1.lower.f[a] + t2.lower.f[a]);
    }
  };
  // S~^-1 on a full-grid odd-supported field: SolveTilEO with b_e = 0 has
  // a degenerate Schur prep, so the odd part of the solution IS
  // Mpc~^-1 b_o.  Unconverged solves abort (a93707d3: silent CG_MAX bias).
  auto SchurSolve = [&](const F &b, F &x) {
    RealD relres = 0;
    const int it = Qdef->SolveTilEO(b, x, tol, cgmax, relres);
    std::cout << GridLogMessage << "[stochpf-CG:schur] iters=" << it
              << " relres=" << relres << std::endl;
    if (relres > tol) {
      std::cout << GridLogMessage
                << "[stochpf] NOT CONVERGED -- aborting (biased trace)"
                << std::endl;
      std::fflush(nullptr);
      std::_Exit(3);
    }
    MaskKeepCb(x, Odd);
  };

  // Custom CG on Mdef^dag Mdef (pattern of dtxqcd_diag.h compute_trminv).
  // Fast path: x = M~^{-1} src via the device Qdef (Schur EO when attached)
  // — identical solution to the CGNR below, ~10-25x the throughput.  The
  // caller must have synced Qdef's profile to the current node (SyncQdef).
  auto SolveDef = [&](const F &src, F &x) {
    if (Qdef && chempot) {
      RealD relres = 0;
      int it;
#ifdef GRID_HAVE_QUDA
      if (use_eo_def && Qdef->HasEO())
        it = Qdef->SolveTilEO(src, x, tol, cgmax, relres);
      else
#endif
        it = Qdef->SolveTil(src, x, tol, cgmax, relres);
      std::cout << GridLogMessage << "[stochpf-CG:quda] iters=" << it
                << " relres=" << relres << std::endl;
      if (relres > tol) {
        std::cout << GridLogMessage << "[stochpf-CG:quda] NOT CONVERGED at "
                  << "CG_MAX=" << cgmax << " (relres " << relres << " > tol "
                  << tol << ") -- aborting: an unconverged solve BIASES the "
                  << "trace estimate" << std::endl;
        Grid_finalize();
        std::_Exit(3);
      }
      return;
    }
    F b(&Grid), r(&Grid), p(&Grid), t1(&Grid), Ap(&Grid);
    MtilDag(src, b);
    x = Zero();
    r = b; p = b;
    RealD r2 = norm2(r), b2 = std::max(norm2(b), 1e-300);
    RealD tol2 = tol * tol * b2;
    int k = 0;
    for (; k < cgmax; ++k) {
      Mtil(p, t1);
      MtilDag(t1, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD rn = norm2(r);
      if (rn < tol2) break;
      RealD beta = rn / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      r2 = rn;
    }
    const RealD relres_h = std::sqrt(norm2(r) / b2);
    if (relres_h > tol) {
      std::cout << GridLogMessage << "[stochpf-CG] NOT CONVERGED at CG_MAX="
                << cgmax << " (relres " << relres_h << " > tol " << tol
                << ") -- aborting: an unconverged solve BIASES the trace "
                << "estimate" << std::endl;
      Grid_finalize();
      std::_Exit(3);
    }
  };

  // OPCHECK=1 (4^4-class only): machine-precision validation of the lattice
  // deformed operator against the INDEPENDENT dense implementation --
  // assemble dense M48 columnwise from the UNDEFORMED Mop, sector-scale it
  // (the construction already validated by the det/gauge tests), and compare
  // H_def . v against Mdef(v) and H_def^dag . v against MdefDag(v) on random
  // vectors.  Proves the weighted-cross corrections exactly, no statistics.
  if (env_int("OPCHECK", 0) != 0) {
    const int V = (int)Grid.gSites();
    GRID_ASSERT(Grid.gSites() == Grid.lSites() && V <= 4096);
    const int dim = 48 * V;
    auto SiteCoor = [&](int lex, Coordinate &c) {
      const Coordinate ld = Grid.GlobalDimensions();
      c.resize(4);
      c[0] = lex % ld[0]; lex /= ld[0];
      c[1] = lex % ld[1]; lex /= ld[1];
      c[2] = lex % ld[2]; lex /= ld[2];
      c[3] = lex;
    };
    typedef typename LatticeFermion::scalar_object FSobj;
    std::vector<std::complex<double>> H((size_t)dim * dim);
    {
      DTXQCDFermionDoubled e(&Grid), y(&Grid);
      std::vector<FSobj> col(V);
      for (int site = 0; site < V; ++site) {
        Coordinate c; SiteCoor(site, c);
        for (int blk = 0; blk < 2; ++blk)
          for (int a = 0; a < DtxqcdNf; ++a)
            for (int sp = 0; sp < Ns; ++sp)
              for (int co = 0; co < Nc; ++co) {
                const int j = ((site * 2 + blk) * DtxqcdNf + a) * 12 + sp * 3 + co;
                for (int aa = 0; aa < DtxqcdNf; ++aa) {
                  e.upper.f[aa] = Zero(); e.lower.f[aa] = Zero();
                }
                FSobj one; one = Zero();
                one()(sp)(co) = ComplexD(1.0, 0.0);
                if (blk == 0) pokeSite(one, e.upper.f[a], c);
                else          pokeSite(one, e.lower.f[a], c);
                Mop.M(e, y);
                for (int b2 = 0; b2 < 2; ++b2)
                  for (int a2 = 0; a2 < DtxqcdNf; ++a2) {
                    unvectorizeToLexOrdArray(col, b2 == 0 ? y.upper.f[a2]
                                                          : y.lower.f[a2]);
                    for (int s2 = 0; s2 < V; ++s2)
                      for (int sp2 = 0; sp2 < Ns; ++sp2)
                        for (int co2 = 0; co2 < Nc; ++co2) {
                          const int i =
                              ((s2 * 2 + b2) * DtxqcdNf + a2) * 12 + sp2 * 3 + co2;
                          auto z = col[s2]()(sp2)(co2);
                          H[(size_t)j * dim + i] =
                              std::complex<double>(z.real(), z.imag());
                        }
                  }
              }
      }
    }
    // Sector-scale (the tent/linear deformation) exactly as the dense test.
    for (int sj = 0; sj < V; ++sj)
      for (int bj = 0; bj < 2; ++bj)
        for (int k = 0; k < 24; ++k) {
          const int j = (sj * 2 + bj) * 24 + k;
          for (int si = 0; si < V; ++si) {
            Coordinate c; SiteCoor(si, c);
            const double t = prof(c[3]);
            for (int bi = 0; bi < 2; ++bi) {
              double f = 1.0;
              if (bi == 0 && bj == 1) f = std::exp(-eta * t);
              if (bi == 1 && bj == 0) f = std::exp(+eta * t);
              if (f == 1.0) continue;
              for (int m = 0; m < 24; ++m)
                H[(size_t)j * dim + (si * 2 + bi) * 24 + m] *= f;
            }
          }
        }
    BuildWeights(eta);
    // Random vector comparison for Mdef and MdefDag.
    auto ToVec = [&](const F &f, std::vector<std::complex<double>> &v) {
      std::vector<FSobj> col(V);
      for (int b = 0; b < 2; ++b)
        for (int a = 0; a < DtxqcdNf; ++a) {
          unvectorizeToLexOrdArray(col, b == 0 ? f.upper.f[a] : f.lower.f[a]);
          for (int s = 0; s < V; ++s)
            for (int sp = 0; sp < Ns; ++sp)
              for (int co = 0; co < Nc; ++co) {
                auto z = col[s]()(sp)(co);
                v[((s * 2 + b) * DtxqcdNf + a) * 12 + sp * 3 + co] =
                    std::complex<double>(z.real(), z.imag());
              }
        }
    };
    F vf(&Grid), yf(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, vf.upper.f[a]);
      gaussian(pRNG, vf.lower.f[a]);
    }
    std::vector<std::complex<double>> vv(dim), yl(dim), yd(dim);
    ToVec(vf, vv);
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 0) Mdef(vf, yf); else MdefDag(vf, yf);
      ToVec(yf, yl);
      // dense: yd = H.v (pass 0) or H^dag.v (pass 1); column-major H.
      for (int i = 0; i < dim; ++i) yd[i] = 0.0;
      for (int j = 0; j < dim; ++j)
        for (int i = 0; i < dim; ++i) {
          if (pass == 0) yd[i] += H[(size_t)j * dim + i] * vv[j];
          else           yd[j] += std::conj(H[(size_t)j * dim + i]) * vv[i];
        }
      double dmax = 0.0, scale = 0.0;
      for (int i = 0; i < dim; ++i) {
        dmax = std::max(dmax, std::abs(yl[i] - yd[i]));
        scale = std::max(scale, std::abs(yd[i]));
      }
      std::cout << GridLogMessage << "OPCHECK " << (pass ? "Mdag" : "M   ")
                << " eta=" << eta << " profile=" << prof_kind
                << ": max|lattice-dense| = " << dmax << "  (scale " << scale
                << ")  " << (dmax < 1e-8 * scale ? "MATCH" : "MISMATCH")
                << std::endl;
    }
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // DERIVCHECK=1: certify the hop-form insertion against the finite
  // difference of the EXACT M~ (G-sandwich of the OPCHECK-validated M_def):
  //   dMhop(v) =?= [M~(eta+d) - M~(eta-d)] v / (2d),  d = 1e-4.
  // Any spin/sign/weight convention error shows as an O(1) mismatch; a
  // correct implementation agrees to O(d^2) ~ 1e-8.
  if (env_int("DERIVCHECK", 0) != 0) {
    const double d = 1e-4;
    F v(&Grid), y1(&Grid), y2(&Grid), yd(&Grid), yh(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, v.upper.f[a]);
      gaussian(pRNG, v.lower.f[a]);
    }
    BuildWeights(eta + d); BuildGWeights(eta + d);
    Mtil(v, y1);
    BuildWeights(eta - d); BuildGWeights(eta - d);
    Mtil(v, y2);
    RealD nrm = 0.0, dev = 0.0;
    BuildHopWeights(eta);
    dMhop(v, yh);
    for (int a = 0; a < DtxqcdNf; ++a) {
      yd.upper.f[a] = (0.5 / d) * (y1.upper.f[a] - y2.upper.f[a]);
      yd.lower.f[a] = (0.5 / d) * (y1.lower.f[a] - y2.lower.f[a]);
      dev += norm2(LatticeFermion(yd.upper.f[a] - yh.upper.f[a]));
      dev += norm2(LatticeFermion(yd.lower.f[a] - yh.lower.f[a]));
      nrm += norm2(yd.upper.f[a]) + norm2(yd.lower.f[a]);
    }
    std::cout << GridLogMessage << "DERIVCHECK eta=" << eta
              << " profile=" << prof_kind
              << ": |dMhop - FD(Mtil)| / |FD| = "
              << std::sqrt(dev / nrm)
              << (std::sqrt(dev / nrm) < 1e-6 ? "   MATCH" : "   MISMATCH")
              << std::endl;
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // ---- TSRC-RATIO mode (DTXQCD_STOCHPF_TSRC_RATIO=M): all-offset
  // reweighting ratios from ONE smooth periodic quadrature.  The contour
  // offset is a CONTINUOUS parameter of the cos profile, and
  //   log[Pf(eta,t0)/Pf(eta,0)] = int_0^t0 ds Tr[S~(s)^-1 dS~/ds],
  // with dS~/ds the hop insertion built by BuildHopWeightsS (coefficients
  // O(eta p'') -- parametrically below the eta-insertion's O(dp/2)).
  // Uniform s-nodes (trapezoid = spectrally accurate for periodic smooth
  // integrands); cumulative partial sums give every offset; periodicity
  // forces the closure sum rule (integral over the full period = 0),
  // printed as TSRCCLOSURE = the live gate + Brownian-bridge corrector.
  // Runs at FIXED strength eta (env ETA is the target, not a quadrature
  // endpoint).  cos + chempot + hop + Qdef EO only; no deflation (the
  // basis is s-dependent; the insertion is small enough not to need it).
  const int tsrc_M = env_int("DTXQCD_STOCHPF_TSRC_RATIO", 0);
  if (tsrc_M > 0) {
    GRID_ASSERT(prof_cos && chempot && hop_insert && Qdef &&
                "TSRC ratio mode needs cos/chempot/hop/Qdef");
#ifdef GRID_HAVE_QUDA
    GRID_ASSERT(Qdef->HasEO());
#endif
    const double e = eta;
    if (env_int("TSRC_DERIVCHECK", 1)) {
      // operator-level FD gate in s: the hop-corrected M~'s ONLY
      // s-dependence is the hop weights, so dMhopS must equal the FD of
      // Mtil(s +- d) to O(d^2) exactly.
      const double s0 = 0.37 * (double)T, dd = 1e-3;
      F v(&Grid), y1(&Grid), y2(&Grid), yh(&Grid);
      GridParallelRNG DR(&Grid);
      DR.SeedFixedIntegers({55, 56, 57, 58});
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(DR, v.upper.f[a]);
        gaussian(DR, v.lower.f[a]);
      }
      prof_shift = s0 + dd;
      BuildWeights(e);
      BuildGWeights(e);
      Mtil(v, y1);
      prof_shift = s0 - dd;
      BuildWeights(e);
      BuildGWeights(e);
      Mtil(v, y2);
      prof_shift = s0;
      BuildWeights(e);
      BuildGWeights(e);
      BuildHopWeightsS(e);
      dMhop(v, yh);
      RealD nrm = 0, dev = 0;
      for (int a = 0; a < DtxqcdNf; ++a) {
        LatticeFermion du(&Grid), dl(&Grid);
        du = (0.5 / dd) * (y1.upper.f[a] - y2.upper.f[a]) - yh.upper.f[a];
        dl = (0.5 / dd) * (y1.lower.f[a] - y2.lower.f[a]) - yh.lower.f[a];
        dev += norm2(du) + norm2(dl);
        du = (0.5 / dd) * (y1.upper.f[a] - y2.upper.f[a]);
        dl = (0.5 / dd) * (y1.lower.f[a] - y2.lower.f[a]);
        nrm += norm2(du) + norm2(dl);
      }
      const RealD r = std::sqrt(dev / nrm);
      std::cout << GridLogMessage
                << "TSRC-DERIVCHECK |dMhopS - FD_s(Mtil)|/|FD| = " << r
                << (r < 1e-5 ? "   MATCH" : "   MISMATCH") << std::endl;
      GRID_ASSERT(r < 1e-5);
    }
    // MG for the s-node solves (the operator changes per node -> setup per
    // node; at fixed eta >= the crossover this dominates plain CG by the
    // measured margins).  Single-GPU battery config: pass
    // DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 (MG-lite L1).
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    const int tsrc_mg =
        env_int("DTXQCD_DEFORM_MG",
                (prof_cos && eta >= env_real("DTXQCD_DEFORM_MG_ETAMIN", 0.15))
                    ? 1 : 0);
    std::unique_ptr<DtxqcdDeformMG<16>> tmg;
    if (tsrc_mg && Qdef->HasEO())
      tmg.reset(new DtxqcdDeformMG<16>(*Qdef, Grid, RBGrid));
#endif
    ComplexD closure(0.0, 0.0);
    double closure_var = 0;
    for (int m = 0; m < tsrc_M; ++m) {
      prof_shift = (double)m * (double)T / (double)tsrc_M;
      BuildWeights(e);
      BuildGWeights(e);
      BuildHopWeightsS(e);
      Qdef->SetProfile(e, [&](int t) { return prof(t); });
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
      if (tmg) {
        tmg->Setup(pRNG);
        Qdef->SetOddSolver([&](const F &b2, F &x2, RealD tol2, int mx,
                               RealD &r2) {
          return tmg->Solve(b2, x2, tol2, mx, r2);
        });
      }
#endif
      ComplexD acc(0.0, 0.0);
      double acc2r = 0, acc2i = 0;
      for (int h = 0; h < n_noise; ++h) {
        F w(&Grid), y(&Grid), x(&Grid);
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(pRNG, w.upper.f[a]);
          gaussian(pRNG, w.lower.f[a]);
        }
        MaskKeepCb(w, Odd);
        dStil(w, y);   // dMhop fields hold the d/ds coefficients here
        SchurSolve(y, x);
        ComplexD est = DtxqcdDeformQuda::finner_(w, x) / 2.0;
        acc += est;
        acc2r += est.real() * est.real();
        acc2i += est.imag() * est.imag();
      }
      ComplexD gm = acc / (double)n_noise;
      const double er2 =
          std::max(0.0, acc2r / n_noise - gm.real() * gm.real()) / n_noise;
      const double ei2 =
          std::max(0.0, acc2i / n_noise - gm.imag() * gm.imag()) / n_noise;
      std::cout << GridLogMessage << "TSRCNODE m=" << m << " s=" << prof_shift
                << " Tr[Sinv dS/ds] = " << gm.real() << " +- "
                << std::sqrt(er2) << "  (im " << gm.imag() << " +- "
                << std::sqrt(ei2) << ")" << std::endl;
      const double wq = (double)T / (double)tsrc_M;
      closure += wq * gm;
      closure_var += wq * wq * er2;
    }
    std::cout << GridLogMessage << "TSRCCLOSURE eta=" << eta
              << " integral over full period = " << closure.real() << " +- "
              << std::sqrt(closure_var)
              << "  (sum rule: 0; offline bridge-corrects the cumulative)"
              << std::endl;
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // PROPER n-point Gauss-Legendre rules on [0,1] (rescaled to [0,eta]).
  // The original code TRUNCATED the 8-point rule to the first n_gl nodes
  // (weights summing to <1, nodes clustered near 0) — every N_GL<8 run
  // before 2026-07-14 integrated a fraction of [0,eta], not the integral
  // (per-node GLNODE integrand samples unaffected).  N_GL now selects a
  // genuine 1/2/4/8-point rule; the audit showed the integrand is smooth,
  // so N_GL=2-4 with the noise budget spent per node is the efficient split.
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

  // ---- DTXQCD_DEFORM_MG: the two-level deform multigrid as the odd-block
  // solver.  Default follows the measured validity domain (same policy as
  // meas_baryon_deform_dtxqcd): ON for eta >= 0.6 on SMOOTH profiles
  // (cos / tent_rounded*), OFF below, always off for sharp tents.  The
  // quadrature visits nodes ep = eta*gl_x[q] < eta, so setup only runs for
  // nodes with ep >= DTXQCD_DEFORM_MG_ETAMIN (default 0.5) — below that
  // plain EO CG is already cheap and a ~min setup per node cannot amortize
  // over 4*N_NOISE solves.  NOTE: Setup draws from pRNG, so the Hutchinson
  // noise sequence differs between MG and non-MG runs (both unbiased).
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  const bool prof_smooth = prof_cos || prof_rounded;
  const int use_mg = env_int("DTXQCD_DEFORM_MG",
                             (eta >= 0.6 && prof_smooth) ? 1 : 0);
  // MG per-node threshold: 0.15 = the measured CG/MG-lite crossover
  // (jobs 1302548/65: x1.5 at ep=0.2 rising to x5.2 at 0.47, setup
  // ~200 s amortizing to ~7 s/solve marginal at ep=0.4).
  const RealD mg_etamin = env_real("DTXQCD_DEFORM_MG_ETAMIN", 0.15);
  std::unique_ptr<DtxqcdDeformMG<16>> mg;
  if (use_mg && use_eo_def && Qdef && Qdef->HasEO()) {
    std::cout << GridLogMessage << "[stochpf] deform-MG ENABLED for nodes "
              << "ep >= " << mg_etamin << std::endl;
    mg.reset(new DtxqcdDeformMG<16>(*Qdef, Grid, RBGrid));
  }
#endif

  // NEYMAN allocation: per-node sample counts N_q ∝ w_q σ(ep_q) at fixed
  // total n_gl*n_noise (variance-optimal for a weighted sum of node
  // means).  σ(ep) = the measured per-sample table from the F-map ladder
  // (comb-4 folded in; the profile is FLAT-ish, so the gain over equal
  // split is modest but free).  DTXQCD_STOCHPF_NEYMAN=0 restores equal.
  auto sigma_ep = [](double x) {
    const double xs[4] = {0.05, 0.16, 0.32, 0.40};
    const double ys[4] = {512.0, 536.0, 621.0, 750.0};
    if (x <= xs[0]) return ys[0];
    for (int i = 0; i < 3; ++i)
      if (x <= xs[i + 1])
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]);
    return ys[3] + (ys[3] - ys[2]) * (x - xs[3]) / (xs[3] - xs[2]);
  };
  std::vector<int> neyman_nq(n_gl, n_noise);
  if (env_int("DTXQCD_STOCHPF_NEYMAN", 1)) {
    double sw = 0;
    for (int q = 0; q < n_gl; ++q)
      sw += gl_w[q] * sigma_ep(eta_lo + (eta - eta_lo) * gl_x[q]);
    for (int q = 0; q < n_gl; ++q) {
      const double frac =
          gl_w[q] * sigma_ep(eta_lo + (eta - eta_lo) * gl_x[q]) / sw;
      neyman_nq[q] = std::max(1, (int)std::lround(n_gl * n_noise * frac));
      std::cout << GridLogMessage << "[stochpf-neyman] node q=" << q
                << " N=" << neyman_nq[q] << std::endl;
    }
  }

  ComplexD dlogdet(0.0, 0.0), dlogdet_var(0.0, 0.0);
  for (int q = 0; q < n_gl; ++q) {
    const double ep = eta_lo + (eta - eta_lo) * gl_x[q];
    const int nn = neyman_nq[q];
    BuildWeights(ep);
    if (chempot) BuildGWeights(ep);
    if (hop_insert) BuildHopWeights(ep);
    // Sync the device M~ to this node's contour.  Generic SetProfile always
    // (REQUIRED for rounded/cos — SetTentProfile bakes the sharp tent; no
    // deflation cache is used here, so tent loses nothing by going generic).
    if (Qdef) Qdef->SetProfile(ep, [&](int t) { return prof(t); });
    // One-time M~ / M~^dag identity gate vs the host sandwich at the first
    // node's contour (catches convention/frame errors at any volume).
    // DTXQCD_DEFORM_QUDA_CHECK=0 skips.
    if (q == 0 && Qdef && chempot &&
        env_int("DTXQCD_DEFORM_QUDA_CHECK", 1) != 0) {
#ifdef GRID_HAVE_QUDA
      // Warm the EO device pools before Grid temporaries grow the elastic
      // MemoryManager cache (the meas_baryon OOM trap, jobs 1301152/84/96).
      if (use_eo_def && Qdef->HasEO()) {
        F w1(&RBGrid), w2(&RBGrid);
        for (int a = 0; a < DtxqcdNf; ++a) {
          w1.upper.f[a] = Zero(); w1.lower.f[a] = Zero();
          w1.upper.f[a].Checkerboard() = Odd;
          w1.lower.f[a].Checkerboard() = Odd;
        }
        Qdef->MpcTil(w1, w2);
      }
#endif
      // Own RNG: drawing from pRNG here would shift the Hutchinson noise
      // sequence vs a DTXQCD_DEFORM_QUDA=0 run and break the A/B gate.
      GridParallelRNG QR(&Grid);
      QR.SeedFixedIntegers({77, 13, 999, 4242});
      F v(&Grid), ag(&Grid), aq(&Grid), dg(&Grid), dq(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(QR, v.upper.f[a]);
        gaussian(QR, v.lower.f[a]);
      }
      Mtil(v, ag);
      Qdef->Mtil(v, aq);
      MtilDag(v, dg);
      Qdef->MtilDag(v, dq);
      RealD nm = 0, nd = 0, nmd = 0, ndd = 0;
      for (int a = 0; a < DtxqcdNf; ++a) {
        nm += norm2(LatticeFermion(ag.upper.f[a] - aq.upper.f[a])) +
              norm2(LatticeFermion(ag.lower.f[a] - aq.lower.f[a]));
        nd += norm2(ag.upper.f[a]) + norm2(ag.lower.f[a]);
        nmd += norm2(LatticeFermion(dg.upper.f[a] - dq.upper.f[a])) +
               norm2(LatticeFermion(dg.lower.f[a] - dq.lower.f[a]));
        ndd += norm2(dg.upper.f[a]) + norm2(dg.lower.f[a]);
      }
      const RealD rM = std::sqrt(nm / nd), rD = std::sqrt(nmd / ndd);
      std::cout << GridLogMessage << "[stochpf-qcheck] |M~_dev - M~_host| "
                << rM << "  dagger " << rD
                << ((rM < 1e-10 && rD < 1e-10) ? "   OK" : "   ** FAIL **")
                << std::endl;
      GRID_ASSERT(rM < 1e-10 && rD < 1e-10);
    }
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (mg) {
      if (ep >= mg_etamin) {
        mg->Setup(pRNG);  // per-node basis: the operator changes with ep
        Qdef->SetOddSolver([&](const F &b2, F &x2, RealD tol2, int mx,
                               RealD &r2) {
          return mg->Solve(b2, x2, tol2, mx, r2);
        });
      } else {
        Qdef->ClearOddSolver();
      }
    }
#endif
    ComplexD g(0.0, 0.0), t_exact(0.0, 0.0);
    double er = 0, ei = 0;
    if (schur_mode) {
      // ---- oblique-projector deflation setup at THIS node ----
      std::vector<F> SVf;   // S~ v_k, full-grid odd-supported
      Eigen::MatrixXcd Ginv;
      int K = 0;
      if (defl_nev > 0) {
#ifdef GRID_HAVE_QUDA
        const int d_ord = env_int("DTXQCD_DEFORM_EO_DEFLATE_ORD", 301);
        const RealD d_lo = env_real("DTXQCD_DEFORM_EO_DEFLATE_LO", 3e-4);
        const RealD d_tol = env_real("DTXQCD_DEFORM_EO_DEFLATE_TOL", 1e-3);
        std::string d_cache;
        if (const char *cd = std::getenv("DTXQCD_DEFORM_EO_DEFLATE_CACHE"))
          if (*cd) {
            char pre[128];
            std::snprintf(pre, sizeof(pre), "/tr%d_lam%.4f_", traj,
                          (double)lam);
            d_cache = std::string(cd) + pre +
                      DtxqcdDeformQuda::EODeflationCacheName(
                          defl_nev, d_tol, d_ord, ep, prof_t0,
                          Grid.GlobalDimensions());
          }
        if (d_cache.empty() || !Qdef->LoadEODeflation(d_cache)) {
          Qdef->BuildEODeflation(
              defl_nev, pRNG, d_ord, d_lo, d_tol,
              env_int("DTXQCD_DEFORM_EO_DEFLATE_RESTARTS", 20));
          if (!d_cache.empty()) Qdef->SaveEODeflation(d_cache);
        }
        const auto &evp = Qdef->EODeflEvecs();
        K = (int)evp.size();
        // Basis: unpack -> struct RB (physical Odd) -> S~ v_k -> full grid.
        std::vector<F> Vf;
        F vrb(&RBGrid), srb(&RBGrid);
        for (int k = 0; k < K; ++k) {
          DtxqcdUnpackDoubled(evp[k], vrb);
          for (int a = 0; a < DtxqcdNf; ++a) {
            vrb.upper.f[a].Checkerboard() = Odd;
            vrb.lower.f[a].Checkerboard() = Odd;
          }
          F vf(&Grid), sf(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            vf.upper.f[a] = Zero();
            vf.lower.f[a] = Zero();
            setCheckerboard(vf.upper.f[a], vrb.upper.f[a]);
            setCheckerboard(vf.lower.f[a], vrb.lower.f[a]);
          }
          Qdef->MpcTil(vrb, srb);
          for (int a = 0; a < DtxqcdNf; ++a) {
            sf.upper.f[a] = Zero();
            sf.lower.f[a] = Zero();
            setCheckerboard(sf.upper.f[a], srb.upper.f[a]);
            setCheckerboard(sf.lower.f[a], srb.lower.f[a]);
          }
          Vf.push_back(vf);
          SVf.push_back(sf);
        }
        // G = (S~V)^dag(S~V);  W_jk = (S~v_j)^dag dS~ v_k;
        // exact term = Tr[W G^-1] -- exact for ANY basis (no triplet bias).
        Eigen::MatrixXcd G(K, K), W(K, K);
        for (int k = 0; k < K; ++k) {
          F dsv(&Grid);
          dStil(Vf[k], dsv);
          for (int j = 0; j < K; ++j) {
            const ComplexD gg = DtxqcdDeformQuda::finner_(SVf[j], SVf[k]);
            const ComplexD ww = DtxqcdDeformQuda::finner_(SVf[j], dsv);
            G(j, k) = std::complex<double>(DtxqcdToStd(gg).real(),
                                           DtxqcdToStd(gg).imag());
            W(j, k) = std::complex<double>(DtxqcdToStd(ww).real(),
                                           DtxqcdToStd(ww).imag());
          }
        }
        Ginv = G.fullPivLu().inverse();
        const std::complex<double> tex = (W * Ginv).trace();
        t_exact = ComplexD(tex.real(), tex.imag());
        std::cout << GridLogMessage << "[stochpf-defl] node ep=" << ep
                  << " K=" << K << " exact term = " << t_exact.real()
                  << " + " << t_exact.imag() << " i" << std::endl;
#endif
      }
      auto Project = [&](F &y) {
        if (K == 0) return;
        Eigen::VectorXcd c(K);
        for (int j = 0; j < K; ++j) {
          const ComplexD z = DtxqcdDeformQuda::finner_(SVf[j], y);
          c(j) = std::complex<double>(DtxqcdToStd(z).real(),
                                      DtxqcdToStd(z).imag());
        }
        Eigen::VectorXcd d = Ginv * c;
        for (int k = 0; k < K; ++k) {
          const ComplexD dk(d(k).real(), d(k).imag());
          for (int a = 0; a < DtxqcdNf; ++a) {
            y.upper.f[a] = y.upper.f[a] - dk * SVf[k].upper.f[a];
            y.lower.f[a] = y.lower.f[a] - dk * SVf[k].lower.f[a];
          }
        }
      };
      // one remainder estimate <w, S~^-1 (1-P) dS~ w>/2 for a given probe
      int tslice_id = 0;
      auto Remainder = [&](const F &w) {
        F y(&Grid), x(&Grid);
        dStil(w, y);
        Project(y);
        SchurSolve(y, x);
        if (tslice_log) {
          // per-timeslice partials s_t (sum_t s_t = the estimate): the
          // sample-covariance of these over the run predicts every
          // t-dilution scheme's variance offline.
          LatticeComplex dens(&Grid);
          dens = Zero();
          for (int a = 0; a < DtxqcdNf; ++a) {
            dens = dens + localInnerProduct(w.upper.f[a], x.upper.f[a]);
            dens = dens + localInnerProduct(w.lower.f[a], x.lower.f[a]);
          }
          std::vector<TComplex> sums;
          sliceSum(dens, sums, Nd - 1);
          std::cout << GridLogMessage << "TSLICE n=" << tslice_id << " re";
          for (int t = 0; t < T; ++t)
            std::cout << " " << 0.5 * TensorRemove(sums[t]).real();
          std::cout << std::endl;
          std::cout << GridLogMessage << "TSLICE n=" << tslice_id << " im";
          for (int t = 0; t < T; ++t)
            std::cout << " " << 0.5 * TensorRemove(sums[t]).imag();
          std::cout << std::endl;
          ++tslice_id;
        }
        return DtxqcdDeformQuda::finner_(w, x) / 2.0;
      };
      // t-parity masks for the eo-t dilution
      LatticeComplex mt0(&Grid), mt1(&Grid);
      if (dilute_eot) {
        mt0 = ComplexD(0.0, 0.0);
        mt1 = ComplexD(0.0, 0.0);
        LatticeComplex one(&Grid);
        one = ComplexD(1.0, 0.0);
        for (int t = 0; t < T; ++t) {
          if (t % 2 == 0)
            mt0 = where(tc == Integer(t), one, mt0);
          else
            mt1 = where(tc == Integer(t), one, mt1);
        }
      }
      if (rho_t0b >= 0) {
        // difference-telescope correlation: same noise through both
        // contour offsets; all weight builders capture prof_t0 by
        // reference, so switching it + rebuilding retargets the operator.
        const int t0a = prof_t0;
        for (int h = 0; h < n_noise; ++h) {
          F w(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            gaussian(pRNG, w.upper.f[a]);
            gaussian(pRNG, w.lower.f[a]);
          }
          MaskKeepCb(w, Odd);
          ComplexD eab[2];
          const int t0s[2] = {t0a, rho_t0b};
          for (int s = 0; s < 2; ++s) {
            prof_t0 = t0s[s];
            BuildWeights(ep);
            BuildGWeights(ep);
            BuildHopWeights(ep);
            Qdef->SetProfile(ep, [&](int t) { return prof(t); });
            eab[s] = Remainder(w);
          }
          prof_t0 = t0a;
          std::cout << GridLogMessage << "RHOPAIR h=" << h << " A "
                    << eab[0].real() << " " << eab[0].imag() << " B "
                    << eab[1].real() << " " << eab[1].imag() << std::endl;
        }
        // restore the t0a operator state for any later nodes
        BuildWeights(ep);
        BuildGWeights(ep);
        BuildHopWeights(ep);
        Qdef->SetProfile(ep, [&](int t) { return prof(t); });
        g = ComplexD(0.0, 0.0);
        er = ei = 0;
        std::cout << GridLogMessage << "GLNODE q=" << q << " eta'=" << ep
                  << " [rhot0 instrument run -- no trace estimate]"
                  << std::endl;
        continue;
      }
      if (tslice_log == 2) {
        // F-MAP instrument: N_NOISE probe pairs x T masked solves each.
        // Offline: F(t,t') = mean over pairs of |f(t,t')|^2 / 4.
        LatticeComplex one(&Grid), mtp(&Grid);
        one = ComplexD(1.0, 0.0);
        for (int h = 0; h < n_noise; ++h) {
          F z(&Grid), zp(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            gaussian(pRNG, z.upper.f[a]);
            gaussian(pRNG, z.lower.f[a]);
            gaussian(pRNG, zp.upper.f[a]);
            gaussian(pRNG, zp.lower.f[a]);
          }
          MaskKeepCb(z, Odd);
          MaskKeepCb(zp, Odd);
          for (int tp = 0; tp < T; ++tp) {
            mtp = ComplexD(0.0, 0.0);
            mtp = where(tc == Integer(tp), one, mtp);
            F zt(&Grid), y(&Grid), x(&Grid);
            for (int a = 0; a < DtxqcdNf; ++a) {
              zt.upper.f[a] = mtp * zp.upper.f[a];
              zt.lower.f[a] = mtp * zp.lower.f[a];
            }
            dStil(zt, y);
            Project(y);
            SchurSolve(y, x);
            LatticeComplex dens(&Grid);
            dens = Zero();
            for (int a = 0; a < DtxqcdNf; ++a) {
              dens = dens + localInnerProduct(z.upper.f[a], x.upper.f[a]);
              dens = dens + localInnerProduct(z.lower.f[a], x.lower.f[a]);
            }
            std::vector<TComplex> sums;
            sliceSum(dens, sums, Nd - 1);
            std::cout << GridLogMessage << "FMAP h=" << h << " tp=" << tp
                      << " re";
            for (int t = 0; t < T; ++t)
              std::cout << " " << TensorRemove(sums[t]).real();
            std::cout << std::endl;
            std::cout << GridLogMessage << "FMAP h=" << h << " tp=" << tp
                      << " im";
            for (int t = 0; t < T; ++t)
              std::cout << " " << TensorRemove(sums[t]).imag();
            std::cout << std::endl;
          }
        }
        g = t_exact;
        er = ei = 0;
        std::cout << GridLogMessage << "GLNODE q=" << q << " eta'=" << ep
                  << " [fmap instrument run -- no trace estimate]"
                  << std::endl;
        dlogdet += (eta - eta_lo) * gl_w[q] * g;
        continue;
      }
      ComplexD acc(0.0, 0.0), acc2r(0.0, 0.0), acc2i(0.0, 0.0);
      for (int h = 0; h < nn; ++h) {
        F w(&Grid);
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(pRNG, w.upper.f[a]);
          gaussian(pRNG, w.lower.f[a]);
        }
        MaskKeepCb(w, Odd);
        auto EstOf = [&](const F &probe) {
          ComplexD e = Remainder(probe);
          if (use_anti) {
            F wg(&Grid);
            wg = probe;
            for (int a = 0; a < DtxqcdNf; ++a)
              wg.lower.f[a] = -wg.lower.f[a];
            e = 0.5 * (e + Remainder(wg));
          }
          return e;
        };
        ComplexD est;
        if (dilute_ts > 0) {
          est = ComplexD(0.0, 0.0);
          LatticeComplex mb(&Grid), one(&Grid);
          one = ComplexD(1.0, 0.0);
          for (int r = 0; r < dilute_ts; ++r) {
            mb = ComplexD(0.0, 0.0);
            for (int t = r; t < T; t += dilute_ts)
              mb = where(tc == Integer(t), one, mb);
            F wb(&Grid);
            for (int a = 0; a < DtxqcdNf; ++a) {
              wb.upper.f[a] = mb * w.upper.f[a];
              wb.lower.f[a] = mb * w.lower.f[a];
            }
            est = est + EstOf(wb);
          }
        } else if (dilute_zp > 0) {
          est = ComplexD(0.0, 0.0);
          LatticeInteger zc(&Grid);
          LatticeCoordinate(zc, 2);
          LatticeComplex mb(&Grid), one(&Grid);
          one = ComplexD(1.0, 0.0);
          for (int r = 0; r < 2; ++r) {
            mb = ComplexD(0.0, 0.0);
            for (int z = r; z < latt[2]; z += 2)
              mb = where(zc == Integer(z), one, mb);
            F wb(&Grid);
            for (int a = 0; a < DtxqcdNf; ++a) {
              wb.upper.f[a] = mb * w.upper.f[a];
              wb.lower.f[a] = mb * w.lower.f[a];
            }
            est = est + EstOf(wb);
          }
        } else if (dilute_tb > 0) {
          est = ComplexD(0.0, 0.0);
          LatticeComplex mb(&Grid), one(&Grid);
          one = ComplexD(1.0, 0.0);
          for (int b = 0; b < T / dilute_tb; ++b) {
            mb = ComplexD(0.0, 0.0);
            for (int t = b * dilute_tb; t < (b + 1) * dilute_tb; ++t)
              mb = where(tc == Integer(t), one, mb);
            F wb(&Grid);
            for (int a = 0; a < DtxqcdNf; ++a) {
              wb.upper.f[a] = mb * w.upper.f[a];
              wb.lower.f[a] = mb * w.lower.f[a];
            }
            est = est + EstOf(wb);
          }
        } else if (dilute_eot) {
          F w0(&Grid), w1(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            w0.upper.f[a] = mt0 * w.upper.f[a];
            w0.lower.f[a] = mt0 * w.lower.f[a];
            w1.upper.f[a] = mt1 * w.upper.f[a];
            w1.lower.f[a] = mt1 * w.lower.f[a];
          }
          est = EstOf(w0) + EstOf(w1);
        } else {
          est = EstOf(w);
        }
        acc += est;
        acc2r += est.real() * est.real();
        acc2i += est.imag() * est.imag();
      }
      ComplexD rmean = acc / (double)nn;
      er = std::sqrt(std::max(0.0, acc2r.real() / nn -
                                   rmean.real() * rmean.real()) / nn);
      ei = std::sqrt(std::max(0.0, acc2i.real() / nn -
                                   rmean.imag() * rmean.imag()) / nn);
      g = t_exact + rmean;
    } else {
    // Hutchinson: Tr[Mdef^{-1} dM] = E<v, Mdef^{-1} dM v> / 2 (gaussian).
    ComplexD acc(0.0, 0.0), acc2r(0.0, 0.0), acc2i(0.0, 0.0);
    for (int h = 0; h < nn; ++h) {
      // Grading-antithetic pair: dM is block-grading ODD, so the odd part of
      // M^{-1} dM dominates the Hutchinson variance yet contributes ZERO to
      // the trace.  Averaging v with Gv (G = block-parity flip of the lower
      // components) cancels it exactly per pair: 2x solves, ~10-40x variance.
      F v(&Grid), y(&Grid), x(&Grid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, v.upper.f[a]);
        gaussian(pRNG, v.lower.f[a]);
      }
      dMtil(v, y);
      SolveDef(y, x);
      ComplexD est1 = innerProduct(v, x) / 2.0;
      for (int a = 0; a < DtxqcdNf; ++a)
        v.lower.f[a] = -v.lower.f[a];                    // v -> Gv
      dMtil(v, y);
      SolveDef(y, x);
      ComplexD est2 = innerProduct(v, x) / 2.0;
      ComplexD est = 0.5 * (est1 + est2);
      acc += est;
      acc2r += est.real() * est.real();
      acc2i += est.imag() * est.imag();
    }
    g = acc / (double)nn;
    er = std::sqrt(std::max(0.0, acc2r.real() / nn -
                                 g.real() * g.real()) / nn);
    ei = std::sqrt(std::max(0.0, acc2i.real() / nn -
                                 g.imag() * g.imag()) / nn);
    }
    std::cout << GridLogMessage << "GLNODE q=" << q << " eta'=" << ep
              << " Tr[Minv dM] = " << g.real() << " +- " << er
              << "  (im " << g.imag() << " +- " << ei << ")"
              << (schur_mode ? "  [schur]" : "") << std::endl;
    dlogdet += (eta - eta_lo) * gl_w[q] * g;
    dlogdet_var += ComplexD((eta - eta_lo) * gl_w[q] * er * (eta - eta_lo) *
                                gl_w[q] * er,
                            (eta - eta_lo) * gl_w[q] * ei * (eta - eta_lo) *
                                gl_w[q] * ei);
  }
  std::cout << GridLogMessage << "STOCHPF traj=" << traj << " eta=" << eta
            << " dlogdet= " << dlogdet.real()
            << " +- " << std::sqrt(dlogdet_var.real())
            << " darg= " << dlogdet.imag()
            << " +- " << std::sqrt(dlogdet_var.imag())
            << " eta_lo= " << eta_lo << std::endl;

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
