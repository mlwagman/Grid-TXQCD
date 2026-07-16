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
  int n_noise = env_int("N_NOISE", 16);
  int n_gl    = env_int("N_GL", 8);
  RealD tol   = env_real("CG_TOL", 1e-8);
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
  const int prof_t0 = env_int("PROF_T0", 0);
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  auto prof = [&](int t) {
    const int dt = ((t - prof_t0) % T + T) % T;
    if (!tent) return (double)dt;
    const double d = (double)dt;
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
  const RealD mg_etamin = env_real("DTXQCD_DEFORM_MG_ETAMIN", 0.5);
  std::unique_ptr<DtxqcdDeformMG<16>> mg;
  if (use_mg && use_eo_def && Qdef && Qdef->HasEO()) {
    std::cout << GridLogMessage << "[stochpf] deform-MG ENABLED for nodes "
              << "ep >= " << mg_etamin << std::endl;
    mg.reset(new DtxqcdDeformMG<16>(*Qdef, Grid, RBGrid));
  }
#endif

  ComplexD dlogdet(0.0, 0.0), dlogdet_var(0.0, 0.0);
  for (int q = 0; q < n_gl; ++q) {
    const double ep = eta * gl_x[q];
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
    // Hutchinson: Tr[Mdef^{-1} dM] = E<v, Mdef^{-1} dM v> / 2 (gaussian).
    ComplexD acc(0.0, 0.0), acc2r(0.0, 0.0), acc2i(0.0, 0.0);
    for (int h = 0; h < n_noise; ++h) {
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
    ComplexD g = acc / (double)n_noise;
    double er = std::sqrt(std::max(0.0, acc2r.real() / n_noise -
                                        g.real() * g.real()) / n_noise);
    double ei = std::sqrt(std::max(0.0, acc2i.real() / n_noise -
                                        g.imag() * g.imag()) / n_noise);
    std::cout << GridLogMessage << "GLNODE q=" << q << " eta'=" << ep
              << " Tr[Minv dM] = " << g.real() << " +- " << er
              << "  (im " << g.imag() << " +- " << ei << ")" << std::endl;
    dlogdet += eta * gl_w[q] * g;
    dlogdet_var += ComplexD(eta * gl_w[q] * er * eta * gl_w[q] * er,
                            eta * gl_w[q] * ei * eta * gl_w[q] * ei);
  }
  std::cout << GridLogMessage << "STOCHPF traj=" << traj << " eta=" << eta
            << " dlogdet= " << dlogdet.real()
            << " +- " << std::sqrt(dlogdet_var.real())
            << " darg= " << dlogdet.imag()
            << " +- " << std::sqrt(dlogdet_var.imag()) << std::endl;

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
