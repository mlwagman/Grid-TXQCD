// meas_baryon_deform_dtxqcd: aux baryon correlator on the eta-DEFORMED
// contour (the tent-profile variance-reduction program, arXiv:2101.12668
// spirit).  Companion to meas_baryon_dtxqcd (undeformed) and
// Test_dtxqcd_eta_stoch_pf (the Pf reweighting factor) -- kept separate
// because the solver styles differ (this one needs the deformed propagator;
// the estimator needs trace insertions).
//
// The deformation d,n -> e^{-eta p(t)}(d,n), d*,n* -> e^{+eta p(t)}(d*,n*),
// p(t) = tent = min(t, T-t) [PROFILE=linear: p=t], factorizes the deformed
// correlator COMPLETELY (2026-07-13 analysis):
//
//   C_def(tk,ts) = e^{-eta [p(tk)-p(ts)]} . [eps d . S_def,uu . eps d*]
//
// with UNDEFORMED d-walls and S_def = M_def^{-1} = G Mtil^{-1} G^{-1}
// computed through the WELL-CONDITIONED chemical-potential frame
// (G = diag(e^{-eta p/2} upper, e^{+eta p/2} lower); Mtil has undeformed aux
// blocks + e^{+-eta/2}-twisted time hops; the G-end factors exactly cancel
// Mtil's leading internal twist, leaving only the diquark's 2-quark-unit
// grading -- coefficient exactly 1 x eta).  The explicit grading factor IS
// BAKED INTO the saved SB matrices (the file is eta-stamped anyway); the
// reweighted ensemble mean of the graded correlator equals the undeformed
// C(t) exactly, while the mean-zero d-wall noise is graded down -- the S/N
// mechanism.
//
// Solves run the estimator's validated Mtil/MtilDag sandwich (OPCHECK
// 1.8e-14, DERIVCHECK 4e-10): CG on Mtil^dag Mtil; NEVER the ill-conditioned
// aux-frame operator, and NEVER the base op's gamma5-based Mdag (the twist
// breaks gamma5-hermiticity).
//
// Gates:
//  - ETA=0 must reproduce meas_baryon_dtxqcd's SB for the same cfg/tsrc to
//    CG tol (different solver path, same operator).
//  - GAUGE_TWIST=1: correlators must be twist-invariant (the deformation
//    factors are colour-trivial scalars; invariance is inherited).
//
// Env: LATT, CFG_DIR, DATA_DIR, TRAJ, ETA (0.2), PROFILE (tent|linear),
//      MEAS_BARYON_NTSRC (4), MASS (-0.245), CSW (prod), STOUT_NSMEAR (1),
//      STOUT_RHO (0.125), CG_TOL (1e-7), CG_MAX (20000), GAUGE_TWIST (0).
//      LAMBDA_DTXQCD used only for output metadata / dir naming.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#endif
// QUDA-backed M~ (StyleC base op + per-timeslice hop corrections) — the
// DTXQCD_DEFORM_QUDA=1 fast path.  See the header for the operator identity
// and the dagger (projector-flip + array-swap) derivation.
#include "dtxqcd_deform_quda.h"

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atoi(v) : d;
}
static std::string env_str(const char *k, const std::string &d) {
  const char *v = std::getenv(k); return (v && *v) ? std::string(v) : d;
}

// ---- baryon machinery (post-Elitzur-fix conventions of meas_baryon) ----

inline LatticeColourMatrix DtxqcdIsoSingletDiquark(const LatticeDtxqcdD &dF) {
  LatticeColourMatrix out(dF.Grid());
  out = Zero();
  static_assert(DtxqcdNf == 2, "isosinglet diquark assumes Nf=2");
  const ComplexD inv_sqrt2 = ComplexD(1.0 / std::sqrt(2.0), 0.0);
  autoView(dv, dF, CpuRead);
  autoView(ov, out, CpuWrite);
  thread_for(ss, dF.Grid()->oSites(), {
    for (int i = 0; i < Nc; ++i)
      for (int j = 0; j < Nc; ++j) {
        auto v = dv[ss]()(0, 1)(i, j) - dv[ss]()(1, 0)(i, j);
        ov[ss]()()(i, j) = inv_sqrt2 * v;
      }
  });
  return out;
}

inline LatticeColourVector DiquarkEpsilonWall(const LatticeColourMatrix &d_iso,
                                              int t_src, bool conj_field) {
  GridBase *g = d_iso.Grid();
  LatticeColourMatrix d_use = conj_field
                                  ? LatticeColourMatrix(conjugate(d_iso))
                                  : d_iso;
  LatticeColourVector wall(g);
  wall = Zero();
  {
    autoView(dv, d_use, CpuRead);
    autoView(wv, wall, CpuWrite);
    thread_for(ss, g->oSites(), {
      auto d = dv[ss]()();
      wv[ss]()()(0) = d(1, 2) - d(2, 1);
      wv[ss]()()(1) = d(2, 0) - d(0, 2);
      wv[ss]()()(2) = d(0, 1) - d(1, 0);
    });
  }
  if (t_src >= 0) {
    LatticeInteger t_coord(g);
    LatticeCoordinate(t_coord, Nd - 1);
    LatticeColourVector zero_cv(g);
    zero_cv = Zero();
    wall = where(t_coord == Integer(t_src), wall, zero_cv);
  }
  return wall;
}

inline LatticeFermion ApplyPosParitySpinProj(const LatticeFermion &in, int sign) {
  Gamma g4(Gamma::Algebra::GammaT);
  Gamma g3(Gamma::Algebra::GammaZ);
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermion pp = 0.5 * (in + g4 * in);
  LatticeFermion g3g5_pp = g3 * (g5 * pp);
  ComplexD i_sign = ComplexD(0.0, (RealD)sign);
  return 0.5 * (pp + i_sign * g3g5_pp);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_baryon_deform_dtxqcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  std::vector<int> latt_dims{16, 16, 16, 48};
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

  RealD mass = env_real("MASS", -0.245);
  RealD csw  = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho  = env_real("STOUT_RHO", 0.125);
  RealD eta  = env_real("ETA", 0.2);
  RealD lam  = env_real("LAMBDA_DTXQCD", 4.0);
  RealD tol  = env_real("CG_TOL", 1e-7);
  int cgmax  = env_int("CG_MAX", 20000);
  int n_tsrc = env_int("MEAS_BARYON_NTSRC", 4);
  const bool tent = env_str("PROFILE", "tent") != "linear";
  std::string cfg_dir = env_str("CFG_DIR",
      "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw");
  std::string data_dir = env_str("DATA_DIR", ".");

  if (n_tsrc <= 0 || n_tsrc > T) n_tsrc = T;
  std::vector<int> tsrc_list;
  for (int j = 0; j < n_tsrc; ++j) tsrc_list.push_back((j * T) / n_tsrc);

  std::cout << GridLogMessage << "==== meas_baryon_deform_dtxqcd ====" << std::endl;
  std::cout << GridLogMessage << " traj=" << traj << " eta=" << eta
            << " profile=" << (tent ? "tent" : "linear")
            << " ntsrc=" << n_tsrc << " cfg=" << cfg_dir << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);

  // GAUGE_TWIST=1 gate (local copy of DtxqcdMaybeGaugeTwist: U->Om U Om^dag,
  // sigma/pi->Om X Om^dag, d/n->Om X Om^T; deformation factors are colour
  // scalars so the deformed correlator must be twist-invariant too).
  if (env_int("GAUGE_TWIST", 0) != 0) {
    LatticeColourMatrix om(&Grid);
    SU<Nc>::RandomGaugeTransform<PeriodicGimplR>(pRNG, U.U, om);
    LatticeColourMatrix omT(&Grid); omT = transpose(om);
    LatticeColourMatrix omA(&Grid); omA = adj(om);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b) {
        { LatticeColourMatrix blk(PeekIndex<1>(U.sigma, a, b));
          blk = om * blk * omA; PokeIndex<1>(U.sigma, blk, a, b); }
        { LatticeColourMatrix blk(PeekIndex<1>(U.pi, a, b));
          blk = om * blk * omA; PokeIndex<1>(U.pi, blk, a, b); }
        { LatticeColourMatrix blk(PeekIndex<1>(U.d, a, b));
          blk = om * blk * omT; PokeIndex<1>(U.d, blk, a, b); }
        { LatticeColourMatrix blk(PeekIndex<1>(U.n, a, b));
          blk = om * blk * omT; PokeIndex<1>(U.n, blk, a, b); }
      }
    std::cout << GridLogMessage << "[GAUGE_TWIST] applied" << std::endl;
  }

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  LatticeColourMatrix d_iso = DtxqcdIsoSingletDiquark(U.d);

  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);

  // ---- DTXQCD_DEFORM_QUDA=1: QUDA-backed M~ (default OFF -> Grid path) ---
  // Base op = the untouched StyleC device M (chem-potential frame keeps aux
  // cross-blocks UNDEFORMED, so the standard operator + aux-static cache
  // apply verbatim); the deformation rides as 4 Grid-side hop-correction
  // terms per apply.  DTXQCD_DEFORM_QUDA_CHECK=1 (default ON when the
  // backend is on) verifies M~ and M~^dag against the Grid sandwich on
  // random vectors before any solve.
  // Default ON since 2026-07-13 (gates: operator identity 1e-15 at t0=0/12
  // and eta=0.2/0.8, twist invariance 2.4e-10, e2e row-0 3.3e-10, ~7x/iter
  // with the native kernel).  =0 restores the pure-Grid sandwich reference.
  // The qcheck (also default ON) re-verifies the operator against the Grid
  // sandwich at startup of EVERY run and _Exit(4)s on drift — tainted
  // solves are structurally impossible.
  const bool use_qdef = env_int("DTXQCD_DEFORM_QUDA", 1) != 0;
  const bool qdef_check =
      env_int("DTXQCD_DEFORM_QUDA_CHECK", use_qdef ? 1 : 0) != 0;
  // EO mode (checked early: it decides which QUDA stack gets built — the
  // full-volume device stack + the EO stack do NOT coexist on one A100;
  // job 1301152 OOM'd in the EO op's lazy init.  In EO mode the full M~ is
  // needed only for Schur prep/backsub + the gate reference, twice per
  // solve — the Grid-path apply (~0.11 s) is ample there, so the adapter
  // is built WITHOUT the QUDA op and the deform driver's full-path legs
  // fall back to the struct reference automatically.)
  // DEFAULT ON for lambda >= 5 since 2026-07-14 (e2882a2e gates: operator
  // 4.4e-16, solution 2.5e-8/2.9e-7, ~2x wall at eta=0.4-0.6, and at
  // eta=0.8 EO CONVERGES where the full path does not).  Keyed off
  // LAMBDA_DTXQCD (metadata env, set by all fleets); =0/1 overrides.
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
      // Aux fields are fixed for the whole measurement: pack the device aux
      // cache once (SetAuxStatic hoist), same as the undeformed meas path.
      Mq->SetAuxStatic(true);
      Hop.reset(new DtxqcdPackedFullHermOp(Mop, *Mq, &Grid, Even));
    } else {
      // EO mode: struct-path full adapter only (see the memory note above).
      Hop.reset(new DtxqcdPackedFullHermOp(Mop, &Grid, Even));
    }
#else
    Hop.reset(new DtxqcdPackedFullHermOp(Mop, &Grid, Even));
#endif
    Qdef.reset(new DtxqcdDeformQuda(*Hop, Grid, Usm, T));
    std::cout << GridLogMessage << "[baryon-def] backend=QUDA-hopcorr"
#ifndef GRID_HAVE_QUDA
              << " (non-QUDA build: struct-path reference fallback)"
#endif
              << std::endl;
  }

  // ---- DTXQCD_DEFORM_EO=1: EO Schur solve (the lambda=5/6 accelerator) --
  // Mpc~ = Moo - M~oe Mee^{-1} M~eo, with M~ee = Mee exactly (deformation
  // rides only the t-hops).  Default OFF pending gates.
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
    std::cout << GridLogMessage << "[baryon-def] EO Schur solve ENABLED "
              << "(Mpc~ device; M~ee = Mee caches verbatim)" << std::endl;
  }
#endif

  // ---- deformed solver stack (validated in Test_dtxqcd_eta_stoch_pf) ----
  //
  // SOURCE-CENTERED PROFILES (2026-07-13): the deformation only SUPPRESSES
  // noise for source->sink pairs aligned with the p-gradient (the ts=24
  // single-config look showed up to 23x AMPLIFICATION for anti-aligned
  // pairs).  So each t_src's propagator is solved on a profile whose tent
  // MINIMUM sits at that source: p(t) = min((t-ts) mod T, T-(t-ts) mod T) --
  // both time directions run uphill from every source, and the baked grading
  // is the translation-invariant e^{-eta min(dt, T-dt)}.  BOOKKEEPING: each
  // offset is a DIFFERENT contour => translation averaging needs a
  // ΔlogPf(t0) reweighting factor PER OFFSET (the estimator has the matching
  // PROF_T0 knob).  PROF_T0 >= 0 overrides with a FIXED offset for all
  // sources (single-contour mode, e.g. to pair with one reweighting factor).
  const int prof_t0_fixed = env_int("PROF_T0", -1);
  LatticeInteger tc(&Grid);
  LatticeCoordinate(tc, Nd - 1);
  int prof_t0 = (prof_t0_fixed >= 0) ? prof_t0_fixed : 0;
  // PROFILE=tent_rounded (PROFILE_W half-width, default 4 slices): the
  // tent apex is a mu-DOMAIN WALL (gradient sign flip) binding the
  // t-localized near-zero mode that gates every eta=0.8 solver (the
  // deform-MG campaign's terminal diagnosis).  A quadratic apex spreads
  // the flip into a linear mu crossing: the bound state LIFTS and WIDENS.
  // Grading sacrifice: p_max drops by w/2 at the source ANTIPODE only —
  // the correlator's least-signal region; the SOURCE-side kink stays
  // sharp (that is where the S/N benefit lives).  This lambda feeds the
  // hop weights, the G masks, SetProfile, AND the baked correlator
  // grading — one definition, all consumers.
  const double prof_w = env_real("PROFILE_W", 4.0);
  const std::string prof_kind = env_str("PROFILE", "tent");
  const bool prof_rounded = prof_kind == "tent_rounded" ||
                            prof_kind == "tent_rounded2";
  // tent_rounded2 ALSO rounds the SOURCE trough: the two tent kinks are
  // a DEGENERATE PAIR of mu-domain walls — the apex binds the
  // upper-block quasiparticle, the source trough binds the CONJUGATE
  // (lower-block) one (mwagman 2026-07-16).  Apex-only rounding splits
  // the pair but leaves sigma_min pinned at the trough state (measured:
  // x1.6 census lift, 3.4% CG relief at w=2).  Trough-rounding's cost:
  // grading RATE reduced only within w slices of the source (short
  // separations, where S/N is best) + the same uniform w/2 offset.
  const bool prof_round_src = prof_kind == "tent_rounded2";
  auto prof = [&](int t) {
    const int dt = ((t - prof_t0) % T + T) % T;
    if (!tent) return (double)dt;
    const double d = (double)dt;
    double v = std::min(d, (double)T - d);
    if (!prof_rounded) return v;
    const double a = std::abs(d - 0.5 * T);
    if (a < prof_w)  // apex parabola, slope-matched at |d-T/2| = w
      v = 0.5 * T - 0.5 * prof_w - a * a / (2.0 * prof_w);
    if (prof_round_src) {
      const double b = std::min(d, (double)T - d);  // distance to source
      if (b < prof_w)  // trough parabola (offset w/2 kept: only
                       // differences enter weights and grading)
        v = 0.5 * prof_w + b * b / (2.0 * prof_w);
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
  LatticeDtxqcdD d_wm(&Grid), d_wp(&Grid);
  LatticeDtxqcdN n_wm(&Grid), n_wp(&Grid);

  typedef DTXQCDFermionDoubled F;
  auto AddCross = [&](const LatticeDtxqcdD &dd, const LatticeDtxqcdN &nn,
                      const F &in, F &out, bool lower_to_upper) {
    DTXQCDFermionNf c(&Grid);
    if (lower_to_upper) {
      DtxqcdApplyDnCross(dd, nn, in.lower, c, /*apply_conj=*/false);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.upper.f[a] = out.upper.f[a] + c.f[a];
    } else {
      DtxqcdApplyDnCross(dd, nn, in.upper, c, /*apply_conj=*/true);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.lower.f[a] = out.lower.f[a] + c.f[a];
    }
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    AddCross(d_wm, n_wm, in, out, true);
    AddCross(d_wp, n_wp, in, out, false);
  };
  auto MdefDag = [&](const F &in, F &out) {
    Mop.Mdag(in, out);
    AddCross(d_wp, n_wp, in, out, true);
    AddCross(d_wm, n_wm, in, out, false);
  };
  LatticeComplex wGu(&Grid), wGl(&Grid), wGui(&Grid), wGli(&Grid);
  // Rebuild ALL deformation weights for the profile centered at t0.
  auto BuildDeformWeights = [&](int t0) {
    prof_t0 = (prof_t0_fixed >= 0) ? prof_t0_fixed : t0;
    WeightField(U.d, d_wm, [&](int t) { return std::exp(-eta * prof(t)) - 1.0; });
    WeightField(U.n, n_wm, [&](int t) { return std::exp(-eta * prof(t)) - 1.0; });
    WeightField(U.d, d_wp, [&](int t) { return std::exp(+eta * prof(t)) - 1.0; });
    WeightField(U.n, n_wp, [&](int t) { return std::exp(+eta * prof(t)) - 1.0; });
    wGu = ComplexD(1.0, 0.0);
    wGl = ComplexD(1.0, 0.0);
    LatticeComplex one(&Grid);
    for (int t = 0; t < T; ++t) {
      one = ComplexD(std::exp(-eta * prof(t) / 2.0), 0.0);
      wGu = where(tc == Integer(t), one, wGu);
      one = ComplexD(std::exp(+eta * prof(t) / 2.0), 0.0);
      wGl = where(tc == Integer(t), one, wGl);
    }
    wGui = ComplexD(1.0, 0.0); wGui = wGui / wGu;
    wGli = ComplexD(1.0, 0.0); wGli = wGli / wGl;
    if (Qdef) {
      if (tent && !prof_rounded)
        Qdef->SetTentProfile(eta, prof_t0);  // tracks t0 for deflation
      else
        // tent_rounded MUST go through the generic SetProfile — the
        // 2026-07-16 w-scan came back bit-identical across widths
        // because SetTentProfile fed the device the built-in SHARP tent
        // (deflation frame-tracking is t0-keyed off SetTentProfile, so
        // rounded profiles are incompatible with the deflation cache).
        Qdef->SetProfile(eta, [&](int t) { return prof(t); });
    }
  };
  auto GApply = [&](F &v, int sign) {
    const LatticeComplex &fu = (sign > 0) ? wGu : wGui;
    const LatticeComplex &fl = (sign > 0) ? wGl : wGli;
    for (int a = 0; a < DtxqcdNf; ++a) {
      v.upper.f[a] = fu * v.upper.f[a];
      v.lower.f[a] = fl * v.lower.f[a];
    }
  };
  auto Mtil = [&](const F &in, F &out) {
    F w(&Grid); w = in;
    GApply(w, +1);
    Mdef(w, out);
    GApply(out, -1);
  };
  auto MtilDag = [&](const F &in, F &out) {
    F w(&Grid); w = in;
    GApply(w, -1);
    MdefDag(w, out);
    GApply(out, +1);
  };
  // x = Mtil^{-1} src via CG on Mtil^dag Mtil.
  auto SolveTil = [&](const F &src, F &x) {
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
    const RealD relres = std::sqrt(norm2(r) / b2);
    std::cout << GridLogMessage << "[deform-CG] iters=" << k
              << " relres=" << relres << std::endl;
    if (relres > tol) {
      std::cout << GridLogMessage << "[deform-CG] NOT CONVERGED at CG_MAX="
                << cgmax << " (relres " << relres << " > tol " << tol
                << ") -- aborting rather than writing tainted correlators"
                << std::endl;
      Grid_finalize();
      std::_Exit(3);
    }
  };

  // ---- QUDA-path operator gates (random-vector, before any solve) --------
  // 1. M~ identity: QUDA hop-correction M~ == Grid sandwich Mtil (exact
  //    operator identity; catches convention/frame errors at any volume).
  // 2. Dagger identity: same for M~^dag (the #1 trap).
  // 3. Adjointness: <u, M~ v> == <M~^dag u, v> within roundoff.
  if (Qdef && qdef_check) {
    BuildDeformWeights(0);
#ifdef GRID_HAVE_QUDA
    // WARM UP the EO device pools FIRST.  Grid's MemoryManager cache is
    // elastic (grows on every temporary, yields only to its own pressure)
    // while QUDA's pools are inelastic: if the qcheck's Grid temporaries
    // grow the cache before the EO op's lazy init (CSFs + ~7 GB site
    // caches), the init starves (jobs 1301152/84/96 all OOM'd here by
    // ~18 MB).  One zero-vector Mpc~ apply allocates everything up front.
    if (use_eo_def && Qdef->HasEO()) {
      F w1(&RBGrid), w2(&RBGrid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        w1.upper.f[a] = Zero(); w1.lower.f[a] = Zero();
        w1.upper.f[a].Checkerboard() = Odd;
        w1.lower.f[a].Checkerboard() = Odd;
      }
      Qdef->MpcTil(w1, w2);
      std::cout << GridLogMessage << "[deform-qcheck:EO] device pools warm"
                << std::endl;
    }
#endif
    // Scope block: the six full-volume temporaries (~0.9 GB) must die
    // before the EO gate allocates its RB set — with BOTH QUDA stacks
    // resident the checks otherwise tip the device budget (job 1301095
    // OOM'd exactly here).
    {
    GridParallelRNG RNG(&Grid);
    RNG.SeedFixedIntegers({77, 13, 999, 4242});
    F v(&Grid), u(&Grid), ag(&Grid), aq(&Grid), dg(&Grid), dq(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(RNG, v.upper.f[a]);
      gaussian(RNG, v.lower.f[a]);
      gaussian(RNG, u.upper.f[a]);
      gaussian(RNG, u.lower.f[a]);
    }
    Mtil(v, ag);
    Qdef->Mtil(v, aq);
    MtilDag(u, dg);
    Qdef->MtilDag(u, dq);
    RealD nm = 0, nd = 0, nmd = 0, ndd = 0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      nm += norm2(LatticeFermion(ag.upper.f[a] - aq.upper.f[a])) +
            norm2(LatticeFermion(ag.lower.f[a] - aq.lower.f[a]));
      nd += norm2(ag.upper.f[a]) + norm2(ag.lower.f[a]);
      nmd += norm2(LatticeFermion(dg.upper.f[a] - dq.upper.f[a])) +
             norm2(LatticeFermion(dg.lower.f[a] - dq.lower.f[a]));
      ndd += norm2(dg.upper.f[a]) + norm2(dg.lower.f[a]);
    }
    const RealD relM = std::sqrt(nm / nd), relD = std::sqrt(nmd / ndd);
    const ComplexD ip1 = DtxqcdDeformQuda::finner_(u, aq);
    const ComplexD ip2 = DtxqcdDeformQuda::finner_(dq, v);
    const RealD adj_rel = DtxqcdAbs(ComplexD(ip1 - ip2)) / DtxqcdAbs(ip1);
    std::cout << GridLogMessage << "[deform-qcheck] |Mtil_quda - Mtil_grid| rel = "
              << relM << std::endl;
    std::cout << GridLogMessage << "[deform-qcheck] |MtilDag_quda - MtilDag_grid| rel = "
              << relD << std::endl;
    std::cout << GridLogMessage << "[deform-qcheck] adjointness <u,Mv> vs <Mdag u,v> rel = "
              << adj_rel << std::endl;
    if (relM > 1e-8 || relD > 1e-8 || adj_rel > 1e-10) {
      std::cout << GridLogMessage
                << "[deform-qcheck] FAILED — refusing to run tainted solves"
                << std::endl;
      Grid_finalize();
      std::_Exit(4);
    }
    std::cout << GridLogMessage << "[deform-qcheck] PASSED" << std::endl;
    }

#ifdef GRID_HAVE_QUDA
    // EO gate: device Mpc~ vs the Schur complement SYNTHESIZED from the
    // (just-verified) full-volume M~ + the cached Mee^{-1}:
    //   M~(v_o ⊕ 0)  ->  odd part = M~oo v, even part = M~eo v
    //   u = Mee^{-1}(M~eo v) ;  M~(0 ⊕ u) -> odd part = M~oe u
    //   ref = M~oo v − M~oe u.
    // Plus adjointness of the device Mpc~/Mpc~dag pair.
    if (use_eo_def && Qdef && Qdef->HasEO()) {
      // The full Schur-synthesis identity needs ~2 GB of check temporaries
      // on top of the EO stack — it fits ONLY at small volumes (the 4^4
      // signhop gate).  At production volume run the cheap device
      // adjointness check (4 RB fields; catches coefficient-array swap /
      // staleness drift, the realistic runtime failure) — production
      // correctness is certified by the solution-level h5 gates
      // (e2882a2e: 2.5e-8/2.9e-7).  DTXQCD_DEFORM_EO_FULLCHECK=1 forces
      // the full identity anywhere (memory permitting).
      const bool full_eo_check =
          env_int("DTXQCD_DEFORM_EO_FULLCHECK",
                  (Grid.gSites() <= 65536) ? 1 : 0) != 0;
      if (!full_eo_check) {
        F vo(&RBGrid), uo(&RBGrid), qo(&RBGrid), qdo(&RBGrid);
        {
          GridParallelRNG RNGf(&Grid);
          RNGf.SeedFixedIntegers({11, 22, 33, 44});
          F tmp_full(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            gaussian(RNGf, tmp_full.upper.f[a]);
            gaussian(RNGf, tmp_full.lower.f[a]);
          }
          for (int a = 0; a < DtxqcdNf; ++a) {
            pickCheckerboard(Odd, vo.upper.f[a], tmp_full.upper.f[a]);
            pickCheckerboard(Odd, vo.lower.f[a], tmp_full.lower.f[a]);
          }
          for (int a = 0; a < DtxqcdNf; ++a) {
            gaussian(RNGf, tmp_full.upper.f[a]);
            gaussian(RNGf, tmp_full.lower.f[a]);
          }
          for (int a = 0; a < DtxqcdNf; ++a) {
            pickCheckerboard(Odd, uo.upper.f[a], tmp_full.upper.f[a]);
            pickCheckerboard(Odd, uo.lower.f[a], tmp_full.lower.f[a]);
          }
        }
        Qdef->MpcTil(vo, qo);
        Qdef->MpcTilDag(uo, qdo);
        const ComplexD a1 = DtxqcdDeformQuda::finner_(uo, qo);
        const ComplexD a2 = DtxqcdDeformQuda::finner_(qdo, vo);
        const RealD adjEO = DtxqcdAbs(ComplexD(a1 - a2)) / DtxqcdAbs(a1);
        std::cout << GridLogMessage
                  << "[deform-qcheck:EO] adjointness (device-only) rel = "
                  << adjEO << std::endl;
        if (adjEO > 1e-10) {
          std::cout << GridLogMessage << "[deform-qcheck:EO] FAILED"
                    << std::endl;
          Grid_finalize();
          std::_Exit(4);
        }
        std::cout << GridLogMessage << "[deform-qcheck:EO] PASSED "
                  << "(adjointness; full identity = the 4^4 gate)"
                  << std::endl;
      } else {
      F vo(&RBGrid), uo(&RBGrid), qo(&RBGrid), qdo(&RBGrid);
      F ref_oo(&RBGrid), ref_eo(&RBGrid), ref_oe(&RBGrid), ue(&RBGrid);
      F zf(&Grid), mzf(&Grid);
      // RB noise = full-grid gaussian + pickCheckerboard (the §5b / house
      // pattern; a GridParallelRNG constructed ON the RB grid hangs — job
      // 1301201 burned its wall in exactly that constructor).
      {
        GridParallelRNG RNGf(&Grid);
        RNGf.SeedFixedIntegers({11, 22, 33, 44});
        F tmp_full(&Grid);
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(RNGf, tmp_full.upper.f[a]);
          gaussian(RNGf, tmp_full.lower.f[a]);
        }
        for (int a = 0; a < DtxqcdNf; ++a) {
          pickCheckerboard(Odd, vo.upper.f[a], tmp_full.upper.f[a]);
          pickCheckerboard(Odd, vo.lower.f[a], tmp_full.lower.f[a]);
        }
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(RNGf, tmp_full.upper.f[a]);
          gaussian(RNGf, tmp_full.lower.f[a]);
        }
        for (int a = 0; a < DtxqcdNf; ++a) {
          pickCheckerboard(Odd, uo.upper.f[a], tmp_full.upper.f[a]);
          pickCheckerboard(Odd, uo.lower.f[a], tmp_full.lower.f[a]);
        }
      }
      // ref = M~oo v − M~oe Mee^{-1} M~eo v via two full-M~ applies.
      for (int a = 0; a < DtxqcdNf; ++a) {
        zf.upper.f[a] = Zero(); zf.lower.f[a] = Zero();
      }
      for (int a = 0; a < DtxqcdNf; ++a) {
        setCheckerboard(zf.upper.f[a], vo.upper.f[a]);
        setCheckerboard(zf.lower.f[a], vo.lower.f[a]);
      }
      Qdef->Mtil(zf, mzf);
      for (int a = 0; a < DtxqcdNf; ++a) {
        pickCheckerboard(Odd, ref_oo.upper.f[a], mzf.upper.f[a]);
        pickCheckerboard(Odd, ref_oo.lower.f[a], mzf.lower.f[a]);
        pickCheckerboard(Even, ref_eo.upper.f[a], mzf.upper.f[a]);
        pickCheckerboard(Even, ref_eo.lower.f[a], mzf.lower.f[a]);
      }
      Dw.MooeeInv(ref_eo, ue);
      for (int a = 0; a < DtxqcdNf; ++a) {
        zf.upper.f[a] = Zero(); zf.lower.f[a] = Zero();
        setCheckerboard(zf.upper.f[a], ue.upper.f[a]);
        setCheckerboard(zf.lower.f[a], ue.lower.f[a]);
      }
      Qdef->Mtil(zf, mzf);
      for (int a = 0; a < DtxqcdNf; ++a) {
        pickCheckerboard(Odd, ref_oe.upper.f[a], mzf.upper.f[a]);
        pickCheckerboard(Odd, ref_oe.lower.f[a], mzf.lower.f[a]);
        ref_oo.upper.f[a] = ref_oo.upper.f[a] - ref_oe.upper.f[a];
        ref_oo.lower.f[a] = ref_oo.lower.f[a] - ref_oe.lower.f[a];
      }
      Qdef->MpcTil(vo, qo);
      RealD nm = 0, nd = 0;
      for (int a = 0; a < DtxqcdNf; ++a) {
        nm += norm2(LatticeFermion(qo.upper.f[a] - ref_oo.upper.f[a])) +
              norm2(LatticeFermion(qo.lower.f[a] - ref_oo.lower.f[a]));
        nd += norm2(ref_oo.upper.f[a]) + norm2(ref_oo.lower.f[a]);
      }
      const RealD relEO = std::sqrt(nm / nd);
      Qdef->MpcTilDag(uo, qdo);
      const ComplexD e1 = DtxqcdDeformQuda::finner_(uo, qo);
      const ComplexD e2 = DtxqcdDeformQuda::finner_(qdo, vo);
      const RealD adjEO = DtxqcdAbs(ComplexD(e1 - e2)) / DtxqcdAbs(e1);
      std::cout << GridLogMessage
                << "[deform-qcheck:EO] |Mpc~_dev - Mpc~_schur| rel = " << relEO
                << std::endl;
      std::cout << GridLogMessage
                << "[deform-qcheck:EO] adjointness rel = " << adjEO
                << std::endl;
      if (relEO > 1e-8 || adjEO > 1e-10) {
        std::cout << GridLogMessage << "[deform-qcheck:EO] FAILED" << std::endl;
        Grid_finalize();
        std::_Exit(4);
      }
      std::cout << GridLogMessage << "[deform-qcheck:EO] PASSED" << std::endl;
      }  // full_eo_check
    }
#endif
  }

  // ---- measurement loop: S_def . (undeformed d* wall), P+ projections ----
  const int NF = 2, NS = 2;
  const int spin_signs[NS] = {+1, -1};
  std::cout << GridLogMessage << "[baryon-def] " << (NF * NS * n_tsrc)
            << " deformed solves" << std::endl;
  std::vector<std::vector<std::vector<std::vector<ComplexD>>>> SB(
      NF, std::vector<std::vector<std::vector<ComplexD>>>(
        NS, std::vector<std::vector<ComplexD>>(T, std::vector<ComplexD>(T))));

  Gamma g5(Gamma::Algebra::Gamma5);
  // MEAS_BARYON_MAXSOLVES=N (0 = unlimited): stop after N solves — timing
  // probes and h5 A/B gates need one representative solve, not the full
  // flavor x spin x t_src set (NTSRC=1 still means 4 solves).
  const int max_solves = env_int("MEAS_BARYON_MAXSOLVES", 0);
  int n_solved = 0;
  for (int t_src : tsrc_list) {
    BuildDeformWeights(t_src);   // source-centered contour for this t_src
    for (int fi = 0; fi < NF; ++fi)
      for (int si = 0; si < NS; ++si) {
        if (max_solves > 0 && n_solved >= max_solves) continue;
        ++n_solved;
        LatticeColourVector wall = DiquarkEpsilonWall(d_iso, t_src,
                                                      /*conj_field=*/true);
        LatticeFermion eta_raw(&Grid);
        eta_raw = Zero();
        for (int beta = 0; beta < Ns; ++beta) pokeSpin(eta_raw, wall, beta);
        LatticeFermion eta_f = ApplyPosParitySpinProj(eta_raw, spin_signs[si]);
        std::cout << GridLogMessage << "[baryon-def] t_src=" << t_src
                  << " f=" << fi << " s=" << si
                  << " ||eta||^2=" << norm2(eta_f) << std::endl;
        F src(&Grid), y(&Grid), x(&Grid);
        src = Zero();
        src.upper.f[fi] = eta_f;
        // S_def = G Mtil^{-1} G^{-1}: scale source by G^{-1}, solve, scale by G.
        GApply(src, -1);
        if (Qdef) {
          RealD relres = 0;
          int it = 0;
          const char *tag = "[deform-CG:quda]";
#ifdef GRID_HAVE_QUDA
          if (use_eo_def && Qdef->HasEO()) {
            // DTXQCD_DEFORM_EO_DEFLATE=nev: build the eigenbasis lazily at
            // the FIRST solve's profile (once per config+eta; later t_srcs
            // are handled by the solve-frame conjugation — even spacings
            // only).  lambda=5 eta=0.8: nev=16 -> 60k iters -> ~1k.
            static const int defl_nev =
                env_int("DTXQCD_DEFORM_EO_DEFLATE", 0);
            if (defl_nev > 0 && tent && !Qdef->HasDeflation()) {
              // LO sets the Chebyshev window edge as a FRACTION of
              // lambda_max and must sit ABOVE the wanted band (modes below
              // lo are amplified; modes inside [lo,hi] are suppressed and
              // never converge — job 1301270 ground 0/16 with lo below the
              // band).  3e-4 * lmax ~ 0.04 lands in the gap between the
              // deformation band (<~1e-2) and the EO bulk edge (~0.1) for
              // lambda=5/6 at eta = 0.6-0.9.
              const int defl_ord = env_int("DTXQCD_DEFORM_EO_DEFLATE_ORD", 301);
              const RealD defl_lo = env_real("DTXQCD_DEFORM_EO_DEFLATE_LO", 3e-4);
              const RealD defl_tol = env_real("DTXQCD_DEFORM_EO_DEFLATE_TOL", 1e-3);
              // DTXQCD_DEFORM_EO_DEFLATE_CACHE=<dir>: reload the ~20-30 min
              // tight eigensolve if a matching sidecar exists (config
              // identity guarded by a gauge fingerprint in the header),
              // else build and save.  The basis is t0-translatable, so one
              // file per (config, eta) serves every t_src AND the
              // sign-reweighting estimator (same Mpc~).  ~75 MB x nev.
              std::string defl_cache;
              if (const char *cd = std::getenv("DTXQCD_DEFORM_EO_DEFLATE_CACHE"))
                if (*cd) {
                  char pre[128];
                  std::snprintf(pre, sizeof(pre), "/tr%d_lam%.4f_", traj,
                                (double)lam);
                  defl_cache = std::string(cd) + pre +
                               DtxqcdDeformQuda::EODeflationCacheName(
                                   defl_nev, defl_tol, defl_ord, eta, t_src,
                                   Grid.GlobalDimensions());
                }
              if (defl_cache.empty() || !Qdef->LoadEODeflation(defl_cache)) {
                Qdef->BuildEODeflation(
                    defl_nev, pRNG, defl_ord, defl_lo, defl_tol,
                    env_int("DTXQCD_DEFORM_EO_DEFLATE_RESTARTS", 20));
                if (!defl_cache.empty()) Qdef->SaveEODeflation(defl_cache);
              }
              if (env_int("DTXQCD_DEFORM_EO_DEFLATE_DIAG", 0))
                Qdef->VerifyDeflationBasis();
            }
            if (Qdef->HasDeflation()) {
              it = Qdef->SolveTilEODeflated(src, y, tol, cgmax, relres);
              tag = "[deform-CG:quda-EO-defl]";
            } else {
              it = Qdef->SolveTilEO(src, y, tol, cgmax, relres);
              tag = "[deform-CG:quda-EO]";
            }
          } else
#endif
          {
            it = Qdef->SolveTil(src, y, tol, cgmax, relres);
          }
          std::cout << GridLogMessage << tag << " iters=" << it
                    << " relres=" << relres << std::endl;
          if (relres > tol) {
            std::cout << GridLogMessage
                      << "[deform-CG:quda] NOT CONVERGED at CG_MAX=" << cgmax
                      << " (relres " << relres << " > tol " << tol
                      << ") -- aborting rather than writing tainted "
                      << "correlators" << std::endl;
            Grid_finalize();
            std::_Exit(3);
          }
        } else {
          SolveTil(src, y);
        }
        GApply(y, +1);
        // Sink contraction: physical weight eps d_iso (localInnerProduct
        // conjugates its first arg -> pass the conjugated wall; the
        // post-Elitzur-fix convention).
        LatticeFermion psi_q = y.upper.f[fi];
        LatticeColourVector swall = DiquarkEpsilonWall(d_iso, -1,
                                                       /*conj_field=*/true);
        LatticeFermion psi_proj = ApplyPosParitySpinProj(psi_q, spin_signs[si]);
        LatticeComplex Cn(&Grid);
        Cn = Zero();
        for (int beta = 0; beta < Ns; ++beta) {
          LatticeColourVector psi_b = peekSpin(psi_proj, beta);
          Cn = Cn + localInnerProduct(swall, psi_b);
        }
        std::vector<TComplex> sl;
        sliceSum(Cn, sl, Nd - 1);
        for (int tk = 0; tk < T; ++tk)
          SB[fi][si][t_src][tk] = TensorRemove(sl[tk]);
      }
  }

  // ---- bake in the explicit contour grading ----
  // Source-centered profiles: p(ts)=0 at each source, so the factor is the
  // translation-invariant e^{-eta p_rel(tk-ts)} (tent: min(dt, T-dt)).  With
  // PROF_T0 fixed, the same formula uses the fixed-profile p differences.
  for (int fi = 0; fi < NF; ++fi)
    for (int si = 0; si < NS; ++si)
      for (int ts : tsrc_list) {
        prof_t0 = (prof_t0_fixed >= 0) ? prof_t0_fixed : ts;
        for (int tk = 0; tk < T; ++tk)
          SB[fi][si][ts][tk] *=
              ComplexD(std::exp(-eta * (prof(tk) - prof(ts))), 0.0);
      }

  std::string outfile = data_dir + "/baryon_deform_dtxqcd_" +
                        std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    const char *fl_nm[NF] = {"p", "n"};
    const char *sp_nm[NS] = {"up", "dn"};
    auto flatten = [&](int fi, int si) {
      std::vector<ComplexD> out(T * T);
      for (int ts = 0; ts < T; ++ts)
        for (int tk = 0; tk < T; ++tk) out[ts * T + tk] = SB[fi][si][ts][tk];
      return out;
    };
    for (int fi = 0; fi < NF; ++fi)
      for (int si = 0; si < NS; ++si)
        write(wr, std::string("baryon_SB_uu_") + fl_nm[fi] + "_" + sp_nm[si] +
                  "_2d", flatten(fi, si));
    write(wr, "traj", traj);
    write(wr, "eta", eta);
    write(wr, "profile_tent", (int)tent);
    write(wr, "prof_t0", prof_t0_fixed);   // -1 = source-centered profiles
    write(wr, "lambda", lam);
    write(wr, "mass", mass);
    write(wr, "plaq", plaq);
    write(wr, "T", T);
    write(wr, "V3", V3);
  }
  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
