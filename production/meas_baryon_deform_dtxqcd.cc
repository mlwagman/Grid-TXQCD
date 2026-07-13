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
  auto prof = [&](int t) {
    const int dt = ((t - prof_t0) % T + T) % T;
    return tent ? (double)std::min(dt, T - dt) : (double)dt;
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

  // ---- measurement loop: S_def . (undeformed d* wall), P+ projections ----
  const int NF = 2, NS = 2;
  const int spin_signs[NS] = {+1, -1};
  std::cout << GridLogMessage << "[baryon-def] " << (NF * NS * n_tsrc)
            << " deformed solves" << std::endl;
  std::vector<std::vector<std::vector<std::vector<ComplexD>>>> SB(
      NF, std::vector<std::vector<std::vector<ComplexD>>>(
        NS, std::vector<std::vector<ComplexD>>(T, std::vector<ComplexD>(T))));

  Gamma g5(Gamma::Algebra::Gamma5);
  for (int t_src : tsrc_list) {
    BuildDeformWeights(t_src);   // source-centered contour for this t_src
    for (int fi = 0; fi < NF; ++fi)
      for (int si = 0; si < NS; ++si) {
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
        SolveTil(src, y);
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
