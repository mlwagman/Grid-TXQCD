// meas_dibaryon_aux_dtxqcd: PURE-AUX dibaryon (triple-diquark) two-point
// functions.  NO inversions, NO Wick contractions.
//
// PRIMARY (I=1, J=0, P=+ -- the 1S0 NN channel):
//   D(x; r1, r3) = [T*_{r1} w]^k(x) . d6_{kl}(x) . [T*_{r3} w]^l(x)
// with
//   w^k     = eps^{kij} d_iso(i,j)      (isosinglet scalar diquark, colour 3bar
//                                        -- the DiquarkEpsilonWall object)
//   d6_{kl} = d(f=0,f=0)_{kl} + d(f=1,f=1)_{kl}
//             (delta_cd flavour trace = [uu]+[dd] isotriplet I3=+-1 mix; by the
//              joint colour+flavour symmetry of d this is colour-SYMMETRIC = 6)
//   T*_r    = product of CONJUGATED smeared links along the straight path r
//             (3bar parallel transport)
// 3bar x 6 x 3bar -> singlet; symmetric in (w1,w3) so the SINGLE-SITE
// operator (r1=r3=0) is non-vanishing (mwagman's eps eps eps_ab delta_cd
// eps_ef structure).  Correlator I3 cross terms <uu (dd)*> vanish, so
// C = 2 x C_{1S0}.
//
// SECONDARY (I=0, J=0): D0(x; r2, r3) = eps^{abc} w_a(x) [T*_{r2}w]_b [T*_{r3}w]_c
// (needs three distinct sites; vanishes locally).  Saved alongside -- free.
//
// CONTACT TERMS: none for dt != 0.  The source is HOLOMORPHIC in d (three
// unconjugated factors) and the aux measure is an ultralocal complex
// Gaussian: <dd>_measure = 0, and source-sink measure pairings need
// coincident sites.  So C(dt>=1) = <(qq)^3 (qbarqbar)^3>/lambda^6 exactly
// (SD identity), and <D> = 0 by baryon number (no vacuum subtraction).
// EXCLUDE dt=0 in analysis.
//
// B=2 => PERIODIC in t (even fermion wrap number): translation-average over
// all T source slices with NO APBC sign.
//
// The contour deformation acts as the OFFLINE multiplicative grading
//   D(t) -> e^{-3 eta p(t)} D(t)
// on the stored slice sums S(t) = (1/V3) sum_xvec D(x): source-centered
// graded correlator = e^{+3 eta p(dt)} S(ts+dt) S*(ts).  Unreweighted
// significance is eta-independent (deterministic rescale); the e^{3 eta p}
// S/N gain enters through the reweighted estimator.
//
// Output: per geometry g a complex T-vector S_g(t); correlators, orientation
// averaging, grading, and reweighting all assemble offline.
//
// GAUGE_TWIST=1: self-contained gate -- computes all S_g, applies a random
// gauge twist (U -> Om U Om^dag, d -> Om d Om^T), recomputes, and compares.
// D is pointwise gauge-invariant so every S_g must match to machine
// precision.  MANDATORY green before first physics use.
//
// Env: LATT (16.16.16.48), CFG_DIR, DATA_DIR (.), TRAJ, STOUT_NSMEAR (1),
//      STOUT_RHO (0.125), LMAX (3), LAMBDA_DTXQCD (5, metadata),
//      GAUGE_TWIST (0).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
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

// isosinglet diquark colour matrix (post-Elitzur-fix convention, as in
// meas_baryon_dtxqcd)
static LatticeColourMatrix IsoSingletDiquark(const LatticeDtxqcdD &dF) {
  GridBase *g = dF.Grid();
  LatticeColourMatrix out(g);
  out = Zero();
  static_assert(DtxqcdNf == 2, "isosinglet diquark assumes Nf=2");
  const ComplexD inv_sqrt2 = ComplexD(1.0 / std::sqrt(2.0), 0.0);
  autoView(dv, dF, CpuRead);
  autoView(ov, out, CpuWrite);
  thread_for(ss, g->oSites(), {
    for (int i = 0; i < Nc; ++i)
      for (int j = 0; j < Nc; ++j) {
        auto v = dv[ss]()(0, 1)(i, j) - dv[ss]()(1, 0)(i, j);
        ov[ss]()()(i, j) = inv_sqrt2 * v;
      }
  });
  return out;
}

// eps-reduced 3bar vector w^a = eps^{abc} d_iso(b,c) (full field, no mask)
static LatticeColourVector EpsReduce(const LatticeColourMatrix &d_iso) {
  GridBase *g = d_iso.Grid();
  LatticeColourVector w(g);
  w = Zero();
  autoView(dv, d_iso, CpuRead);
  autoView(wv, w, CpuWrite);
  thread_for(ss, g->oSites(), {
    auto d = dv[ss]()();
    wv[ss]()()(0) = d(1, 2) - d(2, 1);
    wv[ss]()()(1) = d(2, 0) - d(0, 2);
    wv[ss]()()(2) = d(0, 1) - d(1, 0);
  });
  return w;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_dibaryon_aux_dtxqcd <traj>" << std::endl;
    return 1;
  }
  const int traj = std::atoi(argv[1]);

  std::vector<int> latt_dims{16, 16, 16, 48};
  if (const char *v = std::getenv("LATT"); v && *v) {
    latt_dims.clear();
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  const int T = latt[Nd - 1];
  RealD V3 = 1.0;
  for (int mu = 0; mu < Nd - 1; ++mu) V3 *= (RealD)latt[mu];

  const int nstout = env_int("STOUT_NSMEAR", 1);
  const RealD rho = env_real("STOUT_RHO", 0.125);
  const int LMAX = env_int("LMAX", 3);
  const RealD lam = env_real("LAMBDA_DTXQCD", 5.0);
  const std::string cfg_dir = env_str("CFG_DIR",
      "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw");
  const std::string data_dir = env_str("DATA_DIR", ".");

  std::cout << GridLogMessage << "==== meas_dibaryon_aux_dtxqcd ====" << std::endl;
  std::cout << GridLogMessage << " traj=" << traj << " LMAX=" << LMAX
            << " cfg=" << cfg_dir << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);

  // ---- geometry lists ----------------------------------------------------
  // I=1: unordered displacement pairs (r1, r3) for the two w's; d6 at origin.
  // I=0: (r2, r3) with the first w at origin; all three sites distinct.
  struct Geo { Coordinate r1, r2; };
  std::vector<Geo> geoI1, geoI0;
  auto vec = [&](int mu, int s, int L) {
    Coordinate r(4, 0);
    r[mu] = s * L;
    return r;
  };
  Coordinate zero(4, 0);
  geoI1.push_back({zero, zero});                       // single site
  for (int L = 1; L <= LMAX; ++L) {
    for (int mu = 0; mu < 3; ++mu)
      for (int s : {+1, -1})
        geoI1.push_back({vec(mu, s, L), zero});        // (r, 0)
    for (int mu = 0; mu < 3; ++mu) {
      geoI1.push_back({vec(mu, +1, L), vec(mu, -1, L)});   // collinear
      geoI0.push_back({vec(mu, +1, L), vec(mu, -1, L)});
    }
    for (int mu = 0; mu < 3; ++mu)
      for (int s : {+1, -1})
        geoI1.push_back({vec(mu, s, L), vec(mu, s, L)});   // both same side
    for (int mu = 0; mu < 3; ++mu)
      for (int nu = mu + 1; nu < 3; ++nu)
        for (int s1 : {+1, -1})
          for (int s2 : {+1, -1}) {
            geoI1.push_back({vec(mu, s1, L), vec(nu, s2, L)});  // planar
            geoI0.push_back({vec(mu, s1, L), vec(nu, s2, L)});
          }
  }
  const int NG1 = (int)geoI1.size(), NG0 = (int)geoI0.size();
  std::cout << GridLogMessage << " geometries: I=1 " << NG1
            << ", I=0 " << NG0 << std::endl;

  // ---- everything below recomputed per (possibly twisted) field set ------
  auto ComputeAll = [&](const DTXQCDField &F,
                        std::vector<std::vector<ComplexD>> &S1,
                        std::vector<std::vector<ComplexD>> &S0) {
    // stout smear (after any twist: covariant)
    LatticeGaugeField Usm(F.U);
    if (nstout > 0) {
      Smear_Stout<PeriodicGimplR> stout(rho);
      LatticeGaugeField tmp(&Grid);
      for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
    }
    std::array<LatticeColourMatrix, 3> Umu{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid)};
    for (int mu = 0; mu < 3; ++mu)
      Umu[mu] = PeekIndex<LorentzIndex>(Usm, mu);

    // diquark fields
    LatticeColourMatrix d_iso = IsoSingletDiquark(F.d);
    LatticeColourVector w = EpsReduce(d_iso);
    LatticeColourMatrix d6(&Grid);
    {
      LatticeColourMatrix m00(PeekIndex<1>(F.d, 0, 0));
      LatticeColourMatrix m11(PeekIndex<1>(F.d, 1, 1));
      d6 = m00 + m11;                    // colour-symmetric (6), I3=+-1 mix
    }

    // cached 3bar transports: V[mu][sgn][L] = w transported from x + sgn*L*mu
    // forward step: v(x) = conj(U_mu(x)) v(x+mu);  backward: v(x) =
    // transpose(U_mu(x-mu)) v(x-mu).
    std::map<std::tuple<int, int, int>, LatticeColourVector> V;
    for (int mu = 0; mu < 3; ++mu)
      for (int s : {+1, -1}) {
        LatticeColourVector v = w;
        for (int L = 1; L <= LMAX; ++L) {
          if (s > 0) {
            LatticeColourVector sh = Cshift(v, mu, +1);
            v = LatticeColourMatrix(conjugate(Umu[mu])) * sh;
          } else {
            LatticeColourVector tv =
                LatticeColourMatrix(transpose(Umu[mu])) * v;
            v = Cshift(tv, mu, -1);
          }
          V.emplace(std::make_tuple(mu, s, L), v);
        }
      }
    auto getW = [&](const Coordinate &r) -> const LatticeColourVector & {
      for (int mu = 0; mu < 3; ++mu)
        if (r[mu] != 0)
          return V.at(std::make_tuple(mu, r[mu] > 0 ? +1 : -1, std::abs(r[mu])));
      return w;
    };

    auto slice = [&](const LatticeComplex &D, std::vector<ComplexD> &out) {
      std::vector<typename LatticeComplex::scalar_object> sl;
      sliceSum(D, sl, Nd - 1);
      out.resize(T);
      for (int t = 0; t < T; ++t) out[t] = TensorRemove(sl[t]) / V3;
    };

    // I=1: D = sum_k w1_k (d6 w3)_k   (no conjugation anywhere)
    S1.assign(NG1, {});
    for (int g = 0; g < NG1; ++g) {
      const LatticeColourVector &w1 = getW(geoI1[g].r1);
      const LatticeColourVector &w3 = getW(geoI1[g].r2);
      LatticeColourVector u = d6 * w3;
      LatticeComplex D(&Grid);
      D = Zero();
      for (int k = 0; k < Nc; ++k)
        D = D + peekColour(w1, k) * peekColour(u, k);
      slice(D, S1[g]);
    }

    // I=0: D0 = w . (w2 x w3)  (eps contraction, all unconjugated)
    S0.assign(NG0, {});
    for (int g = 0; g < NG0; ++g) {
      const LatticeColourVector &w2 = getW(geoI0[g].r1);
      const LatticeColourVector &w3 = getW(geoI0[g].r2);
      LatticeComplex b0 = peekColour(w2, 0), b1 = peekColour(w2, 1),
                     b2 = peekColour(w2, 2);
      LatticeComplex c0 = peekColour(w3, 0), c1 = peekColour(w3, 1),
                     c2 = peekColour(w3, 2);
      LatticeComplex D = peekColour(w, 0) * (b1 * c2 - b2 * c1) +
                         peekColour(w, 1) * (b2 * c0 - b0 * c2) +
                         peekColour(w, 2) * (b0 * c1 - b1 * c0);
      slice(D, S0[g]);
    }
  };

  std::vector<std::vector<ComplexD>> S1, S0;
  ComputeAll(U, S1, S0);

  // ---- GAUGE_TWIST gate --------------------------------------------------
  if (env_int("GAUGE_TWIST", 0) != 0) {
    LatticeColourMatrix om(&Grid);
    SU<Nc>::RandomGaugeTransform<PeriodicGimplR>(pRNG, U.U, om);
    LatticeColourMatrix omT(&Grid); omT = transpose(om);
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int b = 0; b < DtxqcdNf; ++b) {
        LatticeColourMatrix blk(PeekIndex<1>(U.d, a, b));
        blk = om * blk * omT;
        PokeIndex<1>(U.d, blk, a, b);
      }
    std::vector<std::vector<ComplexD>> T1, T0;
    ComputeAll(U, T1, T0);
    double dev = 0.0, scale = 0.0;
    for (int g = 0; g < NG1; ++g)
      for (int t = 0; t < T; ++t) {
        dev = std::max(dev, DtxqcdAbs(ComplexD(S1[g][t] - T1[g][t])));
        scale = std::max(scale, DtxqcdAbs(S1[g][t]));
      }
    for (int g = 0; g < NG0; ++g)
      for (int t = 0; t < T; ++t) {
        dev = std::max(dev, DtxqcdAbs(ComplexD(S0[g][t] - T0[g][t])));
        scale = std::max(scale, DtxqcdAbs(S0[g][t]));
      }
    const double rel = dev / std::max(scale, 1e-300);
    std::cout << GridLogMessage << "[dibaryon GAUGE_TWIST] max|dS|/max|S| = "
              << rel << (rel < 1e-8 ? "   PASSED" : "   FAILED") << std::endl;
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(rel < 1e-8 ? 0 : 1);
  }

  // ---- write -------------------------------------------------------------
  if (Grid.IsBoss()) {
    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
    std::string outfile =
        data_dir + "/dibaryon_aux_dtxqcd_" + std::to_string(traj) + ".h5";
    Hdf5Writer wr(outfile);
    write(wr, "T", T);
    write(wr, "V3", V3);
    write(wr, "lambda", lam);
    write(wr, "plaq", plaq);
    write(wr, "stout_nsmear", nstout);
    write(wr, "n_geo_I1", NG1);
    write(wr, "n_geo_I0", NG0);
    std::vector<int> g1meta, g0meta;
    for (auto &g : geoI1)
      for (int c = 0; c < 3; ++c) g1meta.push_back(g.r1[c]);
    for (auto &g : geoI1)
      for (int c = 0; c < 3; ++c) g1meta.push_back(g.r2[c]);
    for (auto &g : geoI0)
      for (int c = 0; c < 3; ++c) g0meta.push_back(g.r1[c]);
    for (auto &g : geoI0)
      for (int c = 0; c < 3; ++c) g0meta.push_back(g.r2[c]);
    write(wr, "geo_I1", g1meta);
    write(wr, "geo_I0", g0meta);
    for (int g = 0; g < NG1; ++g)
      write(wr, "dib_I1_S_" + std::to_string(g), S1[g]);
    for (int g = 0; g < NG0; ++g)
      write(wr, "dib_I0_S_" + std::to_string(g), S0[g]);
    std::cout << GridLogMessage << "wrote " << outfile << std::endl;
  }

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
