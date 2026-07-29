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
// S/N gain enters through the reweighted estimator with the tent trough at
// the d*^3 end (see _dib_graded_demo.py).
//
// Output: per geometry g a complex T-vector S_g(t); correlators, orientation
// averaging, grading, and reweighting all assemble offline.
//
// The measurement core (fields, smearing, geometries, h5 format) lives in
// dtxqcd_dibaryon_core.h, SHARED with the inline HMC observer
// (dtxqcd_dibaryon_inline.h in gen_dtxqcd_cfgs) -- inline and offline run
// bit-identical code.
//
// GAUGE_TWIST=1: self-contained gate -- computes all S_g, applies a random
// gauge twist (U -> Om U Om^dag, d -> Om d Om^T), recomputes, and compares.
// D is pointwise gauge-invariant so every S_g must match to machine
// precision.  MANDATORY green before first physics use.
//
// Env: LATT (16.16.16.48), CFG_DIR, DATA_DIR (.), TRAJ, STOUT_NSMEAR (1),
//      STOUT_RHO (0.125), LMAX (3), LAMBDA_DTXQCD (5, metadata),
//      GAUGE_TWIST (0).

#include "dtxqcd_dibaryon_core.h"

#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>

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

  const int nstout = env_int("STOUT_NSMEAR", 1);
  const RealD rho = env_real("STOUT_RHO", 0.125);
  const int LMAX = env_int("LMAX", 3);
  // SMEARS: comma list of covariant Gaussian smearing iteration counts for
  // the diquark fields (w as 3bar with conjugated links, d6 as colour-6
  // with U d U^T), SPATIAL-only so dt>=1 contact-freedom is preserved.
  // Ascending; 0 = the unsmeared (original) operator set.
  std::vector<int> smears;
  {
    std::stringstream ss(env_str("SMEARS", "0,3,10,30"));
    std::string tok;
    while (std::getline(ss, tok, ',')) smears.push_back(std::atoi(tok.c_str()));
  }
  const RealD salpha = env_real("SMEAR_ALPHA", 0.5);
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

  DtxqcdDibaryon::MeasCore core(Grid, nstout, rho, LMAX, smears, salpha);
  const int NG1 = core.NG1(), NG0 = core.NG0(), NLEV = core.NLEV();
  std::cout << GridLogMessage << " geometries: I=1 " << NG1
            << ", I=0 " << NG0 << std::endl;

  std::vector<std::vector<std::vector<ComplexD>>> S1, S0;
  core.ComputeAll(U, S1, S0);

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
    std::vector<std::vector<std::vector<ComplexD>>> T1, T0;
    core.ComputeAll(U, T1, T0);
    double dev = 0.0, scale = 0.0;
    for (int l = 0; l < NLEV; ++l) {
      for (int g = 0; g < NG1; ++g)
        for (int t = 0; t < T; ++t) {
          dev = std::max(dev, DtxqcdAbs(ComplexD(S1[l][g][t] - T1[l][g][t])));
          scale = std::max(scale, DtxqcdAbs(S1[l][g][t]));
        }
      for (int g = 0; g < NG0; ++g)
        for (int t = 0; t < T; ++t) {
          dev = std::max(dev, DtxqcdAbs(ComplexD(S0[l][g][t] - T0[l][g][t])));
          scale = std::max(scale, DtxqcdAbs(S0[l][g][t]));
        }
    }
    const double rel = dev / std::max(scale, 1e-300);
    std::cout << GridLogMessage << "[dibaryon GAUGE_TWIST] max|dS|/max|S| = "
              << rel << (rel < 1e-8 ? "   PASSED" : "   FAILED") << std::endl;
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(rel < 1e-8 ? 0 : 1);
  }

  // ---- write -------------------------------------------------------------
  // avgPlaquette is a collective: compute on ALL ranks, write on boss only.
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  if (Grid.IsBoss()) {
    std::string outfile =
        data_dir + "/dibaryon_aux_dtxqcd_" + std::to_string(traj) + ".h5";
    core.WriteH5(outfile, lam, plaq, S1, S0);
    std::cout << GridLogMessage << "wrote " << outfile << std::endl;
  }

  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
