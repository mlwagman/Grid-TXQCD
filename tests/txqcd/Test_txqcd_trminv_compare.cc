// Test_txqcd_trminv_compare:
//
// On a saved TXQCD config (gauge + aux: σ, π, s, p, t), compute two estimators
// using Hutchinson stochastic noise:
//
//   Σ_TXQCD = Tr[M_TXQCD^{-1}] / (V · N_F)    (with aux fields)
//   Σ_W     = Tr[M_W^{-1}]     / (V · N_F)    (plain Wilson on gauge cfg)
//
// If Fierz holds at the action's saddle, both should equal Σ_QCD measured
// on a pure QCD ensemble at the same params.  Σ_W on the TXQCD ensemble
// is the Fierz-protected quantity (gauge-only observable) — by Fierz
// it should be unchanged from pure QCD.
//
// Analog of Test_dtxqcd_trminv_block_compare for the non-doubled TXQCD
// operator.  Useful to test whether the saddle issue we see in DTXQCD
// (Σ_M48 ≠ Σ_QCD at small λ) is also present in TXQCD where there's no
// charge-conjugation doubling structure.
//
// Env knobs:
//   CFG_DIR   — directory containing ckpoint_lat.<traj> + ckpoint_lat_aux.<traj>
//   CFG_TRAJ  — trajectory number to load
//   MASS      — Wilson mass (default 0.3, matches TxqcdTest2pt::mass)
//   CSW       — clover coefficient (default 1.0, matches TxqcdTest2pt::csw)
//   N_NOISE   — Hutchinson noise count (default 8)
//   CG_TOL    — CG tolerance (default 1e-6)
//
// Run:
//   CFG_DIR=configs_2pt_txqcd_csw1_optlam CFG_TRAJ=300 \
//     ./tests/txqcd/Test_txqcd_trminv_compare --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace Grid;

// Σ measurement convention: /(2V).
//
// Why /(2V): Grid's gaussian(prng, eta) draws complex entries with
//   Re,Im ~ N(0,1)  →  ⟨η_i^* η_j⟩ = 2 δ_ij   (σ²=2 per complex entry).
// Then ⟨innerProduct(η, M^{-1}η)⟩ = 2·Tr[M^{-1}].  Dividing by 2V absorbs
// the σ²=2 factor and the volume V, giving the "per-site" trace:
//   Σ = Tr[M^{-1}] / V = D_per_site / (m+4)   at U=I, kappa→0
// where D_per_site = Ns·Nc = 12 for plain Wilson (1 flavor) so
//   Σ_W = 12/(m+4) = 4·Nc/(m+4)     ← user's free-field expression
// Same convention matches compute_trminv in Test_dtxqcd_2pt_utils.h
// (and the AUX_INIT_AUTO bisection / saddle relations in gencfgs).
static RealD plain_wilson_clover_trminv(LatticeGaugeField &U, GridCartesian &Grid_,
                                         GridRedBlackCartesian &RBGrid,
                                         GridParallelRNG &prng, RealD mass,
                                         RealD csw, int n_noise, RealD cg_tol) {
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(U, Grid_, RBGrid, mass, csw, csw);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, 30000);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid_), b(&Grid_), x(&Grid_);
    gaussian(prng, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / n_noise;
}

static RealD txqcd_op_trminv(TXQCDField &U, GridCartesian &Grid_,
                              GridRedBlackCartesian &RBGrid,
                              GridParallelRNG &prng, RealD mass, RealD csw,
                              int n_noise, RealD cg_tol) {
  std::array<RealD, TxqcdNf> mass_arr;
  for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass;
  TXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const int cg_max = 30000;
  for (int h = 0; h < n_noise; ++h) {
    TXQCDFermionNf eta(&Grid_), b(&Grid_), x(&Grid_);
    TXQCDFermionNf r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(prng, eta.f[a]);
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < TxqcdNf; ++a) {
      r.f[a] = b.f[a];
      p.f[a] = b.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < TxqcdNf; ++a) {
        x.f[a] = x.f[a] + alpha * p.f[a];
        r.f[a] = r.f[a] - alpha * Ap.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta * p.f[a];
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  // /(2V) convention (above) + /N_F → per-quark Σ.  At U=I, kappa→0:
  //   Σ_TXQCD = (1/N_F)·Tr[M_TX^{-1}]/V = (Nf_internal·Ns·Nc)/(N_F·(m+4))
  //           = (2·12)/(2·(m+4)) = 12/(m+4) = 4·Nc/(m+4)
  // matching Σ_W exactly when Fierz holds.
  return acc / (n_noise * (RealD)TxqcdNf);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);

  GridParallelRNG pRNG(&Grid_);
  GridSerialRNG   sRNG;
  pRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});

  RealD mass = 0.3;
  RealD csw  = 1.0;
  int n_noise = 8;
  RealD cg_tol = 1e-6;
  if (const char *m = std::getenv("MASS"); m && *m)    mass = std::atof(m);
  if (const char *c = std::getenv("CSW");  c && *c)    csw  = std::atof(c);
  if (const char *s = std::getenv("N_NOISE"); s && *s) n_noise = std::atoi(s);
  if (const char *t = std::getenv("CG_TOL"); t && *t)  cg_tol  = std::atof(t);

  const char *cfg_dir_c = std::getenv("CFG_DIR");
  if (!cfg_dir_c || !*cfg_dir_c) {
    std::cerr << "ERROR: CFG_DIR env var required" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfg_dir = cfg_dir_c;

  int traj = -1;
  if (const char *t = std::getenv("CFG_TRAJ"); t && *t) traj = std::atoi(t);
  if (traj < 0) {
    std::cerr << "ERROR: CFG_TRAJ env var required" << std::endl;
    Grid_finalize();
    return 1;
  }

  TXQCDField U(&Grid_);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                 cfg_dir + "/ckpoint_lat",
                                 cfg_dir + "/ckpoint_rng", traj);

  std::cout << GridLogMessage
            << "loaded cfg: dir=" << cfg_dir << " traj=" << traj
            << "  mass=" << mass << "  csw=" << csw
            << "  n_noise=" << n_noise << std::endl;

  std::cout << GridLogMessage
            << "aux  ||σ||²/V=" << (norm2(U.sigma) / Grid_.gSites())
            << "  ||π||²/V="    << (norm2(U.pi)    / Grid_.gSites())
            << "  ||s||²/V="    << (norm2(U.s)     / Grid_.gSites())
            << "  ||p||²/V="    << (norm2(U.p)     / Grid_.gSites())
            << "  ||t||²/V="    << (norm2(U.t)     / Grid_.gSites())
            << std::endl;

  // Seed noise with cfg traj number so different cfgs get independent
  // Hutchinson noise — averaging over cfgs then yields a properly
  // mixed estimator (gauge + aux + noise) instead of a fixed-noise
  // realization of (gauge + aux).
  pRNG.SeedFixedIntegers({100 + traj, 200 + traj, 300 + traj, 400 + traj, 500 + traj});
  RealD sigma_txqcd = txqcd_op_trminv(U, Grid_, RBGrid, pRNG, mass, csw,
                                       n_noise, cg_tol);

  pRNG.SeedFixedIntegers({2100 + traj, 2200 + traj, 2300 + traj, 2400 + traj, 2500 + traj});
  RealD sigma_W = plain_wilson_clover_trminv(U.U, Grid_, RBGrid, pRNG, mass,
                                              csw, n_noise, cg_tol);

  std::cout << GridLogMessage << "Σ_TXQCD (with aux)  = " << sigma_txqcd << std::endl;
  std::cout << GridLogMessage << "Σ_W (plain Wilson)  = " << sigma_W
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage
            << "ratio Σ_TXQCD / Σ_W = " << (sigma_txqcd / sigma_W) << std::endl;

  Grid_finalize();
  return 0;
}
