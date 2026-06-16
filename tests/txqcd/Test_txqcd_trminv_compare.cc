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
static WilsonImplR::ImplParams AntiPeriodicTimeBC() {
  WilsonImplR::ImplParams p;
  p.boundary_phases.resize(Nd, 1.0);
  p.boundary_phases[Nd - 1] = -1.0;
  return p;
}

static RealD plain_wilson_clover_trminv(LatticeGaugeField &U, GridCartesian &Grid_,
                                         GridRedBlackCartesian &RBGrid,
                                         GridParallelRNG &prng, RealD mass,
                                         RealD csw, int n_noise, RealD cg_tol) {
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WilsonImplR::ImplParams ap = AntiPeriodicTimeBC();
  if (const char *p = std::getenv("PBC_TIME"); p && std::atoi(p) != 0) {
    ap.boundary_phases[Nd - 1] = 1.0;
  }
  WCF Dw(U, Grid_, RBGrid, mass, csw, csw, WilsonAnisotropyCoefficients(), ap);
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
  // Default BC = APBC time (matches TXQCDWilsonCloverFermionEO default).
  // PBC_TIME=1 → use periodic in all directions (matches TXQCDWilsonOp / HMC).
  WilsonImplR::ImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  if (const char *p = std::getenv("PBC_TIME"); p && std::atoi(p) != 0) {
    impl_p.boundary_phases[Nd - 1] = 1.0;
    std::cout << GridLogMessage << "PBC_TIME=1 → Σ_TX with periodic time" << std::endl;
  }
  TXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw,
                                 impl_p);
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

  // ZERO_AUX=1 — zero the loaded aux to test M_TX|aux=0 == D_W on 2 copies.
  if (const char *z = std::getenv("ZERO_AUX"); z && std::atoi(z) != 0) {
    U.sigma = Zero(); U.pi = Zero(); U.s = Zero(); U.p = Zero(); U.t = Zero();
    std::cout << GridLogMessage << "ZERO_AUX=1 → all aux zeroed" << std::endl;
  }

  // OP_DIFF=1 — apply M_TX and D_W to the same vector, compare element-wise.
  if (const char *od = std::getenv("OP_DIFF"); od && std::atoi(od) != 0) {
    std::array<RealD, TxqcdNf> mass_arr;
    for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass;
    TXQCDWilsonCloverFermionEO Mtx(U.U, Grid_, RBGrid, mass_arr,
                                    U.sigma, U.pi, U.s, U.p, U.t, csw);
    WilsonFermionD Mw(U.U, Grid_, RBGrid, mass);

    TXQCDFermionNf in_tx(&Grid_), out_tx(&Grid_);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, in_tx.f[a]);
    Mtx.M(in_tx, out_tx);

    // Apply Wilson D_W to each flavor independently.
    LatticeFermion out_w0(&Grid_), out_w1(&Grid_);
    Mw.M(in_tx.f[0], out_w0);
    Mw.M(in_tx.f[1], out_w1);

    LatticeFermion diff0 = out_tx.f[0] - out_w0;
    LatticeFermion diff1 = out_tx.f[1] - out_w1;
    RealD n_tx0 = std::sqrt(norm2(out_tx.f[0]));
    RealD n_w0  = std::sqrt(norm2(out_w0));
    RealD n_d0  = std::sqrt(norm2(diff0));
    RealD n_d1  = std::sqrt(norm2(diff1));
    std::cout << GridLogMessage << "OP_DIFF: ||M_TX·η_0|| = " << n_tx0
              << "   ||D_W·η_0|| = " << n_w0
              << "   ||M_TX·η_0 - D_W·η_0|| = " << n_d0
              << "   rel = " << (n_d0/n_w0)
              << "\n             flavor 1 rel = " << (n_d1/n_w0)
              << std::endl;

    // Direct test: TXQCD's internal Dw_ (mass=0 Wilson) vs fresh WilsonFermion(mass=0)
    WilsonFermionD Mw0(U.U, Grid_, RBGrid, 0.0);
    LatticeFermion out_dwm0(&Grid_), out_mw0(&Grid_);
    Mtx.Wilson().M(in_tx.f[0], out_dwm0);  // TXQCD's internal Wilson kernel
    Mw0.M(in_tx.f[0], out_mw0);
    LatticeFermion diff_w = out_dwm0 - out_mw0;
    RealD rel_w = std::sqrt(norm2(diff_w) / norm2(out_mw0));
    std::cout << GridLogMessage
              << "Dw_inner vs WilsonFermion(m=0) rel = " << rel_w
              << "   (||Dw_inner|| = " << std::sqrt(norm2(out_dwm0))
              << ", ||WilsonFermion|| = " << std::sqrt(norm2(out_mw0)) << ")"
              << std::endl;

    // Decomposition: out_tx = Dw_.M(η) + mass*η + Delta(aux=0)·η = Dw_.M(η) + mass*η
    // Subtract Dw_.M and mass*η from out_tx — what's left should be zero if Delta(aux=0)=0
    LatticeFermion out_check(&Grid_);
    out_check = out_tx.f[0] - out_dwm0 - mass * in_tx.f[0];
    RealD residual = std::sqrt(norm2(out_check));
    std::cout << GridLogMessage
              << "Residual ||M_TX·η - Dw·η - mass·η|| = " << residual
              << "  (should be 0 at aux=0)" << std::endl;
    Grid_finalize();
    return 0;
  }

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

  // Spatial-mean trace VEVs (real + imag) for σ, π, s, p
  {
    RealD V = (RealD)Grid_.gSites();
    auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
    auto tr_pi    = TensorRemove(sum(trace(U.pi)));
    auto tr_s     = TensorRemove(sum(trace(U.s)));
    auto tr_p     = TensorRemove(sum(trace(U.p)));
    std::cout << GridLogMessage << "VEV  ⟨Tr σ⟩ = "
              << (tr_sigma.real()/V) << "  + i*" << (tr_sigma.imag()/V)
              << "\n         ⟨Tr π⟩ = "
              << (tr_pi.real()/V) << "  + i*" << (tr_pi.imag()/V)
              << "\n         ⟨Tr s⟩ = "
              << (tr_s.real()/V) << "  + i*" << (tr_s.imag()/V)
              << "\n         ⟨Tr p⟩ = "
              << (tr_p.real()/V) << "  + i*" << (tr_p.imag()/V)
              << std::endl;
  }

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
