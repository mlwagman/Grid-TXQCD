// Test_dtxqcd_trminv_block_compare:
//
// On a saved DTXQCD config (gauge + aux), compute three estimators
// using Hutchinson stochastic noise on M48:
//
//   Σ_M48     = Tr[M48^{-1}] / (V · 2 · N_F)        (noise in BOTH upper, lower)
//   Σ_upper   = Tr[M_upper^{-1}] / (V · N_F)        (noise in upper only,  lower=0)
//   Σ_lower   = Tr[M_lower^{-1}] / (V · N_F)        (noise in lower only,  upper=0)
//
// If the X → -X symmetry of the action is respected by the saddle (and
// d, n off-diagonals are small or symmetric), then:
//
//   (Σ_upper + Σ_lower) / 2 = Σ_M48                (per-config identity)
//
// and statistically over the ensemble:
//
//   ⟨Σ_upper⟩ = ⟨Σ_M48⟩ = ⟨Σ_lower⟩               (Pfaffian X→-X)
//
// Env knobs:
//   CFG_DIR    — directory containing ckpoint_lat.<traj> + ckpoint_lat_daux.<traj>
//   CFG_TRAJ   — trajectory number to load
//   MASS       — Wilson mass (default 0.3)
//   N_NOISE    — Hutchinson noise count (default 32)
//   CSW        — clover coefficient (default 0.0)
//
// Run:
//   CFG_DIR=configs_2pt_dtxqcd_v2_lam0.1_sweep CFG_TRAJ=50 \
//     ./tests/dtxqcd/Test_dtxqcd_trminv_block_compare --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

using namespace Grid;

// Plain Wilson Σ on gauge cfg alone (aux ignored): Fierz-protected quantity.
// After integrating out aux, the DTXQCD gauge ensemble is supposed to
// equal pure QCD ensemble — so this Σ_W on saved DTXQCD cfgs should
// equal Σ_W on pure QCD cfgs at the same params.
static RealD plain_wilson_trminv(LatticeGaugeField &U, GridCartesian &Grid_,
                                 GridRedBlackCartesian &RBGrid,
                                 GridParallelRNG &prng, RealD mass,
                                 int n_noise, RealD cg_tol) {
  WilsonFermionD Dw(U, Grid_, RBGrid, mass);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
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
  // No /N_F here because this is a single-flavor Wilson; matches the
  // per-block normalization in m48_trminv_masked (mode=UpperOnly).
  return acc / n_noise;
}

enum class NoiseMode { Both, UpperOnly, LowerOnly };

static RealD m48_trminv_masked(DTXQCDField &U, GridCartesian &Grid_,
                               GridRedBlackCartesian &RBGrid,
                               GridParallelRNG &prng, RealD mass, RealD csw,
                               int n_noise, NoiseMode mode) {
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  RealD cg_tol = 1e-6;
  int cg_max = 30000;
  if (const char *t = std::getenv("CG_TOL"); t && *t) cg_tol = std::atof(t);
  if (const char *m = std::getenv("CG_MAX"); m && *m) cg_max = std::atoi(m);
  for (int h = 0; h < n_noise; ++h) {
    DTXQCDFermionDoubled eta(&Grid_), b(&Grid_), x(&Grid_);
    DTXQCDFermionDoubled r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      if (mode == NoiseMode::LowerOnly) {
        eta.upper.f[a] = Zero();
      } else {
        gaussian(prng, eta.upper.f[a]);
      }
      if (mode == NoiseMode::UpperOnly) {
        eta.lower.f[a] = Zero();
      } else {
        gaussian(prng, eta.lower.f[a]);
      }
    }
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < DtxqcdNf; ++a) {
      r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
      p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  // Normalization: each Hutchinson estimate gives Tr[M^{-1}_{αα}] summed
  // over the diagonal entries that the noise covers.
  //   Both:       48 dof per site (Nf×Ns×Nc × 2 blocks) → Tr[M48^{-1}]
  //   UpperOnly:  24 dof per site (Nf×Ns×Nc × upper)    → Tr[M_upper^{-1}]
  //   LowerOnly:  24 dof per site (Nf×Ns×Nc × lower)    → Tr[M_lower^{-1}]
  // To convert to a per-quark Σ (matches plain Wilson on bare gauge at aux=0):
  //   Σ = Tr[M^{-1}] / (V · Nf · dim_block_per_flavor_per_site)
  // where dim_block_per_flavor_per_site = Ns·Nc = 12 for one block, 24 for both.
  // Since we divided by V already and the noise dim cancels out, we just
  // need to normalize by 2·Nf (Both) or Nf (single block) to match.
  RealD denom = (mode == NoiseMode::Both) ? (2.0 * DtxqcdNf) : (RealD)DtxqcdNf;
  return acc / (n_noise * denom);
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
  RealD csw  = 0.0;
  int n_noise = 32;
  if (const char *m = std::getenv("MASS"); m && *m) mass = std::atof(m);
  if (const char *c = std::getenv("CSW"); c && *c)  csw  = std::atof(c);
  if (const char *s = std::getenv("N_NOISE"); s && *s) n_noise = std::atoi(s);

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

  DTXQCDField U(&Grid_);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                 cfg_dir + "/ckpoint_lat",
                                 cfg_dir + "/ckpoint_rng", traj);

  std::cout << GridLogMessage
            << "loaded cfg: dir=" << cfg_dir << " traj=" << traj
            << "  mass=" << mass << "  csw=" << csw
            << "  n_noise=" << n_noise << std::endl;

  // Report aux norms
  std::cout << GridLogMessage
            << "aux  ||σ||²/V=" << (norm2(U.sigma) / Grid_.gSites())
            << "  ||π||²/V="    << (norm2(U.pi)    / Grid_.gSites())
            << "  ||d||²/V="    << (norm2(U.d)     / Grid_.gSites())
            << "  ||n||²/V="    << (norm2(U.n)     / Grid_.gSites())
            << "  ||s||²/V="    << (norm2(U.s)     / Grid_.gSites())
            << "  ||p||²/V="    << (norm2(U.p)     / Grid_.gSites())
            << std::endl;

  // Optional: zero d, n in the loaded aux config — decouples upper/lower
  // blocks of M48.  Then UpperOnly Σ on the modified cfg gives the true
  // Tr[M_upper^{-1}] where M_upper = D_QCD + X (with X = σ + s + (π+p)γ5
  // only).  Useful for testing the user's claim:
  //   Σ_M48 (with d,n) / 2  ≟  Tr[M_upper^{-1}]  (the "non-doubled block")
  bool zero_dn = false;
  if (const char *z = std::getenv("ZERO_DN"); z && std::atoi(z) != 0) {
    zero_dn = true;
    U.d = Zero();
    U.n = Zero();
    std::cout << GridLogMessage << "[ZERO_DN] d, n zeroed → M48 is block-diagonal" << std::endl;
  }

  // INDEPENDENT noise streams for each mode (avoid correlated estimators).
  // Seed depends on cfg traj — different cfgs get independent noise.
  pRNG.SeedFixedIntegers({100 + traj, 200 + traj, 300 + traj, 400 + traj, 500 + traj});
  RealD sigma_full = m48_trminv_masked(U, Grid_, RBGrid, pRNG, mass, csw,
                                       n_noise, NoiseMode::Both);

  pRNG.SeedFixedIntegers({600 + traj, 700 + traj, 800 + traj, 900 + traj, 1000 + traj});
  RealD sigma_upper = m48_trminv_masked(U, Grid_, RBGrid, pRNG, mass, csw,
                                        n_noise, NoiseMode::UpperOnly);

  pRNG.SeedFixedIntegers({1100 + traj, 1200 + traj, 1300 + traj, 1400 + traj, 1500 + traj});
  RealD sigma_lower = m48_trminv_masked(U, Grid_, RBGrid, pRNG, mass, csw,
                                        n_noise, NoiseMode::LowerOnly);

  // Plain Wilson on the gauge cfg ALONE — Fierz-protected quantity.
  // If Fierz holds on the DTXQCD ensemble, this Σ_W should equal Σ_W
  // on a pure QCD ensemble at the same mass + lattice.
  RealD cg_tol_eff = 1e-6;
  if (const char *t = std::getenv("CG_TOL"); t && *t) cg_tol_eff = std::atof(t);
  pRNG.SeedFixedIntegers({2100 + traj, 2200 + traj, 2300 + traj, 2400 + traj, 2500 + traj});
  RealD sigma_W = plain_wilson_trminv(U.U, Grid_, RBGrid, pRNG, mass,
                                       n_noise, cg_tol_eff);

  std::cout << GridLogMessage << "Σ_M48 (both blocks)  = " << sigma_full  << std::endl;
  std::cout << GridLogMessage << "Σ_upper (M24 alone)  = " << sigma_upper << std::endl;
  std::cout << GridLogMessage << "Σ_lower (M24 alone)  = " << sigma_lower << std::endl;
  std::cout << GridLogMessage << "Σ_W (plain Wilson on gauge) = " << sigma_W
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage
            << "(Σ_upper + Σ_lower)/2 = " << ((sigma_upper + sigma_lower) / 2.0)
            << "  (vs Σ_M48 = " << sigma_full
            << ", rel diff = "
            << std::abs((sigma_upper + sigma_lower) / 2.0 - sigma_full)
                 / std::abs(sigma_full)
            << ")" << std::endl;

  Grid_finalize();
  return 0;
}
