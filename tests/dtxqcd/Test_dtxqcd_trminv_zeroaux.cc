// Test_dtxqcd_trminv_zeroaux: verify the M48-based Σ measurement
// (compute_trminv in Test_dtxqcd_2pt_utils.h) gives the right ratio
// vs the plain-Wilson Σ at aux = 0.
//
// At aux = 0 the doubled M48 decouples into 2 (upper, lower) × N_f
// independent copies of the QCD Wilson Dirac operator (the lower
// block uses conjugate(U); same spectrum and same Tr M^-1 as the
// upper).  So:
//
//   Tr M48^{-1} = 2 · N_f · Tr D_QCD^{-1}      (= 4·Tr D for N_f=2)
//
// With the SAME normalization 1/(2V) applied to both the new
// (M48-based) and old (plain Wilson) compute_trminv, the new value
// must be exactly  2 · N_f = 4  times the old one at aux = 0.
//
// If the ratio is anything else, the 1/(2V) factor or the noise
// dimensionality is wrong somewhere.  Note that the same custom-CG
// pattern with the same /(2V) factor is used in the AUX_INIT_AUTO
// bisection in Test_dtxqcd_2pt_gencfgs, so a normalization bug here
// would also be a bug there.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_trminv_zeroaux --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

using namespace Grid;

static RealD plain_wilson_trminv(LatticeGaugeField &U, GridCartesian &Grid_,
                                  GridRedBlackCartesian &RBGrid,
                                  GridParallelRNG &prng, RealD mass,
                                  int n_noise) {
  WilsonFermionD Dw(U, Grid_, RBGrid, mass);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(1e-10, 30000);
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

static RealD m48_trminv(DTXQCDField &U, GridCartesian &Grid_,
                        GridRedBlackCartesian &RBGrid,
                        GridParallelRNG &prng, RealD mass,
                        int n_noise) {
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, /*csw=*/0.0,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const RealD cg_tol = 1e-10;
  for (int h = 0; h < n_noise; ++h) {
    DTXQCDFermionDoubled eta(&Grid_), b(&Grid_), x(&Grid_);
    DTXQCDFermionDoubled r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(prng, eta.upper.f[a]);
      gaussian(prng, eta.lower.f[a]);
    }
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < DtxqcdNf; ++a) {
      r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
      p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    for (int k = 0; k < 30000; ++k) {
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
  return acc / n_noise;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);

  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({1, 2, 3, 4, 5});

  RealD mass = 0.3;
  if (const char *m = std::getenv("MASS"); m && *m) mass = std::atof(m);
  int n_noise = 16;
  if (const char *s = std::getenv("N_NOISE"); s && *s) n_noise = std::atoi(s);

  // Random weak-field gauge (same recipe as TepidConfiguration would use).
  LatticeGaugeField U(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, U);

  DTXQCDField Udt(&Grid_);
  Udt.U = U;
  Udt.sigma = Zero();
  Udt.pi    = Zero();
  Udt.d     = Zero();
  Udt.n     = Zero();
  Udt.s     = Zero();
  Udt.p     = Zero();

  std::cout << GridLogMessage
            << "lattice = " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3]
            << "  mass = " << mass
            << "  n_noise = " << n_noise << std::endl;

  // Each noise iteration uses an independent draw from the pRNG.  To
  // make plain-Wilson and M48 estimators use INDEPENDENT noise streams
  // (not the same), reseed between the two — otherwise the comparison
  // would be confounded by correlated noise.
  pRNG.SeedFixedIntegers({100, 200, 300, 400, 500});
  RealD sigma_old = plain_wilson_trminv(U, Grid_, RBGrid, pRNG, mass, n_noise);

  pRNG.SeedFixedIntegers({600, 700, 800, 900, 1000});
  RealD sigma_new = m48_trminv(Udt, Grid_, RBGrid, pRNG, mass, n_noise);

  RealD expected_ratio = 2.0 * DtxqcdNf;   // = 4 for Nf=2
  RealD observed_ratio = sigma_new / sigma_old;
  RealD rel_err = std::abs(observed_ratio - expected_ratio) / expected_ratio;

  std::cout << GridLogMessage << "plain Wilson Σ        = " << sigma_old << std::endl;
  std::cout << GridLogMessage << "M48 (aux=0) Σ          = " << sigma_new << std::endl;
  std::cout << GridLogMessage << "ratio M48/Wilson       = " << observed_ratio << std::endl;
  std::cout << GridLogMessage << "expected ratio 2·N_f   = " << expected_ratio << std::endl;
  std::cout << GridLogMessage << "relative error         = " << rel_err << std::endl;

  // Tolerance: stochastic Hutchinson at fixed n_noise has fractional
  // error ~ 1/sqrt(n_noise * Vol * dof).  For n_noise=16 on 4^3 × 8 with
  // dof=12 (plain) or 48 (M48), fractional error ~ 0.01 — set tol = 0.05
  // for comfortable margin.
  bool ok = rel_err < 0.05;
  std::cout << GridLogMessage
            << (ok ? "[ok]" : "[FAIL]")
            << " ratio matches 2·N_f within " << (rel_err * 100) << "%"
            << std::endl;

  Grid_finalize();
  return ok ? 0 : 1;
}
