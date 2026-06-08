// Test_dtxqcd_spectrum: estimate lambda_max and lambda_min of
// Mpc^dag Mpc for the doubled Wilson-Clover operator on a typical
// HMC cold-start config (weak gauge + thermal aux at Var = 1/lambda^2).
//
// Reads the same physics knobs as the production driver (MASS_LIGHT_DTXQCD,
// CSW, LAMBDA_DTXQCD, default LATT) so the result directly informs
// RHMC_LO / RHMC_HI for Test_dtxqcd_2pt_gencfgs and gen_dtxqcd_cfgs.
//
// Approach:
//   lambda_max  via power iteration on Mpc^dag Mpc (Rayleigh quotient
//               after ~200 iterations; converges quickly since the
//               spectrum is well-spaced for Wilson on weak gauge).
//   lambda_min  via inverse iteration: each step solves
//               (Mpc^dag Mpc) v_{k+1} = v_k by CG and renormalises.
//               The CG itself uses Mop.M / Mop.Mdag to form Mpc^dag Mpc,
//               with a moderate tolerance (the goal is an order-of-
//               magnitude eigenvalue estimate, not a precise solve).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_spectrum --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>

using namespace Grid;

namespace {
inline RealD env_real(const char *name, RealD def) {
  if (const char *s = std::getenv(name); s && *s) return std::atof(s);
  return def;
}
inline int env_int(const char *name, int def) {
  if (const char *s = std::getenv(name); s && *s) return std::atoi(s);
  return def;
}

// Apply A = Mpc^dag Mpc to a doubled odd-CB fermion: y = M(M(x)) via Mop.
inline void ApplyA(DTXQCDMpcOp &Mop, const DTXQCDFermionDoubled &in,
                   DTXQCDFermionDoubled &out) {
  DTXQCDFermionDoubled tmp(in.Grid());
  Mop.M(in, tmp);
  Mop.Mdag(tmp, out);
}

inline void Scale(DTXQCDFermionDoubled &x, RealD s) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = s * x.upper.f[a];
    x.lower.f[a] = s * x.lower.f[a];
  }
}

// Set x = src on Odd CB.
inline void RandomOdd(GridRedBlackCartesian &rbgrid, GridParallelRNG &pRNG,
                      DTXQCDFermionDoubled &x) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, x.upper.f[a]);
    gaussian(pRNG, x.lower.f[a]);
    x.upper.f[a].Checkerboard() = Odd;
    x.lower.f[a].Checkerboard() = Odd;
  }
}

inline RealD RayleighQuotient(DTXQCDMpcOp &Mop, const DTXQCDFermionDoubled &v) {
  DTXQCDFermionDoubled Av(v.Grid());
  ApplyA(Mop, v, Av);
  return real(innerProduct(v, Av)) / norm2(v);
}
}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1313, 1314, 1315, 1316});

  const RealD mass   = env_real("MASS_LIGHT_DTXQCD", 0.3);
  const RealD csw    = env_real("CSW",               0.0);
  const RealD lambda = env_real("LAMBDA_DTXQCD",     3.0);
  const double wf    = env_real("WF_SCALE",          0.1);
  const int    nmax  = env_int ("N_MAX_ITERS",       200);
  const int    nmin  = env_int ("N_MIN_ITERS",       20);
  const RealD  invtol = env_real("INV_TOL",          1.0e-6);
  const int    invmax = env_int ("INV_MAX",          5000);

  std::cout << GridLogMessage
            << "Spectrum probe: mass=" << mass << " csw=" << csw
            << " lambda=" << lambda << " wf_scale=" << wf << std::endl;

  // Weak gauge + thermal aux init.
  DTXQCDField U(&Grid);
  DTXQCDCompositeImpl::ThermalAuxConfiguration(pRNG, U, lambda, wf);

  std::cout << GridLogMessage
            << "||sigma||^2/V=" << norm2(U.sigma) / Grid.gSites()
            << "  ||pi||^2/V=" << norm2(U.pi) / Grid.gSites()
            << "  ||t||^2/V=" << norm2(U.t) / Grid.gSites()
            << "  ||d||^2/V=" << norm2(U.d) / Grid.gSites()
            << "  ||n||^2/V=" << norm2(U.n) / Grid.gSites()
            << "  (expected ~3/lambda^2=" << 3.0 / (lambda * lambda) << ")"
            << std::endl;

  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid, RBGrid, mass, csw,
                                  U.sigma, U.pi, U.t, U.d, U.n);
  DTXQCDMpcOp Mop(Dw);

  // ---- lambda_max via power iteration ----
  DTXQCDFermionDoubled v(&RBGrid), Av(&RBGrid);
  RandomOdd(RBGrid, pRNG, v);
  Scale(v, 1.0 / std::sqrt(norm2(v)));

  RealD lam_max = 0.0;
  for (int k = 0; k < nmax; ++k) {
    ApplyA(Mop, v, Av);
    lam_max = real(innerProduct(v, Av));
    Scale(Av, 1.0 / std::sqrt(norm2(Av)));
    v = Av;
    if ((k % 20) == 19) {
      std::cout << GridLogMessage << " power iter " << (k + 1)
                << "  lam_max = " << lam_max << std::endl;
    }
  }
  RealD final_lam_max = RayleighQuotient(Mop, v);
  std::cout << GridLogMessage << "lambda_max(Mpc^dag Mpc) ≈ "
            << final_lam_max << std::endl;

  // ---- lambda_min via inverse iteration ----
  // Implement a tiny single-RHS CG on Mpc^dag Mpc for the inverse step.
  // No Grid LinearOperatorBase wiring needed -- we already have ApplyA.
  DTXQCDFermionDoubled w(&RBGrid), x(&RBGrid), r(&RBGrid), p(&RBGrid),
                       Ap(&RBGrid);
  RandomOdd(RBGrid, pRNG, w);
  Scale(w, 1.0 / std::sqrt(norm2(w)));

  RealD lam_min = 0.0;
  for (int k = 0; k < nmin; ++k) {
    // Solve (Mpc^dag Mpc) x = w by CG.
    x = w;  // initial guess
    ApplyA(Mop, x, Ap);
    for (int a = 0; a < DtxqcdNf; ++a) {
      r.upper.f[a] = w.upper.f[a] - Ap.upper.f[a];
      r.lower.f[a] = w.lower.f[a] - Ap.lower.f[a];
      r.upper.f[a].Checkerboard() = Odd;
      r.lower.f[a].Checkerboard() = Odd;
      p.upper.f[a] = r.upper.f[a];
      p.lower.f[a] = r.lower.f[a];
    }
    RealD rsq = norm2(r);
    RealD w2  = norm2(w);
    int iter = 0;
    for (; iter < invmax && rsq > invtol * invtol * w2; ++iter) {
      ApplyA(Mop, p, Ap);
      ComplexD pAp = innerProduct(p, Ap);
      RealD alpha = rsq / real(pAp);
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD rsq_new = norm2(r);
      RealD beta = rsq_new / rsq;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      rsq = rsq_new;
    }
    Scale(x, 1.0 / std::sqrt(norm2(x)));
    w = x;
    lam_min = RayleighQuotient(Mop, w);
    std::cout << GridLogMessage << " inv iter " << (k + 1)
              << "  CG iters=" << iter
              << "  lam_min ≈ " << lam_min << std::endl;
  }

  std::cout << GridLogMessage << "---------- summary ----------" << std::endl;
  std::cout << GridLogMessage << "lambda_min(Mpc^dag Mpc) ≈ " << lam_min
            << std::endl;
  std::cout << GridLogMessage << "lambda_max(Mpc^dag Mpc) ≈ " << final_lam_max
            << std::endl;
  std::cout << GridLogMessage << "Suggested rational bracket: lo ≈ "
            << 0.5 * lam_min << "  hi ≈ " << 1.5 * final_lam_max << std::endl;

  Grid_finalize();
  return 0;
}
