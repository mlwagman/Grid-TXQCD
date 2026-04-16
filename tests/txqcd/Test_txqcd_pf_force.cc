// TXQCD Wilson pseudofermion action: heatbath consistency + finite-difference
// force consistency on every auxiliary slot.
//
//   1. Heatbath: directly draw chi ~ N(0, 1/sqrt 2) per flavor, set
//      Phi = M^dag chi, then verify S(U) == ||chi||^2 to CG tolerance.
//   2. For each aux slot X in {sigma, pi, s, p, t}: pick a random Hermitian
//      direction E (antisymmetric in (mu,nu) for t), evaluate
//        FD = (S(X + h E) - S(X - h E)) / (2 h)
//      and compare to the analytic contraction
//        AN = sum_x Tr( E(x) F_X(x) )
//      with F_X = deriv()'s output for the X slot. Both are real because
//      F_X is Hermitian and E is Hermitian (and the action is real).
//   3. Gauge slot: smoke-check that dSdU.U has nonzero norm.
//
// Lattice 4^4, mass = 0.3 (well inside positive-definite regime), aux drawn
// from HermitianGaussian/GaussianAntisymTensor at full unit variance.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

using namespace Grid;

// Real Frobenius inner product Tr(E F) summed over sites for a Hermitian
// flavor-matrix field. Imaginary part should be ~FP noise.
static RealD HermitianTrInner(const LatticeSigmaField &E,
                              const LatticeSigmaField &F) {
  ComplexD acc = TensorRemove(sum(trace(E * F)));
  return acc.real();
}

static RealD HermitianTrInner(const LatticeSFieldC &E,
                              const LatticeSFieldC &F) {
  ComplexD acc = TensorRemove(sum(trace(E * F)));
  return acc.real();
}

// Sum over (mu<nu) of Re Tr(E_{mu,nu} F_{mu,nu}). One factor per independent
// antisym pair, matching the operator's explicit (mu<nu) loop in
// ApplyDeltaColor.
static RealD TensorTrInner(const LatticeTField &E, const LatticeTField &F) {
  GridBase *grid = E.Grid();
  RealD total = 0.0;
  autoView(Ev, E, CpuRead);
  autoView(Fv, F, CpuRead);
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            total += real(Reduce(Ev[ss]()(mu, nu)(i, j) *
                                 Fv[ss]()(mu, nu)(j, i)));
          }
        }
      }
    }
  });
  grid->GlobalSum(total);
  return total;
}

template <class Field>
static void HermRandom(GridParallelRNG &pRNG, Field &E) {
  HermitianGaussian(pRNG, E);
}
static void TensorRandom(GridParallelRNG &pRNG, LatticeTField &E) {
  GaussianAntisymTensor(pRNG, E);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});  // 2^4 too small for SIMD on this build

  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG sRNG;
  pRNG.SeedFixedIntegers({7, 8, 9, 10});
  sRNG.SeedFixedIntegers({7, 8, 9, 10});

  TXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  // Scale aux down so the Delta insertion is well inside the positive-
  // definite regime of M^dag M; this also tames the large cubic curvature
  // of S(aux) that would otherwise dominate central-difference truncation.
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG, U.t); U.t     = aux_scale * U.t;

  RealD mass = 0.3;
  // Tight CG tol so that S is reproducible to better than h^2 ~ 1e-6.
  TXQCDWilsonPseudoFermionAction action(Grid, RBGrid, mass, 1e-15, 50000);

  int exitcode = 0;

  // ---------------- 1. heatbath consistency ----------------
  {
    TXQCDFermionNf chi(&Grid);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, chi.f[a]);
      chi.f[a] = scale * chi.f[a];
    }
    TXQCDWilsonOp Mop(U.U, Grid, RBGrid, mass, U.sigma, U.pi, U.s, U.p, U.t);
    Mop.Mdag(chi, action.PseudoFermion());
    RealD chi2 = norm2(chi);
    RealD S    = action.S(U);
    RealD rel  = std::abs(S - chi2) / std::max(chi2, 1.0);
    bool pass  = rel < 1e-9;
    std::cout << GridLogMessage << "[heatbath] ||chi||^2=" << chi2
              << " S=" << S << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Now seed Phi via the action's own refresh for the FD tests.
  action.refresh(U, sRNG, pRNG);

  TXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  // Helper: central-difference S along an aux-field direction.
  auto fd_aux = [&](auto &field_ref, auto perturb) -> RealD {
    const RealD h = 1e-4;
    auto saved = field_ref;  // copy
    perturb(field_ref,  h);  RealD Sp = action.S(U);
    field_ref = saved;
    perturb(field_ref, -h);  RealD Sm = action.S(U);
    field_ref = saved;
    return (Sp - Sm) / (2.0 * h);
  };

  // ---------------- 2. FD vs analytic per aux slot ----------------
  auto check = [&](const char *name, RealD an, RealD fd) {
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    bool pass = rel < 1e-3;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  // sigma
  {
    LatticeSigmaField E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.sigma);
    RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h * E; });
    check("sigma", an, fd);
  }
  // pi
  {
    LatticePiField E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.pi);
    RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h) { X = X + h * E; });
    check("pi", an, fd);
  }
  // s
  {
    LatticeSFieldC E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.s);
    RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h) { X = X + h * E; });
    check("s", an, fd);
  }
  // p
  {
    LatticePFieldC E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.p);
    RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h) { X = X + h * E; });
    check("p", an, fd);
  }
  // t (antisym in mu,nu, Hermitian per (mu,nu) block)
  {
    LatticeTField E(&Grid); TensorRandom(pRNG, E);
    RealD an = TensorTrInner(E, dSdU.t);
    RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h) { X = X + h * E; });
    check("t", an, fd);
  }

  // ---------------- 3. gauge force smoke ----------------
  {
    RealD nU = std::sqrt(norm2(dSdU.U));
    bool pass = nU > 0.0 && std::isfinite(nU);
    std::cout << GridLogMessage << "[gauge force] |dSdU.U|=" << nU
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
