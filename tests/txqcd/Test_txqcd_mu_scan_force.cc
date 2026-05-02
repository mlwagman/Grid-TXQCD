// TXQCD mu-parameter validation: AN/FD force agreement at mu != 1.
//
// Uses TXQCDWilsonPseudoFermionAction (same structure as Test_txqcd_pf_force)
// but instantiates the action and operator with a runtime mu sourced from env
// var MU (default 1.0).  Runs the heatbath consistency check and the per-aux
// FD-vs-analytic force checks at the specified mu.
//
// Validates that when M_TXQCD = M_QCD + mu * Delta, both S(U) and dS/daux
// remain consistent for any mu.  Passes regression at mu=1, and verifies the
// chain-rule scaling for mu != 1.
//
// Driver script: scan MU=0.5,1.0,2.0 and check ALL CHECKS PASSED at each.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>

using namespace Grid;

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
static RealD TensorTrInner(const LatticeTField &E, const LatticeTField &F) {
  GridBase *grid = E.Grid();
  RealD total = 0.0;
  autoView(Ev, E, CpuRead);
  autoView(Fv, F, CpuRead);
  for (uint64_t ss = 0; ss < grid->oSites(); ++ss) {
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            total += real(Reduce(Ev[ss]()(mu, nu)(i, j) *
                                 Fv[ss]()(mu, nu)(j, i)));
  }
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

  RealD mu = 1.0;
  if (const char *v = std::getenv("MU")) mu = std::atof(v);
  std::cout << GridLogMessage << "[mu-scan] testing mu=" << mu << std::endl;

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
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
  // Scale aux by 1/mu so the effective Delta amplitude (mu*aux) is fixed
  // across mu values.  This keeps the operator's curvature constant so that
  // central-difference truncation error stays in the same regime; the test
  // exercises the chain-rule mu factor on AN/FD agreement.
  const RealD aux_scale = 0.2 / std::max(mu, 0.5);
  HermitianGaussian(pRNG, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG, U.t); U.t     = aux_scale * U.t;

  RealD mass = 0.3;
  TXQCDWilsonPseudoFermionAction action(Grid, RBGrid, mass, 1e-15, 50000, mu);

  int exitcode = 0;

  // ---- 1. heatbath consistency: Phi = M^dag chi, then S(U) == ||chi||^2 ----
  {
    TXQCDFermionNf chi(&Grid);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, chi.f[a]);
      chi.f[a] = scale * chi.f[a];
    }
    TXQCDWilsonOp Mop(U.U, Grid, RBGrid, mass, U.sigma, U.pi, U.s, U.p, U.t,
                      mu);
    Mop.Mdag(chi, action.PseudoFermion());
    RealD chi2 = norm2(chi);
    RealD S    = action.S(U);
    RealD rel  = std::abs(S - chi2) / std::max(chi2, 1.0);
    bool pass  = rel < 1e-9;
    std::cout << GridLogMessage << "[heatbath] mu=" << mu
              << " ||chi||^2=" << chi2 << " S=" << S << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  action.refresh(U, sRNG, pRNG);

  TXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  auto fd_aux = [&](auto &field_ref, auto perturb) -> RealD {
    const RealD h = 1e-4;
    auto saved = field_ref;
    perturb(field_ref,  h);  RealD Sp = action.S(U);
    field_ref = saved;
    perturb(field_ref, -h);  RealD Sm = action.S(U);
    field_ref = saved;
    return (Sp - Sm) / (2.0 * h);
  };

  auto check = [&](const char *name, RealD an, RealD fd) {
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    bool pass = rel < 1e-3;
    std::cout << GridLogMessage << "[" << name << "] mu=" << mu
              << " AN=" << an << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  {
    LatticeSigmaField E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.sigma);
    RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h * E; });
    check("sigma", an, fd);
  }
  {
    LatticePiField E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.pi);
    RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h) { X = X + h * E; });
    check("pi", an, fd);
  }
  {
    LatticeSFieldC E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.s);
    RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h) { X = X + h * E; });
    check("s", an, fd);
  }
  {
    LatticePFieldC E(&Grid); HermRandom(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.p);
    RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h) { X = X + h * E; });
    check("p", an, fd);
  }
  {
    LatticeTField E(&Grid); TensorRandom(pRNG, E);
    RealD an = TensorTrInner(E, dSdU.t);
    RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h) { X = X + h * E; });
    check("t", an, fd);
  }

  {
    RealD nU = std::sqrt(norm2(dSdU.U));
    bool pass = nU > 0.0 && std::isfinite(nU);
    std::cout << GridLogMessage << "[gauge force] mu=" << mu
              << " |dSdU.U|=" << nU
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage << "[mu-scan] mu=" << mu
            << (exitcode ? "  SOME CHECKS FAILED" : "  ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
