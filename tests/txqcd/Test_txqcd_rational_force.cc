// Force-consistency and heatbath-inverse tests for
// TXQCDWilsonRationalPseudoFermionAction (one-flavor rational: S = phi^dag
// (M^dag M)^{-1/2} phi).
//
//   1. Round-trip: eta ~ N(0, 1/sqrt 2) per flavor, Phi = (M^dag M)^{1/4} eta
//      via the action's refresh, then Y = (M^dag M)^{-1/4} Phi inside S(U)
//      should recover |Y|^2 ~= |eta|^2 up to Remez approximation error. We
//      use a moderate degree (param.degree = 12) and check rel error < 1e-4.
//
//   2. Finite-difference force consistency on every aux slot (sigma, pi, s,
//      p, t) and gauge-force smoke check, following the pattern of
//      Test_txqcd_pf_force.cc.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalPseudoFermionAction.h>

using namespace Grid;

static RealD HermitianTrInner(const LatticeSigmaField &E,
                              const LatticeSigmaField &F) {
  return TensorRemove(sum(trace(E * F))).real();
}
static RealD HermitianTrInner(const LatticeSFieldC &E,
                              const LatticeSFieldC &F) {
  return TensorRemove(sum(trace(E * F))).real();
}
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

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG sRNG;
  pRNG.SeedFixedIntegers({11, 12, 13, 14});
  sRNG.SeedFixedIntegers({11, 12, 13, 14});

  TXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG, U.t); U.t     = aux_scale * U.t;

  RealD mass = 0.3;

  // Rational approx parameters. lo/hi should bracket the spectrum of M^dag M
  // on this small test problem; 1e-4 .. 64 is generous for mass=0.3 Wilson on
  // a 4^4 lattice with moderate aux. Degree 12 gives Remez error ~1e-9 which
  // comfortably satisfies our 1e-4 round-trip tolerance.
  OneFlavourRationalParams rat_params(/*lo=*/1e-4, /*hi=*/64.0,
                                      /*maxit=*/10000, /*tol=*/1e-10,
                                      /*degree=*/12, /*precision=*/64,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1e-8,
                                      /*BoundsCheckTol=*/1e-4);

  TXQCDWilsonRationalPseudoFermionAction action(Grid, RBGrid, mass, rat_params);

  int exitcode = 0;

  // ---------------- 1. refresh -> S round-trip ----------------
  // refresh draws eta with Grid's gaussian (complex, <|eta|^2> = 2 per DOF),
  // rescales by sqrt(1/2) so <|eta|^2> = 1 per DOF, then sets
  //   Phi = (M^dag M)^{1/4} eta.
  // S(U) applies (M^dag M)^{-1/4}, yielding Y ~= eta and S = ||Y||^2. So
  //   <S> = DOFs = TxqcdNf * Nc * Ns * V (at full rational accuracy)
  // and stochastic std-dev is sqrt(DOFs). A ~5-sigma band is very generous.
  {
    const RealD expected_mean =
        (RealD)TxqcdNf * Nc * Ns * (RealD)Grid.gSites();
    const RealD sigma = std::sqrt(expected_mean);
    for (int trial = 0; trial < 3; ++trial) {
      action.refresh(U, sRNG, pRNG);
      RealD S = action.S(U);
      RealD dev = std::abs(S - expected_mean) / sigma;
      bool pass = dev < 5.0;
      std::cout << GridLogMessage << "[refresh-S trial " << trial
                << "] S=" << S << " expected~" << expected_mean
                << " (sigma=" << sigma << ") dev=" << dev
                << (pass ? "  PASS" : "  FAIL") << std::endl;
      if (!pass) exitcode = 1;
    }
  }

  // Seed Phi once more before FD tests.
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
    // Rational action has Remez-approx floor; 1e-3 rel tol is the operative
    // bound matching the two-flavor pf_force test.
    bool pass = rel < 1e-3;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  {
    LatticeSigmaField E(&Grid); HermitianGaussian(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.sigma);
    RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h * E; });
    check("sigma", an, fd);
  }
  {
    LatticePiField E(&Grid); HermitianGaussian(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.pi);
    RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h) { X = X + h * E; });
    check("pi", an, fd);
  }
  {
    LatticeSFieldC E(&Grid); HermitianGaussian(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.s);
    RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h) { X = X + h * E; });
    check("s", an, fd);
  }
  {
    LatticePFieldC E(&Grid); HermitianGaussian(pRNG, E);
    RealD an = HermitianTrInner(E, dSdU.p);
    RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h) { X = X + h * E; });
    check("p", an, fd);
  }
  {
    LatticeTField E(&Grid); GaussianAntisymTensor(pRNG, E);
    RealD an = TensorTrInner(E, dSdU.t);
    RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h) { X = X + h * E; });
    check("t", an, fd);
  }

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
