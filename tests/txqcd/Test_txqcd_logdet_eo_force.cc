// Force-consistency test for TXQCDLogDetEOAction.
//
// 1. Verify S = -ln det(Mee) gives finite, reasonable value
// 2. Finite-difference force check for aux (sigma, pi, s, p, t)
// 3. Gauge force is zero (Mee doesn't depend on gauge links)
// 4. Odd-site force is zero (Mee depends only on even-site aux fields)

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetEOAction.h>

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
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu)
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            total += real(Reduce(Ev[ss]()(mu, nu)(i, j) *
                                 Fv[ss]()(mu, nu)(j, i)));
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
  TXQCDLogDetEOAction action(Grid, RBGrid, mass);

  int exitcode = 0;

  // ---- 1. Action value ----
  RealD S = action.S(U);
  {
    bool pass = std::isfinite(S);
    std::cout << GridLogMessage << "[action value] S=" << S
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ---- 2. Compute analytic force ----
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
    bool pass = rel < 1e-6;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  // ---- FD checks for each aux field ----
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

  // ---- 3. Gauge force should be zero (csw=0) ----
  {
    RealD nU = std::sqrt(norm2(dSdU.U));
    bool pass = nU == 0.0;
    std::cout << GridLogMessage << "[gauge force zero] |dSdU.U|=" << nU
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ---- 4. Odd-site force should be zero ----
  {
    LatticeSigmaField F_odd(&RBGrid);
    pickCheckerboard(Odd, F_odd, dSdU.sigma);
    RealD n_odd = std::sqrt(norm2(F_odd));
    bool pass = n_odd == 0.0;
    std::cout << GridLogMessage << "[sigma odd=0] |F_odd|=" << n_odd
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME LOGDET FORCE CHECKS FAILED"
                         : "ALL LOGDET FORCE CHECKS PASSED")
            << std::endl;

  // ===== Part 2: csw != 0 — clover gauge force FD test =====
  std::cout << GridLogMessage << "===== Clover LogDet force (csw=1.0) =====" << std::endl;
  {
    RealD csw = 1.0;
    TXQCDLogDetEOAction caction(Grid, RBGrid, mass, csw);

    RealD cS = caction.S(U);
    std::cout << GridLogMessage << "[clover action value] S=" << cS << std::endl;

    TXQCDField cdSdU(&Grid);
    caction.deriv(U, cdSdU);

    // Aux force FD checks (same structure, now with clover)
    {
      LatticeSigmaField E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, cdSdU.sigma);
      RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h * E; });
      auto saved_action = &action;
      // Need to use caction for FD
      auto fd2 = [&](auto &field_ref, auto perturb) -> RealD {
        const RealD h = 1e-4;
        auto saved = field_ref;
        perturb(field_ref,  h);  RealD Sp = caction.S(U);
        field_ref = saved;
        perturb(field_ref, -h);  RealD Sm = caction.S(U);
        field_ref = saved;
        return (Sp - Sm) / (2.0 * h);
      };
      fd = fd2(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h * E; });
      check("clover sigma", an, fd);
    }

// Gauge force FD: perturb U_mu(x) -> exp(h * E_mu(x)) U_mu(x)
    {
      std::array<LatticeColourMatrix, 4> Emu{LatticeColourMatrix(&Grid),
          LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
          LatticeColourMatrix(&Grid)};
      for (int mu = 0; mu < Nd; ++mu)
        SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

      RealD an = 0;
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(cdSdU.U, mu);
        an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
      }

      const RealD h = 1e-4;
      LatticeGaugeField Usaved = U.U;

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        LatticeColourMatrix expE = expMat(Emu[mu], h, 12);
        PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
      }
      RealD Sp = caction.S(U);

      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        LatticeColourMatrix expE = expMat(Emu[mu], -h, 12);
        PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
      }
      RealD Sm = caction.S(U);

      U.U = Usaved;

      RealD fd = (Sp - Sm) / (2.0 * h);
      check("clover gauge", an, fd);
    }

    // Gauge force should be non-zero
    {
      RealD nU = std::sqrt(norm2(cdSdU.U));
      bool pass = nU > 1e-10;
      std::cout << GridLogMessage << "[clover gauge force nonzero] |dSdU.U|=" << nU
                << (pass ? "  PASS" : "  FAIL") << std::endl;
      if (!pass) exitcode = 1;
    }

    std::cout << GridLogMessage
              << (exitcode ? "SOME CLOVER LOGDET FORCE CHECKS FAILED"
                           : "ALL CLOVER LOGDET FORCE CHECKS PASSED")
              << std::endl;
  }
  Grid_finalize();
  return exitcode;
}
