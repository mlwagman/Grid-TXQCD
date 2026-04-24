// Force-consistency test for TXQCDWilsonCloverHasenbuschAction (the new
// Hasenbusch rational-ratio monomial used to mass-precondition the TXQCD
// light-quark HMC).  Pattern mirrors Test_txqcd_rational_clover_eo_force:
//
//   1. refresh -> S round-trip consistency (via stored RefreshAction).
//   2. Finite-difference force checks for sigma / pi / s / p / t aux slots
//      and for the gauge link.
//
// Runs at csw=0 first (Wilson-Clover hopping only), and at csw!=0 second
// (the Cmunu clover contribution is a TODO in the Hasenbusch class so the
// csw!=0 gauge/sigma checks are expected to fail until that's added).

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverHasenbuschAction.h>

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

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

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
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG, U.t); U.t     = aux_scale * U.t;

  RealD mass_l = 0.3;
  RealD mass_h = 0.5;   // Hasenbusch shift Δm = +0.2

  OneFlavourRationalParams rat_params(/*lo=*/1e-4, /*hi=*/64.0,
                                      /*maxit=*/10000, /*tol=*/1e-10,
                                      /*degree=*/12, /*precision=*/64,
                                      /*BoundsCheckFreq=*/100,
                                      /*mdtol=*/1e-8,
                                      /*BoundsCheckTol=*/1e-4);

  int exitcode = 0;
  auto check = [&](const char *name, RealD an, RealD fd) {
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    bool pass = rel < 1e-3;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  std::cout << GridLogMessage << "===== Hasenbusch (csw=0) =====" << std::endl;
  {
    TXQCDWilsonCloverHasenbuschAction action(Grid, RBGrid, mass_l, mass_h,
                                              rat_params, /*csw=*/0.0);

    // refresh -> S round-trip
    for (int trial = 0; trial < 3; ++trial) {
      action.refresh(U, sRNG, pRNG);
      RealD S = action.S(U);
      RealD expected = (RealD)TxqcdNf * Nc * Ns * (RealD)RBGrid.gSites();
      RealD sigma = std::sqrt(expected);
      RealD dev = std::abs(S - expected) / sigma;
      bool pass = dev < 5.0;
      std::cout << GridLogMessage << "[refresh-S trial " << trial
                << "] S=" << S << " expected~" << expected
                << " dev=" << dev << (pass ? "  PASS" : "  FAIL") << std::endl;
      if (!pass) exitcode = 1;
    }

    action.refresh(U, sRNG, pRNG);
    TXQCDField dSdU(&Grid);
    action.deriv(U, dSdU);

    auto fd_aux = [&](auto &field_ref, auto perturb) -> RealD {
      const RealD h = 1e-4;
      auto saved = field_ref;
      perturb(field_ref,  h); RealD Sp = action.S(U);
      field_ref = saved;
      perturb(field_ref, -h); RealD Sm = action.S(U);
      field_ref = saved;
      return (Sp - Sm) / (2.0 * h);
    };

    {
      LatticeSigmaField E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, dSdU.sigma);
      RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h*E; });
      check("sigma", an, fd);
    }
    {
      LatticePiField E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, dSdU.pi);
      RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h) { X = X + h*E; });
      check("pi", an, fd);
    }
    {
      LatticeSFieldC E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, dSdU.s);
      RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h) { X = X + h*E; });
      check("s", an, fd);
    }
    {
      LatticePFieldC E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, dSdU.p);
      RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h) { X = X + h*E; });
      check("p", an, fd);
    }
    {
      LatticeTField E(&Grid); GaussianAntisymTensor(pRNG, E);
      RealD an = TensorTrInner(E, dSdU.t);
      RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h) { X = X + h*E; });
      check("t", an, fd);
    }

    // Gauge FD check
    {
      std::array<LatticeColourMatrix, 4> Emu{LatticeColourMatrix(&Grid),
          LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
          LatticeColourMatrix(&Grid)};
      for (int mu = 0; mu < Nd; ++mu)
        SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

      RealD an = 0;
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
        an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
      }
      an *= -2.0;

      const RealD h = 1e-4;
      LatticeGaugeField Usaved = U.U;
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], h, 12) * Umu, mu);
      }
      RealD Sp = action.S(U);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], -h, 12) * Umu, mu);
      }
      RealD Sm = action.S(U);
      U.U = Usaved;

      RealD fd = (Sp - Sm) / (2.0 * h);
      check("gauge", an, fd);
    }
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME HASENBUSCH FORCE CHECKS FAILED (csw=0)"
                         : "ALL HASENBUSCH FORCE CHECKS PASSED (csw=0)")
            << std::endl;

  // Self-consistency: at Δm=0 the ratio operator is identity, so the action
  // is U-independent and all forces should be numerically tiny.
  std::cout << GridLogMessage << "===== Sanity: Δm=0 (csw=1) =====" << std::endl;
  {
    TXQCDWilsonCloverHasenbuschAction zero_dm(Grid, RBGrid, mass_l, mass_l,
                                                rat_params, /*csw=*/1.0);
    zero_dm.refresh(U, sRNG, pRNG);
    TXQCDField dSdU0(&Grid);
    zero_dm.deriv(U, dSdU0);
    std::cout << GridLogMessage
              << "  |dSdU.U|    = " << std::sqrt(norm2(dSdU0.U)) << std::endl;
    std::cout << GridLogMessage
              << "  |dSdU.sigma|= " << std::sqrt(norm2(dSdU0.sigma)) << std::endl;
    std::cout << GridLogMessage
              << "  |dSdU.t|    = " << std::sqrt(norm2(dSdU0.t)) << std::endl;
  }

  std::cout << GridLogMessage << "===== Hasenbusch (csw=1) =====" << std::endl;
  {
    int exit_csw = 0;
    auto check_csw = [&](const char *name, RealD an, RealD fd) {
      RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
      bool pass = rel < 1e-3;
      std::cout << GridLogMessage << "[" << name << "] AN=" << an
                << " FD=" << fd << " rel=" << rel
                << (pass ? "  PASS" : "  FAIL") << std::endl;
      if (!pass) exit_csw = 1;
    };
    TXQCDWilsonCloverHasenbuschAction caction(Grid, RBGrid, mass_l, mass_h,
                                                rat_params, /*csw=*/1.0);
    caction.refresh(U, sRNG, pRNG);
    TXQCDField cdSdU(&Grid);
    caction.deriv(U, cdSdU);

    auto fd_c = [&](auto &field_ref, auto perturb) -> RealD {
      const RealD h = 1e-4;
      auto saved = field_ref;
      perturb(field_ref,  h); RealD Sp = caction.S(U);
      field_ref = saved;
      perturb(field_ref, -h); RealD Sm = caction.S(U);
      field_ref = saved;
      return (Sp - Sm) / (2.0 * h);
    };

    {
      LatticeSigmaField E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, cdSdU.sigma);
      RealD fd = fd_c(U.sigma, [&](LatticeSigmaField &X, RealD h) { X = X + h*E; });
      check_csw("sigma", an, fd);
    }
    {
      LatticePiField E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, cdSdU.pi);
      RealD fd = fd_c(U.pi, [&](LatticePiField &X, RealD h) { X = X + h*E; });
      check_csw("pi", an, fd);
    }
    {
      LatticeSFieldC E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, cdSdU.s);
      RealD fd = fd_c(U.s, [&](LatticeSFieldC &X, RealD h) { X = X + h*E; });
      check_csw("s", an, fd);
    }
    {
      LatticePFieldC E(&Grid); HermitianGaussian(pRNG, E);
      RealD an = HermitianTrInner(E, cdSdU.p);
      RealD fd = fd_c(U.p, [&](LatticePFieldC &X, RealD h) { X = X + h*E; });
      check_csw("p", an, fd);
    }
    {
      LatticeTField E(&Grid); GaussianAntisymTensor(pRNG, E);
      RealD an = TensorTrInner(E, cdSdU.t);
      RealD fd = fd_c(U.t, [&](LatticeTField &X, RealD h) { X = X + h*E; });
      check_csw("t", an, fd);
    }

    // Gauge FD check (csw=1)
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
      an *= -2.0;
      const RealD h = 1e-4;
      LatticeGaugeField Usaved = U.U;
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], h, 12) * Umu, mu);
      }
      RealD Sp = caction.S(U);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
        PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], -h, 12) * Umu, mu);
      }
      RealD Sm = caction.S(U);
      U.U = Usaved;
      RealD fd = (Sp - Sm) / (2.0 * h);
      check_csw("gauge", an, fd);
    }

    std::cout << GridLogMessage
              << (exit_csw ? "SOME HASENBUSCH FORCE CHECKS FAILED (csw=1)"
                           : "ALL HASENBUSCH FORCE CHECKS PASSED (csw=1)")
              << std::endl;
    if (exit_csw) exitcode = 1;
  }

  Grid_finalize();
  return exitcode;
}
