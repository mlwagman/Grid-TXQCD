// TXQCD Möbius rational pseudofermion FD force test.
// Mirrors Test_txqcd_mobius_pf_force.cc but uses the rational
// (M^dag M)^{-1/2} action.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusRationalPseudoFermionAction.h>

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
  }
  grid->GlobalSum(total);
  return total;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt4(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG pRNG4(UGrid); pRNG4.SeedFixedIntegers({7, 8, 9, 10});
  GridParallelRNG pRNG5(FGrid); pRNG5.SeedFixedIntegers({11, 12, 13, 14});
  GridSerialRNG sRNG;            sRNG.SeedFixedIntegers({7, 8, 9, 10});

  TXQCDField U(UGrid);
  SU<Nc>::HotConfiguration(pRNG4, U.U);
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG4, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG4, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG4, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG4, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG4, U.t); U.t     = aux_scale * U.t;

  RealD mass = 0.05, M5 = 1.8, b = 1.5, c = 0.5;
  OneFlavourRationalParams param(/*lo=*/1e-4, /*hi=*/64.0,
                                 /*maxit=*/10000, /*tol=*/1e-10,
                                 /*degree=*/12, /*precision=*/64,
                                 /*BoundsCheckFreq=*/100,
                                 /*mdtol=*/1e-7,
                                 /*BoundsCheckTol=*/1e-4);

  TXQCDMobiusRationalPseudoFermionAction action(*FGrid, *FrbGrid,
                                                 *UGrid, *UrbGrid,
                                                 mass, M5, b, c, param);

  action.refresh(U, sRNG, pRNG5);

  TXQCDField dSdU(UGrid);
  action.deriv(U, dSdU);

  int exitcode = 0;
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
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  { LatticeSigmaField E(UGrid); HermitianGaussian(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.sigma);
    RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h){ X = X + h * E; });
    check("sigma", an, fd); }
  { LatticePiField E(UGrid); HermitianGaussian(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.pi);
    RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h){ X = X + h * E; });
    check("pi", an, fd); }
  { LatticeSFieldC E(UGrid); HermitianGaussian(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.s);
    RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h){ X = X + h * E; });
    check("s", an, fd); }
  { LatticePFieldC E(UGrid); HermitianGaussian(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.p);
    RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h){ X = X + h * E; });
    check("p", an, fd); }
  { LatticeTField E(UGrid); GaussianAntisymTensor(pRNG4, E);
    RealD an = TensorTrInner(E, dSdU.t);
    RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h){ X = X + h * E; });
    check("t", an, fd); }

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
