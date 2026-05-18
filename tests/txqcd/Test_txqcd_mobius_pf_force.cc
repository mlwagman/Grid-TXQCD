// TXQCD Möbius pseudofermion: heatbath consistency + finite-difference force
// validation on every aux slot, plus a smoke check on the gauge force.
//
// Mirrors Test_txqcd_pf_force.cc.  The action is on a 5D fermion grid (Ls × 4D);
// aux fields are 4D.  All FD steps perturb 4D aux fields (or 4D gauge); the
// 5D propagator solve is internal to S(U).

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusPseudoFermionAction.h>

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

template <class Field>
static void HermRandom(GridParallelRNG &pRNG, Field &E) {
  HermitianGaussian(pRNG, E);
}
static void TensorRandom(GridParallelRNG &pRNG, LatticeTField &E) {
  GaussianAntisymTensor(pRNG, E);
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
  TXQCDMobiusPseudoFermionAction action(*FGrid, *FrbGrid, *UGrid, *UrbGrid,
                                         mass, M5, b, c, /*tol=*/1e-15,
                                         /*maxit=*/50000);

  int exitcode = 0;

  // ---------------- 1. heatbath consistency: φ = M† χ ⇒ S = ‖χ‖² ----------------
  {
    TXQCDFermionNf chi(FGrid);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG5, chi.f[a]);
      chi.f[a] = scale * chi.f[a];
    }
    TXQCDMobiusOp Mop(U.U, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                      mass, M5, b, c, U.sigma, U.pi, U.s, U.p, U.t);
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

  // Seed Phi via the action's own refresh for the FD tests.
  action.refresh(U, sRNG, pRNG5);

  TXQCDField dSdU(UGrid);
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
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  // sigma
  { LatticeSigmaField E(UGrid); HermRandom(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.sigma);
    RealD fd = fd_aux(U.sigma, [&](LatticeSigmaField &X, RealD h){ X = X + h * E; });
    check("sigma", an, fd); }
  // pi
  { LatticePiField E(UGrid); HermRandom(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.pi);
    RealD fd = fd_aux(U.pi, [&](LatticePiField &X, RealD h){ X = X + h * E; });
    check("pi", an, fd); }
  // s
  { LatticeSFieldC E(UGrid); HermRandom(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.s);
    RealD fd = fd_aux(U.s, [&](LatticeSFieldC &X, RealD h){ X = X + h * E; });
    check("s", an, fd); }
  // p
  { LatticePFieldC E(UGrid); HermRandom(pRNG4, E);
    RealD an = HermitianTrInner(E, dSdU.p);
    RealD fd = fd_aux(U.p, [&](LatticePFieldC &X, RealD h){ X = X + h * E; });
    check("p", an, fd); }
  // t
  { LatticeTField E(UGrid); TensorRandom(pRNG4, E);
    RealD an = TensorTrInner(E, dSdU.t);
    RealD fd = fd_aux(U.t, [&](LatticeTField &X, RealD h){ X = X + h * E; });
    check("t", an, fd); }

  // gauge force smoke
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
