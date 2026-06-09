// Validate the analytic per-site LU MooeeInv against the PCG oracle.
//
//   1. LU round-trip: LU.Apply(Mooee(x)) ≈ x   (machine precision)
//   2. LU vs PCG: LU.Apply(b) ≈ TXQCDMobiusFermionEO::MooeeInv(b)
//   3. same for the Dag solve
//
// The PCG MooeeInv is the exact oracle; the LU is built by probing the
// already-FD-validated Mooee, so this gates the index/layout plumbing.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusSiteLU.h>

using namespace Grid;

static RealD relerr(const TXQCDFermionNf &a, const TXQCDFermionNf &b) {
  RealD num = 0, den = 0;
  for (int f = 0; f < TxqcdNf; ++f) {
    LatticeFermion d(a.f[f].Grid()); d.Checkerboard() = a.f[f].Checkerboard();
    d = a.f[f] - b.f[f];
    num += norm2(d); den += norm2(a.f[f]);
  }
  return std::sqrt(num) / std::sqrt(std::max(den, 1e-30));
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt4(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG pRNG4(UGrid); pRNG4.SeedFixedIntegers({7, 8, 9, 10});
  GridParallelRNG pRNG5(FGrid); pRNG5.SeedFixedIntegers({11, 12, 13, 14});

  LatticeGaugeField Umu(UGrid);
  SU<Nc>::HotConfiguration(pRNG4, Umu);
  LatticeSigmaField sigma(UGrid); LatticePiField pi(UGrid);
  LatticeSFieldC s(UGrid);        LatticePFieldC p(UGrid);
  LatticeTField  t(UGrid);
  // aux scale via env (default 0.05; HMC sees std ~ 1/lambda)
  const char *as_env = std::getenv("AUX_SCALE");
  const RealD as = (as_env && *as_env) ? std::atof(as_env) : 0.05;
  std::cout << GridLogMessage << "[lu test] AUX_SCALE=" << as << std::endl;
  HermitianGaussian(pRNG4, sigma); sigma = as * sigma;
  HermitianGaussian(pRNG4, pi);    pi    = as * pi;
  HermitianGaussian(pRNG4, s);     s     = as * s;
  HermitianGaussian(pRNG4, p);     p     = as * p;
  GaussianAntisymTensor(pRNG4, t); t     = as * t;

  RealD mass = 0.05, M5 = 1.8, b = 1.5, c = 0.5;
  TXQCDMobiusFermionEO Meo(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                           mass, M5, b, c, sigma, pi, s, p, t);

  TXQCDMobiusSiteLU lu(FrbGrid, Ls);
  lu.Build([&](const TXQCDFermionNf &i, TXQCDFermionNf &o) { Meo.Mooee(i, o); },
           Odd);
  std::cout << GridLogMessage << "[lu] built per-site blocks" << std::endl;

  // random odd 5D rb field
  TXQCDFermionNf x(FrbGrid), bb(FrbGrid), y(FrbGrid), yo(FrbGrid);
  for (int f = 0; f < TxqcdNf; ++f) {
    LatticeFermion tmp(FGrid); gaussian(pRNG5, tmp);
    x.f[f].Checkerboard() = Odd;  pickCheckerboard(Odd, x.f[f], tmp);
    bb.f[f].Checkerboard() = Odd; y.f[f].Checkerboard() = Odd;
    yo.f[f].Checkerboard() = Odd;
  }

  int ec = 0;
  // 1. round trip
  Meo.Mooee(x, bb);
  lu.Apply(bb, y, /*dag=*/false);
  { RealD e = relerr(x, y);
    bool ok = e < 1e-9;
    std::cout << GridLogMessage << "Test 1 (LU.Apply(Mooee x) == x): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  // 2. LU vs PCG oracle
  Meo.MooeeInv(bb, yo);
  { RealD e = relerr(yo, y);
    bool ok = e < 1e-7;
    std::cout << GridLogMessage << "Test 2 (LU vs PCG MooeeInv): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  // 3. dag round trip + vs PCG
  Meo.MooeeDag(x, bb);
  lu.Apply(bb, y, /*dag=*/true);
  { RealD e = relerr(x, y);
    bool ok = e < 1e-9;
    std::cout << GridLogMessage << "Test 3 (LU.Apply^dag(MooeeDag x)==x): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  Meo.MooeeInvDag(bb, yo);
  lu.Apply(bb, y, /*dag=*/true);
  { RealD e = relerr(yo, y);
    bool ok = e < 1e-7;
    std::cout << GridLogMessage << "Test 4 (LU^dag vs PCG MooeeInvDag): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }

  // ---- EVEN cb: the cb Schur Mpc actually uses for Mee^{-1} ----
  TXQCDMobiusSiteLU lu_e(FrbGrid, Ls);
  lu_e.Build([&](const TXQCDFermionNf &i, TXQCDFermionNf &o){ Meo.Mooee(i, o); },
             Even);
  std::cout << GridLogMessage << "[lu] built Even per-site blocks" << std::endl;
  TXQCDFermionNf xe(FrbGrid), bbe(FrbGrid), ye(FrbGrid), yoe(FrbGrid);
  for (int f = 0; f < TxqcdNf; ++f) {
    LatticeFermion tmp(FGrid); gaussian(pRNG5, tmp);
    xe.f[f].Checkerboard() = Even;  pickCheckerboard(Even, xe.f[f], tmp);
    bbe.f[f].Checkerboard() = Even; ye.f[f].Checkerboard() = Even;
    yoe.f[f].Checkerboard() = Even;
  }
  Meo.Mooee(xe, bbe);
  lu_e.Apply(bbe, ye, /*dag=*/false);
  { RealD e = relerr(xe, ye); bool ok = e < 1e-9;
    std::cout << GridLogMessage << "Test 5 (Even LU.Apply(Mooee x)==x): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  Meo.MooeeInv(bbe, yoe);
  { RealD e = relerr(yoe, ye); bool ok = e < 1e-7;
    std::cout << GridLogMessage << "Test 6 (Even LU vs PCG MooeeInv): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  Meo.MooeeDag(xe, bbe);
  lu_e.Apply(bbe, ye, /*dag=*/true);
  { RealD e = relerr(xe, ye); bool ok = e < 1e-9;
    std::cout << GridLogMessage << "Test 7 (Even LU^dag(MooeeDag x)==x): " << e
              << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }
  Meo.MooeeInvDag(bbe, yoe);
  lu_e.Apply(bbe, ye, /*dag=*/true);
  { RealD e = relerr(yoe, ye); bool ok = e < 1e-7;
    std::cout << GridLogMessage << "Test 8 (Even LU^dag vs PCG MooeeInvDag): "
              << e << (ok ? "  PASS" : "  FAIL") << std::endl; if (!ok) ec = 1; }

  std::cout << GridLogMessage
            << (ec ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED") << std::endl;
  Grid_finalize();
  return ec;
}
