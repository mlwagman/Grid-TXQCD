// Test_dtxqcd_site_lu: round-trip + identity tests for the DTXQCD per-site
// 48x48 LU factorization.
//
// Verifies, on a 4^4 lattice scanning every lattice site:
//   1. M48 * Inv(M48) = I (within Eigen LU tolerance).
//   2. log|det(M48)| from the LU equals log|det| via a direct Eigen call —
//      basic sanity for the LU machinery.
//   3. With d = n = 0, log|det(M48)| equals 2 * log|det(M_upper)| at every
//      site (block-diagonal property).
//
// This is the per-site LU functionality the EO operator will use to
// precompute Mooee^{-1} once per ImportAux.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_site_lu --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>

using namespace Grid;
using Eigen::MatrixXcd;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({101, 202, 303, 404});

  int exitcode = 0;

  DtxqcdSpinMatrices spin(Grid);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d(&Grid);
  LatticeDtxqcdN     n(&Grid);
  LatticeDtxqcdS     s(&Grid);
  LatticeDtxqcdP     p(&Grid);
  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s);
  DtxqcdRealScalarGaussian(pRNG, p);
  DtxqcdHermitianCFGaussian(pRNG, d);
  DtxqcdHermitianCFGaussian(pRNG, n);

  const double mass = 0.5;  // safely-invertible diagonal shift

  // Walk every site (lex order) on the local grid; build M48, factor, invert,
  // compare round-trip.  Accumulate worst-case residuals.
  RealD worst_inv = 0.0;
  RealD worst_logdet = 0.0;
  RealD worst_blockdiag = 0.0;

  Coordinate gd(Grid.GlobalDimensions());
  Coordinate coord(Nd, 0);
  for (int x = 0; x < gd[0]; ++x) {
    for (int y = 0; y < gd[1]; ++y) {
      for (int z = 0; z < gd[2]; ++z) {
        for (int tt = 0; tt < gd[3]; ++tt) {
          coord = Coordinate(std::vector<int>{x, y, z, tt});

          DtxqcdSiteAux aux = DtxqcdSiteAux::Extract(sigma, pi, d, n, s, p, coord);

          MatrixXcd M_upper, M_lower, M_off, M48;
          DtxqcdBuildUpperBlock24(mass, aux, spin, M_upper);
          DtxqcdBuildLowerBlock24(mass, aux, spin, M_lower);
          DtxqcdBuildOffDiagBlock24(aux, spin, M_off);
          DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);

          // 1. LU round-trip: M48 * Inv(M48) = I
          auto lu = DtxqcdLU48(M48);
          MatrixXcd Inv = lu.inverse();
          MatrixXcd I = MatrixXcd::Identity(kDtxqcdSiteDim48, kDtxqcdSiteDim48);
          RealD r_inv = (M48 * Inv - I).norm();
          worst_inv = std::max(worst_inv, r_inv);

          // 2. LU log|det| matches direct Eigen log|det|
          RealD ld_lu     = std::log(std::abs(lu.determinant()));
          RealD ld_direct = DtxqcdLogAbsDet48(M48);
          worst_logdet = std::max(worst_logdet,
                                  std::abs(ld_lu - ld_direct));

          // 3. Block-diagonal log|det| with d=n=0 at this site
          DtxqcdSiteAux aux0 = aux;
          aux0.d.setZero();
          aux0.n.setZero();
          MatrixXcd M_off0, M48_0;
          DtxqcdBuildOffDiagBlock24(aux0, spin, M_off0);
          DtxqcdAssembleDoubled48(M_upper, M_lower, M_off0, M48_0);
          RealD ld_48_0 = DtxqcdLogAbsDet48(M48_0);
          RealD ld_up   = DtxqcdLogAbsDet48(M_upper);
          RealD ld_lo   = DtxqcdLogAbsDet48(M_lower);
          worst_blockdiag = std::max(worst_blockdiag,
                                     std::abs(ld_48_0 - (ld_up + ld_lo)));
        }
      }
    }
  }

  auto check = [&](const char* name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " worst = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  check("LU round-trip ||M48*Inv - I||",         worst_inv,       1e-9);
  check("LU log|det| vs direct log|det|",        worst_logdet,    1e-10);
  check("block-diag log|det| (d=n=0 per site)",  worst_blockdiag, 1e-10);

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
