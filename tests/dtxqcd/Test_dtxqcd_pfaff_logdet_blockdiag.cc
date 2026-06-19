// Test_dtxqcd_pfaff_logdet_blockdiag: validate the doubled-space 48x48 site
// matrix assembly by checking that with d = n = 0 the off-diagonal blocks
// vanish and log|det(M48)| = log|det(M_upper)| + log|det(M_lower)|.
//
// Uses DTXQCDSiteMatrix.h for the per-site assembly so the test exercises
// the same code path the EO operator (Phase 2 continuation) will use to
// precompute Mooee.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_pfaff_logdet_blockdiag --grid 4.4.4.4 --mpi 1.1.1.1

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
  pRNG.SeedFixedIntegers({11, 22, 33, 44});

  int exitcode = 0;
  auto check = [&](const char* name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DtxqcdSpinMatrices spin(Grid);
  {
    RealD herm = (spin.gamma5 - spin.gamma5.adjoint()).norm();
    RealD sq   = (spin.gamma5 * spin.gamma5 - Eigen::Matrix4cd::Identity()).norm();
    check("gamma5 Hermiticity", herm, 1e-12);
    check("gamma5^2 = I",       sq,   1e-12);
  }

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

  Coordinate site0(std::vector<int>{0, 0, 0, 0});
  DtxqcdSiteAux aux = DtxqcdSiteAux::Extract(sigma, pi, d, n, s, p, site0);

  const double mass = 0.3;
  MatrixXcd M_upper, M_lower, M_offdiag_full, M_offdiag_zero;
  DtxqcdBuildUpperBlock24(mass, aux, spin, M_upper);
  DtxqcdBuildLowerBlock24(mass, aux, spin, M_lower);
  DtxqcdBuildOffDiagBlock24(aux, spin, M_offdiag_full);

  // Build a "zeroed d, n" aux by copying and zeroing d, n.
  DtxqcdSiteAux aux_zero = aux;
  aux_zero.d.setZero();
  aux_zero.n.setZero();
  DtxqcdBuildOffDiagBlock24(aux_zero, spin, M_offdiag_zero);

  check("M_upper Hermiticity (rel)",
        (M_upper - M_upper.adjoint()).norm() / M_upper.norm(), 1e-12);
  check("M_lower Hermiticity (rel)",
        (M_lower - M_lower.adjoint()).norm() / M_lower.norm(), 1e-12);
  check("offdiag(d=n=0) norm", M_offdiag_zero.norm(), 1e-14);
  check("M_off Hermiticity (rel)",
        (M_offdiag_full - M_offdiag_full.adjoint()).norm()
            / std::max(M_offdiag_full.norm(), 1.0), 1e-12);

  // In v2 (corrected 2026-06-12), both M_upper and M_lower carry +X with
  // the same sigma/pi/s/p insertion.  After the real-symmetric projection
  // of sigma, pi (2026-06-13) the lower block has no transpose flip: the
  // doubled action's antisymmetry (K·M48)^T = -(K·M48) is carried entirely
  // by the block-swap K, not by X^T = -X.  So M_upper - M_lower must be
  // exactly zero from the aux contribution alone (csw=0 here, no clover).
  RealD upper_lower_rel = (M_upper - M_lower).norm()
                        / std::max(M_upper.norm(), 1.0);
  std::cout << GridLogMessage << "||M_upper - M_lower|| / ||M_upper|| = "
            << upper_lower_rel
            << "  (v2: both carry +X; should be ~0 at csw=0)" << std::endl;
  check("M_upper == M_lower at csw=0 (rel)", upper_lower_rel, 1e-12);

  MatrixXcd M48_with_dn, M48_no_dn;
  DtxqcdAssembleDoubled48(M_upper, M_lower, M_offdiag_full, M48_with_dn);
  DtxqcdAssembleDoubled48(M_upper, M_lower, M_offdiag_zero, M48_no_dn);
  check("M48(d,n!=0) Hermiticity (rel)",
        (M48_with_dn - M48_with_dn.adjoint()).norm() / M48_with_dn.norm(),
        1e-12);

  RealD ld_upper   = DtxqcdLogAbsDet48(M_upper);  // works for 24x24 too
  RealD ld_lower   = DtxqcdLogAbsDet48(M_lower);
  RealD ld_48_with = DtxqcdLogAbsDet48(M48_with_dn);
  RealD ld_48_no   = DtxqcdLogAbsDet48(M48_no_dn);

  std::cout << GridLogMessage << "log|det(M_upper)|      = " << ld_upper << std::endl;
  std::cout << GridLogMessage << "log|det(M_lower)|      = " << ld_lower << std::endl;
  std::cout << GridLogMessage << "log|det(M48 d,n!=0)|   = " << ld_48_with << std::endl;
  std::cout << GridLogMessage << "log|det(M48 d=n=0)|    = " << ld_48_no   << std::endl;
  std::cout << GridLogMessage << "upper + lower          = " << (ld_upper + ld_lower) << std::endl;

  RealD residual = std::abs(ld_48_no - (ld_upper + ld_lower));
  RealD ref = std::max(std::abs(ld_48_no), std::abs(ld_upper + ld_lower)) + 1.0;
  check("block-diag log|det| (rel)", residual / ref, 1e-10);

  RealD diff_dn = std::abs(ld_48_with - ld_48_no);
  std::cout << GridLogMessage << "|log|det(d,n!=0)| - log|det(d=n=0)|| = "
            << diff_dn << std::endl;
  if (diff_dn < 1e-6) {
    std::cout << GridLogError << "[FAIL] d, n had no effect on M48 log det"
              << " — off-diagonal assembly may be broken" << std::endl;
    exitcode = 1;
  } else {
    std::cout << GridLogMessage << "[ok] d, n nonzero changes M48 log det"
              << std::endl;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
