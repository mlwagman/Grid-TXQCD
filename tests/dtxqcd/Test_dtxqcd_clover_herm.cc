// Test_dtxqcd_clover_herm: Hermiticity of the upper and lower clover terms
// in the DTXQCD doubled M_ee.
//
// Builds anti-Hermitian random F_{mu,nu}(x) at each (mu<nu) pair (mimicking
// Grid's clover field strength convention), then checks:
//   1. Upper-block clover Hermiticity:
//        <w, D_clover_upper v> = conj(<v, D_clover_upper w>)
//   2. Lower-block clover Hermiticity (same condition).
//   3. Upper != lower in general (the difference confirms the sigma^T-vs-sigma
//      distinction between C D_qcd C^T and D_qcd at the clover term).
//   4. csw = 0 nullity: both blocks return zero.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_clover_herm --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({71, 72, 73, 74});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // Random anti-Hermitian F_{mu,nu} per pair: F = X - adj(X) with X complex
  // Gaussian.  6 pairs in lex order (XY, XZ, XT, YZ, YT, ZT).
  std::vector<LatticeColourMatrix> FS;
  FS.reserve(6);
  for (int k = 0; k < 6; ++k) {
    LatticeColourMatrix X(&Grid);
    gaussian(pRNG, X);
    LatticeColourMatrix F(&Grid);
    F = X - adj(X);
    FS.push_back(std::move(F));
  }
  // Verify F is anti-Hermitian (paranoia)
  {
    RealD worst = 0.0;
    for (auto &F : FS) {
      LatticeColourMatrix sum(&Grid);
      sum = F + adj(F);
      worst = std::max(worst, std::sqrt(norm2(sum)));
    }
    check("F + adj(F) (should be 0)", worst, 1e-12);
  }

  // Random doubled-fermion test vectors (per flavor a).
  DTXQCDFermionNf v(&Grid), w(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  const RealD csw = 1.25;

  // ---------- 1. Upper-block Hermiticity ----------
  {
    DTXQCDFermionNf Dv(&Grid), Dw(&Grid);
    DtxqcdApplyCloverUpper(csw, FS, v, Dv);
    DtxqcdApplyCloverUpper(csw, FS, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    RealD resid = std::abs(wDv - std::conj(vDw));
    RealD ref = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    std::cout << GridLogMessage << "  upper: <w,Dv> = " << wDv
              << "  conj(<v,Dw>) = " << std::conj(vDw) << std::endl;
    check("upper-block Hermiticity (rel)", resid / ref, 1e-12);
  }

  // ---------- 2. Lower-block Hermiticity ----------
  {
    DTXQCDFermionNf Dv(&Grid), Dw(&Grid);
    DtxqcdApplyCloverLower(csw, FS, v, Dv);
    DtxqcdApplyCloverLower(csw, FS, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    RealD resid = std::abs(wDv - std::conj(vDw));
    RealD ref = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    std::cout << GridLogMessage << "  lower: <w,Dv> = " << wDv
              << "  conj(<v,Dw>) = " << std::conj(vDw) << std::endl;
    check("lower-block Hermiticity (rel)", resid / ref, 1e-12);
  }

  // ---------- 3. Upper != lower (sanity) ----------
  {
    DTXQCDFermionNf Up(&Grid), Lo(&Grid);
    DtxqcdApplyCloverUpper(csw, FS, v, Up);
    DtxqcdApplyCloverLower(csw, FS, v, Lo);
    DTXQCDFermionNf diff(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) diff.f[a] = Up.f[a] - Lo.f[a];
    RealD up_norm   = std::sqrt(norm2(Up));
    RealD lo_norm   = std::sqrt(norm2(Lo));
    RealD diff_norm = std::sqrt(norm2(diff));
    std::cout << GridLogMessage << "  ||upper v|| = " << up_norm
              << "  ||lower v|| = " << lo_norm
              << "  ||upper - lower|| = " << diff_norm << std::endl;
    if (diff_norm < 1e-10 * std::max(up_norm, lo_norm)) {
      std::cout << GridLogError << "[FAIL] upper and lower clover agree to 1e-10"
                << " — sigma^T vs sigma distinction may be lost" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "[ok] upper and lower differ as expected"
                << std::endl;
    }
  }

  // ---------- 4. csw = 0 nullity ----------
  {
    DTXQCDFermionNf z_up(&Grid), z_lo(&Grid);
    DtxqcdApplyCloverUpper(0.0, FS, v, z_up);
    DtxqcdApplyCloverLower(0.0, FS, v, z_lo);
    check("csw=0 upper nullity ||.||^2", norm2(z_up), 1e-25);
    check("csw=0 lower nullity ||.||^2", norm2(z_lo), 1e-25);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
