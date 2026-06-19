// Test_dtxqcd_delta_diag_lower (sigmaHerm 2026-06-19): validate that
// DtxqcdApplyDeltaDiagLower applies the joint-transposed aux X^T (σ^T,
// π^T) under DTXQCD_SIGMA_PI_HERMITIAN_ONLY, and that the lower block
// is Hermitian on a Hermitian σ, π configuration.
//
// History:
//   v1 / early v2 (2026-06-12): both blocks carried +X (raw, no
//   transpose), valid under real-symmetric aux projection.
//   sigmaHerm (2026-06-19): σ, π are Hermitian (full DOFs preserved for
//   H-S decoupling); M_lower carries +X^T (joint color+flavor transpose).
//   When σ, π happen to be REAL-symmetric, X^T = X and the two blocks
//   coincide — that's the gate we now test.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({91, 92, 93, 94});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DTXQCDFermionNf v(&Grid), w(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdS     s(&Grid);
  LatticeDtxqcdP     p(&Grid);
  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s);
  DtxqcdRealScalarGaussian(pRNG, p);

  // ---------- 1. upper == lower on REAL-symmetric aux only -----------
  // Under sigmaHerm, M_lower applies +X^T (joint color+flavor transpose
  // of σ, π).  For complex Hermitian aux, σ^T = σ̄ ≠ σ, so the two block
  // applications differ in their imaginary parts — that's required for
  // Pfaffian antisymmetry (gate covered by Test_dtxqcd_pfaffian_antisymmetry).
  // We exhibit the consistency here on REAL-symmetric aux, where σ^T = σ.
  {
    LatticeDtxqcdSigma sigma_rs(sigma);
    LatticeDtxqcdPi    pi_rs(pi);
    DtxqcdRealSymmetricCFInPlace(sigma_rs);
    DtxqcdRealSymmetricCFInPlace(pi_rs);

    DTXQCDFermionNf up(&Grid), lo(&Grid);
    DtxqcdApplyDeltaDiag     (sigma_rs, pi_rs, s, p, v, up);
    DtxqcdApplyDeltaDiagLower(sigma_rs, pi_rs, s, p, v, lo);

    DTXQCDFermionNf diff(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) diff.f[a] = up.f[a] - lo.f[a];

    RealD up_norm   = std::sqrt(norm2(up));
    RealD diff_norm = std::sqrt(norm2(diff));
    std::cout << GridLogMessage << "X-insert (real-symm aux): ||up|| = " << up_norm
              << "  ||up - lo|| = " << diff_norm
              << "  (should be ~0; X^T = X on real-symm aux)" << std::endl;
    check("upper - lower ~ 0 on real-symm aux (rel)",
          diff_norm / std::max(up_norm, 1e-30), 1e-13);
  }

  // ---------- 2. Lower-block Hermiticity on the same aux config -----------
  {
    DTXQCDFermionNf Dv(&Grid), Dw(&Grid);
    DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, v, Dv);
    DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    RealD resid = std::abs(wDv - std::conj(vDw));
    RealD refnorm = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    check("Delta_diag_lower Hermiticity (rel)", resid / refnorm, 1e-12);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
