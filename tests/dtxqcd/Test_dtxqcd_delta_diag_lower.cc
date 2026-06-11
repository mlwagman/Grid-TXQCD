// Test_dtxqcd_delta_diag_lower (v2): validate that DtxqcdApplyDeltaDiagLower
// (-X insertion for the lower block) is exactly the negation of
// DtxqcdApplyDeltaDiag (+X insertion for the upper block) on any aux
// configuration, and that the lower block is Hermitian.
//
// In v2 there is no separate tensor channel: M_lower carries -X^{ij}_{ab}
// uniformly across sigma, pi, s, p; the X^T = X identity (gamma5^T = gamma5
// in Grid's basis) makes the Cstar transpose collapse to a sign flip on X.

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

  // ---------- 1. upper = -lower on any X configuration ----------
  {
    DTXQCDFermionNf up(&Grid), lo(&Grid);
    DtxqcdApplyDeltaDiag     (sigma, pi, s, p, v, up);
    DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, v, lo);

    DTXQCDFermionNf sum(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) sum.f[a] = up.f[a] + lo.f[a];

    RealD up_norm  = std::sqrt(norm2(up));
    RealD sum_norm = std::sqrt(norm2(sum));
    std::cout << GridLogMessage << "X-insert: ||up|| = " << up_norm
              << "  ||up + lo|| = " << sum_norm
              << "  (should be ~0)" << std::endl;
    check("upper + lower ~ 0 (rel)",
          sum_norm / std::max(up_norm, 1e-30), 1e-13);
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
