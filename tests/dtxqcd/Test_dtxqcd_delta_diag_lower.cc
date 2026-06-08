// Test_dtxqcd_delta_diag_lower: validate that DtxqcdApplyDeltaDiagLower
// (Cstar M_22 = C^T X^T C aux insertion for the lower diagonal block)
// differs from DtxqcdApplyDeltaDiag (upper) by exactly twice the tensor
// contribution, with the sigma and pi pieces unchanged.
//
// Specifically:
//   1. With sigma and pi nonzero, t = 0: upper and lower applies agree.
//      (Confirms sigma and pi pieces are unchanged under C^T (.)^T C.)
//   2. With t nonzero, sigma = pi = 0: upper and lower differ by
//      exactly 2 * DtxqcdApplyDeltaTensor(t, v).
//      (Confirms tensor piece sign-flips: upper = +tensor, lower = -tensor.)
//   3. Lower-block Hermiticity on a fermion bilinear:
//      <w, Delta_diag_lower v> = conj(<v, Delta_diag_lower w>).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_delta_diag_lower --grid 4.4.4.4 --mpi 1.1.1.1

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
  LatticeDtxqcdT     t(&Grid);

  // ---------- 1. With t = 0, upper and lower applies agree ----------
  {
    DtxqcdRealGaussian(pRNG, sigma);
    DtxqcdRealGaussian(pRNG, pi);
    t = Zero();

    DTXQCDFermionNf up(&Grid), lo(&Grid);
    DtxqcdApplyDeltaDiag(sigma, pi, t, v, up);
    DtxqcdApplyDeltaDiagLower(sigma, pi, t, v, lo);

    DTXQCDFermionNf diff(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) diff.f[a] = up.f[a] - lo.f[a];

    RealD ref = std::sqrt(std::max(norm2(up), 1e-30));
    RealD rel = std::sqrt(norm2(diff)) / ref;
    check("t=0: ||up - lo|| / ||up||", rel, 1e-14);
  }

  // ---------- 2. With sigma = pi = 0, t nonzero: lo = -up ----------
  {
    sigma = Zero();
    pi    = Zero();
    DtxqcdGaussianAntisymTensor(pRNG, t);

    DTXQCDFermionNf up(&Grid), lo(&Grid);
    DtxqcdApplyDeltaDiag(sigma, pi, t, v, up);
    DtxqcdApplyDeltaDiagLower(sigma, pi, t, v, lo);

    // lo should be -up (only tensor contribution, with sign-flip).
    DTXQCDFermionNf sum(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) sum.f[a] = up.f[a] + lo.f[a];

    RealD up_norm  = std::sqrt(norm2(up));
    RealD lo_norm  = std::sqrt(norm2(lo));
    RealD sum_norm = std::sqrt(norm2(sum));
    std::cout << GridLogMessage << "tensor-only: ||up|| = " << up_norm
              << "  ||lo|| = " << lo_norm
              << "  ||up + lo|| = " << sum_norm
              << "  (should be ~0)" << std::endl;
    check("tensor-only: ||up + lo|| / ||up||",
          sum_norm / std::max(up_norm, 1e-30), 1e-13);
  }

  // ---------- 3. Lower-block Hermiticity on a random aux config ----------
  {
    DtxqcdRealGaussian(pRNG, sigma);
    DtxqcdRealGaussian(pRNG, pi);
    DtxqcdGaussianAntisymTensor(pRNG, t);

    DTXQCDFermionNf Dv(&Grid), Dw(&Grid);
    DtxqcdApplyDeltaDiagLower(sigma, pi, t, v, Dv);
    DtxqcdApplyDeltaDiagLower(sigma, pi, t, w, Dw);
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
