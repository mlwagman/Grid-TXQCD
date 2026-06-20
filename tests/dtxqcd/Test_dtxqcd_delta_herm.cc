// Test_dtxqcd_delta_herm (v2): unit checks for the diagonal-block X insertion
// (DtxqcdApplyDeltaDiag) on the v2 aux roster.
//
// X^{ij}_{ab} = sigma + s delta delta + (pi + p delta delta) gamma5  is
// Hermitian whenever sigma, pi are CF Hermitian and s, p are real.  We
// verify <w, X v> = conj(<v, X w>) on random fermions for several aux
// configurations.

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
  pRNG.SeedFixedIntegers({11, 22, 33, 44});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DTXQCDFermionNf v(&Grid), w(&Grid), Dv(&Grid), Dw(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  auto herm_check = [&](const char *name,
                        const DTXQCDFermionNf &Dv,
                        const DTXQCDFermionNf &Dw) {
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    RealD herm_resid = DtxqcdAbs(wDv - DtxqcdConj(vDw));
    RealD ref = std::max({DtxqcdAbs(wDv), DtxqcdAbs(vDw), 1.0});
    check(name, herm_resid / ref, 1e-10);
  };

  // 1. aux=0 nullity.
  {
    LatticeDtxqcdSigma sigma(&Grid); sigma = Zero();
    LatticeDtxqcdPi    pi(&Grid);    pi    = Zero();
    LatticeDtxqcdS     s(&Grid);     s     = Zero();
    LatticeDtxqcdP     p(&Grid);     p     = Zero();
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, v, Dv);
    check("aux=0 nullity ||X v||^2", norm2(Dv), 1e-20);
  }

  // 2. Hermiticity with sigma + pi only (singlets zero).
  {
    LatticeDtxqcdSigma sigma(&Grid);
    LatticeDtxqcdPi    pi(&Grid);
    LatticeDtxqcdS     s(&Grid);     s = Zero();
    LatticeDtxqcdP     p(&Grid);     p = Zero();
    DtxqcdHermitianCFGaussian(pRNG, sigma);
    DtxqcdHermitianCFGaussian(pRNG, pi);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, v, Dv);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, w, Dw);
    herm_check("Hermiticity X (sigma+pi)", Dv, Dw);
  }

  // 3. Hermiticity with singlets s, p only.
  {
    LatticeDtxqcdSigma sigma(&Grid); sigma = Zero();
    LatticeDtxqcdPi    pi(&Grid);    pi    = Zero();
    LatticeDtxqcdS     s(&Grid);
    LatticeDtxqcdP     p(&Grid);
    DtxqcdRealScalarGaussian(pRNG, s);
    DtxqcdRealScalarGaussian(pRNG, p);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, v, Dv);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, w, Dw);
    herm_check("Hermiticity X (s+p)", Dv, Dw);
  }

  // 4. Hermiticity with full X (all four pieces).
  {
    LatticeDtxqcdSigma sigma(&Grid);
    LatticeDtxqcdPi    pi(&Grid);
    LatticeDtxqcdS     s(&Grid);
    LatticeDtxqcdP     p(&Grid);
    DtxqcdHermitianCFGaussian(pRNG, sigma);
    DtxqcdHermitianCFGaussian(pRNG, pi);
    DtxqcdRealScalarGaussian(pRNG, s);
    DtxqcdRealScalarGaussian(pRNG, p);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, v, Dv);
    DtxqcdApplyDeltaDiag(sigma, pi, s, p, w, Dw);
    herm_check("Hermiticity X (full)", Dv, Dw);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
