// Test_dtxqcd_delta_herm: unit checks for the DTXQCD diagonal-block Delta
// insertion (sigma^A, pi^A, t^A_{mu,nu}) — see Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h.
//
// Three checks on a 4^4 lattice:
//   1. Aux=0 nullity: with sigma^A = pi^A = t^A = 0, Delta(v) = 0 for any v.
//   2. Hermiticity (sigma,pi piece): on random sigma^A, pi^A and random v, w,
//      < w | Delta_{sigma,pi} v > = < v | Delta_{sigma,pi} w > (up to FP).
//   3. Hermiticity (tensor piece): same with t^A_{mu,nu} only.
//   4. Hermiticity (combined): both with all three pieces active.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_delta_herm --grid 4.4.4.4 --mpi 1.1.1.1

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

  // ---------- random fermion test pair ----------
  DTXQCDFermionNf v(&Grid), w(&Grid), Dv(&Grid), Dw(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  // ---------- 1. Aux=0 nullity ----------
  {
    LatticeDtxqcdSigma sigma(&Grid); sigma = Zero();
    LatticeDtxqcdPi    pi(&Grid);    pi    = Zero();
    LatticeDtxqcdT     t(&Grid);     t     = Zero();
    DtxqcdApplyDeltaDiag(sigma, pi, t, v, Dv);
    RealD r = norm2(Dv);
    check("aux=0 nullity ||Delta v||^2", r, 1e-20);
  }

  // ---------- 2. Hermiticity of (sigma,pi) piece ----------
  {
    LatticeDtxqcdSigma sigma(&Grid);
    LatticeDtxqcdPi    pi(&Grid);
    DtxqcdRealGaussian(pRNG, sigma);
    DtxqcdRealGaussian(pRNG, pi);
    DtxqcdApplyDeltaSigmaPi(sigma, pi, v, Dv);
    DtxqcdApplyDeltaSigmaPi(sigma, pi, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    // Hermiticity: <w, D v> = conj(<v, D w>) for D = D^dag.
    RealD herm_resid = std::abs(wDv - std::conj(vDw));
    RealD ref = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    check("Hermiticity Delta_{sigma,pi} | wDv - vDw |", herm_resid / ref, 1e-10);
  }

  // ---------- 3. Hermiticity of tensor piece ----------
  {
    LatticeDtxqcdT t(&Grid);
    DtxqcdGaussianAntisymTensor(pRNG, t);
    DtxqcdApplyDeltaTensor(t, v, Dv);
    DtxqcdApplyDeltaTensor(t, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    // Hermiticity: <w, D v> = conj(<v, D w>) for D = D^dag.
    RealD herm_resid = std::abs(wDv - std::conj(vDw));
    RealD ref = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    check("Hermiticity Delta_t | wDv - vDw |", herm_resid / ref, 1e-10);
  }

  // ---------- 4. Hermiticity of full diagonal Delta ----------
  {
    LatticeDtxqcdSigma sigma(&Grid);
    LatticeDtxqcdPi    pi(&Grid);
    LatticeDtxqcdT     t(&Grid);
    DtxqcdRealGaussian(pRNG, sigma);
    DtxqcdRealGaussian(pRNG, pi);
    DtxqcdGaussianAntisymTensor(pRNG, t);
    DtxqcdApplyDeltaDiag(sigma, pi, t, v, Dv);
    DtxqcdApplyDeltaDiag(sigma, pi, t, w, Dw);
    ComplexD wDv = innerProduct(w, Dv);
    ComplexD vDw = innerProduct(v, Dw);
    // Hermiticity: <w, D v> = conj(<v, D w>) for D = D^dag.
    RealD herm_resid = std::abs(wDv - std::conj(vDw));
    RealD ref = std::max({std::abs(wDv), std::abs(vDw), 1.0});
    check("Hermiticity Delta_diag | wDv - vDw |", herm_resid / ref, 1e-10);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
