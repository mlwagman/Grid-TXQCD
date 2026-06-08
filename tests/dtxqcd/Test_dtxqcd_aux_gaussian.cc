// Test_dtxqcd_aux_gaussian: FD vs analytic force for DTXQCDAuxiliaryFieldGaussianAction.
//
// The Gaussian prior is quadratic in each aux slot, so dS/dh in any aux
// direction Y has zero higher-order error and FD should match analytic
// to round-off (rel ~ 1e-10 at h = 1e-4).  Drop-in port of the FD-test
// pattern used in Test_dtxqcd_logdet_aux_force / _rational_aux_force,
// without the rational / multi-shift CG overhead.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_aux_gaussian --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>

using namespace Grid;

static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(localInnerProduct(A.sigma, B.sigma))).real();
  r += TensorRemove(sum(localInnerProduct(A.pi,    B.pi))).real();
  r += 0.5 * TensorRemove(sum(localInnerProduct(A.t, B.t))).real();
  r += TensorRemove(sum(trace(A.d * transpose(B.d)))).real();
  r += TensorRemove(sum(trace(A.n * transpose(B.n)))).real();
  return r;
}

static void PerturbAux(const DTXQCDField &U, const DTXQCDField &Y, RealD scale,
                       DTXQCDField &out) {
  out.U     = U.U;
  out.sigma = U.sigma + scale * Y.sigma;
  out.pi    = U.pi    + scale * Y.pi;
  out.t     = U.t     + scale * Y.t;
  out.d     = U.d     + scale * Y.d;
  out.n     = U.n     + scale * Y.n;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({131, 132, 133, 134});

  int exitcode = 0;
  auto check = [&](const char *name, RealD num, RealD ana, RealD tol) {
    RealD rel = std::abs(num - ana)
              / std::max({std::abs(num), std::abs(ana), 1.0});
    bool ok = (rel < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " :  numeric = " << num << "  analytic = " << ana
              << "  rel = " << rel << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdRealGaussian(pRNG, U.sigma);
  DtxqcdRealGaussian(pRNG, U.pi);
  DtxqcdGaussianAntisymTensor(pRNG, U.t);
  DtxqcdHermitianGaussian(pRNG, U.d);
  DtxqcdHermitianGaussian(pRNG, U.n);

  DTXQCDField Y(&Grid);
  Y.U = Zero();
  DtxqcdRealGaussian(pRNG, Y.sigma);
  DtxqcdRealGaussian(pRNG, Y.pi);
  DtxqcdGaussianAntisymTensor(pRNG, Y.t);
  DtxqcdHermitianGaussian(pRNG, Y.d);
  DtxqcdHermitianGaussian(pRNG, Y.n);

  const RealD lambda = 3.0;
  const RealD h      = 1e-4;

  DTXQCDAuxiliaryFieldGaussianAction action(lambda);
  DTXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  auto check_piece = [&](const char *name, std::function<void(DTXQCDField&)> mask) {
    DTXQCDField Y_piece(&Grid);
    Y_piece = Zero();
    Y_piece.sigma = Y.sigma;  Y_piece.pi = Y.pi;
    Y_piece.t = Y.t;          Y_piece.d  = Y.d;          Y_piece.n  = Y.n;
    Y_piece.U = Zero();
    mask(Y_piece);

    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbAux(U, Y_piece, +h, Up);
    PerturbAux(U, Y_piece, -h, Um);
    RealD num = (action.S(Up) - action.S(Um)) / (2.0 * h);
    RealD ana = AuxInnerReal(dSdU, Y_piece);
    check(name, num, ana, 1e-8);
  };

  check_piece("sigma-only", [](DTXQCDField &P) {
    P.pi = Zero(); P.t = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("pi-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.t = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("t-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero();
  });
  check_piece("d-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.t = Zero(); P.n = Zero();
  });
  check_piece("n-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.t = Zero(); P.d = Zero();
  });
  check_piece("all-aux", [](DTXQCDField &) {});

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
