// Test_dtxqcd_aux_gaussian: FD vs analytic force for DTXQCDAuxiliaryFieldGaussianAction (v2).
//
// The Gaussian prior is quadratic in each aux slot, so dS/dh in any aux
// direction Y has zero higher-order error and FD should match analytic
// to round-off (rel ~ 1e-10 at h = 1e-4).
//
// v2 roster: sigma, pi, d, n (CF Hermitian) + s, p (singlet scalars).  No t.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_aux_gaussian --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>

using namespace Grid;

// AuxInnerReal must match the dS/dX transpose convention chosen by the action
// (see DTXQCDAuxGaussianAction::deriv()): all four CF Hermitian fields use the
// natural-Wirtinger Re Tr(F Y^T) form; scalars use the plain product.
static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(trace(A.sigma * transpose(B.sigma)))).real();
  r += TensorRemove(sum(trace(A.pi    * transpose(B.pi))))   .real();
  r += TensorRemove(sum(trace(A.d     * transpose(B.d))))    .real();
  r += TensorRemove(sum(trace(A.n     * transpose(B.n))))    .real();
  r += TensorRemove(sum(localInnerProduct(A.s, B.s))).real();
  r += TensorRemove(sum(localInnerProduct(A.p, B.p))).real();
  return r;
}

static void PerturbAux(const DTXQCDField &U, const DTXQCDField &Y, RealD scale,
                       DTXQCDField &out) {
  out.U     = U.U;
  out.sigma = U.sigma + scale * Y.sigma;
  out.pi    = U.pi    + scale * Y.pi;
  out.d     = U.d     + scale * Y.d;
  out.n     = U.n     + scale * Y.n;
  out.s     = U.s     + scale * Y.s;
  out.p     = U.p     + scale * Y.p;
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
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);

  DTXQCDField Y(&Grid);
  Y.U = Zero();
  DtxqcdHermitianCFGaussian(pRNG, Y.sigma);
  DtxqcdHermitianCFGaussian(pRNG, Y.pi);
  DtxqcdHermitianCFGaussian(pRNG, Y.d);
  DtxqcdHermitianCFGaussian(pRNG, Y.n);
  DtxqcdRealScalarGaussian(pRNG, Y.s);
  DtxqcdRealScalarGaussian(pRNG, Y.p);

  const RealD lambda = 3.0;
  const RealD h      = 1e-4;

  DTXQCDAuxiliaryFieldGaussianAction action(lambda);
  DTXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  auto check_piece = [&](const char *name, std::function<void(DTXQCDField&)> mask) {
    DTXQCDField Y_piece(&Grid);
    Y_piece = Zero();
    Y_piece.sigma = Y.sigma;  Y_piece.pi = Y.pi;
    Y_piece.d     = Y.d;      Y_piece.n  = Y.n;
    Y_piece.s     = Y.s;      Y_piece.p  = Y.p;
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
    P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("pi-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("d-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("n-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("s-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.p = Zero();
  });
  check_piece("p-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero();
  });
  check_piece("all-aux", [](DTXQCDField &) {});

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
