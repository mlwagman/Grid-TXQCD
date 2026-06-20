// Test_dtxqcd_mooee_herm: validate Hermiticity of the doubled M_ee at the
// fermion-application level (DtxqcdApplyMooeeDoubled).
//
// For Hermitian M:  <w, M v> = conj(<v, M w>), where the doubled inner
// product is < (w_u, w_l) , (v_u, v_l) > = <w_u, v_u> + <w_l, v_l>.
//
// Together with Test_dtxqcd_mooee_consistency (op vs SiteMatrix) and the
// Hermiticity of the 48x48 site matrix (Test_dtxqcd_pfaff_logdet_blockdiag),
// this confirms that the entire DTXQCD M_ee stack — Lattice fields, Pauli
// flavor structure, d/n off-diagonals — produces a Hermitian operator on
// doubled fermions.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_mooee_herm --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>

using namespace Grid;

static ComplexD doubled_inner(const DTXQCDFermionNf &w_u,
                              const DTXQCDFermionNf &w_l,
                              const DTXQCDFermionNf &v_u,
                              const DTXQCDFermionNf &v_l) {
  return innerProduct(w_u, v_u) + innerProduct(w_l, v_l);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({41, 42, 43, 44});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

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
  if (DtxqcdDnComplexSymmetric()) {
    if (!DtxqcdSigmaPiHermitianOnly()) {
      DtxqcdRealSymmetricCFInPlace(sigma);
      DtxqcdRealSymmetricCFInPlace(pi);
    }
    DtxqcdComplexSymmetricCFGaussian(pRNG, d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, n);
  }

  DTXQCDFermionNf w_u(&Grid), w_l(&Grid), v_u(&Grid), v_l(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, w_u.f[a]);
    gaussian(pRNG, w_l.f[a]);
    gaussian(pRNG, v_u.f[a]);
    gaussian(pRNG, v_l.f[a]);
  }

  const double mass = 0.4;

  // Apply M_ee to v -> M v
  DTXQCDFermionNf Mv_u(&Grid), Mv_l(&Grid);
  DtxqcdApplyMooeeDoubled(mass, sigma, pi, d, n, s, p, v_u, v_l, Mv_u, Mv_l);

  // Apply M_ee to w -> M w
  DTXQCDFermionNf Mw_u(&Grid), Mw_l(&Grid);
  DtxqcdApplyMooeeDoubled(mass, sigma, pi, d, n, s, p, w_u, w_l, Mw_u, Mw_l);

  ComplexD wMv = doubled_inner(w_u, w_l, Mv_u, Mv_l);
  ComplexD vMw = doubled_inner(v_u, v_l, Mw_u, Mw_l);

  std::cout << GridLogMessage << "<w, M v>           = " << wMv << std::endl;
  std::cout << GridLogMessage << "conj(<v, M w>)     = " << DtxqcdConj(vMw) << std::endl;

  RealD herm = DtxqcdAbs(wMv - DtxqcdConj(vMw));
  RealD ref  = std::max({DtxqcdAbs(wMv), DtxqcdAbs(vMw), 1.0});
  check("M_ee Hermiticity |<w,Mv> - conj(<v,Mw>)| (rel)", herm / ref, 1e-12);

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
