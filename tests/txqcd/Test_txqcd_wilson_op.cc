// Correctness tests for the TXQCD Wilson Dirac operator M = D_W + Delta.
//
// Two checks:
//   1. Free-field consistency. With aux fields zero, M acting per flavor must
//      reproduce stock WilsonFermion::M on each flavor. Difference is FP-zero.
//   2. gamma5-Hermiticity (notes Eq. 9). Equivalently, gamma5*M is composite-
//      Hermitian:  <w, gamma5 M v> = conj(<v, gamma5 M w>).
//      We verify this on random gauge + aux + fermion fields.
//
// Both checks use random Hermitian aux fields and a random SU(N) gauge link.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>

using namespace Grid;

static ComplexD CompositeInner(const TXQCDFermionNf &x,
                               const TXQCDFermionNf &y) {
  ComplexD acc = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) acc += innerProduct(x.f[a], y.f[a]);
  return acc;
}

static inline RealD cmag(const ComplexD &z) {
  return std::abs(std::complex<double>(z.real(), z.imag()));
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({101, 202, 303, 404});

  LatticeGaugeField Umu(&Grid);
  SU<Nc>::HotConfiguration(pRNG, Umu);

  LatticeSigmaField sigma(&Grid);
  LatticePiField    pi(&Grid);
  LatticeSFieldC    s(&Grid);
  LatticePFieldC    p(&Grid);
  LatticeTField     t(&Grid);

  TXQCDFermionNf v(&Grid), w(&Grid);
  for (int a = 0; a < TxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  RealD mass = 0.1;
  int exitcode = 0;

  // ------------------------------------------------------------
  // 1. Free-field consistency: aux = 0  =>  M agrees with stock Wilson.
  // ------------------------------------------------------------
  {
    sigma = Zero(); pi = Zero(); s = Zero(); p = Zero(); t = Zero();
    TXQCDWilsonOp Mop(Umu, Grid, RBGrid, mass, sigma, pi, s, p, t);

    TXQCDFermionNf Mv(&Grid);
    Mop.M(v, Mv);

    WilsonFermion<WilsonImplR> Dw(Umu, Grid, RBGrid, mass);
    RealD resid = 0.0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion ref(&Grid);
      Dw.M(v.f[a], ref);
      LatticeFermion diff = Mv.f[a] - ref;
      resid += norm2(diff);
    }
    resid = std::sqrt(resid);
    bool pass = resid < 1e-12;
    std::cout << GridLogMessage << "[free-field aux=0] |M_TXQCD - D_W| = "
              << resid << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ------------------------------------------------------------
  // 2. gamma5-Hermiticity: <w, gamma5 M v> = conj(<v, gamma5 M w>).
  // ------------------------------------------------------------
  {
    HermitianGaussian(pRNG, sigma);
    HermitianGaussian(pRNG, pi);
    HermitianGaussian(pRNG, s);
    HermitianGaussian(pRNG, p);
    GaussianAntisymTensor(pRNG, t);

    TXQCDWilsonOp Mop(Umu, Grid, RBGrid, mass, sigma, pi, s, p, t);

    TXQCDFermionNf Mv(&Grid), Mw(&Grid);
    Mop.M(v, Mv);
    Mop.M(w, Mw);

    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf g5Mv(&Grid), g5Mw(&Grid);
    for (int a = 0; a < TxqcdNf; ++a) {
      g5Mv.f[a] = g5 * Mv.f[a];
      g5Mw.f[a] = g5 * Mw.f[a];
    }

    ComplexD lhs = CompositeInner(w, g5Mv);
    ComplexD rhs = conjugate(CompositeInner(v, g5Mw));
    RealD err = cmag(lhs - rhs) / std::max(cmag(lhs), 1.0);
    bool pass = err < 1e-12;
    std::cout << GridLogMessage << "[gamma5-herm] <w,g5 M v>=" << lhs
              << " <v,g5 M w>*=" << rhs << " rel=" << err
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ------------------------------------------------------------
  // 3. Mdag consistency: M^dag should equal gamma5 M gamma5 (which is how
  //    we implement it). Also verify <w, M v> = conj(<v, Mdag w>).
  // ------------------------------------------------------------
  {
    TXQCDWilsonOp Mop(Umu, Grid, RBGrid, mass, sigma, pi, s, p, t);
    TXQCDFermionNf Mv(&Grid), Mdw(&Grid);
    Mop.M(v, Mv);
    Mop.Mdag(w, Mdw);
    ComplexD lhs = CompositeInner(w, Mv);
    ComplexD rhs = conjugate(CompositeInner(v, Mdw));
    RealD err = cmag(lhs - rhs) / std::max(cmag(lhs), 1.0);
    bool pass = err < 1e-12;
    std::cout << GridLogMessage << "[Mdag] <w,Mv>=" << lhs
              << " <v,Mdag w>*=" << rhs << " rel=" << err
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
