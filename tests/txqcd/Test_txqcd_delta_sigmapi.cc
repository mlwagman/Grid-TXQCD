// Hermiticity test for the sigma/pi flavor sector of the TXQCD Delta-operator.
//
// With sigma and pi Hermitian Nf x Nf flavor matrices, the local insertion
//   Delta_{sigma,pi}(x) v[a] = sum_b sigma_{a,b}(x) v[b] + pi_{a,b}(x) gamma5 v[b]
// is manifestly Hermitian in the combined (flavor x spin x color x site) inner
// product, since sigma^dag = sigma, pi^dag = pi and gamma5^dag = gamma5. This
// test draws random aux fields, fermion fields v and w, and checks
//   <w, Delta v> == <v, Delta w>*
// which (after conjugation) is the Hermiticity statement.
//
// This is a narrow but non-trivial check that the flavor index wiring,
// PeekIndex on the Hermitian flavor-matrix tensor type, and gamma5
// application to each flavor component all line up correctly.

#include <Grid/Grid.h>

using namespace Grid;

static ComplexD CompositeInner(const TXQCDFermionNf &x,
                               const TXQCDFermionNf &y) {
  ComplexD acc = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) acc += innerProduct(x.f[a], y.f[a]);
  return acc;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  LatticeSigmaField sigma(&Grid), pi(&Grid);
  HermitianGaussian(pRNG, sigma);
  HermitianGaussian(pRNG, pi);

  TXQCDFermionNf v(&Grid), w(&Grid), Dv(&Grid), Dw(&Grid);
  for (int a = 0; a < TxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  ApplyDeltaSigmaPi(sigma, pi, v, Dv);
  ApplyDeltaSigmaPi(sigma, pi, w, Dw);

  ComplexD lhs = CompositeInner(w, Dv);   // <w, Dv>
  ComplexD rhs = conjugate(CompositeInner(v, Dw));  // <v, Dw>*
  RealD err = std::abs(lhs - rhs) / std::max(std::abs(lhs), 1.0);

  std::cout << GridLogMessage << "<w,Dv>    = " << lhs << std::endl;
  std::cout << GridLogMessage << "<v,Dw>^*  = " << rhs << std::endl;
  std::cout << GridLogMessage << "rel diff  = " << err
            << (err < 1e-12 ? "  PASS" : "  FAIL") << std::endl;

  int exitcode = (err < 1e-12) ? 0 : 1;
  Grid_finalize();
  return exitcode;
}
