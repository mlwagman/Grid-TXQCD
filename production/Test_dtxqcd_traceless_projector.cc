// Unit test: DtxqcdMakeTracelessCFFastInPlace must (1) zero the CF trace,
// (2) preserve every off-diagonal entry and the traceless part of the
// diagonal, (3) be idempotent.  Guards against the iMatrix-iScalar ET
// broadcast subtracting from ALL entries instead of the diagonal.
#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt({4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian Grid_(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({11, 22, 33, 44, 55});

  LatticeDtxqcdSigma X(&Grid_);
  DtxqcdHermitianCFGaussian(pRNG, X);

  typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SObj;
  Coordinate site({1, 2, 3, 4});
  SObj before;
  peekSite(before, X, site);

  LatticeDtxqcdSigma X0(X);
  DtxqcdMakeTracelessCFFastInPlace(X);

  SObj after;
  peekSite(after, X, site);

  // trace before/after at the probe site
  ComplexD tr_b(0, 0), tr_a(0, 0);
  for (int a = 0; a < DtxqcdNf; ++a)
    for (int i = 0; i < Nc; ++i) {
      tr_b += static_cast<ComplexD>(before()(a, a)(i, i));
      tr_a += static_cast<ComplexD>(after()(a, a)(i, i));
    }
  const ComplexD c_expect = tr_b / RealD(DtxqcdNf * Nc);

  int fails = 0;
  auto check = [&](const std::string &what, RealD dev, RealD tol) {
    bool ok = dev < tol;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ") << what
              << "  dev=" << dev << "  tol=" << tol << std::endl;
    if (!ok) ++fails;
  };

  check("site trace zeroed", abs(tr_a), 1e-13);

  // every entry: diagonal must shift by exactly c_expect, off-diag unchanged
  RealD max_offdiag_dev = 0.0, max_diag_dev = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a)
    for (int b = 0; b < DtxqcdNf; ++b)
      for (int i = 0; i < Nc; ++i)
        for (int j = 0; j < Nc; ++j) {
          ComplexD vb = static_cast<ComplexD>(before()(a, b)(i, j));
          ComplexD va = static_cast<ComplexD>(after()(a, b)(i, j));
          if (a == b && i == j)
            max_diag_dev = std::max(max_diag_dev, abs(va - (vb - c_expect)));
          else
            max_offdiag_dev = std::max(max_offdiag_dev, abs(va - vb));
        }
  check("off-diagonal entries preserved", max_offdiag_dev, 1e-14);
  check("diagonal shifted by exactly Tr/6", max_diag_dev, 1e-14);

  // idempotence
  LatticeDtxqcdSigma X2(X);
  DtxqcdMakeTracelessCFFastInPlace(X2);
  X2 = X2 - X;
  check("idempotent", std::sqrt(norm2(X2) / norm2(X0)), 1e-14);

  // global trace via lattice sum
  auto trsum = TensorRemove(sum(trace(X)));
  check("global sum of trace zero",
        abs(ComplexD(trsum.real(), trsum.imag())) / std::sqrt(norm2(X0)), 1e-13);

  std::cout << GridLogMessage << "==== traceless projector "
            << (fails ? "FAILED" : "PASSED") << " (" << fails << ") ===="
            << std::endl;
  Grid_finalize();
  std::_Exit(fails ? 1 : 0);
}
