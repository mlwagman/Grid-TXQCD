// M-wrap.6: 4⁴ bit-equivalence between DTXQCD_MULTISHIFT_QUDA=0 (Grid path)
// and DTXQCD_MULTISHIFT_QUDA=1 (DTXQCDMultiShiftCGQUDA) at the deriv()
// level for DTXQCDWilsonCloverRationalFullAction.
//
// Approach:
//   1. Build action + seed Phi via refresh() with env unset.
//   2. Call deriv() with env unset → dSdU_grid.
//   3. Set DTXQCD_MULTISHIFT_QUDA=1.
//   4. Call deriv() again on the SAME Phi → dSdU_quda.
//   5. Slot-by-slot rel diff: U (gauge), σ, π, d, n, s, p (aux).
//
// Gate: max slot rel ≤ 1e-7 (Remez floor — both backends solve the same
// shifted system to mdtol=1e-8, so the dominant deviation is FP reordering
// inside QUDA's MatQuda vs Grid's WilsonFermion::M plus aux-roundtrip
// rounding).
//
// Run:
//   ./Test_dtxqcd_multishift_quda_fd --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/util/QudaInit.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace Grid;

namespace {

// Per-channel rel norm: ||A - B|| / max(||A||, ||B||).
template <class Field>
RealD rel_field(const Field &A, const Field &B) {
  Field d(A.Grid());
  d = A - B;
  RealD nA = std::sqrt(norm2(A));
  RealD nB = std::sqrt(norm2(B));
  RealD nD = std::sqrt(norm2(d));
  RealD den = std::max({nA, nB, 1e-30});
  return nD / den;
}

void check_slot(const char *name, RealD rel, RealD tol, int &exitcode) {
  bool ok = (rel < tol);
  std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] "
            << name << " : rel = " << std::scientific << rel
            << " (tol " << tol << ")" << std::endl;
  if (!ok) exitcode = 1;
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
#ifdef GRID_HAVE_QUDA
  Grid::Quda::initialize();
#endif

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);  pRNG.SeedFixedIntegers({701, 702, 703, 704});
  GridSerialRNG   sRNG;         sRNG.SeedFixedIntegers({711, 712, 713, 714});

  // Random gauge + aux at production csw, mass.
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(U.sigma);
    DtxqcdRealSymmetricCFInPlace(U.pi);
    DtxqcdRealSymmetricCFInPlace(U.d);
    DtxqcdRealSymmetricCFInPlace(U.n);
  }
  // λ=3 amplitude on aux (matches production setting).
  const RealD lam = 3.0;
  U.sigma = (1.0 / lam) * U.sigma;
  U.pi    = (1.0 / lam) * U.pi;
  U.d     = (1.0 / lam) * U.d;
  U.n     = (1.0 / lam) * U.n;
  U.s     = (1.0 / lam) * U.s;
  U.p     = (1.0 / lam) * U.p;

  // Action params — small degree for fast test, production csw + mass.
  // Ctor order: (lo, hi, maxit, tol, degree, precision, BoundsFreq, BoundsTol).
  OneFlavourRationalParams p(/*lo=*/1e-2, /*hi=*/2.5, /*maxit=*/3000,
                             /*tol=*/1e-10, /*degree=*/5, /*precision=*/40);
  p.mdtolerance  = 1e-8;
  const RealD mass = -0.245;
  const RealD csw  =  1.249;

  DTXQCDWilsonCloverRationalFullAction action(Grid, RBGrid, mass, p, csw);

  // Seed Phi with env unset (Grid path) so both subsequent deriv() calls
  // see the same Phi.
  unsetenv("DTXQCD_MULTISHIFT_QUDA");
  std::cout << GridLogMessage << "[fd] refresh() (Grid path) ..." << std::endl;
  action.refresh(U, sRNG, pRNG);

  // ---- Pass 1: deriv() via Grid path ----
  unsetenv("DTXQCD_MULTISHIFT_QUDA");
  DTXQCDField dSdU_grid(&Grid);
  std::cout << GridLogMessage << "[fd] deriv() Grid path ..." << std::endl;
  action.deriv(U, dSdU_grid);

  // ---- Pass 2: deriv() via QUDA path ----
  setenv("DTXQCD_MULTISHIFT_QUDA", "1", /*overwrite=*/1);
  DTXQCDField dSdU_quda(&Grid);
  std::cout << GridLogMessage << "[fd] deriv() QUDA path ..." << std::endl;
  action.deriv(U, dSdU_quda);

  // ---- Compare slot-by-slot ----
  int exitcode = 0;
  const RealD tol = 1e-7;
  std::cout << GridLogMessage << "=== M-wrap.6 FD slot comparison ===" << std::endl;
  check_slot("U     (gauge)", rel_field(dSdU_grid.U,     dSdU_quda.U),     tol, exitcode);
  check_slot("sigma (aux)",   rel_field(dSdU_grid.sigma, dSdU_quda.sigma), tol, exitcode);
  check_slot("pi    (aux)",   rel_field(dSdU_grid.pi,    dSdU_quda.pi),    tol, exitcode);
  check_slot("d     (aux)",   rel_field(dSdU_grid.d,     dSdU_quda.d),     tol, exitcode);
  check_slot("n     (aux)",   rel_field(dSdU_grid.n,     dSdU_quda.n),     tol, exitcode);
  check_slot("s     (aux)",   rel_field(dSdU_grid.s,     dSdU_quda.s),     tol, exitcode);
  check_slot("p     (aux)",   rel_field(dSdU_grid.p,     dSdU_quda.p),     tol, exitcode);

  std::cout << GridLogMessage << "Test_dtxqcd_multishift_quda_fd: "
            << (exitcode == 0 ? "PASS" : "FAIL") << std::endl;

#ifdef GRID_HAVE_QUDA
  Grid::Quda::finalize();
#endif
  Grid_finalize();
  return exitcode;
}
