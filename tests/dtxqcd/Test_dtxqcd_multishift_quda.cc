// Test_dtxqcd_multishift_quda: 4⁴ bit-equivalence between Grid DTXQCDMultiShiftCG
// (all-DP, all-Grid) and DTXQCDMultiShiftCGQUDA (DP outer + QUDA-DP inner).
//
// Per-shift gates:
//   ||(M†M + poles_s)·psi_quda[s] − b|| / ||b||  ≤ 1e-9   (residual)
//   ||psi_quda[s] − psi_grid[s]|| / ||psi_grid[s]||  ≤ 1e-9   (bit-equiv)
//
// Also reports per-shift CG iter counts for Grid vs QUDA.
//
// Run:
//   ./Test_dtxqcd_multishift_quda --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA.h>
#include <Grid/util/QudaInit.h>
#include <iomanip>
#include <iostream>

using namespace Grid;

namespace {
RealD rel_diff(const DTXQCDFermionDoubled &a, const DTXQCDFermionDoubled &b) {
  RealD num = 0.0, den = 0.0;
  for (int aa = 0; aa < DtxqcdNf; ++aa) {
    LatticeFermion du(a.upper.f[aa].Grid()), dl(a.lower.f[aa].Grid());
    du = a.upper.f[aa] - b.upper.f[aa];
    dl = a.lower.f[aa] - b.lower.f[aa];
    num += norm2(du) + norm2(dl);
    den += norm2(b.upper.f[aa]) + norm2(b.lower.f[aa]);
  }
  if (den == 0.0) return 0.0;
  return std::sqrt(num / den);
}

void randomize_doubled(GridParallelRNG &pRNG, DTXQCDFermionDoubled &v) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.upper.f[a]);
    gaussian(pRNG, v.lower.f[a]);
  }
}

// Compute ||(M†M + pole)·psi - b|| / ||b||  via Grid HermOp.
template <class DTXQCDMop>
RealD shifted_residual(DTXQCDMop &Mop, RealD pole,
                       const DTXQCDFermionDoubled &psi,
                       const DTXQCDFermionDoubled &b) {
  GridBase *grid = b.Grid();
  DTXQCDFermionDoubled tmp(grid), mmp(grid), r(grid);
  Mop.M(psi, tmp);
  Mop.Mdag(tmp, mmp);
  for (int a = 0; a < DtxqcdNf; ++a) {
    mmp.upper.f[a] = mmp.upper.f[a] + pole * psi.upper.f[a];
    mmp.lower.f[a] = mmp.lower.f[a] + pole * psi.lower.f[a];
    r.upper.f[a]   = b.upper.f[a]   - mmp.upper.f[a];
    r.lower.f[a]   = b.lower.f[a]   - mmp.lower.f[a];
  }
  RealD num = norm2(r);
  RealD den = norm2(b);
  if (den == 0.0) return 0.0;
  return std::sqrt(num / den);
}
}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

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
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({501, 502, 503, 504});

  // ---------- Random gauge + aux (production csw / mass / λ=3 amplitude) -----
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD aux_scale = 1.0 / 3.0;   // λ=3 production amplitude

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d_field(&Grid);
  LatticeDtxqcdN     n_field(&Grid);
  LatticeDtxqcdS     s_field(&Grid);
  LatticeDtxqcdP     p_field(&Grid);

  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s_field);
  DtxqcdRealScalarGaussian(pRNG, p_field);
  DtxqcdHermitianCFGaussian(pRNG, d_field);
  DtxqcdHermitianCFGaussian(pRNG, n_field);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(sigma);
    DtxqcdRealSymmetricCFInPlace(pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, d_field);
    DtxqcdComplexSymmetricCFGaussian(pRNG, n_field);
  }
  sigma   = aux_scale * sigma;
  pi      = aux_scale * pi;
  s_field = aux_scale * s_field;
  p_field = aux_scale * p_field;
  d_field = aux_scale * d_field;
  n_field = aux_scale * n_field;

  // ---------- Build both operators ------------------------------------------
  DTXQCDWilsonCloverFermionEO Dw(U, Grid, RBGrid, mass, csw,
                                  sigma, pi, d_field, n_field, s_field, p_field);
  DTXQCDMOp Mop_grid(Dw);

  DTXQCDMpcOpQUDA Mop_quda(U, Grid, RBGrid, mass, csw,
                            sigma, pi, d_field, n_field, s_field, p_field);

  // ---------- Multi-shift setup ---------------------------------------------
  // N=5 shifts spanning a modest range; production uses N≈10.  Sorted ascending
  // with poles[0] the smallest (per DTXQCDMultiShiftCG contract).
  std::vector<RealD> poles = {0.01, 0.05, 0.1, 0.25, 0.5};
  std::vector<RealD> tol  (poles.size(), 1.0e-10);
  const int MaxIter = 3000;

  // ---------- Random source --------------------------------------------------
  DTXQCDFermionDoubled b(&Grid);
  randomize_doubled(pRNG, b);

  // ---------- Solve via Grid all-DP multishift ------------------------------
  std::vector<DTXQCDFermionDoubled> psi_grid;
  psi_grid.reserve(poles.size());
  for (size_t s = 0; s < poles.size(); ++s) psi_grid.emplace_back(&Grid);
  std::cout << GridLogMessage << "---- Grid multishift solve ----" << std::endl;
  DTXQCDMultiShiftCG(Mop_grid, poles, tol, b, psi_grid, MaxIter);

  // ---------- Solve via QUDA-inner multishift -------------------------------
  std::vector<DTXQCDFermionDoubled> psi_quda;
  psi_quda.reserve(poles.size());
  for (size_t s = 0; s < poles.size(); ++s) psi_quda.emplace_back(&Grid);
  std::cout << GridLogMessage << "---- QUDA multishift solve ----" << std::endl;
  DTXQCDMultiShiftCGQUDA(Mop_grid, Mop_quda, poles, tol, b, psi_quda, MaxIter,
                          /*ReliableUpdateFreq=*/50);

  // ---------- Per-shift gates -----------------------------------------------
  const RealD gate_res = 1.0e-9;
  const RealD gate_rel = 1.0e-9;
  int exitcode = 0;
  std::cout << GridLogMessage << "---- Per-shift gates ----" << std::endl;
  for (size_t s = 0; s < poles.size(); ++s) {
    RealD res_grid = shifted_residual(Mop_grid, poles[s], psi_grid[s], b);
    RealD res_quda = shifted_residual(Mop_grid, poles[s], psi_quda[s], b);
    RealD rd       = rel_diff(psi_quda[s], psi_grid[s]);
    bool ok = (res_quda < gate_res) && (rd < gate_rel);
    std::cout << GridLogMessage
              << "shift " << s << "  pole=" << std::setw(7) << std::setprecision(4) << poles[s]
              << "  res_grid=" << std::scientific << std::setprecision(3) << res_grid
              << "  res_quda=" << res_quda
              << "  ||psi_quda-psi_grid||/||psi_grid||=" << rd
              << "  " << (ok ? "PASS" : "FAIL") << std::endl;
    if (!ok) exitcode = 1;
  }

  std::cout << GridLogMessage << "---- Overall ----" << std::endl;
  std::cout << GridLogMessage << "Test_dtxqcd_multishift_quda: "
            << (exitcode == 0 ? "PASS" : "FAIL") << std::endl;

  Grid::Quda::finalize();
  Grid_finalize();
  return exitcode;
}
