// Test_dtxqcd_mpc_op_quda: 4⁴ bit-equivalence between DTXQCDMOp (Grid full
// path) and DTXQCDMpcOpQUDA (M-wrap.4 QUDA-Mat + γ_2·conj + aux-kernel
// wrapper).
//
// Gate: per-slot rel diff ≤ 1e-12 across 10 random DTXQCDFermionDoubled
// inputs, plus 1 Mdag probe.
//
// Run:
//   ./Test_dtxqcd_mpc_op_quda --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/util/QudaInit.h>
#include <iomanip>
#include <iostream>

using namespace Grid;

namespace {
RealD rel_diff(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion diff(a.Grid());
  diff = a - b;
  RealD nb = norm2(b);
  if (nb == 0.0) return 0.0;
  return std::sqrt(norm2(diff) / nb);
}

void randomize_doubled(GridParallelRNG &pRNG, DTXQCDFermionDoubled &v) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.upper.f[a]);
    gaussian(pRNG, v.lower.f[a]);
  }
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
  pRNG.SeedFixedIntegers({401, 402, 403, 404});

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

  // ---------- Grid reference (DTXQCDMOp::M wraps DTXQCDWilsonCloverFermionEO::M)
  DTXQCDWilsonCloverFermionEO Dw(U, Grid, RBGrid, mass, csw,
                                  sigma, pi, d_field, n_field, s_field, p_field);
  DTXQCDMOp Mop_grid(Dw);

  // ---------- M-wrap.4 wrapper ----------------------------------------------
  DTXQCDMpcOpQUDA Mop_quda(U, Grid, RBGrid, mass, csw,
                            sigma, pi, d_field, n_field, s_field, p_field);

  // ---------- 10-spinor M comparison ----------------------------------------
  const int Ntrials = 10;
  RealD max_rel[4] = {0.0, 0.0, 0.0, 0.0};   // upper.f[0,1] + lower.f[0,1]
  int exitcode = 0;

  for (int t = 0; t < Ntrials; ++t) {
    DTXQCDFermionDoubled v(&Grid), out_grid(&Grid), out_quda(&Grid);
    randomize_doubled(pRNG, v);
    Mop_grid.M(v, out_grid);
    Mop_quda.M(v, out_quda);

    RealD r_u0 = rel_diff(out_quda.upper.f[0], out_grid.upper.f[0]);
    RealD r_u1 = rel_diff(out_quda.upper.f[1], out_grid.upper.f[1]);
    RealD r_l0 = rel_diff(out_quda.lower.f[0], out_grid.lower.f[0]);
    RealD r_l1 = rel_diff(out_quda.lower.f[1], out_grid.lower.f[1]);
    max_rel[0] = std::max(max_rel[0], r_u0);
    max_rel[1] = std::max(max_rel[1], r_u1);
    max_rel[2] = std::max(max_rel[2], r_l0);
    max_rel[3] = std::max(max_rel[3], r_l1);

    std::cout << GridLogMessage << "trial " << t << "  rel: "
              << std::scientific << std::setprecision(3)
              << "u0=" << r_u0 << " u1=" << r_u1
              << " l0=" << r_l0 << " l1=" << r_l1 << std::endl;
  }

  std::cout << GridLogMessage << "----" << std::endl;
  std::cout << GridLogMessage << "M MAX REL: "
            << "upper.f[0]=" << max_rel[0]
            << "  upper.f[1]=" << max_rel[1]
            << "  lower.f[0]=" << max_rel[2]
            << "  lower.f[1]=" << max_rel[3] << std::endl;

  const RealD tol = 1e-12;
  bool ok = (max_rel[0] < tol && max_rel[1] < tol && max_rel[2] < tol && max_rel[3] < tol);
  std::cout << GridLogMessage << "M GATE: " << (ok ? "PASS" : "FAIL")
            << "  (tol " << tol << ")" << std::endl;
  if (!ok) exitcode = 1;

  // ---------- Mdag probe ----------------------------------------------------
  DTXQCDFermionDoubled v(&Grid), out_grid_dag(&Grid), out_quda_dag(&Grid);
  randomize_doubled(pRNG, v);
  Mop_grid.Mdag(v, out_grid_dag);
  Mop_quda.Mdag(v, out_quda_dag);
  RealD rd_u0 = rel_diff(out_quda_dag.upper.f[0], out_grid_dag.upper.f[0]);
  RealD rd_u1 = rel_diff(out_quda_dag.upper.f[1], out_grid_dag.upper.f[1]);
  RealD rd_l0 = rel_diff(out_quda_dag.lower.f[0], out_grid_dag.lower.f[0]);
  RealD rd_l1 = rel_diff(out_quda_dag.lower.f[1], out_grid_dag.lower.f[1]);
  std::cout << GridLogMessage << "Mdag rel: u0=" << rd_u0 << " u1=" << rd_u1
            << " l0=" << rd_l0 << " l1=" << rd_l1 << std::endl;
  RealD mdag_max = std::max({rd_u0, rd_u1, rd_l0, rd_l1});
  bool mdag_ok = (mdag_max < tol);
  std::cout << GridLogMessage << "Mdag GATE: " << (mdag_ok ? "PASS" : "FAIL")
            << "  (max=" << mdag_max << " tol=" << tol << ")" << std::endl;
  if (!mdag_ok) exitcode = 1;

  Grid::Quda::finalize();
  Grid_finalize();
  return exitcode;
}
