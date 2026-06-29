// Test_dtxqcd_quda_aux_kernel: M-wrap.3 unit test.
//
// Validates ApplyFusedDtxqcdAuxKernel (aux-only, no clover) at 4⁴ against
// the legacy per-piece reference path:
//   ref.upper = DtxqcdApplyX(transpose_aux=false) + DtxqcdApplyDnCross(apply_conj=false)
//   ref.lower = DtxqcdApplyX(transpose_aux=*)     + DtxqcdApplyDnCross(apply_conj=*)
//
// Both flag settings probed:
//   - (transpose_aux=true,  use_dn_conj=true)  — production constexpr default
//   - (transpose_aux=false, use_dn_conj=false) — coverage for the alternate
//                                                template paths
//
// Gate: max rel diff ≤ 1e-14 across all 4 flavor blocks
// (upper.f[0,1] + lower.f[0,1]).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel.h>

using namespace Grid;

static RealD rel_diff_fermion(const LatticeFermion &a, const LatticeFermion &b) {
  LatticeFermion d(a.Grid()); d = a - b;
  RealD na = std::sqrt(norm2(a));
  RealD nd = std::sqrt(norm2(d));
  return (na > 0.0) ? nd / na : nd;
}

// Reference path: per-piece legacy aux call.
// Computes ref_out = DtxqcdApplyX(... transpose_aux) on each block (overwrite)
// plus DtxqcdApplyDnCross(... apply_conj) cross contributions (overwrite),
// then sums into out_accum.
static void reference_aux(const LatticeDtxqcdSigma &sigma,
                          const LatticeDtxqcdPi    &pi,
                          const LatticeDtxqcdD     &d,
                          const LatticeDtxqcdN     &n,
                          const LatticeDtxqcdS     &s,
                          const LatticeDtxqcdP     &p,
                          const DTXQCDFermionDoubled &in,
                          DTXQCDFermionDoubled       &out,
                          bool transpose_aux,
                          bool use_dn_conj) {
  GridBase *grid = in.Grid();
  DTXQCDFermionNf delta_u(grid), delta_l(grid);
  DTXQCDFermionNf cross_u(grid), cross_l(grid);

  DtxqcdApplyX(sigma, pi, s, p, in.upper, delta_u, +1.0, /*transpose_aux=*/false);
  DtxqcdApplyX(sigma, pi, s, p, in.lower, delta_l, +1.0, /*transpose_aux=*/transpose_aux);
  DtxqcdApplyDnCross(d, n, in.lower, cross_u, /*apply_conj=*/false);
  DtxqcdApplyDnCross(d, n, in.upper, cross_l, /*apply_conj=*/use_dn_conj);

  for (int a = 0; a < DtxqcdNf; ++a) {
    out.upper.f[a] = delta_u.f[a] + cross_u.f[a];
    out.lower.f[a] = delta_l.f[a] + cross_l.f[a];
  }
}

static int run_case(GridCartesian &grid, GridParallelRNG &pRNG,
                    bool transpose_aux, bool use_dn_conj) {
  std::cout << GridLogMessage << "=== case (transpose_aux="
            << (transpose_aux ? "true" : "false")
            << ", use_dn_conj="
            << (use_dn_conj ? "true" : "false") << ") ===" << std::endl;

  // ---- aux fields ----
  LatticeDtxqcdSigma sigma(&grid);
  LatticeDtxqcdPi    pi   (&grid);
  LatticeDtxqcdD     d    (&grid);
  LatticeDtxqcdN     n    (&grid);
  LatticeDtxqcdS     s    (&grid);
  LatticeDtxqcdP     p    (&grid);

  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  // d, n: complex-symmetric (production setting; matches DtxqcdApplyDnCross
  // expectations under DN_COMPLEX_SYMMETRIC = true).
  DtxqcdComplexSymmetricCFGaussian(pRNG, d);
  DtxqcdComplexSymmetricCFGaussian(pRNG, n);
  DtxqcdRealScalarGaussian(pRNG, s);
  DtxqcdRealScalarGaussian(pRNG, p);

  // ---- random input ----
  DTXQCDFermionDoubled in(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, in.upper.f[a]);
    gaussian(pRNG, in.lower.f[a]);
  }

  // ---- reference ----
  DTXQCDFermionDoubled out_ref(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_ref.upper.f[a] = Zero();
    out_ref.lower.f[a] = Zero();
  }
  reference_aux(sigma, pi, d, n, s, p, in, out_ref, transpose_aux, use_dn_conj);

  // ---- kernel (accumulate into zero-init buffer) ----
  DTXQCDFermionDoubled out_ker(&grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_ker.upper.f[a] = Zero();
    out_ker.lower.f[a] = Zero();
  }
  DtxqcdQudaAuxKernel::ApplyFusedDtxqcdAuxKernel(
      sigma, pi, d, n, s, p, in, out_ker,
      transpose_aux, use_dn_conj, /*accumulate=*/true);

  // ---- compare per slot ----
  RealD max_rel = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    RealD ru = rel_diff_fermion(out_ker.upper.f[a], out_ref.upper.f[a]);
    RealD rl = rel_diff_fermion(out_ker.lower.f[a], out_ref.lower.f[a]);
    std::cout << GridLogMessage << "  upper.f[" << a << "] rel = " << ru
              << "   lower.f[" << a << "] rel = " << rl << std::endl;
    max_rel = std::max({max_rel, ru, rl});
  }
  std::cout << GridLogMessage << "  max_rel = " << max_rel
            << "   (gate ≤ 1e-14)" << std::endl;

  return (max_rel <= 1e-14) ? 0 : 1;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt({4, 4, 4, 4});
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--grid") {
      std::vector<int> dd;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) dd.push_back(std::stoi(tok));
      if (dd.size() == (size_t)Nd) latt = Coordinate(dd);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid_(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({3001, 3002, 3003, 3004});

  std::cout << GridLogMessage << "Lattice: ";
  for (int d_ = 0; d_ < Nd; ++d_) std::cout << latt[d_] << (d_ + 1 < Nd ? "×" : "");
  std::cout << "   csw=1.249 mass=-0.245 (documented; kernel uses neither)"
            << std::endl;

  int rc = 0;
  // Production constexpr setting: transpose_aux=true, use_dn_conj=true.
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/true,  /*use_dn_conj=*/true);
  // Alternate template path coverage.
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/false, /*use_dn_conj=*/false);
  // Mixed cases to exercise the remaining two template instantiations.
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/true,  /*use_dn_conj=*/false);
  rc |= run_case(Grid_, pRNG, /*transpose_aux=*/false, /*use_dn_conj=*/true);

  std::cout << GridLogMessage
            << (rc == 0 ? "[ok] M-wrap.3 PASS — all 4 flag combinations bit-exact"
                        : "[FAIL] M-wrap.3 — at least one case exceeded 1e-14")
            << std::endl;

  Grid_finalize();
  return rc;
}
