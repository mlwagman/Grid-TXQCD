// Test_dtxqcd_mooeeinv_n: agreement between single-RHS MooeeInv (SIMD per-
// oSite gemv) and the multi-RHS MooeeInvN (per-site Eigen 48 x NRHS gemm)
// paths on the cached doubled-Wilson-Clover EO operator.
//
// Generates NRHS = 5 random doubled fermions on each CB, applies MooeeInv
// once-per-RHS via the SIMD path AND once via the batched-RHS path, and
// requires the outputs to agree to machine precision (the two paths share
// the same underlying inverse cache; any disagreement is an indexing or
// pack/unpack bug).  Same check for MooeeInvDag / MooeeInvDagN.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_mooeeinv_n --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>

using namespace Grid;

static RealD MaxAbsDiff(const DTXQCDFermionDoubled &a,
                        const DTXQCDFermionDoubled &b) {
  RealD m = 0.0;
  for (int af = 0; af < DtxqcdNf; ++af) {
    LatticeFermion du = a.upper.f[af] - b.upper.f[af];
    LatticeFermion dl = a.lower.f[af] - b.lower.f[af];
    m = std::max(m, std::sqrt(norm2(du)));
    m = std::max(m, std::sqrt(norm2(dl)));
  }
  return m;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({901, 902, 903, 904});

  int exitcode = 0;
  auto check = [&](const char *name, RealD diff, RealD tol) {
    bool ok = (diff < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << "  max |single - batched| = " << diff
              << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // Random gauge + aux to fix the cached EO inverse.
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdRealGaussian(pRNG, U.sigma);
  DtxqcdRealGaussian(pRNG, U.pi);
  DtxqcdGaussianAntisymTensor(pRNG, U.t);
  DtxqcdHermitianGaussian(pRNG, U.d);
  DtxqcdHermitianGaussian(pRNG, U.n);

  const RealD mass = 0.4;

  auto run_csw = [&](RealD csw, const char *tag) {
    DTXQCDWilsonCloverFermionEO Dw(U.U, Grid, RBGrid, mass, csw,
                                   U.sigma, U.pi, U.t, U.d, U.n);

    const int NRHS = 5;
    for (int cb_idx = 0; cb_idx < 2; ++cb_idx) {
      int cb = (cb_idx == 0) ? Even : Odd;
      const char *cbname = (cb_idx == 0) ? "Even" : "Odd";

      // NRHS random CB-tagged doubled fermions on the requested parity.
      std::vector<DTXQCDFermionDoubled> ins;        ins.reserve(NRHS);
      std::vector<DTXQCDFermionDoubled> outs_single; outs_single.reserve(NRHS);
      std::vector<DTXQCDFermionDoubled> outs_batched; outs_batched.reserve(NRHS);
      for (int k = 0; k < NRHS; ++k) {
        ins.emplace_back(&RBGrid);
        outs_single.emplace_back(&RBGrid);
        outs_batched.emplace_back(&RBGrid);
        for (int a = 0; a < DtxqcdNf; ++a) {
          LatticeFermion fu(&Grid), fl(&Grid);
          gaussian(pRNG, fu);
          gaussian(pRNG, fl);
          pickCheckerboard(cb, ins[k].upper.f[a], fu);
          pickCheckerboard(cb, ins[k].lower.f[a], fl);
        }
      }

      // Single-RHS path: per-RHS MooeeInv.
      for (int k = 0; k < NRHS; ++k) Dw.MooeeInv(ins[k], outs_single[k]);

      // Batched path: one MooeeInvN call.
      std::vector<const DTXQCDFermionDoubled *> in_ptrs(NRHS);
      std::vector<DTXQCDFermionDoubled *>       out_ptrs(NRHS);
      for (int k = 0; k < NRHS; ++k) {
        in_ptrs[k]  = &ins[k];
        out_ptrs[k] = &outs_batched[k];
      }
      Dw.MooeeInvN(in_ptrs, out_ptrs);

      RealD maxd = 0.0;
      for (int k = 0; k < NRHS; ++k)
        maxd = std::max(maxd, MaxAbsDiff(outs_single[k], outs_batched[k]));
      check((std::string(tag) + " " + cbname + " MooeeInv  vs MooeeInvN").c_str(),
            maxd, 1e-10);

      // Same comparison for the dag direction.
      std::vector<DTXQCDFermionDoubled> outs_dag_single;  outs_dag_single.reserve(NRHS);
      std::vector<DTXQCDFermionDoubled> outs_dag_batched; outs_dag_batched.reserve(NRHS);
      for (int k = 0; k < NRHS; ++k) {
        outs_dag_single.emplace_back(&RBGrid);
        outs_dag_batched.emplace_back(&RBGrid);
      }
      for (int k = 0; k < NRHS; ++k) Dw.MooeeInvDag(ins[k], outs_dag_single[k]);
      std::vector<DTXQCDFermionDoubled *> out_dag_ptrs(NRHS);
      for (int k = 0; k < NRHS; ++k) out_dag_ptrs[k] = &outs_dag_batched[k];
      Dw.MooeeInvDagN(in_ptrs, out_dag_ptrs);

      RealD maxd_dag = 0.0;
      for (int k = 0; k < NRHS; ++k)
        maxd_dag = std::max(maxd_dag,
                            MaxAbsDiff(outs_dag_single[k], outs_dag_batched[k]));
      check((std::string(tag) + " " + cbname + " MooeeInvDag vs MooeeInvDagN").c_str(),
            maxd_dag, 1e-10);
    }
  };

  run_csw(/*csw=*/0.0,  "csw=0   ");
  run_csw(/*csw=*/1.25, "csw=1.25");

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
