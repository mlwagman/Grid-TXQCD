// Test_dtxqcd_multishift_quda_eo — Gate 2 of the EO Style C port:
// 10-shift multishift solve on the odd-CB Schur operator, Grid-DP CG
// (DTXQCDMultiShiftCG on DTXQCDMpcOp) vs the templated Style C solver
// (DTXQCDMultiShiftCGQUDA_StyleC instantiated on DTXQCDMpcOpQUDA_EO),
// same source.  Gates:
//   per-shift ||psi_grid - psi_quda|| / ||psi_grid||  <= 1e-10
//   per-shift true residual ||(Mpc†Mpc + pole)psi - src|| / ||src|| <= 10*tol
//
// Run: ./Test_dtxqcd_multishift_quda_eo --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC.h>
#include <Grid/util/QudaInit.h>
#include <cstdlib>

using namespace Grid;

namespace {
RealD rel_diff(const DTXQCDFermionDoubled &a, const DTXQCDFermionDoubled &b) {
  RealD num = 0, den = 0;
  for (int f = 0; f < DtxqcdNf; ++f) {
    LatticeFermion du(a.upper.f[f].Grid()), dl(a.upper.f[f].Grid());
    du = a.upper.f[f] - b.upper.f[f];
    dl = a.lower.f[f] - b.lower.f[f];
    num += norm2(du) + norm2(dl);
    den += norm2(b.upper.f[f]) + norm2(b.lower.f[f]);
  }
  return std::sqrt(num / std::max(den, 1e-300));
}
RealD dnorm2(const DTXQCDFermionDoubled &a) {
  RealD n = 0;
  for (int f = 0; f < DtxqcdNf; ++f)
    n += norm2(a.upper.f[f]) + norm2(a.lower.f[f]);
  return n;
}
}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Grid::Quda::initialize();

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dv; std::stringstream ss(argv[i + 1]); std::string t;
      while (std::getline(ss, t, '.')) dv.push_back(std::stoi(t));
      if (dv.size() == (size_t)Nd) latt = Coordinate(dv);
    }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridParallelRNG pRNG(&Grid_); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  auto env_real = [](const char *n, RealD d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atof(e) : d;
  };
  const RealD lambda = env_real("LAMBDA", 4.0);
  const RealD mass   = env_real("MASS", -0.245);
  const RealD csw    = env_real("CSW", 1.24930970916466);

  DTXQCDField U(&Grid_);
  PeriodicGimplR::TepidConfiguration(pRNG, U.U);
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, 0.0);

  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMpcOp mpc(Dw);
  DTXQCDMpcOpQUDA_EO Mq(Dw, U.U, Grid_, RBGrid, mass, csw,
                        U.sigma, U.pi, U.d, U.n, U.s, U.p);

  // 10-shift geometric ladder (x^{-1/4}-like) + uniform tight tolerance.
  const int nshift = 10;
  std::vector<RealD> poles(nshift), tols(nshift, 1e-10);
  for (int s = 0; s < nshift; ++s)
    poles[s] = 0.05 * std::pow(2.2, s);   // 0.05 ... ~25

  DTXQCDFermionDoubled full(&Grid_), src(&RBGrid);
  for (int f = 0; f < DtxqcdNf; ++f) {
    gaussian(pRNG, full.upper.f[f]);
    gaussian(pRNG, full.lower.f[f]);
  }
  for (int f = 0; f < DtxqcdNf; ++f) {
    pickCheckerboard(Odd, src.upper.f[f], full.upper.f[f]);
    pickCheckerboard(Odd, src.lower.f[f], full.lower.f[f]);
  }

  std::vector<DTXQCDFermionDoubled> psi_grid, psi_quda;
  for (int s = 0; s < nshift; ++s) {
    psi_grid.emplace_back(&RBGrid);
    psi_quda.emplace_back(&RBGrid);
  }

  std::cout << GridLogMessage << "[gate2-eo] Grid-DP multishift..." << std::endl;
  DTXQCDMultiShiftCG(mpc, poles, tols, src, psi_grid, 20000);

  std::cout << GridLogMessage << "[gate2-eo] Style C (EO device) multishift..."
            << std::endl;
  DTXQCDMultiShiftCGQUDA_StyleC(mpc, Mq, poles, tols, src, psi_quda, 20000);

  int fails = 0;
  for (int s = 0; s < nshift; ++s) {
    RealD rd = rel_diff(psi_quda[s], psi_grid[s]);
    // true residual of the Style C solution
    DTXQCDFermionDoubled t1(&RBGrid), t2(&RBGrid), r(&RBGrid);
    mpc.M(psi_quda[s], t1);
    mpc.Mdag(t1, t2);
    for (int f = 0; f < DtxqcdNf; ++f) {
      r.upper.f[f] = src.upper.f[f] - t2.upper.f[f] - poles[s] * psi_quda[s].upper.f[f];
      r.lower.f[f] = src.lower.f[f] - t2.lower.f[f] - poles[s] * psi_quda[s].lower.f[f];
    }
    RealD res = std::sqrt(dnorm2(r) / dnorm2(src));
    bool ok = (rd < 1e-10) && (res < 10 * tols[s]);
    if (!ok) ++fails;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ")
              << "shift " << s << " pole=" << poles[s]
              << "  |dpsi|rel=" << rd << "  true_res=" << res << std::endl;
  }
  std::cout << GridLogMessage << "==== gate2-eo "
            << (fails ? "FAILED" : "PASSED") << " (" << fails << ") ===="
            << std::endl;
  Grid_finalize();
  std::_Exit(fails ? 1 : 0);   // skip QUDA static-teardown crash
}
