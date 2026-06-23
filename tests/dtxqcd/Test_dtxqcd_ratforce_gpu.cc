// Test_dtxqcd_ratforce_gpu: bit-exact agreement between the CPU thread_for
// reference and the GPU device kernel (DTXQCDRationalForceGpuKernel.h) for
// AccumulateSiteForces (the per-pole, per-CB aux + clover-sigma site loop).
//
// Both paths feed the same (X, Y) random fermion bilinear through identical
// math; the GPU path is byte-for-byte the device twin of the CPU path per the
// kernel header.  This test fires AccumulateSiteForces twice within one
// process by setenv("DTXQCD_RATFORCE_GPU", "0|1", 1) — needs the per-call env
// read that RatForceGpuEnabled() now does (no more static cache).
//
// Compares the 6 aux output channels (.sigma, .pi, .d, .n, .s, .p on dSdU)
// and the 6 (mu, nu) clover_sigma_full ColourMatrix lattices, per CB
// (Even/Odd) per csw (0, 1.25), totalling 4 cases × 12 channel norms.
//
// Run:
//   ./tests/dtxqcd/Test_dtxqcd_ratforce_gpu --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>

using namespace Grid;

// Subclass to expose the protected AccumulateSiteForces for test use.
class ActionExposer : public DTXQCDWilsonCloverRationalEOAction {
 public:
  using Base = DTXQCDWilsonCloverRationalEOAction;
  using Base::Base;
  using Base::AccumulateSiteForces;
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
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
  pRNG.SeedFixedIntegers({921, 922, 923, 924});

  int exitcode = 0;
  auto check = [&](const char *name, RealD diff, RealD tol) {
    bool ok = (diff < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << "  max |cpu - gpu| = " << diff
              << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // Random gauge + aux to fix the cached EO operator.
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);

  const RealD mass = -0.245;
  OneFlavourRationalParams rp(/*lo=*/0.1, /*hi=*/64.0,
                              /*MaxIter=*/2000, /*tolerance=*/1e-10,
                              /*degree=*/10, /*precision=*/50,
                              /*BoundsCheckFreq=*/0, /*mdtolerance=*/1e-8);

  auto run_csw = [&](RealD csw, const char *tag) {
    ActionExposer act(Grid, RBGrid, mass, rp, csw);

    for (int cb_idx = 0; cb_idx < 2; ++cb_idx) {
      int cb = (cb_idx == 0) ? Even : Odd;
      bool odd_cb = (cb == Odd);
      const char *cbname = (cb == Even) ? "Even" : "Odd ";

      // Random X, Y on this CB (DTXQCDFermionDoubled).
      DTXQCDFermionDoubled X(&RBGrid), Y(&RBGrid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        LatticeFermion fu(&Grid), fl(&Grid);
        gaussian(pRNG, fu);
        gaussian(pRNG, fl);
        pickCheckerboard(cb, X.upper.f[a], fu);
        pickCheckerboard(cb, X.lower.f[a], fl);
        gaussian(pRNG, fu);
        gaussian(pRNG, fl);
        pickCheckerboard(cb, Y.upper.f[a], fu);
        pickCheckerboard(cb, Y.lower.f[a], fl);
      }

      const RealD ak = 1.0;

      // CPU reference: DTXQCD_RATFORCE_GPU=0.
      setenv("DTXQCD_RATFORCE_GPU", "0", 1);
      DTXQCDField dSdU_cpu(&Grid);
      dSdU_cpu = Zero();
      std::vector<LatticeColourMatrix> cs_cpu(6, LatticeColourMatrix(&Grid));
      for (int k = 0; k < 6; ++k) cs_cpu[k] = Zero();
      act.AccumulateSiteForces(X, Y, odd_cb, ak, dSdU_cpu, cs_cpu);

      // GPU device kernel: DTXQCD_RATFORCE_GPU=1.
      setenv("DTXQCD_RATFORCE_GPU", "1", 1);
      DTXQCDField dSdU_gpu(&Grid);
      dSdU_gpu = Zero();
      std::vector<LatticeColourMatrix> cs_gpu(6, LatticeColourMatrix(&Grid));
      for (int k = 0; k < 6; ++k) cs_gpu[k] = Zero();
      act.AccumulateSiteForces(X, Y, odd_cb, ak, dSdU_gpu, cs_gpu);

      // Compare 6 aux channels via norm-of-difference / sqrt(volume scale).
      auto diff_lat = [](auto &a, auto &b) -> RealD {
        std::decay_t<decltype(a)> d = a;
        d = a - b;
        return std::sqrt(norm2(d));
      };
      RealD d_sig = diff_lat(dSdU_cpu.sigma, dSdU_gpu.sigma);
      RealD d_pi  = diff_lat(dSdU_cpu.pi,    dSdU_gpu.pi);
      RealD d_d   = diff_lat(dSdU_cpu.d,     dSdU_gpu.d);
      RealD d_n   = diff_lat(dSdU_cpu.n,     dSdU_gpu.n);
      RealD d_s   = diff_lat(dSdU_cpu.s,     dSdU_gpu.s);
      RealD d_p   = diff_lat(dSdU_cpu.p,     dSdU_gpu.p);

      const std::string base = std::string(tag) + " " + cbname;
      check((base + " aux.sigma").c_str(), d_sig, 1e-10);
      check((base + " aux.pi   ").c_str(), d_pi,  1e-10);
      check((base + " aux.d    ").c_str(), d_d,   1e-10);
      check((base + " aux.n    ").c_str(), d_n,   1e-10);
      check((base + " aux.s    ").c_str(), d_s,   1e-10);
      check((base + " aux.p    ").c_str(), d_p,   1e-10);

      if (csw != 0.0) {
        RealD cs_max = 0.0;
        for (int k = 0; k < 6; ++k)
          cs_max = std::max(cs_max, diff_lat(cs_cpu[k], cs_gpu[k]));
        check((base + " clover_sigma[mn]").c_str(), cs_max, 1e-10);
      }
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
