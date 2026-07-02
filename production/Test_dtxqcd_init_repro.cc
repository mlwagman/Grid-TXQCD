// Test_dtxqcd_init_repro: reproduce the EXACT production fresh-start aux init.
//
// gen_dtxqcd_cfgs.cc InitFreshDtxqcdField re-seeds pRNG to {6,7,8,9,10}
// immediately before the FINAL FillAuxFields (line 268), so the production aux
// Gaussian draw starts at RNG stream position ZERO.  The
// Test_dtxqcd_aux_drive_symmetry reconstruction instead drew the weak-field
// gauge first (consuming stream) and then filled aux -> its init is a
// DIFFERENT Gaussian sample, and its w = <p> - <Trpi> = -0.081 is not the
// chain's init.  This test prints both orderings so the chain's true init
// (expected w ~ -0.13, matching traj 1 of the lambda=0.1 chain up to one
// trajectory of momentum diffusion ~0.003) can be confirmed.
//
// Run: ./Test_dtxqcd_init_repro --grid 16.16.16.48 --mpi 1.1.1.1
//      (SIGMA_INIT=0.221654383 LAMBDA=0.1 by default)

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <cstdlib>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{16, 16, 16, 48});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dvec;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) dvec.push_back(std::stoi(tok));
      if (dvec.size() == (size_t)Nd) latt = Coordinate(dvec);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  auto env_real = [](const char *name, RealD dflt) {
    const char *e = std::getenv(name);
    return (e && *e) ? std::atof(e) : dflt;
  };
  const RealD lambda = env_real("LAMBDA", 0.1);
  const RealD Sigma  = env_real("SIGMA_INIT", 0.221654383);

  auto report = [&](const char *tag, DTXQCDField &U) {
    RealD V = (RealD)Grid.gSites();
    RealD trsig = TensorRemove(sum(trace(U.sigma))).real() / V;
    RealD trpi  = TensorRemove(sum(trace(U.pi))).real() / V;
    RealD s     = TensorRemove(sum(U.s)).real() / V;
    RealD p     = TensorRemove(sum(U.p)).real() / V;
    std::cout << GridLogMessage << tag
              << "  <Trsig>=" << trsig << "  <s>=" << s
              << "  w_sig=s-Trsig=" << (s - trsig) << std::endl;
    std::cout << GridLogMessage << tag
              << "  <Trpi>=" << trpi << "  <p>=" << p
              << "  w_pi=p-Trpi=" << (p - trpi) << std::endl;
    std::cout << GridLogMessage << tag
              << "  ||pi||^2/V=" << norm2(U.pi) / V
              << "  ||sigma||^2/V=" << norm2(U.sigma) / V << std::endl;
  };

  // ---- A: production driver order (reseed before final fill) ----
  {
    GridParallelRNG pRNG(&Grid);
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    DTXQCDField U(&Grid);
    DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});   // driver line 268
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma);
    report("[driver order]", U);
  }

  // ---- B: drive-symmetry-test order (no reseed; gauge consumed stream) ----
  {
    GridParallelRNG pRNG(&Grid);
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    DTXQCDField U(&Grid);
    DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma);
    report("[test order  ]", U);
  }

  Grid_finalize();
  return 0;
}
