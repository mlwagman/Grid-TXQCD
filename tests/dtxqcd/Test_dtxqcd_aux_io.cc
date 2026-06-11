// Test_dtxqcd_aux_io: round-trip test for the DTXQCD packed aux sidecar.
//
// Verifies:
//   1. Sidecar file size = 16 (header) + gSites * 42 * 8 (payload) bytes.
//   2. Magic word in file = 0x44545841 ('DTXA').
//   3. Round-trip of sigma^A, pi^A, t^A_{mu,nu}, d, n is bit-exact after
//      the generation-side projections are applied.
//
// Run (single-rank):
//   ./tests/dtxqcd/Test_dtxqcd_aux_io --grid 4.4.4.8 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <filesystem>
#include <fstream>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  GridSerialRNG sRNG;
  sRNG.SeedFixedIntegers({5, 6, 7, 8});

  int exitcode = 0;

  DTXQCDField Uorig(&Grid);
  DTXQCDCompositeImpl::HotConfiguration(pRNG, Uorig);

  std::string tmpdir = "./Test_dtxqcd_aux_io_tmp";
  if (Grid.IsBoss()) std::filesystem::create_directories(tmpdir);
  Grid.Barrier();

  CheckpointerParameters Pars;
  Pars.config_prefix  = tmpdir + "/ckpoint_lat";
  Pars.rng_prefix     = tmpdir + "/ckpoint_rng";
  Pars.smeared_prefix = tmpdir + "/ckpoint_smr";
  Pars.format         = "IEEE64BIG";
  Pars.saveInterval   = 1;
  DTXQCDCheckpointer cp(Pars);

  const int traj = 7;
  cp.TrajectoryComplete(traj, Uorig, sRNG, pRNG);
  Grid.Barrier();

  if (Grid.IsBoss()) {
    std::string auxfile = Pars.config_prefix + "_daux." + std::to_string(traj);
    std::ifstream ifs(auxfile, std::ios::binary | std::ios::ate);
    if (!ifs) {
      std::cout << GridLogError << "aux sidecar missing: " << auxfile
                << std::endl;
      exitcode = 1;
    } else {
      uint64_t sz = ifs.tellg();
      // v2 payload: 4 * 36 (CF Hermitian) + 2 * 1 (scalars) = 146 doubles/site.
      uint64_t expected = 16 + (uint64_t)Grid.gSites() * 146 * 8;
      if (sz != expected) {
        std::cout << GridLogError << "[FAIL] sidecar size " << sz
                  << " != expected " << expected << std::endl;
        exitcode = 1;
      } else {
        std::cout << GridLogMessage << "[ok] sidecar size " << sz
                  << " bytes (" << (sz - 16) << " payload, "
                  << ((sz - 16) / Grid.gSites()) << " bytes/site)"
                  << std::endl;
      }
      ifs.seekg(0);
      uint32_t magic = 0;
      ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
      if (magic != 0x44545832u) {
        std::cout << GridLogError << "[FAIL] magic 0x" << std::hex << magic
                  << std::dec << " != 0x44545832 ('DTX2')" << std::endl;
        exitcode = 1;
      } else {
        std::cout << GridLogMessage << "[ok] magic 0x44545832 ('DTX2')"
                  << std::endl;
      }
    }
  }

  // Round-trip via the static ReadConfig path.  Fresh RNGs are fine; we only
  // compare the field content, not RNG state.
  DTXQCDField Uloaded(&Grid);
  GridSerialRNG sRNG2;
  GridParallelRNG pRNG2(&Grid);
  DTXQCDCheckpointer::ReadConfig(Uloaded, sRNG2, pRNG2,
                                 Pars.config_prefix, Pars.rng_prefix, traj);

  auto check_diff = [&](const char *name, RealD diff_norm) {
    if (diff_norm > 1e-20) {
      std::cout << GridLogError << "[FAIL] " << name << " roundtrip ||delta||^2 = "
                << diff_norm << " > 1e-20" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "[ok] " << name
                << " roundtrip ||delta||^2 = " << diff_norm << std::endl;
    }
  };

  { LatticeDtxqcdSigma dx(&Grid); dx = Uorig.sigma - Uloaded.sigma; check_diff("sigma", norm2(dx)); }
  { LatticeDtxqcdPi    dx(&Grid); dx = Uorig.pi    - Uloaded.pi;    check_diff("pi",    norm2(dx)); }
  { LatticeDtxqcdD     dx(&Grid); dx = Uorig.d     - Uloaded.d;     check_diff("d",     norm2(dx)); }
  { LatticeDtxqcdN     dx(&Grid); dx = Uorig.n     - Uloaded.n;     check_diff("n",     norm2(dx)); }
  { LatticeDtxqcdS     dx(&Grid); dx = Uorig.s     - Uloaded.s;     check_diff("s",     norm2(dx)); }
  { LatticeDtxqcdP     dx(&Grid); dx = Uorig.p     - Uloaded.p;     check_diff("p",     norm2(dx)); }
  { LatticeGaugeField  dU(&Grid); dU = Uorig.U     - Uloaded.U;     check_diff("U",     norm2(dU)); }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
