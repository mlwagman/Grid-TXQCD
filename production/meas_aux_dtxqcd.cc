// meas_aux_dtxqcd: standalone per-cfg aux-field wall-wall correlator
// measurement for DTXQCD v2.  Sibling of meas_aux_txqcd.cc.
//
// Loads gauge + daux + RNG from a DTXQCD checkpoint and writes wall-wall
// correlators for:
//   Isovector pion triplet (pi+/pi-/pi0 from color-traced pi field)
//   Isovector a0 triplet (a0+/a0-/a00 from color-traced sigma field)
//   Isoscalar quartet (s, p, Tr sigma, Tr pi)
//   Mixed isoscalar (Tr sigma * s, Tr pi * p) -- for SD identity check
// to HDF5.  Uses the shared kernel in
// Grid/qcd/action/dtxqcd/DTXQCDAuxCorrelator.h (also used by the per-traj
// HMC diagnostic inside Test_dtxqcd_2pt_gencfgs).
//
// Usage:
//   ./meas_aux_dtxqcd <traj> [--grid LxLyLzLt] [--mpi mxmymzmt]
//
// Env knobs (in addition to params.h LAMBDA_DTXQCD / MASS_LIGHT_DTXQCD):
//   CFG_DIR  : override default cfgs/dtxqcd_<lambda> input directory
//   DATA_DIR : override default meas_2pt/dtxqcd_<lambda> output directory

#include "params.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxCorrelator.h>

using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_aux_dtxqcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  std::string cfg_dir = dtxqcd_cfg_dir();
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = std::string(d);
  std::string data_dir = dtxqcd_data_dir();
  if (const char *d = std::getenv("DATA_DIR"); d && *d) data_dir = std::string(d);
  mkdir_p(data_dir);

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= (RealD)latt[mu];
  RealD V3 = V4 / RealD(T);
  const RealD lam   = lambda_dtxqcd;
  const RealD lam4  = lam * lam * lam * lam;

  std::cout << GridLogMessage << "======== meas_aux_dtxqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lam
            << "    lambda^4 = " << lam4 << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3]
            << "    V4 = " << V4 << "    V3 = " << V3
            << "    T = " << T << std::endl;
  std::cout << GridLogMessage << "  Nf      = " << DtxqcdNf
            << "    Nc = " << Nc << "    Nd = " << Nd << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << cfg_dir << std::endl;
  std::cout << GridLogMessage << "  out dir = " << data_dir << std::endl;
  std::cout << GridLogMessage << "==================================" << std::endl;

  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);

  // All channels computed via the shared header (also used by the per-traj
  // HMC diagnostic in Test_dtxqcd_2pt_gencfgs).  Slice sums are collective:
  // must run on every rank.
  DtxqcdAuxWallCorrelators awc = DtxqcdComputeAuxWallCorrelators(U, lam);

  std::string outfile = data_dir + "/aux_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);

    // Raw correlators (lambda^4 scaled).  Analysis subtracts disconnected
    // <W><W*> pieces using ensemble means of wall_*_slice.
    write(wr, "aux_C_pi_plus",  awc.C_pi_plus);
    write(wr, "aux_C_pi_minus", awc.C_pi_minus);
    write(wr, "aux_C_pi_zero",  awc.C_pi_zero);
    write(wr, "aux_C_a0_plus",  awc.C_a0_plus);
    write(wr, "aux_C_a0_minus", awc.C_a0_minus);
    write(wr, "aux_C_a0_zero",  awc.C_a0_zero);
    write(wr, "aux_C_s",        awc.C_s);
    write(wr, "aux_C_p",        awc.C_p);
    write(wr, "aux_C_trsig",    awc.C_trsig);
    write(wr, "aux_C_trpi",     awc.C_trpi);
    write(wr, "aux_C_trsig_s",  awc.C_trsig_s);
    write(wr, "aux_C_trpi_p",   awc.C_trpi_p);

    // Wall slices (raw, not lambda-scaled)
    write(wr, "aux_wall_sig_ab", awc.wall_sig_ab_flat);
    write(wr, "aux_wall_pi_ab",  awc.wall_pi_ab_flat);
    write(wr, "aux_wall_d_ab",   awc.wall_d_ab_flat);
    write(wr, "aux_wall_n_ab",   awc.wall_n_ab_flat);
    write(wr, "aux_wall_s",      awc.wall_s);
    write(wr, "aux_wall_p",      awc.wall_p);
    write(wr, "aux_wall_trsig",  awc.wall_trsig);
    write(wr, "aux_wall_trpi",   awc.wall_trpi);
    write(wr, "aux_wall_trd",    awc.wall_trd);
    write(wr, "aux_wall_trn",    awc.wall_trn);

    write(wr, "traj",   traj);
    write(wr, "lambda", lam);
    write(wr, "Nf",     DtxqcdNf);
    write(wr, "Nc",     Nc);
    write(wr, "Nd",     Nd);
    write(wr, "T",      T);
    write(wr, "V3",     V3);
    write(wr, "V4",     V4);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
