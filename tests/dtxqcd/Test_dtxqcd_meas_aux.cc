// Test_dtxqcd_meas_aux: standalone per-cfg aux-field correlator measurement.
//
// Mirrors production/meas_aux_txqcd.cc.  Loads gauge + daux + RNG from a
// DTXQCD checkpoint and writes wall-wall correlators (isovector π±/π⁰,
// isovector a0±/a0⁰, isoscalar s, p, Tr σ, Tr π, and mixed Tr σ·s / Tr π·p)
// to HDF5.  Use to run on existing config streams; the same correlators are
// also recorded every trajectory by DtxqcdDiagnostics (Test_dtxqcd_2pt_utils.h).
//
// Usage:
//   ./Test_dtxqcd_meas_aux <traj> --grid <Lx.Ly.Lz.Lt> --mpi <m...>
//
// Env knobs:
//   CFG_DIR (in utils.h): override default configs_2pt_dtxqcd directory
//   LAMBDA:               override compile-time lambda (must match HMC stream)
//   OUT_DIR:              output directory (default <CFG_DIR>/aux_meas)

#include "Test_dtxqcd_2pt_utils.h"

using namespace DtxqcdTest2pt;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: Test_dtxqcd_meas_aux <traj> --grid <L> --mpi <m>"
              << std::endl;
    Grid_finalize();
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  // LAMBDA env override; defaults to compile-time params lambda.
  RealD lambda_run = lambda;
  if (const char *l = std::getenv("LAMBDA"); l && *l) {
    lambda_run = std::atof(l);
  }

  std::string cfg_dir = dtxqcd_cfg_dir();
  std::string out_dir = cfg_dir + "/aux_meas";
  if (const char *o = std::getenv("OUT_DIR"); o && *o) out_dir = std::string(o);
  mkdir_p(out_dir);

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= (RealD)latt[mu];
  RealD V3 = V4 / RealD(T);

  std::cout << GridLogMessage << "======== Test_dtxqcd_meas_aux ========"
            << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lambda_run
            << "    lambda^4 = "
            << lambda_run * lambda_run * lambda_run * lambda_run << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3]
            << "    V4 = " << V4 << "    V3 = " << V3
            << "    T = " << T << std::endl;
  std::cout << GridLogMessage << "  Nf      = " << DtxqcdNf
            << "    Nc = " << Nc << "    Nd = " << Nd << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << cfg_dir << std::endl;
  std::cout << GridLogMessage << "  out dir = " << out_dir << std::endl;
  std::cout << GridLogMessage << "======================================="
            << std::endl;

  DTXQCDField U(&Grid);
  LoadDtxqcdConfig(U, sRNG, pRNG, traj, cfg_dir);

  // All channels computed via the shared header (also used by the per-traj
  // HMC diagnostic).  Slice sums are collective — must run on every rank.
  DtxqcdAuxWallCorrelators awc =
      DtxqcdComputeAuxWallCorrelators(U, lambda_run);

  std::string outfile = out_dir + "/aux_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);

    // Raw correlators (λ⁴-scaled).  Analysis subtracts disconnected
    // ⟨W⟩⟨W*⟩ pieces using ensemble means of wall slices.
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

    // Wall slices (raw, not λ-scaled)
    write(wr, "aux_wall_sig_ab", awc.wall_sig_ab_flat);
    write(wr, "aux_wall_pi_ab",  awc.wall_pi_ab_flat);
    write(wr, "aux_wall_s",      awc.wall_s);
    write(wr, "aux_wall_p",      awc.wall_p);
    write(wr, "aux_wall_trsig",  awc.wall_trsig);
    write(wr, "aux_wall_trpi",   awc.wall_trpi);

    write(wr, "traj",   traj);
    write(wr, "lambda", lambda_run);
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
