#include "params.h"
#include "aux_correlator.h"

using namespace TXQCDProduction;

// ---- Main -----------------------------------------------------------------

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_aux_txqcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  mkdir_p(txqcd_data_dir());

  int T = latt[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= latt[mu];
  RealD V3 = V4 / RealD(T);
  const RealD lam4 = lambda * lambda * lambda * lambda;

  // --- Startup banner ---
  std::cout << GridLogMessage << "======== meas_aux_txqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lambda
            << "    lambda^4 = " << lam4 << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3]
            << "    V4 = " << V4 << "    V3 = " << V3
            << "    T = " << T << std::endl;
  std::cout << GridLogMessage << "  Nf      = " << TxqcdNf
            << "    Nc = " << Nc << "    Nd = " << Nd << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << txqcd_cfg_dir() << std::endl;
  std::cout << GridLogMessage << "  out dir = " << txqcd_data_dir() << std::endl;
  std::cout << GridLogMessage << "================================" << std::endl;

  TXQCDField U(&Grid);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                txqcd_cfg_dir() + "/ckpoint_lat",
                                txqcd_cfg_dir() + "/ckpoint_rng", traj);

  // All channels computed via the shared helper (also used by the HMC
  // diagnostic block in gen_txqcd_cfgs_2plus1.cc).  Slice sums are
  // collective — must run on every rank.
  AuxWallCorrelators awc = ComputeAuxWallCorrelators(U);

  // ----- Write h5 ----------------------------------------------------------
  std::string outfile = txqcd_data_dir() + "/aux_txqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);

    // RAW correlators only — no per-cfg vac-sub (biased for zero-mean fields).
    // Analysis must compute ensemble mean of wall_*_slice across cfgs and
    // form C_clean(τ) = C_raw(τ) - (λ⁴/V₃·T) Σ_t ⟨W(t+τ)⟩·⟨W*(t)⟩.

    write(wr, "aux_pi_raw", awc.C_pi_iso);
    write(wr, "wall_pi_tr_slice", awc.wall_pi_tr);
    write(wr, "aux_pi_ab_raw", awc.C_pi_ab_flat);
    write(wr, "wall_pi_ab_slice", awc.wall_pi_ab_flat);
    write(wr, "aux_sigma_raw", awc.C_sigma);
    write(wr, "wall_sigma_slice", awc.wall_sigma);
    write(wr, "aux_s_raw", awc.C_s);
    write(wr, "wall_s_slice", awc.wall_s);
    write(wr, "aux_p_raw", awc.C_p);
    write(wr, "wall_p_slice", awc.wall_p);
    write(wr, "aux_t_raw", awc.C_t_flat);
    write(wr, "wall_t_slice", awc.wall_t_flat);

    // metadata
    write(wr, "traj", traj);
    write(wr, "lambda", lambda);
    write(wr, "Nf", TxqcdNf);
    write(wr, "Nc", Nc);
    write(wr, "Nd", Nd);
    write(wr, "T", T);
    write(wr, "V3", V3);
    write(wr, "V4", V4);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
