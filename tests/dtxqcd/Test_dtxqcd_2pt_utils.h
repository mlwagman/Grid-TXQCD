#pragma once
// Shared utilities for the DTXQCD 2pt Fierz-equivalence test suite.
// Mirror of Test_txqcd_2pt_utils.h with the field roster updated for the
// diquark-tensor variant.  Re-uses the QCD reference ensemble that
// Test_txqcd_2pt_gencfgs generates -- the DTXQCD vs QCD comparison shares
// the same QCD configs (configs_2pt_qcd_nf2), so a 2pt suite run only needs
// to generate the DTXQCD half.
//
// Configs land in configs_2pt_dtxqcd/ with the standard NERSC gauge file
// (ckpoint_lat.N) plus the DTXQCDCheckpointer aux sidecar
// (ckpoint_lat_daux.N, magic = DTXA = 0x44545841) and Grid RNG state
// (ckpoint_rng.N).

#include "../txqcd/Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxCorrelator.h>

namespace DtxqcdTest2pt {

using namespace TxqcdTest2pt;  // shared params (beta, lambda, mass, etc.) +
                                // file helpers + correlator math + QCD diag

// DTXQCD-specific config directory.  Sits beside configs_2pt_txqcd /
// configs_2pt_qcd_nf2 so a single run of the 2pt suite produces all three
// ensembles at the same trajectory range.
inline std::string dtxqcd_cfg_dir() {
  if (const char *d = std::getenv("CFG_DIR"); d && *d) return std::string(d);
  return "configs_2pt_dtxqcd";
}

// Existence check: in addition to gauge + rng, the DTXQCD sidecar has the
// "_daux" suffix (not "_aux", which is the TXQCD sidecar).
inline bool dtxqcd_configs_exist(const std::string &dir) {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(dir + "/ckpoint_lat."      + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_lat_daux." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng."      + std::to_string(t))) return false;
  }
  return true;
}
inline bool dtxqcd_configs_exist() { return dtxqcd_configs_exist(dtxqcd_cfg_dir()); }

inline int latest_dtxqcd_checkpoint(const std::string &dir) {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    if (file_exists(dir + "/ckpoint_lat."      + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_lat_daux." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng."      + std::to_string(t)))
      latest = t;
  }
  return latest;
}
inline int latest_dtxqcd_checkpoint() {
  return latest_dtxqcd_checkpoint(dtxqcd_cfg_dir());
}

inline void LoadDtxqcdConfig(DTXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG, int traj,
                              const std::string &dir) {
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  dir + "/ckpoint_lat",
                                  dir + "/ckpoint_rng", traj);
}
inline void LoadDtxqcdConfig(DTXQCDField &U, GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG, int traj) {
  LoadDtxqcdConfig(U, sRNG, pRNG, traj, dtxqcd_cfg_dir());
}

// Per-trajectory HMC diagnostics for DTXQCD.  Mirrors TxqcdDiagnostics but
// the aux field roster is (sigma^A, pi^A, t^A, d, n) instead of TXQCD's
// (sigma, pi, s, p, t) -- so the recorded VEVs are scalar norm-squared
// sums per slot (the Pauli triplet sigma^A is not a flavor matrix where
// Tr applies, and there's no s/p field).  Tr M^{-1} on the gauge field
// works identically to QCD/TXQCD's path.
struct DtxqcdDiagnostics : HmcDiagWriter<DTXQCDField> {
  GridCartesian         &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG       &prng_;
  RealD mass_;
  RealD lambda_;
  int   n_vev_noise_;
  std::vector<RealD> norm_sigma_, norm_pi_, norm_d_, norm_n_, norm_s_, norm_p_;
  std::vector<RealD> vev_trminv_;
  // Per-traj aux wall-wall correlators (length T or Nf²·T each).  Outer dim
  // accumulates trajectories since last flush; flushed at meas_skip intervals.
  std::vector<std::vector<ComplexD>>
      aux_C_pi_plus_, aux_C_pi_minus_, aux_C_pi_zero_,
      aux_C_a0_plus_, aux_C_a0_minus_, aux_C_a0_zero_,
      aux_C_s_, aux_C_p_, aux_C_trsig_, aux_C_trpi_,
      aux_C_trsig_s_, aux_C_trpi_p_;
  std::vector<std::vector<ComplexD>>
      aux_wall_sig_ab_, aux_wall_pi_ab_,
      aux_wall_s_, aux_wall_p_, aux_wall_trsig_, aux_wall_trpi_;

  DtxqcdDiagnostics(const std::string &prefix, int interval,
                    std::vector<ActionRef> acts,
                    GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                    GridParallelRNG &prng, RealD mass, RealD lambda,
                    int n_vev_noise)
      : HmcDiagWriter(prefix, interval, std::move(acts), true),
        grid_(grid), rbgrid_(rbgrid), prng_(prng),
        mass_(mass), lambda_(lambda), n_vev_noise_(n_vev_noise) {}

  RealD get_plaq(DTXQCDField &U) override {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  }

  // Volume-averaged Tr M^{-1} via Hutchinson stochastic estimator, same
  // structure as TXQCD / QCD diagnostics.  Acts on the gauge field only,
  // so the DTXQCD aux fields don't enter here -- this is a "QCD"-like
  // observable evaluated on the DTXQCD-generated gauge background.
  RealD compute_trminv(LatticeGaugeField &Uvev) {
    WilsonFermionD Dw(Uvev, grid_, rbgrid_, mass_);
    MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    RealD V = (RealD)grid_.gSites();
    RealD acc = 0.0;
    for (int h = 0; h < n_vev_noise_; ++h) {
      LatticeFermion eta(&grid_), b(&grid_), x(&grid_);
      gaussian(prng_, eta);
      Dw.Mdag(eta, b);
      x = Zero();
      CG(HermOp, b, x);
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    return acc / n_vev_noise_;
  }

  void record_aux(DTXQCDField &U) override {
    RealD V = (RealD)U.Grid()->gSites();
    norm_sigma_.push_back(norm2(U.sigma) / V);
    norm_pi_   .push_back(norm2(U.pi)    / V);
    norm_d_    .push_back(norm2(U.d)     / V);
    norm_n_    .push_back(norm2(U.n)     / V);
    norm_s_    .push_back(norm2(U.s)     / V);
    norm_p_    .push_back(norm2(U.p)     / V);
    vev_trminv_.push_back(compute_trminv(U.U));
    // Aux wall-wall correlators (collective sliceSum — must run on every
    // rank).  Cost ≪1% of a trajectory.
    DtxqcdAuxWallCorrelators awc =
        DtxqcdComputeAuxWallCorrelators(U, lambda_);
    aux_C_pi_plus_  .push_back(std::move(awc.C_pi_plus));
    aux_C_pi_minus_ .push_back(std::move(awc.C_pi_minus));
    aux_C_pi_zero_  .push_back(std::move(awc.C_pi_zero));
    aux_C_a0_plus_  .push_back(std::move(awc.C_a0_plus));
    aux_C_a0_minus_ .push_back(std::move(awc.C_a0_minus));
    aux_C_a0_zero_  .push_back(std::move(awc.C_a0_zero));
    aux_C_s_        .push_back(std::move(awc.C_s));
    aux_C_p_        .push_back(std::move(awc.C_p));
    aux_C_trsig_    .push_back(std::move(awc.C_trsig));
    aux_C_trpi_     .push_back(std::move(awc.C_trpi));
    aux_C_trsig_s_  .push_back(std::move(awc.C_trsig_s));
    aux_C_trpi_p_   .push_back(std::move(awc.C_trpi_p));
    aux_wall_sig_ab_.push_back(std::move(awc.wall_sig_ab_flat));
    aux_wall_pi_ab_ .push_back(std::move(awc.wall_pi_ab_flat));
    aux_wall_s_     .push_back(std::move(awc.wall_s));
    aux_wall_p_     .push_back(std::move(awc.wall_p));
    aux_wall_trsig_ .push_back(std::move(awc.wall_trsig));
    aux_wall_trpi_  .push_back(std::move(awc.wall_trpi));
  }

  void flush(int traj) override {
    if (traj_.empty()) return;
    std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
    Hdf5Writer wr(fname);
    write(wr, "traj", traj_);
    write(wr, "plaq", plaq_);
    write(wr, "force_avg", force_avg_);
    write(wr, "force_max", force_max_);
    write(wr, "fdt_avg",   fdt_avg_);
    write(wr, "fdt_max",   fdt_max_);
    write(wr, "norm_sigma", norm_sigma_);
    write(wr, "norm_pi",    norm_pi_);
    write(wr, "norm_d",     norm_d_);
    write(wr, "norm_n",     norm_n_);
    write(wr, "norm_s",     norm_s_);
    write(wr, "norm_p",     norm_p_);
    write(wr, "vev_trminv", vev_trminv_);
    // Aux wall-wall correlators (one row per traj since last flush)
    write(wr, "aux_C_pi_plus",  aux_C_pi_plus_);
    write(wr, "aux_C_pi_minus", aux_C_pi_minus_);
    write(wr, "aux_C_pi_zero",  aux_C_pi_zero_);
    write(wr, "aux_C_a0_plus",  aux_C_a0_plus_);
    write(wr, "aux_C_a0_minus", aux_C_a0_minus_);
    write(wr, "aux_C_a0_zero",  aux_C_a0_zero_);
    write(wr, "aux_C_s",        aux_C_s_);
    write(wr, "aux_C_p",        aux_C_p_);
    write(wr, "aux_C_trsig",    aux_C_trsig_);
    write(wr, "aux_C_trpi",     aux_C_trpi_);
    write(wr, "aux_C_trsig_s",  aux_C_trsig_s_);
    write(wr, "aux_C_trpi_p",   aux_C_trpi_p_);
    write(wr, "aux_wall_sig_ab", aux_wall_sig_ab_);
    write(wr, "aux_wall_pi_ab",  aux_wall_pi_ab_);
    write(wr, "aux_wall_s",      aux_wall_s_);
    write(wr, "aux_wall_p",      aux_wall_p_);
    write(wr, "aux_wall_trsig",  aux_wall_trsig_);
    write(wr, "aux_wall_trpi",   aux_wall_trpi_);
    std::vector<std::string> names;
    for (auto &a : actions_) names.push_back(a.name);
    write(wr, "action_names", names);

    traj_.clear(); plaq_.clear();
    force_avg_.clear(); force_max_.clear();
    fdt_avg_.clear();  fdt_max_.clear();
    norm_sigma_.clear(); norm_pi_.clear();
    norm_d_.clear();     norm_n_.clear();
    norm_s_.clear();     norm_p_.clear();
    vev_trminv_.clear();
    aux_C_pi_plus_.clear();  aux_C_pi_minus_.clear(); aux_C_pi_zero_.clear();
    aux_C_a0_plus_.clear();  aux_C_a0_minus_.clear(); aux_C_a0_zero_.clear();
    aux_C_s_.clear();        aux_C_p_.clear();
    aux_C_trsig_.clear();    aux_C_trpi_.clear();
    aux_C_trsig_s_.clear();  aux_C_trpi_p_.clear();
    aux_wall_sig_ab_.clear(); aux_wall_pi_ab_.clear();
    aux_wall_s_.clear();      aux_wall_p_.clear();
    aux_wall_trsig_.clear();  aux_wall_trpi_.clear();

    std::cout << GridLogMessage << "HMC diagnostics written to " << fname
              << std::endl;
  }
};

}  // namespace DtxqcdTest2pt
