#pragma once
#include <Grid/Grid.h>

using namespace Grid;

namespace TXQCDProduction {

// ===== Lattice geometry =====
inline Coordinate lattice_size() { return Coordinate(std::vector<int>{16, 16, 16, 48}); }

// ===== Quark masses =====
constexpr RealD mass_light   = -0.2450;
constexpr RealD mass_strange = -0.2450;
constexpr RealD csw = 1.24930970916466;

// ===== Gauge action =====
constexpr RealD beta = 6.1;
constexpr RealD u0 = 0.832605301399891;
constexpr RealD lambda = 0.5;

// ===== Stout smearing for inversions =====
constexpr RealD stout_rho_inv = 0.125;
constexpr int   stout_nsmear_inv = 1;

// ===== Stout smearing for source/sink construction =====
constexpr RealD stout_rho_src = 0.16;
constexpr int   stout_nsmear_src = 3;

// ===== Gaussian source/sink smearing =====
constexpr RealD gauss_width = 2.1;
constexpr int   gauss_niter = 20;

// ===== Connected source grid =====
constexpr int space_src_per_dim = 4;
constexpr int time_src_per_dim  = 12;
inline Coordinate src_grid_origin() { return Coordinate(std::vector<int>{0, 0, 0, 0}); }

// ===== VEV monitoring =====
constexpr int n_vev_noise = 8;

// ===== Disconnected =====
constexpr int n_noise_disco = 32;

// ===== HMC =====
constexpr int n_therm = 100;
constexpr int n_prod = 1000;
constexpr int meas_skip = 10;

// ===== Solver =====
constexpr RealD cg_tol = 1e-8;
constexpr int   cg_max = 30000;

// ===== Config paths =====
inline std::string lambda_tag() {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(4) << lambda;
  return "lam" + ss.str();
}
inline std::string txqcd_cfg_dir()  { return "cfgs/txqcd_" + lambda_tag(); }
inline std::string txqcd_data_dir() { return "meas_2pt/txqcd_" + lambda_tag(); }
inline std::string qcd_cfg_dir()    { return "cfgs/qcd"; }
inline std::string qcd_data_dir()   { return "meas_2pt/qcd"; }

// ===== Measurement trajectories =====
inline std::vector<int> meas_trajs() {
  std::vector<int> v;
  for (int t = n_therm; t < n_therm + n_prod; t += meas_skip)
    v.push_back(t);
  return v;
}

// ===== Utility =====
inline void mkdir_p(const std::string &path) {
  std::string cmd = "mkdir -p " + path;
  system(cmd.c_str());
}

inline bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

}  // namespace TXQCDProduction
