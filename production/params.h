#pragma once
#include <Grid/Grid.h>

using namespace Grid;

namespace TXQCDProduction {

// ===== Lattice geometry =====
inline Coordinate lattice_size() { return Coordinate(std::vector<int>{16, 16, 16, 32}); }

// ===== Quark masses =====
constexpr RealD mass_light = -0.05;
constexpr RealD mass_strange = 0.04;
constexpr RealD csw = 1.0;

// ===== Gauge action =====
constexpr RealD beta = 6.0;
constexpr RealD u0 = 0.843;
constexpr RealD lambda = 0.5;

// ===== Stout smearing for inversions =====
constexpr RealD stout_rho_inv = 0.1;
constexpr int   stout_nsmear_inv = 3;

// ===== Stout smearing for source/sink construction =====
constexpr RealD stout_rho_src = 0.1;
constexpr int   stout_nsmear_src = 3;

// ===== Gaussian source/sink smearing =====
constexpr RealD gauss_width = 4.0;
constexpr int   gauss_niter = 50;

// ===== Connected source grid =====
constexpr int src_per_dim = 2;
inline Coordinate src_grid_origin() { return Coordinate(std::vector<int>{0, 0, 0, 0}); }

// ===== Disconnected =====
constexpr int n_noise_disco = 32;

// ===== HMC =====
constexpr int n_therm = 300;
constexpr int n_prod = 500;
constexpr int meas_skip = 10;

// ===== Solver =====
constexpr RealD cg_tol = 1e-8;
constexpr int   cg_max = 30000;

// ===== Config paths =====
inline std::string txqcd_cfg_dir() { return "cfgs/txqcd"; }
inline std::string qcd_cfg_dir()   { return "cfgs/qcd"; }
inline std::string data_dir()      { return "meas_2pt"; }

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
