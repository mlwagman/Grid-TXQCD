#pragma once
#include <Grid/Grid.h>
#include <cstdlib>

using namespace Grid;

namespace TXQCDProduction {

namespace detail {
inline RealD env_real(const char *name, RealD def) {
  if (const char *s = std::getenv(name); s && *s) return std::atof(s);
  return def;
}
inline int env_int(const char *name, int def) {
  if (const char *s = std::getenv(name); s && *s) return std::atoi(s);
  return def;
}
inline Coordinate env_latt(const char *name,
                            const std::vector<int> &def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return Coordinate(def);
  std::vector<int> v; std::string t;
  for (const char *p = s; *p; ++p) {
    if (*p == '.') { if (!t.empty()) v.push_back(std::atoi(t.c_str())); t.clear(); }
    else t.push_back(*p);
  }
  if (!t.empty()) v.push_back(std::atoi(t.c_str()));
  if ((int)v.size() != Nd) return Coordinate(def);
  return Coordinate(v);
}
}

// ===== Lattice geometry (override: LATT=L.L.L.T) =====
inline Coordinate lattice_size() {
  return detail::env_latt("LATT", std::vector<int>{16, 16, 16, 48});
}

// Runtime-overridable physics params (env var in parentheses).  Defaults are
// production values; set the env var to test small-lattice / bisection points.
inline const RealD mass_light   = detail::env_real("MASS_LIGHT",   -0.2450); // MASS_LIGHT
inline const RealD mass_strange = detail::env_real("MASS_STRANGE", -0.2450); // MASS_STRANGE
inline const RealD csw          = detail::env_real("CSW",           1.24930970916466); // CSW

// ===== Gauge action =====
inline const RealD beta   = detail::env_real("BETA",   6.1);               // BETA
inline const RealD u0     = detail::env_real("U0",     0.832605301399891); // U0
inline const RealD lambda = detail::env_real("LAMBDA", 0.5);               // LAMBDA

// ===== Stout smearing for inversions =====
inline const RealD stout_rho_inv   = detail::env_real("STOUT_RHO",    0.125); // STOUT_RHO
inline const int   stout_nsmear_inv = detail::env_int("STOUT_NSMEAR", 1);      // STOUT_NSMEAR

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
