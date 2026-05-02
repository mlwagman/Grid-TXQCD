#pragma once
// Pure Wilson Nf=2 TXQCD vs QCD Fierz test on a larger 8^3 x 24 lattice
// with lighter quark mass.  Distinguishing features vs the base
// Test_txqcd_2pt_* suite:
//   - 8^3 x 24 lattice (vs 4^3 x 8)
//   - csw = 0 (pure Wilson, no clover term) -- the existing base suite is
//     sometimes labelled "Wilson" but reads csw=1.0 from shared utils;
//     this header re-exports the Wilson constants with csw forced to 0.
//   - beta = 5.7, well past the bulk Aoki transition for plain Wilson
//   - m_l = 0.0 (lighter than the base m=0.3) but heavy enough to avoid
//     exceptional configurations from the lack of clover improvement.
//
// Reuses the Nf=2 framework (TXQCDWilsonRationalEOAction / TXQCDLogDetEOAction
// with mass scalar; aux fields with TxqcdNf=2 flavor matrices).  This means
// we keep the default Nf=2 build -- no -DTXQCD_Nf=3.

#include "Test_txqcd_2pt_utils.h"

namespace TxqcdTest2ptWilson8x24 {

using TxqcdTest2pt::lambda;          // 3.0; LAMBDA env var override below
using TxqcdTest2pt::n_therm;
using TxqcdTest2pt::n_prod;
using TxqcdTest2pt::meas_skip;
using TxqcdTest2pt::meas_tol;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_noise;
using TxqcdTest2pt::n_vev_noise;
using TxqcdTest2pt::mkdir_p;
using TxqcdTest2pt::file_exists;
using TxqcdTest2pt::CorrelatorFromSlice;
using TxqcdTest2pt::vmean;
using TxqcdTest2pt::vstderr;
using TxqcdTest2pt::TxqcdSmearedDiagnostics;
using TxqcdTest2pt::QcdSmearedDiagnostics;
using TxqcdTest2pt::TxqcdDiagnostics;
using TxqcdTest2pt::QcdDiagnostics;
using TxqcdTest2pt::QcdCheckpointer;

// Override lattice and physical params for this suite.
constexpr RealD beta = 5.7;
constexpr RealD mass = 0.0;          // lighter than the base m=0.3
constexpr RealD csw  = 0.0;          // pure Wilson, no clover term

inline Coordinate default_latt() {
  return Coordinate(std::vector<int>{8, 8, 8, 24});
}

inline Coordinate src_site() {
  return Coordinate(std::vector<int>{0, 0, 0, 0});
}

// Runtime override of n_therm / n_prod to match the gencfgs env vars, so
// configs_exist and meas_trajs see the extended ensemble length (e.g.
// N_PROD=700 -> trajectories up to 800).
inline int n_therm_runtime() {
  const char *v = std::getenv("N_THERM"); return (v && *v) ? std::atoi(v) : n_therm;
}
inline int n_prod_runtime() {
  const char *v = std::getenv("N_PROD");  return (v && *v) ? std::atoi(v) : n_prod;
}

// Trajectory list — runtime overridable via env vars TRAJ_START, TRAJ_END,
// TRAJ_SKIP (defaults: n_therm_runtime, n_therm_runtime + n_prod_runtime,
// meas_skip).  For "measure last 200 of an 800-traj ensemble": set
// TRAJ_START=610 (and N_PROD=700 if not already in env).
inline std::vector<int> meas_trajs() {
  auto envi = [](const char *n, int def) {
    const char *v = std::getenv(n); return (v && *v) ? std::atoi(v) : def;
  };
  int start = envi("TRAJ_START", n_therm_runtime());
  int end   = envi("TRAJ_END",   n_therm_runtime() + n_prod_runtime());
  int skip  = envi("TRAJ_SKIP",  meas_skip);
  std::vector<int> v;
  for (int t = start; t < end; t += skip) v.push_back(t);
  return v;
}

// Runtime LAMBDA override.  Cfg/meas dirs include the lambda when set so
// optlam vs reference runs do not collide.
inline RealD lambda_runtime() {
  const char *v = std::getenv("LAMBDA");
  return (v && *v) ? std::atof(v) : lambda;
}
inline std::string lambda_suffix() {
  RealD l = lambda_runtime();
  if (std::abs(l - lambda) < 1e-6) return "";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "_lam%.4f", l);
  return std::string(buf);
}

// Runtime MASS override.  Cfg/meas dirs include the mass when set so
// lighter-pion runs do not collide with the default m=0 ensemble.
inline RealD mass_runtime() {
  const char *v = std::getenv("MASS");
  return (v && *v) ? std::atof(v) : mass;
}
inline std::string mass_suffix() {
  RealD m = mass_runtime();
  if (std::abs(m - mass) < 1e-6) return "";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "_m%+.4f", m);
  return std::string(buf);
}

// Runtime MU override.  M_TXQCD = M_QCD + mu * Delta; default 1.0 reproduces
// the original operator.  Cfg/meas dirs include mu when not equal to 1 so
// scans (lambda, mu) with the same mean-field bias don't collide.
inline RealD mu_runtime() {
  const char *v = std::getenv("MU");
  return (v && *v) ? std::atof(v) : 1.0;
}
inline std::string mu_suffix() {
  RealD m = mu_runtime();
  if (std::abs(m - 1.0) < 1e-6) return "";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "_mu%.4f", m);
  return std::string(buf);
}

inline std::string txqcd_cfg_dir() {
  return "configs_2pt_txqcd_wilson8x24" + mass_suffix() + lambda_suffix() +
         mu_suffix();
}
inline std::string qcd_cfg_dir() {
  // QCD has no lambda or mu dependence — all TXQCD-(lambda,mu) runs share
  // one QCD ensemble.  meas_dir() still includes the suffixes so TXQCD
  // measurement outputs at different (lambda, mu) don't collide.
  return "configs_2pt_qcd_wilson8x24" + mass_suffix();
}
inline std::string meas_dir() {
  return "meas_2pt_wilson8x24" + mass_suffix() + lambda_suffix() +
         mu_suffix();
}

// Local versions that use the wilson8x24 meas_trajs (env-overridable),
// so an extend run (N_PROD=700 over an existing 600-traj ensemble) sees
// the missing cfgs and resumes generation rather than skipping.
inline bool txqcd_configs_exist() {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(txqcd_cfg_dir() + "/ckpoint_lat." + std::to_string(t)))
      return false;
  }
  return !trajs.empty();
}
inline bool qcd_configs_exist() {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(qcd_cfg_dir() + "/ckpoint_lat." + std::to_string(t)))
      return false;
  }
  return !trajs.empty();
}
inline int latest_txqcd_checkpoint() {
  return TxqcdTest2pt::latest_txqcd_checkpoint(txqcd_cfg_dir());
}
inline int latest_qcd_checkpoint() {
  return TxqcdTest2pt::latest_qcd_checkpoint(qcd_cfg_dir());
}

inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj) {
  TxqcdTest2pt::LoadTxqcdConfig(U, sRNG, pRNG, traj, txqcd_cfg_dir());
}
inline void LoadQcdConfig(LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG, int traj) {
  TxqcdTest2pt::LoadQcdConfig(U, sRNG, pRNG, traj, qcd_cfg_dir());
}

}  // namespace TxqcdTest2ptWilson8x24
