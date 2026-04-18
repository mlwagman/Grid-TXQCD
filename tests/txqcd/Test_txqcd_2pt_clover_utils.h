#pragma once
// Shared utilities for the TXQCD 2pt clover test suite.
// Reuses physical parameters and I/O helpers from Test_txqcd_2pt_utils.h
// but provides clover-specific config directories and csw parameter.

#include "Test_txqcd_2pt_utils.h"

namespace TxqcdTest2ptClover {

// Import constants and helpers from base namespace that don't conflict
using TxqcdTest2pt::beta;
using TxqcdTest2pt::lambda;
using TxqcdTest2pt::mass;
using TxqcdTest2pt::n_therm;
using TxqcdTest2pt::n_prod;
using TxqcdTest2pt::meas_skip;
using TxqcdTest2pt::meas_tol;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_noise;
using TxqcdTest2pt::default_latt;
using TxqcdTest2pt::src_site;
using TxqcdTest2pt::meas_trajs;
using TxqcdTest2pt::mkdir_p;
using TxqcdTest2pt::file_exists;
using TxqcdTest2pt::CorrelatorFromSlice;
using TxqcdTest2pt::WriteMeasReal;
using TxqcdTest2pt::WriteMeasComplex;
using TxqcdTest2pt::WriteMeasScalar;
using TxqcdTest2pt::ReadMeasReal;
using TxqcdTest2pt::ReadMeasComplex;
using TxqcdTest2pt::ReadMeasScalar;
using TxqcdTest2pt::vmean;
using TxqcdTest2pt::vstderr;

constexpr RealD csw = 1.0;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_csw1"; }
inline std::string qcd_cfg_dir()   { return "configs_2pt_qcd_nf2_csw1"; }
inline std::string meas_dir()      { return "meas_2pt_csw1"; }

inline bool txqcd_configs_exist() {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    std::string dir = txqcd_cfg_dir();
    if (!file_exists(dir + "/ckpoint_lat." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_lat_aux." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng." + std::to_string(t))) return false;
  }
  return true;
}
inline bool qcd_configs_exist() {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    std::string dir = qcd_cfg_dir();
    if (!file_exists(dir + "/ckpoint_lat." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng." + std::to_string(t))) return false;
  }
  return true;
}

inline int latest_txqcd_checkpoint() {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    std::string dir = txqcd_cfg_dir();
    if (file_exists(dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_lat_aux." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }
  return latest;
}
inline int latest_qcd_checkpoint() {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    std::string dir = qcd_cfg_dir();
    if (file_exists(dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }
  return latest;
}

inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj) {
  CheckpointerParameters CPp;
  CPp.config_prefix = txqcd_cfg_dir() + "/ckpoint_lat";
  CPp.rng_prefix    = txqcd_cfg_dir() + "/ckpoint_rng";
  CPp.saveInterval = 1; CPp.format = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);
  ckpt.CheckpointRestore(traj, U, sRNG, pRNG);
}

inline void LoadQcdConfig(LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG, int traj) {
  std::string dir = qcd_cfg_dir();
  std::string cf = dir + "/ckpoint_lat." + std::to_string(traj);
  std::string rf = dir + "/ckpoint_rng." + std::to_string(traj);
  FieldMetaData header;
  NerscIO::readRNGState(sRNG, pRNG, header, rf);
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
  NerscIO::readConfiguration<GaugeStats>(U, header, cf);
}

}  // namespace TxqcdTest2ptClover
