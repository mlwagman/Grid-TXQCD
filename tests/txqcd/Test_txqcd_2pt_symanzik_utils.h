#pragma once
// Shared utilities for the TXQCD 2pt tadpole-improved Symanzik (Lüscher-Weisz) test suite.
// Same as stout suite but with improved gauge action and different config directory names.

#include "Test_txqcd_2pt_stout_utils.h"

namespace TxqcdTest2ptSymanzik {

using TxqcdTest2ptStout::beta;
using TxqcdTest2ptStout::lambda;
using TxqcdTest2ptStout::mass;
using TxqcdTest2ptStout::mass_s;
using TxqcdTest2ptStout::csw;
using TxqcdTest2ptStout::u0;
using TxqcdTest2ptStout::stout_rho;
using TxqcdTest2ptStout::stout_nsmear;
using TxqcdTest2ptStout::n_therm;
using TxqcdTest2ptStout::n_prod;
using TxqcdTest2ptStout::meas_skip;
using TxqcdTest2ptStout::meas_tol;
using TxqcdTest2ptStout::cg_max;
using TxqcdTest2ptStout::n_noise;
using TxqcdTest2ptStout::default_latt;
using TxqcdTest2ptStout::src_site;
using TxqcdTest2ptStout::meas_trajs;
using TxqcdTest2ptStout::mkdir_p;
using TxqcdTest2ptStout::file_exists;
using TxqcdTest2ptStout::CorrelatorFromSlice;
using TxqcdTest2ptStout::vmean;
using TxqcdTest2ptStout::vstderr;
using TxqcdTest2ptStout::TxqcdDiagnostics;
using TxqcdTest2ptStout::QcdDiagnostics;
using TxqcdTest2ptStout::QcdCheckpointer;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_symanzik"; }
inline std::string qcd_cfg_dir()   { return "configs_2pt_qcd_nf2_symanzik"; }
inline std::string meas_dir()      { return "meas_2pt_symanzik"; }

inline bool txqcd_configs_exist() { return TxqcdTest2pt::txqcd_configs_exist(txqcd_cfg_dir()); }
inline bool qcd_configs_exist()   { return TxqcdTest2pt::qcd_configs_exist(qcd_cfg_dir()); }
inline int  latest_txqcd_checkpoint() { return TxqcdTest2pt::latest_txqcd_checkpoint(txqcd_cfg_dir()); }
inline int  latest_qcd_checkpoint()   { return TxqcdTest2pt::latest_qcd_checkpoint(qcd_cfg_dir()); }

inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj) {
  TxqcdTest2pt::LoadTxqcdConfig(U, sRNG, pRNG, traj, txqcd_cfg_dir());
}
inline void LoadQcdConfig(LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG, int traj) {
  TxqcdTest2pt::LoadQcdConfig(U, sRNG, pRNG, traj, qcd_cfg_dir());
}

}  // namespace TxqcdTest2ptSymanzik
