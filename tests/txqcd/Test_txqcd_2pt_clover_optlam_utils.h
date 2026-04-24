#pragma once
// Optimal-lambda variant of the TXQCD 2pt clover test suite.

#include "Test_txqcd_2pt_clover_utils.h"

namespace TxqcdTest2ptCloverOptlam {

using TxqcdTest2ptClover::beta;
constexpr RealD lambda = 5.33;  // optimal lambda for Clover Nf=2
using TxqcdTest2ptClover::mass;
using TxqcdTest2ptClover::mass_s;
using TxqcdTest2ptClover::csw;
using TxqcdTest2ptClover::u0;
using TxqcdTest2ptClover::stout_rho;
using TxqcdTest2ptClover::stout_nsmear;
using TxqcdTest2ptClover::n_therm;
using TxqcdTest2ptClover::n_prod;
using TxqcdTest2ptClover::meas_skip;
using TxqcdTest2ptClover::meas_tol;
using TxqcdTest2ptClover::cg_max;
using TxqcdTest2ptClover::n_noise;
using TxqcdTest2ptClover::n_vev_noise;
using TxqcdTest2ptClover::default_latt;
using TxqcdTest2ptClover::src_site;
using TxqcdTest2ptClover::meas_trajs;
using TxqcdTest2ptClover::mkdir_p;
using TxqcdTest2ptClover::file_exists;
using TxqcdTest2ptClover::CorrelatorFromSlice;
using TxqcdTest2ptClover::vmean;
using TxqcdTest2ptClover::vstderr;
using TxqcdTest2ptClover::QcdCheckpointer;
using TxqcdTest2ptClover::TxqcdDiagnostics;
using TxqcdTest2ptClover::QcdDiagnostics;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_csw1_optlam"; }
inline std::string qcd_cfg_dir()   { return TxqcdTest2ptClover::qcd_cfg_dir(); }
inline std::string meas_dir()      { return "meas_2pt_csw1_optlam"; }

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

}  // namespace TxqcdTest2ptCloverOptlam
