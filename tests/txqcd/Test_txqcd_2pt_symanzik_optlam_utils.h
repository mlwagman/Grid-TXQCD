#pragma once
// Optimal-lambda variant of the TXQCD 2pt Symanzik Nf=2+1 test suite.

#include "Test_txqcd_2pt_symanzik_utils.h"

namespace TxqcdTest2ptSymanzikOptlam {

using TxqcdTest2ptSymanzik::beta;
constexpr RealD lambda = 4.71;  // optimal lambda for Symanzik Nf=2+1
using TxqcdTest2ptSymanzik::mass;
using TxqcdTest2ptSymanzik::mass_s;
using TxqcdTest2ptSymanzik::csw;
using TxqcdTest2ptSymanzik::u0;
using TxqcdTest2ptSymanzik::stout_rho;
using TxqcdTest2ptSymanzik::stout_nsmear;
using TxqcdTest2ptSymanzik::n_therm;
using TxqcdTest2ptSymanzik::n_prod;
using TxqcdTest2ptSymanzik::meas_skip;
using TxqcdTest2ptSymanzik::meas_tol;
using TxqcdTest2ptSymanzik::cg_max;
using TxqcdTest2ptSymanzik::n_noise;
using TxqcdTest2ptSymanzik::n_vev_noise;
using TxqcdTest2ptSymanzik::default_latt;
using TxqcdTest2ptSymanzik::src_site;
using TxqcdTest2ptSymanzik::meas_trajs;
using TxqcdTest2ptSymanzik::mkdir_p;
using TxqcdTest2ptSymanzik::file_exists;
using TxqcdTest2ptSymanzik::CorrelatorFromSlice;
using TxqcdTest2ptSymanzik::vmean;
using TxqcdTest2ptSymanzik::vstderr;
using TxqcdTest2ptSymanzik::QcdCheckpointer;
using TxqcdTest2ptSymanzik::TxqcdDiagnostics;
using TxqcdTest2ptSymanzik::QcdDiagnostics;
template <class SmearPolicy>
using TxqcdSmearedDiagnostics =
    TxqcdTest2pt::TxqcdSmearedDiagnostics<SmearPolicy, TxqcdTest2ptStout::TxqcdCloverDiag>;
template <class SmearPolicy>
using QcdSmearedDiagnostics =
    TxqcdTest2pt::QcdSmearedDiagnostics<SmearPolicy, TxqcdTest2ptStout::QcdCloverDiag>;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_symanzik_optlam"; }
inline std::string qcd_cfg_dir()   { return TxqcdTest2ptSymanzik::qcd_cfg_dir(); }
inline std::string meas_dir()      { return "meas_2pt_symanzik_optlam"; }

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

}  // namespace TxqcdTest2ptSymanzikOptlam
