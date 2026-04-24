#pragma once
// Optimal-lambda variant of the TXQCD 2pt stout-smeared Nf=2+1 test suite.

#include "Test_txqcd_2pt_stout_utils.h"

namespace TxqcdTest2ptStoutOptlam {

using TxqcdTest2ptStout::beta;
constexpr RealD lambda = 4.65;  // optimal lambda for Stout Nf=2+1
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
using TxqcdTest2ptStout::n_vev_noise;
using TxqcdTest2ptStout::default_latt;
using TxqcdTest2ptStout::src_site;
using TxqcdTest2ptStout::meas_trajs;
using TxqcdTest2ptStout::mkdir_p;
using TxqcdTest2ptStout::file_exists;
using TxqcdTest2ptStout::CorrelatorFromSlice;
using TxqcdTest2ptStout::vmean;
using TxqcdTest2ptStout::vstderr;
using TxqcdTest2ptStout::QcdCheckpointer;
using TxqcdTest2ptStout::TxqcdDiagnostics;
using TxqcdTest2ptStout::QcdDiagnostics;
template <class SmearPolicy>
using TxqcdSmearedDiagnostics =
    TxqcdTest2pt::TxqcdSmearedDiagnostics<SmearPolicy, TxqcdTest2ptStout::TxqcdDiagnostics>;
template <class SmearPolicy>
using QcdSmearedDiagnostics =
    TxqcdTest2pt::QcdSmearedDiagnostics<SmearPolicy, TxqcdTest2ptStout::QcdDiagnostics>;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_stout_optlam"; }
inline std::string qcd_cfg_dir()   { return TxqcdTest2ptStout::qcd_cfg_dir(); }
inline std::string meas_dir()      { return "meas_2pt_stout_optlam"; }

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

}  // namespace TxqcdTest2ptStoutOptlam
