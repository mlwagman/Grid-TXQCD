#pragma once
// Shared utilities for the TXQCD 2pt stout-smeared test suite.
// Reuses physical parameters and I/O helpers from Test_txqcd_2pt_utils.h
// but provides stout-smeared config directories and clover VEV operator.

#include "Test_txqcd_2pt_utils.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

namespace TxqcdTest2ptStout {

using TxqcdTest2pt::beta;
using TxqcdTest2pt::lambda;
using TxqcdTest2pt::mass;
using TxqcdTest2pt::mass_s;
using TxqcdTest2pt::csw;
using TxqcdTest2pt::u0;
using TxqcdTest2pt::stout_rho;
using TxqcdTest2pt::stout_nsmear;
constexpr int n_therm = 300;
using TxqcdTest2pt::n_prod;
using TxqcdTest2pt::meas_skip;
using TxqcdTest2pt::meas_tol;
using TxqcdTest2pt::cg_max;
using TxqcdTest2pt::n_noise;
using TxqcdTest2pt::n_vev_noise;
using TxqcdTest2pt::default_latt;
using TxqcdTest2pt::src_site;

inline std::vector<int> meas_trajs() {
  std::vector<int> v;
  for (int t = n_therm; t < n_therm + n_prod; t += meas_skip)
    v.push_back(t);
  return v;
}
using TxqcdTest2pt::mkdir_p;
using TxqcdTest2pt::file_exists;
using TxqcdTest2pt::CorrelatorFromSlice;
using TxqcdTest2pt::vmean;
using TxqcdTest2pt::vstderr;
using TxqcdTest2pt::QcdCheckpointer;

struct TxqcdCloverDiag : TxqcdTest2pt::TxqcdDiagnostics {
  using TxqcdTest2pt::TxqcdDiagnostics::TxqcdDiagnostics;
  RealD compute_trminv(LatticeGaugeField &Uvev) override {
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(Uvev, grid_, rbgrid_, mass_, csw_, csw_);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, TxqcdTest2pt::cg_max);
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
};

struct QcdCloverDiag : TxqcdTest2pt::QcdDiagnostics {
  using TxqcdTest2pt::QcdDiagnostics::QcdDiagnostics;
  RealD compute_trminv(LatticeGaugeField &Uvev) override {
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(Uvev, grid_, rbgrid_, mass_, csw_, csw_);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, TxqcdTest2pt::cg_max);
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
};

using TxqcdDiagnostics = TxqcdCloverDiag;
using QcdDiagnostics = QcdCloverDiag;

template <class SmearPolicy>
using TxqcdSmearedDiagnostics =
    TxqcdTest2pt::TxqcdSmearedDiagnostics<SmearPolicy, TxqcdCloverDiag>;

template <class SmearPolicy>
using QcdSmearedDiagnostics =
    TxqcdTest2pt::QcdSmearedDiagnostics<SmearPolicy, QcdCloverDiag>;

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd_stout"; }
inline std::string qcd_cfg_dir()   { return "configs_2pt_qcd_nf2_stout"; }
inline std::string meas_dir()      { return "meas_2pt_stout"; }

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

}  // namespace TxqcdTest2ptStout
