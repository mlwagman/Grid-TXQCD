#pragma once
// Shared utilities for the TXQCD 2pt symanzik Nf=3 test suite.
// Mirrors symanzik utils but targets a single Nf=3 TXQCD operator with
// mass = {m_l, m_l, m_s} instead of the Nf=2 TXQCD + Nf=1 QCD-wrap setup.
//
// All compilation units including this header must be compiled with
// -DTXQCD_Nf=3 so the TXQCD types use a 3x3 flavor matrix.
//
// Reuses the same QCD Nf=2+1 reference ensemble produced by the symanzik
// test (configs_2pt_qcd_nf2_symanzik); only generates a parallel TXQCD Nf=3
// stream.

#ifndef TXQCD_Nf
#error "symanzik_nf3 tests must be compiled with -DTXQCD_Nf=3"
#endif

#include "Test_txqcd_2pt_symanzik_utils.h"

namespace TxqcdTest2ptSymanzikNf3 {

using TxqcdTest2ptSymanzik::beta;
using TxqcdTest2ptSymanzik::lambda;
using TxqcdTest2ptSymanzik::mass;     // m_l (light)
using TxqcdTest2ptSymanzik::mass_s;   // m_s (strange)
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
using TxqcdTest2ptSymanzik::default_latt;
using TxqcdTest2ptSymanzik::src_site;
using TxqcdTest2ptSymanzik::meas_trajs;
using TxqcdTest2ptSymanzik::mkdir_p;
using TxqcdTest2ptSymanzik::file_exists;
using TxqcdTest2ptSymanzik::CorrelatorFromSlice;
using TxqcdTest2ptSymanzik::vmean;
using TxqcdTest2ptSymanzik::vstderr;
using TxqcdTest2ptSymanzik::TxqcdSmearedDiagnostics;
using TxqcdTest2ptSymanzik::n_vev_noise;

// Diagonal mass matrix m = (m_l, m_l, m_s).
inline std::array<RealD, 3> nf3_mass() { return {mass, mass, mass_s}; }

// Runtime LAMBDA override (default = compile-time symanzik lambda = 3.0).
// Lets us run two ensembles (optlam λ≈4.71, reference λ=3.0) from the same
// binary; the cfg/measurement directories include lambda so they don't clash.
inline RealD lambda_runtime() {
  const char *v = std::getenv("LAMBDA");
  return (v && *v) ? std::atof(v) : lambda;
}

inline std::string lambda_suffix() {
  RealD l = lambda_runtime();
  if (std::abs(l - lambda) < 1e-6) return "";  // default → no suffix
  char buf[32];
  std::snprintf(buf, sizeof(buf), "_lam%.4f", l);
  return std::string(buf);
}

inline std::string txqcd_nf3_cfg_dir() {
  return "configs_2pt_txqcd_symanzik_nf3" + lambda_suffix();
}
inline std::string meas_dir() {
  return "meas_2pt_symanzik_nf3" + lambda_suffix();
}

// Reuses QCD reference from the Nf=2+1 symanzik test.
inline std::string qcd_cfg_dir() { return TxqcdTest2ptSymanzik::qcd_cfg_dir(); }

inline bool txqcd_configs_exist() {
  return TxqcdTest2pt::txqcd_configs_exist(txqcd_nf3_cfg_dir());
}
inline int latest_txqcd_checkpoint() {
  return TxqcdTest2pt::latest_txqcd_checkpoint(txqcd_nf3_cfg_dir());
}

inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj) {
  TxqcdTest2pt::LoadTxqcdConfig(U, sRNG, pRNG, traj, txqcd_nf3_cfg_dir());
}

}  // namespace TxqcdTest2ptSymanzikNf3
