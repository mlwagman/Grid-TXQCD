#pragma once
// Shared utilities for the TXQCD 2pt test suite.
// Five programs use this header:
//   gencfgs -> conn -> disc -> aux -> compare
// Each step reads configs or measurement files produced by earlier steps.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace Grid;

namespace TxqcdTest2pt {

// ---- Physical parameters (shared by all programs) ----
constexpr RealD beta   = 5.6;
constexpr RealD lambda = 3.0;
constexpr RealD mass   = 0.3;
constexpr int   n_therm   = 100;
constexpr int   n_prod    = 500;
constexpr int   meas_skip = 10;

constexpr RealD meas_tol = 1e-10;
constexpr int   cg_max   = 10000;
constexpr int   n_noise  = 32;

inline Coordinate default_latt() { return Coordinate(std::vector<int>{4, 4, 4, 8}); }
inline Coordinate src_site()     { return Coordinate(std::vector<int>{0, 0, 0, 0}); }

inline std::string txqcd_cfg_dir() { return "configs_2pt_txqcd"; }
inline std::string qcd_cfg_dir()   { return "configs_2pt_qcd_nf2"; }
inline std::string meas_dir()      { return "meas_2pt"; }

inline std::vector<int> meas_trajs() {
  std::vector<int> v;
  for (int t = n_therm; t < n_therm + n_prod; t += meas_skip)
    v.push_back(t);
  return v;
}

// ---- File helpers ----
inline void mkdir_p(const std::string &d) {
  if (!d.empty()) mkdir(d.c_str(), 0755);
}
inline bool file_exists(const std::string &f) {
  struct stat st;
  return stat(f.c_str(), &st) == 0;
}

// ---- Config I/O ----
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

// ---- Volume-averaged correlator from per-timeslice data ----
//   C(dt) = (1/V4) sum_{t0} f(t0+dt) * conj(f(t0))
// Averages over all source locations via the sum over t0.
inline std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  int T = (int)s.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += s[(t0 + dt) % T] * std::conj(s[t0]);
  for (auto &c : C) c /= V4;
  return C;
}

// ---- Measurement file I/O ----
// Real: one line per config, T space-separated values.
inline void WriteMeasReal(const std::string &fname,
                          const std::vector<std::vector<RealD>> &data, int T) {
  std::ofstream f(fname);
  f << "# T=" << T << " Ncfg=" << data.size() << "\n" << std::setprecision(16);
  for (auto &row : data) {
    for (int t = 0; t < T; ++t) f << (t ? " " : "") << row[t];
    f << "\n";
  }
}
// Complex: pairs of (re im) per timeslice.
inline void WriteMeasComplex(const std::string &fname,
                             const std::vector<std::vector<ComplexD>> &data,
                             int T) {
  std::ofstream f(fname);
  f << "# T=" << T << " Ncfg=" << data.size() << "\n" << std::setprecision(16);
  for (auto &row : data) {
    for (int t = 0; t < T; ++t)
      f << (t ? " " : "") << row[t].real() << " " << row[t].imag();
    f << "\n";
  }
}
// Scalar per config.
inline void WriteMeasScalar(const std::string &fname,
                            const std::vector<RealD> &data) {
  std::ofstream f(fname);
  f << "# Ncfg=" << data.size() << "\n" << std::setprecision(16);
  for (auto v : data) f << v << "\n";
}

inline std::vector<std::vector<RealD>>
ReadMeasReal(const std::string &fname, int &T) {
  std::ifstream f(fname);
  if (!f) { std::cerr << "Cannot open " << fname << "\n"; std::exit(1); }
  std::string line;
  std::getline(f, line);
  T = 0;
  sscanf(line.c_str(), "# T=%d", &T);
  std::vector<std::vector<RealD>> data;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<RealD> row(T);
    for (int t = 0; t < T; ++t) ss >> row[t];
    data.push_back(row);
  }
  return data;
}
inline std::vector<std::vector<ComplexD>>
ReadMeasComplex(const std::string &fname, int &T) {
  std::ifstream f(fname);
  if (!f) { std::cerr << "Cannot open " << fname << "\n"; std::exit(1); }
  std::string line;
  std::getline(f, line);
  T = 0;
  sscanf(line.c_str(), "# T=%d", &T);
  std::vector<std::vector<ComplexD>> data;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<ComplexD> row(T);
    for (int t = 0; t < T; ++t) {
      RealD re, im;
      ss >> re >> im;
      row[t] = ComplexD(re, im);
    }
    data.push_back(row);
  }
  return data;
}
inline std::vector<RealD> ReadMeasScalar(const std::string &fname) {
  std::ifstream f(fname);
  if (!f) { std::cerr << "Cannot open " << fname << "\n"; std::exit(1); }
  std::string line;
  std::getline(f, line);  // header
  std::vector<RealD> data;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    data.push_back(std::stod(line));
  }
  return data;
}

// ---- Stats ----
inline RealD vmean(const std::vector<RealD> &v) {
  RealD s = 0;
  for (auto x : v) s += x;
  return s / v.size();
}
inline RealD vstderr(const std::vector<RealD> &v) {
  RealD m = vmean(v), s2 = 0;
  for (auto x : v) s2 += (x - m) * (x - m);
  return std::sqrt(s2 / (v.size() * (v.size() - 1)));
}

}  // namespace TxqcdTest2pt
