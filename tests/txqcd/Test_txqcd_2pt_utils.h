#pragma once
// Shared utilities for the TXQCD 2pt test suite.
// Five programs use this header:
//   gencfgs -> conn -> disc -> aux -> compare
// Each step reads configs or measurement files produced by earlier steps.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <sys/stat.h>

using namespace Grid;

namespace TxqcdTest2pt {

// ---- Physical parameters (shared by all programs) ----
constexpr RealD beta   = 5.6;
constexpr RealD lambda = 3.0;
constexpr RealD mass   = 0.3;
constexpr RealD mass_s = 0.4;
constexpr RealD csw    = 1.0;
constexpr RealD u0     = 0.843;
constexpr RealD stout_rho   = 0.1;
constexpr int   stout_nsmear = 3;
constexpr int   n_therm   = 100;
constexpr int   n_prod    = 500;
constexpr int   meas_skip = 10;

constexpr RealD meas_tol = 1e-10;
constexpr int   cg_max   = 10000;
constexpr int   n_noise  = 32;
constexpr int   n_vev_noise = 8;

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

// ---- Config I/O (parameterized by directory) ----
inline bool txqcd_configs_exist(const std::string &dir) {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(dir + "/ckpoint_lat." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_lat_aux." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng." + std::to_string(t))) return false;
  }
  return true;
}
inline bool qcd_configs_exist(const std::string &dir) {
  auto trajs = meas_trajs();
  for (int t : trajs) {
    if (!file_exists(dir + "/ckpoint_lat." + std::to_string(t))) return false;
    if (!file_exists(dir + "/ckpoint_rng." + std::to_string(t))) return false;
  }
  return true;
}

inline int latest_txqcd_checkpoint(const std::string &dir) {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    if (file_exists(dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_lat_aux." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }
  return latest;
}
inline int latest_qcd_checkpoint(const std::string &dir) {
  int latest = -1;
  for (int t = meas_skip; t <= n_therm + n_prod; t += meas_skip) {
    if (file_exists(dir + "/ckpoint_lat." + std::to_string(t)) &&
        file_exists(dir + "/ckpoint_rng." + std::to_string(t)))
      latest = t;
  }
  return latest;
}

inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj,
                            const std::string &dir) {
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                dir + "/ckpoint_lat",
                                dir + "/ckpoint_rng", traj);
}

inline void LoadQcdConfig(LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG, int traj,
                          const std::string &dir) {
  std::string cf = dir + "/ckpoint_lat." + std::to_string(traj);
  std::string rf = dir + "/ckpoint_rng." + std::to_string(traj);
  FieldMetaData header;
  NerscIO::readRNGState(sRNG, pRNG, header, rf);
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
  NerscIO::readConfiguration<GaugeStats>(U, header, cf);
}

// Convenience wrappers using default directories
inline bool txqcd_configs_exist() { return txqcd_configs_exist(txqcd_cfg_dir()); }
inline bool qcd_configs_exist()   { return qcd_configs_exist(qcd_cfg_dir()); }
inline int  latest_txqcd_checkpoint() { return latest_txqcd_checkpoint(txqcd_cfg_dir()); }
inline int  latest_qcd_checkpoint()   { return latest_qcd_checkpoint(qcd_cfg_dir()); }
inline void LoadTxqcdConfig(TXQCDField &U, GridSerialRNG &sRNG,
                            GridParallelRNG &pRNG, int traj) {
  LoadTxqcdConfig(U, sRNG, pRNG, traj, txqcd_cfg_dir());
}
inline void LoadQcdConfig(LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG, int traj) {
  LoadQcdConfig(U, sRNG, pRNG, traj, qcd_cfg_dir());
}

// ---- QCD checkpointer (NerscIO, shared by all gencfgs) ----
struct QcdCheckpointer : public HmcObservable<LatticeGaugeField> {
  std::string cfg_prefix, rng_prefix;
  int save_interval;
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
  void TrajectoryComplete(int t, LatticeGaugeField &U, GridSerialRNG &sR,
                          GridParallelRNG &pR) override {
    if (t % save_interval != 0) return;
    NerscIO::writeRNGState(sR, pR,
                           rng_prefix + "." + std::to_string(t));
    NerscIO::writeConfiguration<GaugeStats>(
        U, cfg_prefix + "." + std::to_string(t), 0, 1);
  }
};

// ---- Volume-averaged correlator from per-timeslice data ----
inline std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  int T = (int)s.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += s[(t0 + dt) % T] * conjugate(s[t0]);
  for (auto &c : C) c /= V4;
  return C;
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

// ---- HMC diagnostics (chunk-per-checkpoint HDF5, no restart logic needed) ----

template <class Field>
class HmcDiagWriter : public HmcObservable<Field> {
public:
  struct ActionRef {
    std::string name;
    Action<Field> *action;
  };

protected:
  std::string prefix_;
  int interval_;
  std::vector<ActionRef> actions_;
  bool has_aux_;

  std::vector<int>    traj_;
  std::vector<RealD>  plaq_;
  std::vector<RealD>  vev_sigma_, vev_s_;
  std::vector<std::vector<RealD>> force_avg_, force_max_;
  std::vector<std::vector<RealD>> fdt_avg_, fdt_max_;

  virtual RealD get_plaq(Field &U) = 0;
  virtual void record_aux(Field &U) {}

  virtual void flush(int traj) {
    if (traj_.empty()) return;
    std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
    Hdf5Writer wr(fname);
    write(wr, "traj", traj_);
    write(wr, "plaq", plaq_);
    write(wr, "force_avg", force_avg_);
    write(wr, "force_max", force_max_);
    write(wr, "fdt_avg", fdt_avg_);
    write(wr, "fdt_max", fdt_max_);
    if (has_aux_) {
      write(wr, "vev_sigma", vev_sigma_);
      write(wr, "vev_s", vev_s_);
    }
    std::vector<std::string> names;
    for (auto &a : actions_) names.push_back(a.name);
    write(wr, "action_names", names);

    traj_.clear(); plaq_.clear();
    force_avg_.clear(); force_max_.clear();
    fdt_avg_.clear(); fdt_max_.clear();
    vev_sigma_.clear(); vev_s_.clear();

    std::cout << GridLogMessage << "HMC diagnostics written to " << fname << std::endl;
  }

public:
  HmcDiagWriter(const std::string &prefix, int interval,
                std::vector<ActionRef> acts, bool aux = false)
      : prefix_(prefix), interval_(interval),
        actions_(std::move(acts)), has_aux_(aux) {}

  void TrajectoryComplete(int traj, Field &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    traj_.push_back(traj);
    RealD plaq_now = get_plaq(U);
    plaq_.push_back(plaq_now);
    std::cout << GridLogMessage << "Traj " << traj
              << " plaq = " << plaq_now << std::endl;

    int na = (int)actions_.size();
    std::vector<RealD> fa(na), fm(na), fdta(na), fdtm(na);
    for (int i = 0; i < na; ++i) {
      fa[i]   = actions_[i].action->deriv_norm_average();
      fm[i]   = actions_[i].action->deriv_max_average();
      fdta[i] = actions_[i].action->Fdt_norm_average();
      fdtm[i] = actions_[i].action->Fdt_max_average();
    }
    force_avg_.push_back(fa);
    force_max_.push_back(fm);
    fdt_avg_.push_back(fdta);
    fdt_max_.push_back(fdtm);

    record_aux(U);

    if (traj % interval_ == 0) flush(traj);
  }
};

struct TxqcdDiagnostics : HmcDiagWriter<TXQCDField> {
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG &prng_;
  RealD mass_, csw_;
  int n_vev_noise_;
  std::vector<RealD> vev_trminv_;

  TxqcdDiagnostics(const std::string &prefix, int interval,
            std::vector<ActionRef> acts,
            GridCartesian &grid, GridRedBlackCartesian &rbgrid,
            GridParallelRNG &prng, RealD mass, RealD csw, int n_vev_noise)
      : HmcDiagWriter(prefix, interval, std::move(acts), true),
        grid_(grid), rbgrid_(rbgrid), prng_(prng),
        mass_(mass), csw_(csw), n_vev_noise_(n_vev_noise) {}

  RealD get_plaq(TXQCDField &U) override {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  }

  virtual LatticeGaugeField get_vev_gauge(TXQCDField &U) { return U.U; }

  virtual RealD compute_trminv(LatticeGaugeField &Uvev) {
    WilsonFermionD Dw(Uvev, grid_, rbgrid_, mass_);
    MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
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

  void record_aux(TXQCDField &U) override {
    RealD V = (RealD)U.Grid()->gSites();
    vev_sigma_.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
    vev_s_.push_back(TensorRemove(sum(trace(U.s))).real() / V);

    LatticeGaugeField Uvev = get_vev_gauge(U);
    vev_trminv_.push_back(compute_trminv(Uvev));
  }

  void flush(int traj) override {
    if (traj_.empty()) return;
    std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
    Hdf5Writer wr(fname);
    write(wr, "traj", traj_);
    write(wr, "plaq", plaq_);
    write(wr, "force_avg", force_avg_);
    write(wr, "force_max", force_max_);
    write(wr, "fdt_avg", fdt_avg_);
    write(wr, "fdt_max", fdt_max_);
    write(wr, "vev_sigma", vev_sigma_);
    write(wr, "vev_s", vev_s_);
    write(wr, "vev_trminv", vev_trminv_);
    std::vector<std::string> names;
    for (auto &a : actions_) names.push_back(a.name);
    write(wr, "action_names", names);

    traj_.clear(); plaq_.clear();
    force_avg_.clear(); force_max_.clear();
    fdt_avg_.clear(); fdt_max_.clear();
    vev_sigma_.clear(); vev_s_.clear();
    vev_trminv_.clear();

    std::cout << GridLogMessage << "HMC diagnostics written to " << fname << std::endl;
  }
};

struct QcdDiagnostics : HmcDiagWriter<LatticeGaugeField> {
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  GridParallelRNG &prng_;
  RealD mass_, csw_;
  int n_vev_noise_;
  std::vector<RealD> vev_trminv_;

  QcdDiagnostics(const std::string &prefix, int interval,
          std::vector<ActionRef> acts,
          GridCartesian &grid, GridRedBlackCartesian &rbgrid,
          GridParallelRNG &prng, RealD mass, RealD csw, int n_vev_noise)
      : HmcDiagWriter(prefix, interval, std::move(acts), false),
        grid_(grid), rbgrid_(rbgrid), prng_(prng),
        mass_(mass), csw_(csw), n_vev_noise_(n_vev_noise) {}

  RealD get_plaq(LatticeGaugeField &U) override {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
  }

  virtual LatticeGaugeField get_vev_gauge(LatticeGaugeField &U) { return U; }

  virtual RealD compute_trminv(LatticeGaugeField &Uvev) {
    WilsonFermionD Dw(Uvev, grid_, rbgrid_, mass_);
    MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
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

  void measure_vev(LatticeGaugeField &U) {
    LatticeGaugeField Uvev = get_vev_gauge(U);
    vev_trminv_.push_back(compute_trminv(Uvev));
  }

  void TrajectoryComplete(int traj, LatticeGaugeField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    measure_vev(U);
    HmcDiagWriter<LatticeGaugeField>::TrajectoryComplete(traj, U, sRNG, pRNG);
  }

  void flush(int traj) override {
    if (traj_.empty()) return;
    std::string fname = prefix_ + "." + std::to_string(traj) + ".h5";
    Hdf5Writer wr(fname);
    write(wr, "traj", traj_);
    write(wr, "plaq", plaq_);
    write(wr, "force_avg", force_avg_);
    write(wr, "force_max", force_max_);
    write(wr, "fdt_avg", fdt_avg_);
    write(wr, "fdt_max", fdt_max_);
    write(wr, "vev_trminv", vev_trminv_);
    std::vector<std::string> names;
    for (auto &a : actions_) names.push_back(a.name);
    write(wr, "action_names", names);

    traj_.clear(); plaq_.clear();
    force_avg_.clear(); force_max_.clear();
    fdt_avg_.clear(); fdt_max_.clear();
    vev_trminv_.clear();

    std::cout << GridLogMessage << "HMC diagnostics written to " << fname << std::endl;
  }
};

// ---- Smeared VEV monitoring ----
// Overrides get_vev_gauge to measure Tr M^{-1} on stout-smeared links.

template <class SmearPolicy, class Base = TxqcdDiagnostics>
struct TxqcdSmearedDiagnostics : Base {
  SmearPolicy &smear_;
  using ActionRef = typename HmcDiagWriter<TXQCDField>::ActionRef;

  TxqcdSmearedDiagnostics(const std::string &prefix, int interval,
                           std::vector<ActionRef> acts,
                           SmearPolicy &smear,
                           GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                           GridParallelRNG &prng,
                           RealD mass, RealD csw, int n_vev_noise)
      : Base(prefix, interval, std::move(acts),
             grid, rbgrid, prng, mass, csw, n_vev_noise),
        smear_(smear) {}

  LatticeGaugeField get_vev_gauge(TXQCDField &U) override {
    smear_.set_Field(U);
    return smear_.get_SmearedU().U;
  }
};

template <class SmearPolicy, class Base = QcdDiagnostics>
struct QcdSmearedDiagnostics : Base {
  SmearPolicy &smear_;
  using ActionRef = typename HmcDiagWriter<LatticeGaugeField>::ActionRef;

  QcdSmearedDiagnostics(const std::string &prefix, int interval,
                         std::vector<ActionRef> acts,
                         SmearPolicy &smear,
                         GridCartesian &grid, GridRedBlackCartesian &rbgrid,
                         GridParallelRNG &prng,
                         RealD mass, RealD csw, int n_vev_noise)
      : Base(prefix, interval, std::move(acts),
             grid, rbgrid, prng, mass, csw, n_vev_noise),
        smear_(smear) {}

  LatticeGaugeField get_vev_gauge(LatticeGaugeField &U) override {
    smear_.set_Field(U);
    return smear_.get_SmearedU();
  }
};

}  // namespace TxqcdTest2pt
