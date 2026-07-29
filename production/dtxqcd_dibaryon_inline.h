// dtxqcd_dibaryon_inline.h: every-trajectory PURE-AUX dibaryon measurement
// as an HMC observer.  No inversions, no contractions -- cost ~2.5 s/config
// at 16^3x48 vs ~574 s/trajectory, so it runs every trajectory by default
// and turns the HMC stream itself into the dibaryon dataset (~5x effective
// statistics vs the 10-traj checkpoint cadence; tau_int ~ 0.5 measured at
// 10-traj spacing, so finer sampling adds real information at small t).
//
// Shares MeasCore with meas_dibaryon_aux_dtxqcd.cc, so the h5 files are
// format-identical to the offline binary's and the existing analyzers
// (analyze_dibaryon_aux.py, analyze_dibaryon_sm.py, _dib_graded_demo.py)
// work unchanged on the output directory.
//
// The observer receives the THIN-link field (same as the checkpointer) and
// applies its own stout internally via MeasCore -- construct with the
// generator's STOUT_NSMEAR/STOUT_RHO so inline == offline-on-checkpoint.
//
// Env knobs (read at construction):
//   DIBARYON_INLINE          (default 1)  master switch
//   DIBARYON_INLINE_INTERVAL (default 1)  measure every N trajectories
//   DIBARYON_INLINE_SMEARS   (default "0,60,100")  covariant Gaussian levels
//                            (the locked structure: peak N~60-100, alpha=0.5)
//   DIBARYON_INLINE_LMAX     (default 3)  displacement ladder extent
//   SMEAR_ALPHA              (default 0.5)

#pragma once

#include <sys/stat.h>
#include "dtxqcd_dibaryon_core.h"

class DtxqcdDibaryonInline : public Grid::HmcObservable<Grid::DTXQCDField> {
 public:
  DtxqcdDibaryonInline(Grid::GridCartesian &Grid_, const std::string &out_dir,
                       Grid::RealD lambda, int nstout, Grid::RealD rho)
      : out_dir_(out_dir), lambda_(lambda),
        interval_(env_int("DIBARYON_INLINE_INTERVAL", 1)),
        core_(Grid_, nstout, rho, env_int("DIBARYON_INLINE_LMAX", 3),
              parse_smears(), env_real("SMEAR_ALPHA", 0.5)) {
    if (interval_ < 1) interval_ = 1;
    if (Grid_.IsBoss()) ::mkdir(out_dir_.c_str(), 0755);
    std::cout << Grid::GridLogMessage << "[dibaryon-inline] ENABLED  dir="
              << out_dir_ << "  interval=" << interval_
              << "  geoms I1/I0=" << core_.NG1() << "/" << core_.NG0()
              << "  levels=" << core_.NLEV() << std::endl;
  }

  void TrajectoryComplete(int traj, Grid::DTXQCDField &U,
                          Grid::GridSerialRNG &sRNG,
                          Grid::GridParallelRNG &pRNG) override {
    (void)sRNG;
    (void)pRNG;
    if (traj % interval_ != 0) return;
    double t0 = Grid::usecond();
    std::vector<std::vector<std::vector<Grid::ComplexD>>> S1, S0;
    core_.ComputeAll(U, S1, S0);
    Grid::RealD plaq =
        Grid::WilsonLoops<Grid::PeriodicGimplR>::avgPlaquette(U.U);
    if (U.U.Grid()->IsBoss()) {
      const std::string outfile =
          out_dir_ + "/dibaryon_aux_dtxqcd_" + std::to_string(traj) + ".h5";
      core_.WriteH5(outfile, lambda_, plaq, S1, S0);
    }
    std::cout << Grid::GridLogMessage << "[dibaryon-inline] traj " << traj
              << "  " << (Grid::usecond() - t0) * 1e-6 << " s" << std::endl;
  }

 private:
  static int env_int(const char *k, int d) {
    const char *v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : d;
  }
  static Grid::RealD env_real(const char *k, Grid::RealD d) {
    const char *v = std::getenv(k);
    return (v && *v) ? std::atof(v) : d;
  }
  static std::vector<int> parse_smears() {
    const char *v = std::getenv("DIBARYON_INLINE_SMEARS");
    std::stringstream ss(v && *v ? v : "0,60,100");
    std::vector<int> out;
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::atoi(tok.c_str()));
    return out;
  }

  std::string out_dir_;
  Grid::RealD lambda_;
  int interval_;
  DtxqcdDibaryon::MeasCore core_;
};
