// Read NERSC (our ckpoint_lat.*) or ILDG/SciDAC LIME (chroma) gauge configs
// and compute plaquette.  Auto-detects format from the file's magic bytes.
//
// Usage:
//   ./compute_plaq <cfg>...  --grid L.L.L.T --mpi mx.my.mz.mt
//
// NERSC files start with the text "BEGIN_HEADER"; LIME files start with the
// 4-byte magic 0x45 0x67 0x89 0xAB.
//
// Optional env vars for gauge-action reporting (matches chroma's LW_TREE_GAUGEACT):
//   BETA   — β (default 0, skips action reporting)
//   U0     — tadpole u0 (default 0.832605301399891)
// When BETA>0, prints S(U) = PlaqPlusRectangleAction(β, −β/(20·u0²)).S(U)
// and the derived chroma-side S computed analytically from <P>, <R>.

#include <Grid/Grid.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/smearing/StoutSmearing.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Grid;

enum class CfgFormat { NERSC, LIME_ILDG, UNKNOWN };

static CfgFormat detect_format(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return CfgFormat::UNKNOWN;
  unsigned char buf[16] = {0};
  std::fread(buf, 1, sizeof(buf), f);
  std::fclose(f);
  if (std::memcmp(buf, "BEGIN_HEADER", 12) == 0) return CfgFormat::NERSC;
  if (buf[0] == 0x45 && buf[1] == 0x67 && buf[2] == 0x89 && buf[3] == 0xAB)
    return CfgFormat::LIME_ILDG;
  return CfgFormat::UNKNOWN;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  // Collect filenames from CLI (skip --grid / --mpi and their values).
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) { ++i; continue; }
    files.push_back(a);
  }
  if (files.empty()) {
    std::cerr << "Usage: compute_plaq <cfg>... --grid L.L.L.T --mpi mx.my.mz.mt\n";
    Grid_finalize();
    return 1;
  }

  LatticeGaugeField Umu(&Grid);
  typedef GaugeStatistics<PeriodicGimplR> GS;
  for (const auto &f : files) {
    CfgFormat fmt = detect_format(f);
    FieldMetaData header;

    if (fmt == CfgFormat::NERSC) {
      NerscIO::readConfiguration<GS>(Umu, header, f);
    } else if (fmt == CfgFormat::LIME_ILDG) {
      IldgReader reader;
      reader.open(f);
      reader.readConfiguration(Umu, header);
      reader.close();
    } else {
      std::cerr << "compute_plaq: unknown format for " << f << std::endl;
      continue;
    }

    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
    RealD link = WilsonLoops<PeriodicGimplR>::linkTrace(Umu);
    std::cout << GridLogMessage
              << f << "\tplaq = " << std::setprecision(10) << plaq
              << "\tlink_trace = " << link << std::endl;

    // Optional: report stout-smeared plaquette for comparison with chroma.
    // Env var STOUT_RHO (default 0.125), STOUT_NSMEAR (default 1).
    const char *sr_env = std::getenv("STOUT_RHO");
    if (sr_env && *sr_env) {
      RealD srho = std::atof(sr_env);
      int nsmear = 1;
      if (const char *ns = std::getenv("STOUT_NSMEAR"); ns && *ns)
        nsmear = std::atoi(ns);
      std::cout << GridLogMessage
                << "  [stout] smearing rho=" << srho << " nsmear=" << nsmear
                << std::endl;
      Smear_Stout<PeriodicGimplR> Stout(srho);
      LatticeGaugeField Usm(&Grid), Utmp(&Grid);
      Utmp = Umu;
      for (int n = 0; n < nsmear; ++n) {
        Stout.smear(Usm, Utmp);
        Utmp = Usm;
        RealD psm = WilsonLoops<PeriodicGimplR>::avgPlaquette(Utmp);
        std::cout << GridLogMessage
                  << "  [stout] after smear #" << (n + 1)
                  << " plaq = " << std::setprecision(12) << psm << std::endl;
      }
    }

    // Optional: also report gauge action value under LW_TREE coefficients.
    const char *beta_env = std::getenv("BETA");
    if (beta_env && *beta_env) {
      RealD beta = std::atof(beta_env);
      if (beta != 0.0) {
        const char *u0_env = std::getenv("U0");
        RealD u0 = (u0_env && *u0_env) ? std::atof(u0_env) : 0.832605301399891;
        RealD c_plaq = beta;
        RealD c_rect = -beta / (20.0 * u0 * u0);
        PlaqPlusRectangleAction<PeriodicGimplR> Sgauge(c_plaq, c_rect);
        RealD S_grid = Sgauge.S(Umu);
        RealD rect = WilsonLoops<PeriodicGimplR>::avgRectangle(Umu);
        RealD V = (RealD)Umu.Grid()->gSites();
        // Chroma's U-dependent action (per PlaqGaugeAct::S + RectGaugeAct::S):
        //   S_chroma = -(c0/Nc)·ΣReTrP - (c1/Nc)·ΣReTrR
        // With <P> = ΣReTrP / (V·Nc·N_plaq), <R> = ΣReTrR / (V·Nc·N_rect):
        //   S_chroma = -c0·N_plaq·V·<P> - c1·N_rect·V·<R>    (for the U-dep part)
        // Grid's S(U) on the same U:
        //   S_grid = c_plaq·(1-<P>)·N_plaq·V + c_rect·(1-<R>)·N_rect·V
        // The U-independent offset between the two (when c_plaq=c0, c_rect=c1):
        //   offset = c0·N_plaq·V + c1·N_rect·V
        // so  S_grid − offset = -c0·N_plaq·V·<P> - c1·N_rect·V·<R> = S_chroma_Udep.
        RealD N_plaq = (RealD)Nd * (Nd - 1) / 2.0;  // 6
        RealD N_rect = (RealD)Nd * (Nd - 1);         // 12
        RealD offset = c_plaq * N_plaq * V + c_rect * N_rect * V;
        RealD S_grid_Udep = S_grid - offset;
        RealD S_chroma_Udep = -c_plaq * N_plaq * V * plaq
                              - c_rect * N_rect * V * rect;
        std::cout << GridLogMessage
                  << "  [action] β=" << beta << " u0=" << u0
                  << " c_plaq=" << c_plaq << " c_rect=" << c_rect << std::endl;
        std::cout << GridLogMessage
                  << "  [action] avg rect = " << std::setprecision(10) << rect
                  << std::endl;
        std::cout << GridLogMessage
                  << "  [action] Grid S(U)        = " << std::setprecision(15)
                  << S_grid << std::endl;
        std::cout << GridLogMessage
                  << "  [action] Grid S(U) − offset = " << std::setprecision(15)
                  << S_grid_Udep << std::endl;
        std::cout << GridLogMessage
                  << "  [action] chroma S(U) U-dep = " << std::setprecision(15)
                  << S_chroma_Udep << std::endl;
        RealD rel = std::fabs(S_grid_Udep - S_chroma_Udep) /
                    std::max(std::fabs(S_chroma_Udep), 1.0);
        std::cout << GridLogMessage
                  << "  [action] relative diff = " << rel
                  << (rel < 1e-10 ? "  PASS" : "  NOTE") << std::endl;
      }
    }
  }

  Grid_finalize();
  return 0;
}
