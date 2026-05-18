// bench_gauge_force.cc — time PlaqPlusRectangleAction::deriv() on the
// production 16³×48 gauge field. Helps isolate the per-call gauge-force
// cost from the multi-rate integrator overhead seen in HMC trajectories.
//
// Usage:
//   IMPORT_CFG=<lime/nersc> N_REPS=20 ./bench_gauge_force --grid 16.16.16.48 --mpi 1.1.1.1

#include "params.h"
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace Grid;
using namespace TXQCDProduction;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), mpi);

  LatticeGaugeField Umu(&Grid);
  std::cout << GridLogMessage << "Loading gauge cfg..." << std::endl;
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(Umu, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(Umu, header);
      IR.close();
    }
  } else {
    SU<Nc>::HotConfiguration(*new GridParallelRNG(&Grid), Umu);
  }

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
  std::cout << GridLogMessage << "plaq = " << plaq << std::endl;

  // Production gauge-action coefficients (Lüscher-Weisz tree-level, β=6.1, u0=0.949).
  PlaqPlusRectangleAction<PeriodicGimplR> GaugeAction(beta, -beta / (20.0 * u0 * u0));

  int n_reps = 20;
  if (const char *t = std::getenv("N_REPS"); t && *t) n_reps = std::atoi(t);
  std::cout << GridLogMessage << "N_REPS = " << n_reps << std::endl;

  LatticeGaugeField dSdU(&Grid);

  // Warm-up.
  GaugeAction.deriv(Umu, dSdU);

  std::vector<double> times(n_reps);
  for (int r = 0; r < n_reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    GaugeAction.deriv(Umu, dSdU);
    auto t1 = std::chrono::steady_clock::now();
    times[r] = std::chrono::duration<double>(t1 - t0).count();
    std::cout << GridLogMessage << "  rep " << r << "  deriv time = " << times[r] << " s,"
              << "  ||dSdU|| = " << norm2(dSdU) << std::endl;
  }

  double sum = 0, sum2 = 0;
  for (auto t : times) { sum += t; sum2 += t*t; }
  double mean = sum / n_reps;
  double var  = sum2 / n_reps - mean*mean;
  std::cout << GridLogMessage << "Plaq+Rect deriv avg = " << mean
            << " s, stddev = " << std::sqrt(std::max(var, 0.0)) << " s" << std::endl;

  Grid_finalize();
  return 0;
}
