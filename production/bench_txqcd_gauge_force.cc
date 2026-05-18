// bench_txqcd_gauge_force.cc — time GaugeActionAdapter<PlaqPlusRectangleAction>
// on a TXQCDField (the wrapped form actually used in TXQCD HMC). Compares
// against bench_gauge_force.cc to isolate adapter overhead from kernel cost.

#include "params.h"
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/GaugeActionAdapter.h>
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

  TXQCDField U(&Grid);
  std::cout << GridLogMessage << "Loading gauge cfg..." << std::endl;
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    FILE *fp = std::fopen(ic, "rb"); char magic[16] = {0};
    if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(U.U, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(U.U, header);
      IR.close();
    }
  } else {
    GridParallelRNG pRNG(&Grid);
    pRNG.SeedFixedIntegers({1,2,3,4});
    SU<Nc>::HotConfiguration(pRNG, U.U);
  }
  U.sigma = Zero();
  U.pi    = Zero();
  U.s     = Zero();
  U.p     = Zero();
  U.t     = Zero();

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage << "plaq = " << plaq << std::endl;

  // Production gauge-action coefficients.
  GaugeActionAdapter<PlaqPlusRectangleAction<PeriodicGimplR>>
      GaugeAction(beta, -beta / (20.0 * u0 * u0));
  GaugeAction.is_smeared = false;

  int n_reps = 2;
  if (const char *t = std::getenv("N_REPS"); t && *t) n_reps = std::atoi(t);
  std::cout << GridLogMessage << "N_REPS = " << n_reps << std::endl;

  TXQCDField dSdU(&Grid);

  // Warm-up.
  GaugeAction.deriv(U, dSdU);
  std::cout << GridLogMessage << "warmup done" << std::endl;

  std::vector<double> times(n_reps);
  for (int r = 0; r < n_reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    GaugeAction.deriv(U, dSdU);
    auto t1 = std::chrono::steady_clock::now();
    times[r] = std::chrono::duration<double>(t1 - t0).count();
    std::cout << GridLogMessage << "  rep " << r << " deriv time = " << times[r] << " s,"
              << "  ||dSdU.U|| = " << norm2(dSdU.U) << std::endl;
  }

  double sum = 0;
  for (auto t : times) sum += t;
  std::cout << GridLogMessage << "TXQCD-adapter Plaq+Rect deriv avg = "
            << sum / n_reps << " s" << std::endl;

  Grid_finalize();
  return 0;
}
