// Force-vs-FD consistency test for the production LW tree-level gauge action.
//
// Loads any cfg (LIME or NERSC), constructs PlaqPlusRectangleAction with our
// (β, -β/(20 u0²)) coefficients, samples random momentum p, then checks
//
//   dS_actual = S(U_+) - S(U_-)            (numeric, ε-stepped)
//   dS_pred   = -2 ε Σ_μ Re tr(p_μ U_μ ∂S/∂U_μ)   (from action.deriv)
//
// using the same convention as Grid's existing tests/forces/Test_rect_force.
// Reports ratio dS_actual / (2 dS_pred ε) for several ε.  Equality at small ε
// proves analytic deriv is consistent with action.  Failure pinpoints a
// gauge-force bug masked by numerically-correct S(U).
//
// Usage:
//   ./test_force_gauge <cfg> --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048

#include <Grid/Grid.h>
#include <cstdio>
#include <cstring>

#include "params.h"

using namespace Grid;
using namespace TXQCDProduction;

enum class CfgFormat { NERSC, LIME_ILDG, UNKNOWN };
static CfgFormat detect_format(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return CfgFormat::UNKNOWN;
  char magic[16] = {0};
  std::fread(magic, 1, sizeof(magic), f);
  std::fclose(f);
  if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) return CfgFormat::NERSC;
  return CfgFormat::LIME_ILDG;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::string cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) { ++i; continue; }
    cfg = a;
  }
  if (cfg.empty()) {
    std::cerr << "Usage: test_force_gauge <cfg> --grid L.L.L.T --mpi mx.my.mz.mt\n";
    Grid_finalize();
    return 1;
  }

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({77, 78, 79, 80, 81});

  LatticeGaugeField Umu(&Grid);
  CfgFormat fmt = detect_format(cfg);
  FieldMetaData header;
  typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
  if (fmt == CfgFormat::NERSC) {
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cfg);
  } else {
    IldgReader reader;
    reader.open(cfg);
    reader.readConfiguration(Umu, header);
    reader.close();
  }

  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
  std::cout << GridLogMessage << "loaded " << cfg << " plaq=" << std::setprecision(12) << plaq << std::endl;

  // Production gauge action: chroma's LW_TREE_GAUGEACT, c0=β, c1=-β/(20 u0²).
  PlaqPlusRectangleAction<PeriodicGimplR> Action(beta, -beta / (20.0 * u0 * u0));

  // Sample a random Lie-algebra momentum.
  LatticeGaugeField mom(&Grid);
  LatticeColourMatrix mommu(&Grid);
  for (int mu = 0; mu < Nd; ++mu) {
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
    PokeIndex<LorentzIndex>(mom, mommu, mu);
  }

  // Analytic prediction: dS_pred = -2 trace(p · U · ∂S/∂U).
  // Conventions: Grid's deriv returns U·dS/dU in the antihermitian projection;
  // for U' = (1 + ε p) U, dS = -2 ε sum_μ Re trace(p_μ · UdSdU_μ).
  LatticeGaugeField UdSdU(&Grid);
  Action.deriv(Umu, UdSdU);

  ComplexD dSpred(0.0, 0.0);
  for (int mu = 0; mu < Nd; ++mu) {
    auto UdSdUmu = PeekIndex<LorentzIndex>(UdSdU, mu);
    auto pmu     = PeekIndex<LorentzIndex>(mom, mu);
    LatticeComplex dS_mu = -2.0 * trace(pmu * UdSdUmu);
    dSpred += TensorRemove(sum(dS_mu));
  }

  RealD S0 = Action.S(Umu);
  std::cout << GridLogMessage << "S(U) = " << std::setprecision(15) << S0 << std::endl;
  std::cout << GridLogMessage << "dS_pred (per ε) = " << dSpred << std::endl;

  // Numerical check: S(U±εp) where U' = (1 + ε p) U.  Linear in ε to leading
  // order, so dS_actual / ε should equal dS_pred at small ε up to O(ε²) error.
  for (RealD eps : {1e-2, 1e-3, 1e-4, 1e-5}) {
    LatticeGaugeField Up(&Grid), Um(&Grid);
    {
      autoView(Up_v, Up, CpuWrite);
      autoView(Um_v, Um, CpuWrite);
      autoView(U_v,  Umu, CpuRead);
      autoView(p_v,  mom, CpuRead);
      thread_foreach(i, p_v, {
        for (int mu = 0; mu < Nd; ++mu) {
          Up_v[i](mu) = U_v[i](mu) + p_v[i](mu) * U_v[i](mu) * eps;
          Um_v[i](mu) = U_v[i](mu) - p_v[i](mu) * U_v[i](mu) * eps;
        }
      });
    }
    RealD Sp = Action.S(Up);
    RealD Sm = Action.S(Um);
    RealD dS_actual = (Sp - Sm);  // = 2 ε * dS/dε  (leading order)
    ComplexD ratio = dS_actual / (2.0 * eps * dSpred);
    std::cout << GridLogMessage
              << "ε=" << std::scientific << std::setprecision(2) << eps
              << "  dS_actual/2ε=" << std::setprecision(10) << dS_actual / (2.0 * eps)
              << "  dS_pred=" << dSpred
              << "  ratio=" << ratio << std::endl;
  }

  Grid_finalize();
  return 0;
}
