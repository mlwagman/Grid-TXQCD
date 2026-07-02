// Test_qcd_quenched_plaq: PURE-GAUGE (quenched) HMC at the production gauge
// action point — PlaqPlusRectangleAction with c_plaq = beta = 6.1 and
// c_rect = -0.439968 (the exact values the DTXQCD chains log, i.e.
// beta/(20 u0^2) with the chroma-matched u0).
//
// Purpose (2026-07-01, plaq-deficit investigation): the lambda<=0.25 DTXQCD
// non-EO chains plateau at plaq ~= 0.4534, well below the dynamical QCD
// reference 0.5133.  Hypothesis: at small lambda the huge aux disorder
// (width 1/lambda) suppresses the sea-quark feedback on the gauge like
// O(lambda^2), so the sampled gauge ensemble sits at the QUENCHED fixed
// point.  If quenched plaq at this action point ~= 0.4534, the hypothesis is
// confirmed quantitatively.
//
// Run (two streams to bracket equilibrium):
//   ./Test_qcd_quenched_plaq --grid 16.16.16.48 --mpi 1.1.1.1 \
//       --StartingType HotStart  --Trajectories 400 --Thermalizations 0
//   ./Test_qcd_quenched_plaq ... --StartingType ColdStart ...
// Env: BETA (6.1), C_RECT (-0.439968), MDSTEPS (20), TRAJL (1.0),
//      CKPT_PREFIX (checkpoint path prefix, default ./ckpoint_lat)

#include <Grid/Grid.h>
#include <cstdlib>

int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);

  auto env_real = [](const char *name, RealD dflt) {
    const char *e = std::getenv(name);
    return (e && *e) ? std::atof(e) : dflt;
  };
  auto env_int = [](const char *name, int dflt) {
    const char *e = std::getenv(name);
    return (e && *e) ? std::atoi(e) : dflt;
  };

  typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");

  CheckpointerParameters CPparams;
  std::string prefix = "./ckpoint_lat";
  if (const char *p = std::getenv("CKPT_PREFIX")) prefix = p;
  CPparams.config_prefix = prefix;
  CPparams.rng_prefix    = prefix + "_rng";
  CPparams.saveInterval  = env_int("CKPT_INTERVAL", 100);
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds   = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();

  const RealD beta   = env_real("BETA", 6.1);
  const RealD c_rect = env_real("C_RECT", -0.439968);
  std::cout << GridLogMessage << "[quenched] PlaqPlusRectangle c_plaq=" << beta
            << " c_rect=" << c_rect << std::endl;
  PlaqPlusRectangleActionR Gaction(beta, c_rect);

  ActionLevel<HMCWrapper::Field> Level1(1);
  Level1.push_back(&Gaction);
  TheHMC.TheAction.push_back(Level1);

  TheHMC.Parameters.MD.MDsteps = env_int("MDSTEPS", 20);
  TheHMC.Parameters.MD.trajL   = env_real("TRAJL", 1.0);

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Run();

  Grid_finalize();
  return 0;
}
