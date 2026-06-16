// Test_qcd_freefield_qbarq:
//
// Vanilla QCD HMC reference for the {TXQCD, DTXQCD} free-field test
// progression.  Plain Wilson fermion PF action with configurable Nf:
//
//   Nf=2 (DTXQCD effective):  1 TwoFlavourPseudoFermionAction → |det M|²
//   Nf=4 (TXQCD effective):   2 TwoFlavourPseudoFermionAction → |det M|⁴
//
// Step 1 (U frozen via QCD_FREEZE_GAUGE=1, default): gauge stays at U=I.
//   Output ⟨q̄q⟩ via stochastic Hutchinson on the same cfg used by the
//   {TXQCD, DTXQCD} runs.  Comparison is then per-cfg trivial — but the
//   reference Σ_W is computed in the same convention (Test_txqcd_trminv_compare).
//
// Step 2 (QCD_FREEZE_GAUGE=0): gauge dynamics on with Wilson gauge action,
//   provides the Σ ensemble mean to compare to TXQCD/DTXQCD with U unfrozen.
//
// Env knobs:
//   NF        — 2 or 4 (default 2)
//   MASS      — Wilson mass (default 0.3)
//   BETA      — Wilson gauge β (default 6.0)
//   MDSTEPS   — integrator MD steps (default 10)
//   TRAJL     — trajectory length (default 1.0)
//   N_PROD    — production trajs (default 30)
//   N_THERM   — thermalization trajs (default 10)
//   MEAS_SKIP — checkpoint interval (default 5)
//   CFG_DIR   — output dir
//   QCD_FREEZE_GAUGE — 1 to keep U=I, 0 for dynamical gauge (default 1)

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavour.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

// Custom GimplR variant: zeros gauge momentum + force when frozen.
// Used by HMC integrator template; otherwise identical to PeriodicGimplR.
struct PeriodicGimplFreezable : public PeriodicGimplR {
  static int freeze_gauge() {
    static int v = []() {
      const char *e = std::getenv("QCD_FREEZE_GAUGE");
      // Default: frozen.
      if (!e || !*e) return 1;
      return std::atoi(e);
    }();
    return v;
  }

  static void generate_momenta(LatticeGaugeField &P, GridSerialRNG &sRNG,
                               GridParallelRNG &pRNG) {
    PeriodicGimplR::generate_momenta(P, sRNG, pRNG);
    if (freeze_gauge()) P = Zero();
  }
  static LatticeGaugeField projectForce(LatticeGaugeField &F) {
    if (freeze_gauge()) {
      LatticeGaugeField out(F.Grid());
      out = Zero();
      return out;
    }
    return PeriodicGimplR::projectForce(F);
  }
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  int nf = 2;
  RealD mass = 0.3;
  RealD beta = 6.0;
  int mdsteps = 10;
  RealD trajL = 1.0;
  int n_therm = 10;
  int n_prod  = 30;
  int meas_skip = 5;
  std::string cfg_dir = "free_qcd";

  if (const char *v = std::getenv("NF");        v && *v) nf = std::atoi(v);
  if (const char *v = std::getenv("MASS");      v && *v) mass = std::atof(v);
  if (const char *v = std::getenv("BETA");      v && *v) beta = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");   v && *v) mdsteps = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");     v && *v) trajL = std::atof(v);
  if (const char *v = std::getenv("N_THERM");   v && *v) n_therm = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");    v && *v) n_prod = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");   v && *v) cfg_dir = v;
  // Default QCD_FREEZE_GAUGE=1 if unset
  if (!std::getenv("QCD_FREEZE_GAUGE")) setenv("QCD_FREEZE_GAUGE", "1", 1);

  if (nf != 2 && nf != 4) {
    std::cerr << "ERROR: NF must be 2 or 4" << std::endl;
    Grid_finalize();
    return 1;
  }

  std::cout << GridLogMessage
            << "QCD FREE-FIELD test: Nf=" << nf
            << " mass=" << mass << " beta=" << beta
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " FREEZE_GAUGE=" << PeriodicGimplFreezable::freeze_gauge()
            << " cfg_dir=" << cfg_dir << std::endl;

  std::vector<int> latt_dims{4,4,4,8};
  Coordinate latt(latt_dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  // Cold U = I
  LatticeGaugeField U(&Grid);
  SU<Nc>::ColdConfiguration(U);
  std::cout << GridLogMessage
            << "  U cold start: plaq = "
            << WilsonLoops<PeriodicGimplR>::avgPlaquette(U)
            << " (expect 1)" << std::endl;

  // Wilson fermion (csw=0)
  WilsonFermion<WilsonImplR> Dw(U, Grid, RBGrid, mass);
  ConjugateGradient<LatticeFermion> CG_S(1e-12, 30000);
  ConjugateGradient<LatticeFermion> CG_F(1e-8, 30000);

  // Nf=2 PF: 1 instance.  Nf=4: 2 instances (each gives |det M|²).
  TwoFlavourPseudoFermionAction<WilsonImplR> PF_a(Dw, CG_S, CG_F);
  TwoFlavourPseudoFermionAction<WilsonImplR> PF_b(Dw, CG_S, CG_F);  // ignored if nf=2

  WilsonGaugeActionR GaugeAction(beta);

  typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
  ActionLevel<LatticeGaugeField, Reps> L1(1);
  L1.push_back(&PF_a);
  if (nf == 4) L1.push_back(&PF_b);
  ActionLevel<LatticeGaugeField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<LatticeGaugeField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = mdsteps;
  MD.trajL   = trajL;

  int total_traj = n_therm + n_prod;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = 0;
  HMCp.Trajectories        = total_traj - n_therm;
  HMCp.NoMetropolisUntil   = n_therm;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  NoSmearing<PeriodicGimplFreezable> Smear;
  typedef ForceGradient<PeriodicGimplFreezable,
                        NoSmearing<PeriodicGimplFreezable>, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  // mkdir
  mkdir(cfg_dir.c_str(), 0755);

  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  NerscHmcCheckpointer<PeriodicGimplFreezable> ckpt(CPp);

  std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  Grid_finalize();
  return 0;
}
