// Test_dtxqcd_fierz_full_qcd:
//
// Full-QCD Fierz equivalence test on a small lattice (4^4 default).  Unlike
// the freefield_qbarq variants which freeze gauge at U=I, this driver runs
// TWO full HMCs back-to-back at identical physical params:
//
//   1) DTXQCD HMC (Wilson gauge + Nf=2 Wilson rational PF on M48 + aux):
//      records per-traj plaquette AND Σ_DTX / Σ_W via FierzAvg observer.
//
//   2) Pure QCD HMC (Wilson gauge + Nf=2 Wilson rational PF, no aux):
//      records per-traj plaquette as the reference.
//
// At large λ the two HMCs should produce statistically identical
// plaquettes (the aux's effect on the gauge integral measure scales as
// 1/λ²).  When this test passes, any plaquette drift seen in production
// is a code-path issue, not a fundamental property of the action.
//
// Defaults: λ=10, m=0.3, csw=0, β=6.0, lattice 4^4, MDsteps=20, trajL=√2,
//   N_THERM=200, N_PROD=200, FIERZ_AVG_N_NOISE=16, FIERZ_AVG_MEAS_STRIDE=5.
//
// Env knobs: LAMBDA, MASS, CSW, BETA, MDSTEPS, TRAJL, N_THERM, N_PROD,
//   FIERZ_AVG_N_NOISE, FIERZ_AVG_MEAS_STRIDE, CFG_DIR, PASS_TOL,
//   SKIP_QCD=1 (skip phase 2 for a DTXQCD-only run).

#include "Test_dtxqcd_2pt_utils.h"
#include "Test_dtxqcd_fierz_check_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

// Per-traj plaquette observer.  Templated on the gauge-field type so we
// can use it for both DTXQCDField (extracts .U) and LatticeGaugeField.
namespace {

template <class Field>
struct PlaqExtract;

template <>
struct PlaqExtract<DTXQCDField> {
  static RealD plaq(const DTXQCDField &U) {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  }
};

template <>
struct PlaqExtract<LatticeGaugeField> {
  static RealD plaq(const LatticeGaugeField &U) {
    return WilsonLoops<PeriodicGimplR>::avgPlaquette(U);
  }
};

template <class Field>
class PlaqLogger : public HmcObservable<Field> {
 public:
  std::vector<RealD> samples;
  int n_skip;
  std::string tag;
  PlaqLogger(int n_skip_, std::string tag_) : n_skip(n_skip_), tag(tag_) {}
  void TrajectoryComplete(int traj, Field &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    RealD p = PlaqExtract<Field>::plaq(U);
    std::cout << GridLogMessage << "[" << tag << " plaq traj " << traj
              << "] " << p << std::endl;
    if (traj >= n_skip) samples.push_back(p);
  }
  // Bookkeeping summary.
  void summary(const std::string &name) const {
    int N = (int)samples.size();
    if (N == 0) {
      std::cout << GridLogMessage << "[" << name << "] no samples"
                << std::endl;
      return;
    }
    RealD mean = 0.0;
    for (auto p : samples) mean += p;
    mean /= N;
    RealD var = 0.0;
    for (auto p : samples) var += (p - mean) * (p - mean);
    RealD se = (N > 1) ? std::sqrt(var / (N * (N - 1))) : 0.0;
    std::cout << GridLogMessage << "[" << name
              << "] mean plaq = " << mean << " ± " << se
              << " (N=" << N << ")" << std::endl;
  }
  RealD mean() const {
    if (samples.empty()) return 0.0;
    RealD s = 0.0; for (auto p : samples) s += p; return s / samples.size();
  }
  RealD se() const {
    int N = (int)samples.size();
    if (N < 2) return 0.0;
    RealD m = mean();
    RealD v = 0.0; for (auto p : samples) v += (p - m) * (p - m);
    return std::sqrt(v / (N * (N - 1)));
  }
};

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ---- defaults (200 warm + 200 prod by default) ----
  RealD lambda_run = 10.0;
  RealD mass_run   = 0.3;
  RealD csw_run    = 0.0;
  RealD beta_run   = 6.0;
  int mdsteps      = 20;
  RealD trajL      = std::sqrt(2.0);
  int n_therm_run  = 200;
  int n_prod_run   = 200;
  int meas_skip_run = 10;
  bool skip_qcd    = false;
  std::string cfg_dir = "fierz_full_qcd";

  if (const char *v = std::getenv("LAMBDA");    v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");      v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW");       v && *v) csw_run    = std::atof(v);
  if (const char *v = std::getenv("BETA");      v && *v) beta_run   = std::atof(v);
  if (const char *v = std::getenv("MDSTEPS");   v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");     v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");   v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");    v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP"); v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");   v && *v) cfg_dir = v;
  if (const char *v = std::getenv("SKIP_QCD");  v && *v) skip_qcd = (std::atoi(v) != 0);

  setenv("USE_FULL_PF", "1", 1);  // gauge evolves via WilsonGaugeAction

  std::cout << GridLogMessage
            << "DTXQCD FIERZ FULL-QCD test: lambda=" << lambda_run
            << " mass=" << mass_run << " csw=" << csw_run
            << " beta=" << beta_run
            << " MDsteps=" << mdsteps << " trajL=" << trajL
            << " N_THERM=" << n_therm_run << " N_PROD=" << n_prod_run
            << " cfg_dir=" << cfg_dir << std::endl;

  // ---- shared lattice geometry ----
  Coordinate latt_default(std::vector<int>{4, 4, 4, 4});
  Coordinate latt = GridDefaultLatt();
  if (latt.size() == 0) latt = latt_default;
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  std::cout << GridLogMessage << "Lattice = " << latt[0] << "."
            << latt[1] << "." << latt[2] << "." << latt[3] << std::endl;

  TxqcdTest2pt::mkdir_p(cfg_dir);

  // APBC time matches DTXQCD default
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;

  RealD pass_tol = 0.02;
  if (const char *v = std::getenv("PASS_TOL"); v && *v) pass_tol = std::atof(v);

  // SOLVER=HMC (default) uses standard Nf=2 TwoFlavourPseudoFermionAction
  // (single regular CG on M^†M).  SOLVER=RHMC uses 2× OneFlavourRational
  // (multishift CG + Remez per PF).  Same Boltzmann weight, but RHMC ~10×
  // slower with clover; spotting any small numerical residue is the use case.
  std::string qcd_solver = "HMC";
  if (const char *v = std::getenv("SOLVER"); v && *v) qcd_solver = v;
  bool qcd_use_hmc = (qcd_solver == "HMC" || qcd_solver == "hmc");
  std::cout << GridLogMessage << "QCD-phase solver: "
            << (qcd_use_hmc ? "HMC (TwoFlavour, regular CG)"
                            : "RHMC (2× OneFlavourRational, multishift CG)")
            << std::endl;

  // ============================================================
  // Phase 1 : pure QCD reference HMC (faster — runs first so we get
  //          the reference plaq before the slower DTXQCD chain finishes).
  //          Skip with SKIP_QCD=1.
  // ============================================================
  PlaqLogger<LatticeGaugeField> plaq_qcd(n_therm_run, "QCD");
  if (!skip_qcd) {
    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);
    // Different seed so the two ensembles aren't trivially correlated.
    sRNG.SeedFixedIntegers({111, 112, 113, 114, 115});
    pRNG.SeedFixedIntegers({211, 212, 213, 214, 215});

    LatticeGaugeField U(&Grid);
    SU<Nc>::ColdConfiguration(U);

    WilsonGaugeActionR GaugeAction(beta_run);
    // Honor csw — Wilson-Clover when csw>0 to match the DTXQCD phase.
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    std::unique_ptr<WilsonFermion<WilsonImplR>>  Dw_plain;
    std::unique_ptr<WCF>                         Dw_clover;
    FermionOperator<WilsonImplR>                *Dw = nullptr;
    if (csw_run == 0.0) {
      Dw_plain.reset(new WilsonFermion<WilsonImplR>(U, Grid, RBGrid, mass_run, ip));
      Dw = Dw_plain.get();
    } else {
      Dw_clover.reset(new WCF(U, Grid, RBGrid, mass_run, csw_run, csw_run,
                              WilsonAnisotropyCoefficients(), ip));
      Dw = Dw_clover.get();
    }

    // Two PF variants — SOLVER=HMC (default) and SOLVER=RHMC.
    ConjugateGradient<LatticeFermion> CG_action(1e-8, 10000);
    ConjugateGradient<LatticeFermion> CG_deriv (1e-6, 10000);
    std::unique_ptr<TwoFlavourPseudoFermionAction<WilsonImplR>> PF_hmc;
    std::unique_ptr<OneFlavourRationalPseudoFermionAction<WilsonImplR>> PF_rhmc1, PF_rhmc2;
    OneFlavourRationalParams rp(/*lo=*/0.05, /*hi=*/200.0, /*MaxIter=*/10000,
                                /*tol=*/1e-8, /*degree=*/12, /*precision=*/64,
                                /*BCFreq=*/100, /*mdtol=*/1e-6);
    if (const char *v = std::getenv("RAT_LO"); v && *v) rp.lo = std::atof(v);
    if (const char *v = std::getenv("RAT_HI"); v && *v) rp.hi = std::atof(v);
    if (qcd_use_hmc) {
      PF_hmc.reset(new TwoFlavourPseudoFermionAction<WilsonImplR>(
                       *Dw, CG_deriv, CG_action));
    } else {
      PF_rhmc1.reset(new OneFlavourRationalPseudoFermionAction<WilsonImplR>(*Dw, rp));
      PF_rhmc2.reset(new OneFlavourRationalPseudoFermionAction<WilsonImplR>(*Dw, rp));
    }

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    if (qcd_use_hmc) {
      L1.push_back(PF_hmc.get());
    } else {
      L1.push_back(PF_rhmc1.get());
      L1.push_back(PF_rhmc2.get());
    }
    ActionLevel<LatticeGaugeField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1); Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name = "ForceGradient";
    MD.MDsteps = mdsteps;
    MD.trajL   = trajL;

    HMCparameters HMCp;
    HMCp.StartTrajectory    = 0;
    HMCp.Trajectories       = n_therm_run + n_prod_run;
    HMCp.NoMetropolisUntil  = n_therm_run;
    HMCp.MetropolisTest     = true;
    HMCp.PerformRandomShift = false;
    HMCp.StartingType       = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    Smear.set_Field(U);
    typedef ForceGradient<PeriodicGimplR, NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs;
    Obs.push_back(&plaq_qcd);

    std::cout << GridLogMessage << "===== Phase 1: pure QCD HMC =====" << std::endl;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();

    plaq_qcd.summary("Phase 1 QCD ");
    // Print headline result as soon as QCD finishes — DTXQCD chain is
    // slower and the user wants this number first.
    std::cout << GridLogMessage << std::endl
              << "===== QCD REFERENCE PLAQ (Phase 1 complete) =====" << std::endl
              << "  β=" << beta_run << "  m=" << mass_run
              << "  lattice=" << latt[0] << "." << latt[1] << "." << latt[2]
              << "." << latt[3] << std::endl
              << "  QCD plaq = " << plaq_qcd.mean()
              << " ± " << plaq_qcd.se()
              << "  (N=" << plaq_qcd.samples.size() << ")"
              << std::endl
              << "=====================================================" << std::endl;
  }

  // ============================================================
  // Phase 2 : DTXQCD HMC
  // ============================================================
  PlaqLogger<DTXQCDField> plaq_dtxqcd(n_therm_run, "DTXQCD");
  DtxqcdFierzCheckResult fierz_result{0, 0, 0, 0, false};
  RealD sigma_w_dtxqcd_mean = 0.0;  // for the cross-comparison summary
  {
    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({101, 102, 103, 104, 105});
    pRNG.SeedFixedIntegers({201, 202, 203, 204, 205});

    DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_run);
    DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);

    RealD rat_lo = 0.05, rat_hi = 200.0;
    int   rat_degree = 12;
    if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
    if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
    if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
    OneFlavourRationalParams rp(rat_lo, rat_hi, /*MaxIter=*/10000, /*tol=*/1e-8,
                                rat_degree, 64, /*BCFreq=*/100, /*mdtol=*/1e-6);

    DTXQCDWilsonCloverRationalFullAction
        PF_full(Grid, RBGrid, mass_run, rp, csw_run);

    typedef Representations<EmptyRep<DTXQCDField>> Reps;
    ActionLevel<DTXQCDField, Reps> L1(1);
    L1.push_back(&PF_full);
    L1.push_back(&AuxAction);
    ActionLevel<DTXQCDField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    ActionSet<DTXQCDField, Reps> Aset;
    Aset.push_back(L1); Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name = "ForceGradient";
    MD.MDsteps = mdsteps;
    MD.trajL   = trajL;

    DTXQCDField U(&Grid);
    DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);

    HMCparameters HMCp;
    HMCp.StartTrajectory    = 0;
    HMCp.Trajectories       = n_therm_run + n_prod_run;
    HMCp.NoMetropolisUntil  = n_therm_run;
    HMCp.MetropolisTest     = true;
    HMCp.PerformRandomShift = false;
    HMCp.StartingType       = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<DTXQCDCompositeImpl> Smear;
    typedef ForceGradient<DTXQCDCompositeImpl,
                          NoSmearing<DTXQCDCompositeImpl>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(U);

    std::vector<HmcObservable<DTXQCDField> *> Obs;
    Obs.push_back(&plaq_dtxqcd);

    std::unique_ptr<DtxqcdFierzAveragingObserver> avg_obs;
    int fierz_avg_n_noise = 16;
    if (const char *v = std::getenv("FIERZ_AVG_N_NOISE"); v && *v)
      fierz_avg_n_noise = std::atoi(v);
    if (fierz_avg_n_noise > 0) {
      avg_obs.reset(new DtxqcdFierzAveragingObserver(Grid, RBGrid, mass_run,
                                                      csw_run, lambda_run,
                                                      n_therm_run,
                                                      fierz_avg_n_noise,
                                                      /*cg_tol=*/1e-8));
      Obs.push_back(avg_obs.get());
    }

    std::cout << GridLogMessage << "===== Phase 1: DTXQCD HMC =====" << std::endl;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();

    if (avg_obs) {
      fierz_result = avg_obs->finalize(pass_tol,
                                        "Test_dtxqcd_fierz_full_qcd");
      sigma_w_dtxqcd_mean = fierz_result.sigma_w;
    }
    plaq_dtxqcd.summary("Phase 2 DTXQCD");
  }

  // ============================================================
  // Cross-comparison summary
  // ============================================================
  std::cout << GridLogMessage << std::endl
            << "===== CROSS-COMPARISON SUMMARY =====" << std::endl;
  std::cout << GridLogMessage
            << "Lattice " << latt[0] << "." << latt[1] << "." << latt[2]
            << "." << latt[3] << "  β=" << beta_run << "  m=" << mass_run
            << "  csw=" << csw_run << "  λ=" << lambda_run << std::endl;
  std::cout << GridLogMessage
            << "  DTXQCD plaq = " << plaq_dtxqcd.mean()
            << " ± " << plaq_dtxqcd.se()
            << " (N=" << plaq_dtxqcd.samples.size() << ")" << std::endl;
  if (!skip_qcd) {
    std::cout << GridLogMessage
              << "  QCD    plaq = " << plaq_qcd.mean()
              << " ± " << plaq_qcd.se()
              << " (N=" << plaq_qcd.samples.size() << ")" << std::endl;
    RealD dplaq = plaq_dtxqcd.mean() - plaq_qcd.mean();
    RealD se_diff = std::sqrt(plaq_dtxqcd.se()*plaq_dtxqcd.se()
                            + plaq_qcd.se()*plaq_qcd.se());
    RealD nsig = (se_diff > 0) ? std::fabs(dplaq) / se_diff : 0.0;
    std::cout << GridLogMessage
              << "  Δplaq (DTXQCD − QCD) = " << dplaq
              << " ± " << se_diff
              << "  (" << nsig << " σ)" << std::endl;
  }
  if (fierz_result.pass || sigma_w_dtxqcd_mean != 0.0) {
    std::cout << GridLogMessage
              << "  Fierz Σ_DTX / Σ_W = " << fierz_result.ratio
              << "  |dev|=" << fierz_result.dev
              << "  pass=" << (fierz_result.pass ? "YES" : "NO")
              << std::endl;
  }

  Grid_finalize();
  return fierz_result.pass ? 0 : 1;
}
