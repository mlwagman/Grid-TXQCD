// Test_dtxqcd_fierz_prod_action:
//
// Production-style Fierz / plaq equivalence test.  Same structure as
// Test_dtxqcd_fierz_full_qcd but with the production action stack:
//   - Lüscher-Weisz tadpole-improved Symanzik gauge action on THIN links
//     (chroma's LW_TREE_GAUGEACT: c0 = β, c1 = −β/(20·u0²))
//   - Stout-smeared fermion (rho = 0.125, n = 1 step) — chroma convention
//   - Wilson-clover at csw = 1.24930970916466, m = −0.245
//   - β = 6.1, u0 = 0.832605301399891
//
// Three phases run sequentially (each skippable):
//   1. Pure QCD HMC — LW gauge (thin) + stout-smeared Nf=2 Wilson-clover RHMC
//   2. DTXQCD non-EO HMC (USE_FULL_PF=1) — single rational PF on M48
//   3. DTXQCD EO HMC — EO Schur rational PF + LogDet
//
// All three log plaquette per traj.  Phases 2 and 3 additionally install the
// FierzAvg observer (Σ_DTX / Σ_W).  Cross-comparison printed at the end.
//
// Defaults: λ=10, m=−0.245, csw=1.24930970916466, β=6.1, u0=0.832605…,
//   stout ρ=0.125 n=1, lattice 4⁴, MDsteps=20, trajL=√2,
//   N_THERM=200, N_PROD=200, FIERZ_AVG_N_NOISE=16, FIERZ_AVG_MEAS_STRIDE=5.
//
// Skips: SKIP_QCD=1, SKIP_NONEO=1, SKIP_EO=1 to omit phases.

#include "Test_dtxqcd_2pt_utils.h"
#include "Test_dtxqcd_fierz_check_utils.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/smearing/StoutSmearing.h>
#include <Grid/qcd/smearing/GaugeConfiguration.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace Grid;

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
  void summary(const std::string &name) const {
    std::cout << GridLogMessage << "[" << name << "] mean plaq = "
              << mean() << " ± " << se()
              << " (N=" << samples.size() << ")" << std::endl;
  }
};

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // ---- production defaults ----
  RealD lambda_run = 10.0;
  RealD mass_run   = -0.2450;
  RealD csw_run    = 1.24930970916466;
  RealD beta_run   = 6.1;
  RealD u0_run     = 0.832605301399891;
  RealD stout_rho  = 0.125;
  int   stout_n    = 1;
  int   mdsteps    = 20;
  RealD trajL      = std::sqrt(2.0);
  int n_therm_run  = 200;
  int n_prod_run   = 200;
  int meas_skip_run = 10;
  bool skip_qcd    = false;
  bool skip_noneo  = false;
  bool skip_eo     = false;
  std::string cfg_dir = "fierz_prod_action";

  if (const char *v = std::getenv("LAMBDA");      v && *v) lambda_run = std::atof(v);
  if (const char *v = std::getenv("MASS");        v && *v) mass_run   = std::atof(v);
  if (const char *v = std::getenv("CSW");         v && *v) csw_run    = std::atof(v);
  if (const char *v = std::getenv("BETA");        v && *v) beta_run   = std::atof(v);
  if (const char *v = std::getenv("U0");          v && *v) u0_run     = std::atof(v);
  if (const char *v = std::getenv("STOUT_RHO");   v && *v) stout_rho  = std::atof(v);
  if (const char *v = std::getenv("STOUT_NSMEAR");v && *v) stout_n    = std::atoi(v);
  if (const char *v = std::getenv("MDSTEPS");     v && *v) mdsteps    = std::atoi(v);
  if (const char *v = std::getenv("TRAJL");       v && *v) trajL      = std::atof(v);
  if (const char *v = std::getenv("N_THERM");     v && *v) n_therm_run = std::atoi(v);
  if (const char *v = std::getenv("N_PROD");      v && *v) n_prod_run  = std::atoi(v);
  if (const char *v = std::getenv("MEAS_SKIP");   v && *v) meas_skip_run = std::atoi(v);
  if (const char *v = std::getenv("CFG_DIR");     v && *v) cfg_dir = v;
  if (const char *v = std::getenv("SKIP_QCD");    v && *v) skip_qcd   = (std::atoi(v) != 0);
  if (const char *v = std::getenv("SKIP_NONEO");  v && *v) skip_noneo = (std::atoi(v) != 0);
  if (const char *v = std::getenv("SKIP_EO");     v && *v) skip_eo    = (std::atoi(v) != 0);

  // Production-grade DTXQCD requires Remez autoscale from the start at this
  // mass / csw — without it the rational bracket misses the M48 spectrum
  // and dH blows up.  Default ON; user can disable via RAT_AUTO_HI=0.
  setenv("RAT_AUTO_HI", "1", /*overwrite=*/0);

  std::cout << GridLogMessage
            << "DTXQCD PROD-ACTION FIERZ test: λ=" << lambda_run
            << " m=" << mass_run << " csw=" << csw_run
            << " β=" << beta_run << " u0=" << u0_run
            << " stout=" << stout_n << "x ρ=" << stout_rho
            << " MDS=" << mdsteps << " trajL=" << trajL
            << " N_THERM=" << n_therm_run << " N_PROD=" << n_prod_run
            << std::endl;

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

  // APBC time (chroma convention) for the fermion impl
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;

  RealD pass_tol = 0.05;  // looser for the noisy m=-0.245 light-mass regime
  if (const char *v = std::getenv("PASS_TOL"); v && *v) pass_tol = std::atof(v);

  // SOLVER=HMC (default) → TwoFlavourPseudoFermionAction (regular CG).
  // SOLVER=RHMC → 2× OneFlavourRationalPseudoFermionAction (multishift+Remez).
  // Same Boltzmann weight; the duplicate lets us spot numerical residue.
  std::string qcd_solver = "HMC";
  if (const char *v = std::getenv("SOLVER"); v && *v) qcd_solver = v;
  bool qcd_use_hmc = (qcd_solver == "HMC" || qcd_solver == "hmc");
  std::cout << GridLogMessage << "QCD-phase solver: "
            << (qcd_use_hmc ? "HMC (TwoFlavour, regular CG)"
                            : "RHMC (2× OneFlavourRational, multishift CG)")
            << std::endl;

  // LW c1 = −β/(20·u0²)
  const RealD c1 = -beta_run / (20.0 * u0_run * u0_run);

  // Phase results
  PlaqLogger<LatticeGaugeField> plaq_qcd  (n_therm_run, "QCD");
  PlaqLogger<DTXQCDField>       plaq_noneo(n_therm_run, "DTXQCD-noneo");
  PlaqLogger<DTXQCDField>       plaq_eo   (n_therm_run, "DTXQCD-eo");
  DtxqcdFierzCheckResult fierz_noneo{0,0,0,0,false}, fierz_eo{0,0,0,0,false};

  // ============================================================
  // Phase 1 : pure QCD ref (LW gauge thin + stout-smeared WilsonClover)
  // ============================================================
  if (!skip_qcd) {
    GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({111,112,113,114,115});
    pRNG.SeedFixedIntegers({211,212,213,214,215});

    // Weak-field init (NOT cold) — stout-smeared Wilson-clover at m=-0.245
    // is unstable at U=I (clover term zero ⇒ supercritical-κ Wilson; rational
    // PF goes NaN at first action eval).  wf=0.1 matches chroma's WEAK_FIELD
    // convention (plaq ≈ 0.997 → bounds the spectrum away from zero).
    LatticeGaugeField U(&Grid);
    {
      LatticeColourMatrix Ulink(&Grid);
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::LieRandomize(pRNG, Ulink, /*wf=*/0.1);
        PokeIndex<LorentzIndex>(U, Ulink, mu);
      }
    }
    std::cout << GridLogMessage << "QCD init: weak-field gauge wf=0.1, plaq="
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U) << std::endl;

    PlaqPlusRectangleAction<PeriodicGimplR> GaugeAction(beta_run, c1);
    GaugeAction.is_smeared = false;  // LW on thin links

    WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>>
        Dw(U, Grid, RBGrid, mass_run, csw_run, csw_run,
           WilsonAnisotropyCoefficients(), ip);

    ConjugateGradient<LatticeFermion> CG_action(1e-8, 10000);
    ConjugateGradient<LatticeFermion> CG_deriv (1e-6, 10000);
    OneFlavourRationalParams rp(/*lo=*/0.001, /*hi=*/100.0, /*MaxIter=*/10000,
                                /*tol=*/1e-8, /*degree=*/12, /*precision=*/64,
                                /*BCFreq=*/100, /*mdtol=*/1e-6);
    if (const char *v = std::getenv("RAT_LO"); v && *v) rp.lo = std::atof(v);
    if (const char *v = std::getenv("RAT_HI"); v && *v) rp.hi = std::atof(v);

    std::unique_ptr<TwoFlavourPseudoFermionAction<WilsonImplR>> PF_hmc;
    std::unique_ptr<OneFlavourRationalPseudoFermionAction<WilsonImplR>> PF_rhmc1, PF_rhmc2;
    if (qcd_use_hmc) {
      PF_hmc.reset(new TwoFlavourPseudoFermionAction<WilsonImplR>(
                       Dw, CG_deriv, CG_action));
      PF_hmc->is_smeared = true;
    } else {
      PF_rhmc1.reset(new OneFlavourRationalPseudoFermionAction<WilsonImplR>(Dw, rp));
      PF_rhmc2.reset(new OneFlavourRationalPseudoFermionAction<WilsonImplR>(Dw, rp));
      PF_rhmc1->is_smeared = true;
      PF_rhmc2->is_smeared = true;
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

    Smear_Stout<PeriodicGimplR> StoutOp(stout_rho);
    SmearedConfiguration<PeriodicGimplR> Smear(&Grid, (unsigned int)stout_n, StoutOp);
    Smear.set_Field(U);

    typedef ForceGradient<PeriodicGimplR,
                          SmearedConfiguration<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs;
    Obs.push_back(&plaq_qcd);

    std::cout << GridLogMessage << "===== Phase 1: pure QCD HMC (LW gauge + stout-smeared W-clover) =====" << std::endl;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();

    plaq_qcd.summary("Phase 1 QCD ");
    std::cout << GridLogMessage << std::endl
              << "===== QCD REFERENCE PLAQ (Phase 1 complete) =====" << std::endl
              << "  β=" << beta_run << "  u0=" << u0_run
              << "  m=" << mass_run << "  csw=" << csw_run
              << "  stout=" << stout_n << "x ρ=" << stout_rho
              << "  lattice=" << latt[0] << "." << latt[1] << "." << latt[2]
              << "." << latt[3] << std::endl
              << "  QCD plaq = " << plaq_qcd.mean()
              << " ± " << plaq_qcd.se()
              << "  (N=" << plaq_qcd.samples.size() << ")"
              << std::endl
              << "=====================================================" << std::endl;
  }

  // Lambda to run one DTXQCD phase (EO or non-EO based on use_full_pf).
  auto run_dtxqcd_phase = [&](bool use_full_pf,
                              PlaqLogger<DTXQCDField> &plaq_log,
                              DtxqcdFierzCheckResult &fierz_out,
                              const std::string &phase_tag) {
    if (use_full_pf) setenv("USE_FULL_PF", "1", 1);
    else             setenv("USE_FULL_PF", "0", 1);

    GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid);
    sRNG.SeedFixedIntegers({121,122,123,124,125});
    pRNG.SeedFixedIntegers({221,222,223,224,225});

    // Weak-field init for the stout-smeared production action (same reason
    // as Phase 1 — cold U=I + clover + supercritical-κ → NaN rational PF).
    DTXQCDField U(&Grid);
    DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);  // zeroes aux, sets U=I
    DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, /*wf=*/0.1);
    std::cout << GridLogMessage << "DTXQCD init: weak-field gauge wf=0.1, plaq="
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;

    DTXQCDGaugeActionAdapter<PlaqPlusRectangleAction<PeriodicGimplR>>
        GaugeAction(beta_run, c1);
    GaugeAction.is_smeared = false;  // LW on thin

    DTXQCDAuxiliaryFieldGaussianAction AuxAction(lambda_run);

    RealD rat_lo = 0.001, rat_hi = 100.0;
    int   rat_degree = 12;
    if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
    if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
    if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
    OneFlavourRationalParams rp(rat_lo, rat_hi, /*MaxIter=*/10000, /*tol=*/1e-8,
                                rat_degree, 64, /*BCFreq=*/100, /*mdtol=*/1e-6);

    DTXQCDLogDetCloverEOAction           LogDet(Grid, RBGrid, mass_run, csw_run);
    DTXQCDWilsonCloverRationalFullAction PF_full(Grid, RBGrid, mass_run, rp, csw_run);
    DTXQCDWilsonCloverRationalEOAction   PF_eo  (Grid, RBGrid, mass_run, rp, csw_run);

    typedef Representations<EmptyRep<DTXQCDField>> Reps;
    ActionLevel<DTXQCDField, Reps> L1(1);
    if (use_full_pf) {
      L1.push_back(&PF_full);
    } else {
      L1.push_back(&PF_eo);
      L1.push_back(&LogDet);
    }
    L1.push_back(&AuxAction);
    ActionLevel<DTXQCDField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    ActionSet<DTXQCDField, Reps> Aset;
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

    Smear_Stout<PeriodicGimplR> StoutOp(stout_rho);
    DTXQCDSmearedConfiguration Smear(&Grid, (unsigned int)stout_n, StoutOp);
    Smear.set_Field(U);
    PF_full.is_smeared = true;
    PF_eo.is_smeared   = true;
    LogDet.is_smeared  = true;

    typedef ForceGradient<DTXQCDCompositeImpl,
                          DTXQCDSmearedConfiguration, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);

    std::vector<HmcObservable<DTXQCDField> *> Obs;
    Obs.push_back(&plaq_log);

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

    std::cout << GridLogMessage << "===== " << phase_tag
              << " (use_full_pf=" << (use_full_pf ? "1" : "0") << ") =====" << std::endl;
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();

    if (avg_obs) fierz_out = avg_obs->finalize(pass_tol, phase_tag);
    plaq_log.summary(phase_tag);
  };

  if (!skip_noneo)
    run_dtxqcd_phase(/*use_full_pf=*/true, plaq_noneo, fierz_noneo,
                     "Phase 2 DTXQCD non-EO");
  if (!skip_eo)
    run_dtxqcd_phase(/*use_full_pf=*/false, plaq_eo, fierz_eo,
                     "Phase 3 DTXQCD EO");

  // ============================================================
  // Cross-comparison summary
  // ============================================================
  std::cout << GridLogMessage << std::endl
            << "===== CROSS-COMPARISON SUMMARY (production action) =====" << std::endl
            << "  λ=" << lambda_run << "  m=" << mass_run << "  csw=" << csw_run
            << "  β=" << beta_run << "  u0=" << u0_run
            << "  stout=" << stout_n << "x ρ=" << stout_rho << std::endl;
  auto print_plaq = [&](const std::string &name,
                        const PlaqLogger<LatticeGaugeField> *lf,
                        const PlaqLogger<DTXQCDField> *df) {
    if (lf) std::cout << GridLogMessage << "  " << name
                       << " plaq = " << lf->mean() << " ± " << lf->se()
                       << " (N=" << lf->samples.size() << ")" << std::endl;
    if (df) std::cout << GridLogMessage << "  " << name
                       << " plaq = " << df->mean() << " ± " << df->se()
                       << " (N=" << df->samples.size() << ")" << std::endl;
  };
  if (!skip_qcd)   print_plaq("QCD          ", &plaq_qcd,   nullptr);
  if (!skip_noneo) print_plaq("DTXQCD non-EO", nullptr,     &plaq_noneo);
  if (!skip_eo)    print_plaq("DTXQCD    EO ", nullptr,     &plaq_eo);
  if (!skip_noneo)
    std::cout << GridLogMessage
              << "  Fierz non-EO Σ_DTX/Σ_W = " << fierz_noneo.ratio
              << "  dev=" << fierz_noneo.dev
              << "  pass=" << (fierz_noneo.pass ? "YES" : "NO") << std::endl;
  if (!skip_eo)
    std::cout << GridLogMessage
              << "  Fierz    EO Σ_DTX/Σ_W = " << fierz_eo.ratio
              << "  dev=" << fierz_eo.dev
              << "  pass=" << (fierz_eo.pass ? "YES" : "NO") << std::endl;

  Grid_finalize();
  bool ok = (skip_noneo || fierz_noneo.pass) && (skip_eo || fierz_eo.pass);
  return ok ? 0 : 1;
}
