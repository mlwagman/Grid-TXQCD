// Symanzik Nf=3 gauge generation: a single TXQCD operator with mass =
// diag(m_l, m_l, m_s).  Avoids the conceptual issue of mixing a Nf=2
// TXQCD action with a Nf=1 QCD-wrapped strange in the same HMC, which
// gives kaon-like propagators where light and strange feel different
// auxiliary backgrounds.
//
// Compile with -DTXQCD_Nf=3.

#ifndef TXQCD_Nf
#error "Compile with -DTXQCD_Nf=3"
#endif
static_assert(TXQCD_Nf == 3, "expects TXQCD_Nf=3");

#include "Test_txqcd_2pt_symanzik_nf3_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptSymanzikNf3;

// Env-var overrides so the same binary can do a tiny pipeline test
// (e.g. N_THERM=20 N_PROD=30) and the full symanzik-matched run.
static int env_int(const char *name, int def) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}

// Auto-measure Σ = vev_trminv/2 at given mass on a stout-smeared gauge
// field, matching the production AUX_INIT_AUTO path.  Used to set per-flavor
// Σ_l and Σ_s for the Nf=3 ThermalAuxConfiguration init.
static RealD AutoMeasureSigma(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                              LatticeGaugeField &Usm, RealD m_q,
                              GridParallelRNG &noisePRNG, int n_noise,
                              RealD tol, int maxit) {
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>>
      MeasFermOp;
  MeasFermOp Dw(Usm, Grid, RBGrid, m_q, csw, csw,
                WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<MeasFermOp, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);
  RealD V = (RealD)Grid.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
    gaussian(noisePRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  RealD vev_trminv = acc / n_noise;
  return vev_trminv / 2.0;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // LATT env var override (e.g. LATT="8.8.8.16") for production-volume smoke
  // testing without rebuilding default_latt().  Otherwise honor --grid CLI;
  // fall back to the test's small default if neither is given.
  Coordinate latt;
  if (const char *e = std::getenv("LATT"); e && *e) {
    int Lx, Ly, Lz, Lt;
    if (std::sscanf(e, "%d.%d.%d.%d", &Lx, &Ly, &Lz, &Lt) == 4)
      latt = Coordinate(std::vector<int>{Lx, Ly, Lz, Lt});
    else
      latt = default_latt();
  } else {
    auto cli = GridDefaultLatt();
    if (cli.size() == 4 && cli[0] > 0) latt = cli;
    else                                latt = default_latt();
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  std::cout << GridLogMessage << "Lattice: " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3] << std::endl;
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int n_therm_use = env_int("N_THERM", n_therm);
  int n_prod_use  = env_int("N_PROD",  n_prod);
  int total_traj  = n_therm_use + n_prod_use;

  if (txqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "TXQCD symanzik Nf=3 configs already exist, skipping."
              << std::endl;
    Grid_finalize();
    return 0;
  }

  std::array<RealD, 3> mass_diag = nf3_mass();
  std::cout << GridLogMessage
            << "Generating TXQCD Nf=3 symanzik configs (mass={"
            << mass_diag[0] << "," << mass_diag[1] << "," << mass_diag[2]
            << "}, csw=" << csw
            << ", rho=" << stout_rho << ", Nsmear=" << stout_nsmear
            << ", trajL=0.5, MDsteps=10)..." << std::endl;
  mkdir_p(txqcd_nf3_cfg_dir());

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int start_traj = 0;
  int latest = latest_txqcd_checkpoint();

  RealD cg_tol = 1e-8;
  OneFlavourRationalParams rat_params(1e-4, 200.0, cg_max, cg_tol, 16, 64,
                                      100, 1e-6, 1e-4);

  typedef SymanzikGaugeAction<PeriodicGimplR> SymanzikR;
  GaugeActionAdapter<SymanzikR> GaugeAction(beta, u0);
  GaugeAction.is_smeared = true;

  AuxiliaryFieldGaussianAction AuxAction(lambda_runtime());

  // Single Nf=3 TXQCD action with diag mass = (m_l, m_l, m_s).
  TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass_diag, rat_params,
                                       csw);
  PF.is_smeared = true;
  TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass_diag, csw);
  LogDet.is_smeared = true;

  TXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage << "Resuming from checkpoint at traj "
              << latest << std::endl;
    LoadTxqcdConfig(U, sRNG, pRNG, latest);
    start_traj = latest;
  } else {
    sRNG.SeedFixedIntegers({71, 72, 73, 74, 75});
    pRNG.SeedFixedIntegers({76, 77, 78, 79, 80});

    // Three input modes for per-flavor Σ in priority order:
    //   AUX_INIT_AUTO=1 (default) → measure Σ_l (at m_l) and Σ_s (at m_s) on
    //                                the just-generated weak-field gauge with
    //                                production stout smearing.
    //   SIGMA_L=v SIGMA_S=v       → set explicitly.
    auto envd = [](const char *n, double d) {
      const char *v = std::getenv(n); return (v && *v) ? std::atof(v) : d;
    };
    bool auto_init = true;
    if (std::getenv("SIGMA_L") || std::getenv("SIGMA_S")) auto_init = false;
    if (const char *a = std::getenv("AUX_INIT_AUTO");
        a && *a && std::atoi(a) == 0) auto_init = false;

    double wf_scale = envd("WEAK_FIELD_SCALE", 0.1);
    TXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, wf_scale);

    std::array<RealD, 3> Sigma;
    if (auto_init) {
      // Stout-smear the weak-field gauge with the same params the HMC will
      // use, then measure Σ separately at m_l and m_s.
      Smear_Stout<PeriodicGimplR> StoutMeas(stout_rho);
      SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear,
                                                      StoutMeas);
      SmearMeas.set_Field(U.U);
      LatticeGaugeField Usm = SmearMeas.get_SmearedU();
      // Use a separate noise RNG so the main pRNG state remains identical to
      // a non-AUX_INIT_AUTO run with the same seed (deterministic aux init).
      GridParallelRNG noisePRNG(&Grid);
      noisePRNG.SeedFixedIntegers({81, 82, 83, 84, 85});
      RealD Sigma_l = AutoMeasureSigma(Grid, RBGrid, Usm, mass,
                                       noisePRNG, n_vev_noise, 1e-8, cg_max);
      RealD Sigma_s = AutoMeasureSigma(Grid, RBGrid, Usm, mass_s,
                                       noisePRNG, n_vev_noise, 1e-8, cg_max);
      Sigma = {Sigma_l, Sigma_l, Sigma_s};
      std::cout << GridLogMessage
                << "[AUX_INIT_AUTO] Σ_l=" << Sigma_l
                << " (at m_l=" << mass << ")  Σ_s=" << Sigma_s
                << " (at m_s=" << mass_s << ")" << std::endl;
    } else {
      Sigma = {envd("SIGMA_L", 2.685), envd("SIGMA_L", 2.685),
               envd("SIGMA_S", 2.635)};
      std::cout << GridLogMessage << "[AUX_INIT manual] Σ=("
                << Sigma[0] << "," << Sigma[1] << "," << Sigma[2] << ")"
                << std::endl;
    }
    std::cout << GridLogMessage
              << "Filling aux: <σ_aa>=Σ_a/λ²={"
              << Sigma[0] / (lambda_runtime() * lambda_runtime()) << ","
              << Sigma[1] / (lambda_runtime() * lambda_runtime()) << ","
              << Sigma[2] / (lambda_runtime() * lambda_runtime()) << "}, λ=" << lambda_runtime()
              << std::endl;
    TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_runtime(), Sigma);
  }

  typedef Representations<EmptyRep<TXQCDField>> Reps;
  ActionLevel<TXQCDField, Reps> L1(1);
  L1.push_back(&PF);
  L1.push_back(&LogDet);
  L1.push_back(&AuxAction);
  ActionLevel<TXQCDField, Reps> L2(4);
  L2.push_back(&GaugeAction);
  ActionSet<TXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);

  IntegratorParameters MD;
  MD.name = "ForceGradient";
  MD.MDsteps = 10;
  MD.trajL = 0.5;

  int no_metrop = (start_traj < n_therm_use) ? (n_therm_use - start_traj) : 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = total_traj - no_metrop - start_traj;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  TXQCDSmearedConfiguration Smear(&Grid, stout_nsmear, Stout);

  typedef ForceGradient<TXQCDCompositeImpl,
                        TXQCDSmearedConfiguration, Reps> IntT;
  IntT MDyn(&Grid, MD, Aset, Smear);
  Smear.set_Field(U);

  CheckpointerParameters CPp;
  CPp.config_prefix = txqcd_nf3_cfg_dir() + "/ckpoint_lat";
  CPp.rng_prefix    = txqcd_nf3_cfg_dir() + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(CPp);

  TxqcdSmearedDiagnostics<TXQCDSmearedConfiguration> diag(
      txqcd_nf3_cfg_dir() + "/hmc_diagnostics", meas_skip, {
      {"PseudoFermion", &PF},
      {"LogDet", &LogDet},
      {"AuxGaussian", &AuxAction},
      {"Gauge", &GaugeAction}
  }, Smear, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

  std::vector<HmcObservable<TXQCDField> *> Obs = {&ckpt, &diag};
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();

  std::cout << GridLogMessage
            << "Symanzik Nf=3 gen complete: " << txqcd_nf3_cfg_dir()
            << std::endl;
  Grid_finalize();
  return 0;
}
