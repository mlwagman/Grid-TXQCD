// DTXQCD production HMC driver — Nf=2 Wilson-clover with diquark-tensor
// auxiliary fields, sibling of gen_txqcd_cfgs but with the DTXQCD action
// roster (Pauli-triplet sigma/pi/t + Hermitian color d/n) and 1/4-root
// Pfaffian RHMC pseudofermion.
//
// Action layers:
//   Level 1 (inner): DTXQCDWilsonCloverRationalEOAction + DTXQCDLogDetCloverEOAction
//   Level 2 (gauge multiplier): plaquette gauge action via DTXQCDGaugeActionAdapter
//   Level 3 (aux multiplier):   DTXQCDAuxiliaryFieldGaussianAction
//
// Initial scope is deliberately minimal (no stout smearing, no QUDA, no
// diagnostic observer beyond the checkpointer) -- enough for a smoke run
// confirming dH finite and acceptance plausible.  The aux-correlator
// observers, eigenvalue diagnostics, and stout chain from gen_txqcd_cfgs_2plus1
// can graft on once the bare HMC trajectory is stable.
//
// Env overrides:
//   LATT=L.L.L.T     lattice size (default from params.h)
//   MASS_LIGHT_DTXQCD, LAMBDA_DTXQCD, CSW, BETA     physics
//   MDSTEPS, TRAJL, INTEGRATOR     integrator
//   RHMC_LO, RHMC_HI, RHMC_DEG, CG_TOL, CG_MAX     rational solver
//   N_SKIP           checkpoint save interval (default 10)
//   N_THERM, N_PROD overrides via params.h constexprs (recompile to change)

#include "params.h"
#include <cstdio>   // IMPORT_CFG file-magic sniff
#include <cstring>  // memcmp
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOActionQudaPrimitive.h>
#endif
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSmearedConfiguration.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>
// Strange quark (Nf=1, ADD_STRANGE=1): plain Wilson-clover QCD action wrapped
// into the composite field via DTXQCDQCDActionAdapter, exactly as TXQCD's
// Nf=2+1 generator.  The chroma reference plaq is Nf=2+1, so a fair plaq/VEV
// comparison REQUIRES the spectator strange.
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalAction.h>
#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionMP.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include "dtxqcd_diag.h"   // signed-Pfaffian + aux diagnostics observer

using namespace TXQCDProduction;

// Fresh-start DTXQCD field initialisation, ported from
// Test_dtxqcd_2pt_gencfgs.cc (csw-parametrized for production):
//   - weak-field gauge (GenerateWeakFieldGauge, wf=0.1),
//   - aux drawn at AUX_FLUCT_LAMBDA width (default 10) + a saddle shift Σ,
//   - AUX_INIT=value  → Σ fixed explicitly,
//   - AUX_INIT_AUTO=1 → self-consistent saddle by bisection on
//                       g(Σ)=Σ−Σ_DTXQCD(Σ) (avoids the slow singlet
//                       equilibration / cold-aux=0 LogDet divergence),
//   - ZERO_DN_INIT / ZERO_ALL_AUX post-init overrides.
static void InitFreshDtxqcdField(Grid::GridCartesian &Grid_,
                                 Grid::GridRedBlackCartesian &RBGrid_,
                                 Grid::DTXQCDField &U,
                                 Grid::GridSerialRNG &sRNG,
                                 Grid::GridParallelRNG &pRNG,
                                 Grid::RealD mass, Grid::RealD csw,
                                 Grid::RealD lambda, int cg_max,
                                 bool gauge_preloaded = false) {
  using namespace Grid;
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  // Fluctuation width for FillAuxFields, decoupled from the physical lambda
  // (FLUCT=10 keeps the doubled M well-conditioned on the cold gauge).
  if (std::getenv("AUX_FLUCT_LAMBDA") == nullptr)
    setenv("AUX_FLUCT_LAMBDA", "10.0", 0);

  RealD Sigma_init = 0.0;
  bool sigma_auto = false;
  if (const char *si = std::getenv("AUX_INIT"); si && *si) {
    Sigma_init = std::atof(si);
  } else if (const char *sa = std::getenv("AUX_INIT_AUTO");
             sa && std::atoi(sa) != 0) {
    sigma_auto = true;
  }

  // Step 1: gauge.  Fresh start draws a weak-field gauge; an imported start
  // (IMPORT_CFG) already has U.U loaded -> skip and measure Sigma on it.
  if (!gauge_preloaded) {
    DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, /*wf=*/0.1);
  } else {
    std::cout << GridLogMessage
              << "[InitFreshDtxqcdField] using preloaded (imported) gauge"
              << std::endl;
  }

  if (sigma_auto) {
    // Bare-Σ seed: Hutchinson Tr M^{-1} on plain Wilson (csw=0, periodic) —
    // only the bisection upper bracket; the operator below uses the real csw.
    WilsonImplParams impl_p;
    typedef WilsonFermion<WilsonImplR> MeasFermOp;
    MeasFermOp Dw(U.U, Grid_, RBGrid_, mass, impl_p);
    MdagMLinearOperator<MeasFermOp, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    const RealD V = (RealD)Grid_.gSites();
    GridParallelRNG noisePRNG(&Grid_);
    noisePRNG.SeedFixedIntegers({110, 120, 130, 140, 150});
    const int n_noise = 8;
    RealD acc = 0.0;
    for (int h = 0; h < n_noise; ++h) {
      LatticeFermion eta(&Grid_), b(&Grid_), x(&Grid_);
      gaussian(noisePRNG, eta);
      Dw.Mdag(eta, b);
      x = Zero();
      CG(HermOp, b, x);
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    Sigma_init = acc / n_noise;
    std::cout << GridLogMessage << "[AUX_INIT_AUTO] Sigma_bare = vev_trminv = "
              << Sigma_init << std::endl;
  } else if (Sigma_init != 0.0) {
    std::cout << GridLogMessage << "[AUX_INIT] Sigma = " << Sigma_init << std::endl;
  }

  // Step 2: fill aux with Gaussian + saddle shift.
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_init);

  // Step 2b (AUX_INIT_AUTO): bisect g(Σ)=Σ−Σ_DTXQCD(Σ) on the FULL doubled op.
  if (sigma_auto) {
    auto measure_sigma_dtxqcd = [&](RealD Sigma_at) -> RealD {
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_at);
      DTXQCDWilsonCloverFermionEO Dw_dtxqcd(U.U, Grid_, RBGrid_, mass, csw,
                                            U.sigma, U.pi, U.d, U.n, U.s, U.p);
      const RealD V = (RealD)Grid_.gSites();
      GridParallelRNG noisePRNG(&Grid_);
      noisePRNG.SeedFixedIntegers({400, 410, 420, 430, 440});
      const int n_noise_iter = 4;
      RealD acc = 0.0;
      for (int h = 0; h < n_noise_iter; ++h) {
        DTXQCDFermionDoubled eta(&Grid_), b(&Grid_), x(&Grid_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(noisePRNG, eta.upper.f[a]);
          gaussian(noisePRNG, eta.lower.f[a]);
        }
        Dw_dtxqcd.Mdag(eta, b);
        DTXQCDFermionDoubled r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
        x = Zero();
        for (int a = 0; a < DtxqcdNf; ++a) {
          r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
          p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
        }
        RealD r2 = norm2(r), b2 = norm2(b);
        RealD cg_tol2 = 1e-8 * b2;  // Sigma estimate to ~1e-4, ample for AUX_INIT_TOL
        for (int k = 0; k < cg_max; ++k) {
          Dw_dtxqcd.M(p, tmp);
          Dw_dtxqcd.Mdag(tmp, Ap);
          RealD pAp = innerProduct(p, Ap).real();
          RealD alpha = r2 / pAp;
          for (int a = 0; a < DtxqcdNf; ++a) {
            x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
            x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
            r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
            r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
          }
          RealD r2_new = norm2(r);
          if (r2_new < cg_tol2) break;
          RealD beta = r2_new / r2;
          for (int a = 0; a < DtxqcdNf; ++a) {
            p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
            p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
          }
          r2 = r2_new;
        }
        acc += innerProduct(eta, x).real() / (2.0 * V);
      }
      return (acc / n_noise_iter) / (2.0 * DtxqcdNf);
    };

    int aux_iter_max = 15;
    if (const char *m = std::getenv("AUX_INIT_MAX_ITER"); m && *m)
      aux_iter_max = std::atoi(m);
    RealD aux_iter_tol = 0.1;  // 10% is plenty for an HMC init -- the trajectory
                               // equilibrates the residual; tighten via AUX_INIT_TOL
    if (const char *t = std::getenv("AUX_INIT_TOL"); t && *t)
      aux_iter_tol = std::atof(t);
    bool aux_done = false;
    // --- Picard fixed-point first (cheap): Sigma_{n+1} = Sigma_DTXQCD(Sigma_n)
    //     from Sigma_0 = 0.  At large lambda the map Jacobian ~ N_F/lambda^2
    //     << 1, so it contracts to the saddle in 1-2 evaluations.  Bail to
    //     bisection if it is NOT contracting (small lambda, Jacobian > 1, where
    //     Picard oscillates/diverges -- the regime the bisection guards).
    const int picard_max =
        TXQCDProduction::detail::env_int("AUX_INIT_PICARD_MAX", 4);
    if (TXQCDProduction::detail::env_int("AUX_INIT_PICARD", 1) != 0 &&
        picard_max > 0) {
      RealD Scur = measure_sigma_dtxqcd(0.0);  // mean-field condensate (aux=0)
      std::cout << GridLogMessage << "[AUX_INIT Picard 0] Sigma_DTXQCD(0) = "
                << Scur << "  (mean-field)" << std::endl;
      RealD prev_delta = 1e300;
      for (int it = 1; it <= picard_max; ++it) {
        RealD Snew  = measure_sigma_dtxqcd(Scur);
        RealD delta = std::abs(Snew - Scur);
        RealD rel   = delta / std::max(std::abs(Snew), 1e-30);
        std::cout << GridLogMessage << "[AUX_INIT Picard " << it << "] Sigma = "
                  << Snew << "  rel_change = " << rel << std::endl;
        if (rel < aux_iter_tol) { Sigma_init = Snew; aux_done = true; break; }
        if (delta >= prev_delta) {  // not contracting -> small-lambda regime
          std::cout << GridLogMessage
                    << "[AUX_INIT Picard] not contracting -> bisection fallback"
                    << std::endl;
          break;
        }
        prev_delta = delta;
        Scur = Snew;
      }
    }
    // --- Bisection fallback (robust at small lambda) ---
    if (!aux_done) {
      RealD Sigma_lo = 0.0;
      RealD g_lo = Sigma_lo - measure_sigma_dtxqcd(Sigma_lo);
      RealD Sigma_hi = (Sigma_init > 1e-12) ? Sigma_init : 1.0;
      RealD g_hi = Sigma_hi - measure_sigma_dtxqcd(Sigma_hi);
      // The saddle can sit just ABOVE Sigma_bare; grow the upper bound until g
      // changes sign.
      for (int gi = 0; gi < 6 && g_lo * g_hi > 0.0; ++gi) {
        Sigma_hi *= 1.8;
        g_hi = Sigma_hi - measure_sigma_dtxqcd(Sigma_hi);
      }
      std::cout << GridLogMessage << "[AUX_INIT bracket] g(0)=" << g_lo
                << "  g(" << Sigma_hi << ")=" << g_hi << std::endl;
      if (g_lo * g_hi > 0.0) {
        Sigma_init = (std::abs(g_hi) <= std::abs(g_lo)) ? Sigma_hi : Sigma_lo;
        std::cout << GridLogMessage
                  << "[AUX_INIT_AUTO] no sign change after growth — nearest "
                     "endpoint Sigma = " << Sigma_init << std::endl;
      } else {
        for (int it = 0; it < aux_iter_max; ++it) {
          RealD Sigma_mid = 0.5 * (Sigma_lo + Sigma_hi);
          RealD g_mid = Sigma_mid - measure_sigma_dtxqcd(Sigma_mid);
          std::cout << GridLogMessage << "[AUX_INIT iter " << it << "] Sigma_mid="
                    << Sigma_mid << "  g=" << g_mid << std::endl;
          if (g_mid * g_lo < 0.0) { Sigma_hi = Sigma_mid; g_hi = g_mid; }
          else { Sigma_lo = Sigma_mid; g_lo = g_mid; }
          if ((Sigma_hi - Sigma_lo) /
                  std::max(0.5 * (Sigma_hi + Sigma_lo), 1e-30) < aux_iter_tol)
            break;
        }
        Sigma_init = 0.5 * (Sigma_lo + Sigma_hi);
      }
    }
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_init);
    std::cout << GridLogMessage << "[AUX_INIT_AUTO converged] Sigma* = "
              << Sigma_init << "  -> <s>* = 2 N_F Sigma*/lambda^2 = "
              << (2.0 * DtxqcdNf * Sigma_init / (lambda * lambda)) << std::endl;
  }

  if (const char *z = std::getenv("ZERO_DN_INIT"); z && std::atoi(z) != 0) {
    std::cout << GridLogMessage << "Zeroing d, n diquark fields at init" << std::endl;
    U.d = Zero(); U.n = Zero();
  }
  if (const char *z = std::getenv("ZERO_ALL_AUX"); z && std::atoi(z) != 0) {
    std::cout << GridLogMessage << "Zeroing ALL aux fields at init (cold)" << std::endl;
    U.sigma = Zero(); U.pi = Zero(); U.d = Zero();
    U.n = Zero(); U.s = Zero(); U.p = Zero();
  }
}

// Build the chosen integrator + HMC and evolve.  Templated on the integrator
// (ForceGradient | MinimumNorm2) and the smearing policy (NoSmearing |
// DTXQCDSmearedConfiguration) so the 2x2 runtime selection in main() shares one
// body.
template <template <class, class, class> class IntegratorT, class SmearPolicy>
static void RunDtxqcdHMC(
    Grid::GridCartesian &Grid_, Grid::IntegratorParameters &MD,
    Grid::ActionSet<Grid::DTXQCDField,
                    Grid::Representations<Grid::EmptyRep<Grid::DTXQCDField>>> &Aset,
    SmearPolicy &Smear, Grid::HMCparameters &HMCp, Grid::GridSerialRNG &sRNG,
    Grid::GridParallelRNG &pRNG,
    std::vector<Grid::HmcObservable<Grid::DTXQCDField> *> &Obs,
    Grid::DTXQCDField &U) {
  using namespace Grid;
  typedef IntegratorT<DTXQCDCompositeImpl, SmearPolicy,
                      Representations<EmptyRep<DTXQCDField>>> IntT;
  IntT MDyn(&Grid_, MD, Aset, Smear);
  HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
  HMC.evolve();
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: "
            << GridThread::GetThreads() << std::endl;

  // ---- geometry ----
  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  // RNG seeding deferred: DTXQCDCheckpointer::ReadConfig restores it on resume;
  // InitFreshDtxqcdField seeds it on a fresh start.

  // ---- physics parameters (env-overridable via params.h knobs) ----
  const RealD mass  = mass_light_dtxqcd;
  const RealD csw_  = csw;
  const RealD beta_ = beta;
  const RealD lam   = lambda_dtxqcd;
  const RealD tol   = TXQCDProduction::detail::env_real("CG_TOL", 1e-10);
  const int   cgmax = TXQCDProduction::detail::env_int ("CG_MAX", 10000);
  std::cout << GridLogMessage
            << "DTXQCD physics: mass=" << mass << " csw=" << csw_
            << " beta=" << beta_ << " lambda=" << lam
            << " cg_tol=" << tol << " cg_max=" << cgmax << std::endl;

  // ---- rational params for the RHMC pseudofermion ----
  OneFlavourRationalParams rp(
      /*lo=*/         TXQCDProduction::detail::env_real("RHMC_LO",   1.0e-1),
      /*hi=*/         TXQCDProduction::detail::env_real("RHMC_HI",   64.0),
      /*MaxIter=*/    cgmax,
      /*tolerance=*/  tol,
      /*degree=*/     TXQCDProduction::detail::env_int ("RHMC_DEG",  12),
      /*precision=*/  50,
      /*BoundsCheckFreq=*/ 0,
      /*mdtolerance=*/ tol);
  std::cout << GridLogMessage
            << "RHMC rational: lo=" << rp.lo << " hi=" << rp.hi
            << " degree=" << rp.degree << " tol=" << rp.tolerance << std::endl;

  // ---- checkpoint directory + resume scan ----
  // TRAJ is a TARGET total (not an increment): a resubmit with TRAJ <= latest
  // exits immediately.  Sidecar suffix is "_daux" (DTX3), per DTXQCDCheckpointer.
  const std::string cfg_dir = dtxqcd_cfg_dir();
  mkdir_p(cfg_dir);
  const int total_traj =
      TXQCDProduction::detail::env_int("TRAJ", n_therm + n_prod);
  int start_traj = 0, latest = -1;
  for (int t = meas_skip; t <= total_traj; t += meas_skip) {
    if (file_exists(cfg_dir + "/ckpoint_lat."      + std::to_string(t)) &&
        file_exists(cfg_dir + "/ckpoint_lat_daux." + std::to_string(t)) &&
        file_exists(cfg_dir + "/ckpoint_rng."      + std::to_string(t)))
      latest = t;
  }

  // ---- composite field (gauge + aux): resume or fresh start ----
  DTXQCDField U(&Grid);
  if (latest > 0) {
    std::cout << GridLogMessage
              << "Resuming DTXQCD from checkpoint at traj " << latest << std::endl;
    DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                   cfg_dir + "/ckpoint_lat",
                                   cfg_dir + "/ckpoint_rng", latest);
    start_traj = latest;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    // Fork from an external thermalized gauge config (chroma LIME or NERSC):
    // load U.U, then run the aux saddle init (AUX_INIT_AUTO) on the imported
    // gauge instead of a weak field.  ILDG (.lime) needs HAVE_LIME at build.
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic
              << " (fork from external gauge)" << std::endl;
    std::FILE *fp = std::fopen(ic, "rb");
    char magic[16] = {0};
    if (fp) { (void)std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(U.U, header, std::string(ic));
    } else {
#ifdef HAVE_LIME
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(U.U, header);
      IR.close();
#else
      std::cerr << "IMPORT_CFG=" << ic << ": non-NERSC (ILDG/.lime) gauge "
                   "requires a LIME-enabled build (--with-lime)." << std::endl;
      std::abort();
#endif
    }
    std::cout << GridLogMessage << "Imported gauge plaquette = "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;
    InitFreshDtxqcdField(Grid, RBGrid, U, sRNG, pRNG, mass, csw_, lam, cgmax,
                         /*gauge_preloaded=*/true);
  } else {
    std::cout << GridLogMessage
              << "Fresh DTXQCD start (weak-field gauge + aux saddle init)"
              << std::endl;
    InitFreshDtxqcdField(Grid, RBGrid, U, sRNG, pRNG, mass, csw_, lam, cgmax);
  }

  // ---- actions ----
  DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_);
  DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lam);
  DTXQCDLogDetCloverEOAction                   LogDet(Grid, RBGrid, mass, csw_);

  // USE_FULL_PF=1 swaps the EO Schur 1/4-root + LogDet pair for a SINGLE non-EO
  // 1/4-root pseudofermion on the full doubled M^dag M.  The full operator
  // absorbs the LogDet, so no LogDet companion is pushed in this branch (the EO
  // Schur Mpc diverges below aux_std ~ 0.5, i.e. the low-lambda regime; the full
  // M stays mild).  Mirrors Test_dtxqcd_2pt_gencfgs.cc.  Default = EO Schur.
  bool use_full_pf = false;
  if (const char *u = std::getenv("USE_FULL_PF"); u && *u)
    use_full_pf = (std::atoi(u) != 0);
  std::cout << GridLogMessage
            << "DTXQCD pseudofermion = "
            << (use_full_pf ? "FULL (non-EO, LogDet folded in)" : "EO Schur + LogDet")
            << std::endl;

  // DTXQCD_QUDA_HYBRID=1 swaps in the QUDA Wilson-hopping hybrid action (the
  // dominant force-assembly cost moves to QUDA); default = pure-Grid rational.
  // (QUDA hybrid is EO-only; ignored when USE_FULL_PF=1.)
  std::unique_ptr<DTXQCDWilsonCloverRationalEOAction>   PF_grid_holder;
  std::unique_ptr<DTXQCDWilsonCloverRationalFullAction> PF_full_holder;
  Action<DTXQCDField> *PFActionPtr = nullptr;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<DTXQCDWilsonCloverRationalEOActionQudaPrimitive> PF_quda_holder;
  if (!use_full_pf) {
    if (const char *e = std::getenv("DTXQCD_QUDA_HYBRID"); e && std::atoi(e) != 0) {
      PF_quda_holder =
          std::make_unique<DTXQCDWilsonCloverRationalEOActionQudaPrimitive>(
              Grid, RBGrid, mass, rp, csw_);
      PFActionPtr = PF_quda_holder.get();
      std::cout << GridLogMessage
                << "[DTXQCD] PF action: QUDA Wilson-hopping hybrid "
                   "(DTXQCD_QUDA_HYBRID=1)" << std::endl;
    }
  }
#endif
  if (!PFActionPtr && use_full_pf) {
    PF_full_holder = std::make_unique<DTXQCDWilsonCloverRationalFullAction>(
        Grid, RBGrid, mass, rp, csw_);
    PFActionPtr = PF_full_holder.get();
  }
  if (!PFActionPtr) {
    PF_grid_holder = std::make_unique<DTXQCDWilsonCloverRationalEOAction>(
        Grid, RBGrid, mass, rp, csw_);
    PFActionPtr = PF_grid_holder.get();
  }
  Action<DTXQCDField> &PFAction = *PFActionPtr;

  // ---- optional spectator strange quark (Nf=2+1), gated by ADD_STRANGE=1 ----
  // DTXQCD's doubled operator is the Nf=2 LIGHT sector; the chroma reference
  // plaquette/VEV is Nf=2+1, so a fair comparison REQUIRES a plain-QCD Nf=1
  // strange — exactly as gen_txqcd_cfgs_2plus1.cc.  EO factorization:
  // det(M)=det(Mee)·det(Mpc); QCDLogDetCloverEOAction(nf=1) handles det(Mee),
  // OneFlavourSchurCloverRationalAction handles det(Mpc†Mpc)^{1/2}.  Both run on
  // a Wilson-clover op at mass_strange/csw with APBC time, wrapped into the
  // composite field via DTXQCDQCDActionAdapter (aux force slots zeroed -- the
  // QCD strange has no σ,π,d,n,s,p dependence) and smeared with the gauge.
  // Mixed precision: full-DP action + single-precision sibling op for the MD
  // force multishift solve (~2x faster than full-DP on the strange force eval,
  // which is the dominant per-traj cost after the light deriv); refresh and S
  // stay full-DP.  Mirrors gen_txqcd_cfgs_2plus1.cc; identical physics to DP.
  const bool add_strange =
      TXQCDProduction::detail::env_int("ADD_STRANGE", 0) != 0;
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCFstrange;
  typedef WilsonCloverFermion<WilsonImplF, CloverHelpers<WilsonImplF>> WCFstrangeF;
  WilsonImplParams strange_impl_p;
  strange_impl_p.boundary_phases.resize(Nd, 1.0);
  strange_impl_p.boundary_phases[Nd - 1] = -1.0;   // APBC time (chroma)
  OneFlavourRationalParams strange_rat(1.0e-4, 100.0, cgmax, tol, 20, 64,
                                       100, 1e-6, 1e-4);
  std::unique_ptr<GridCartesian>          StrangeGridF;
  std::unique_ptr<GridRedBlackCartesian>  StrangeRBGridF;
  std::unique_ptr<LatticeGaugeFieldF>     StrangeUmuF;
  std::unique_ptr<WCFstrange>             StrangeFermOp;
  std::unique_ptr<WCFstrangeF>            StrangeFermOpF;
  std::unique_ptr<QCDLogDetCloverEOAction<WilsonImplR>>             StrangeLogDet;
  std::unique_ptr<OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>
      StrangeSchur;
  std::unique_ptr<DTXQCDQCDActionAdapter>                           StrangeLogDetAd;
  std::unique_ptr<DTXQCDQCDActionAdapter>                           StrangeSchurAd;
  if (add_strange) {
    // single-precision sibling grids + gauge (the MP action re-imports the
    // smeared gauge to BOTH precisions each deriv, so this initial fill is just
    // for construction).
    StrangeGridF   = std::make_unique<GridCartesian>(
        latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
    StrangeRBGridF = std::make_unique<GridRedBlackCartesian>(StrangeGridF.get());
    StrangeUmuF    = std::make_unique<LatticeGaugeFieldF>(StrangeGridF.get());
    { LatticeColourMatrix  U_d(&Grid);
      LatticeColourMatrixF U_f(StrangeGridF.get());
      for (int mu = 0; mu < Nd; ++mu) {
        U_d = PeekIndex<LorentzIndex>(U.U, mu);
        precisionChange(U_f, U_d);
        PokeIndex<LorentzIndex>(*StrangeUmuF, U_f, mu);
      } }
    StrangeFermOp  = std::make_unique<WCFstrange>(
        U.U, Grid, RBGrid, mass_strange, csw_, csw_,
        WilsonAnisotropyCoefficients(), strange_impl_p);
    StrangeFermOpF = std::make_unique<WCFstrangeF>(
        *StrangeUmuF, *StrangeGridF, *StrangeRBGridF, mass_strange, csw_, csw_,
        WilsonAnisotropyCoefficients(), strange_impl_p);
    StrangeLogDet  = std::make_unique<QCDLogDetCloverEOAction<WilsonImplR>>(
        *StrangeFermOp, /*nf=*/1);
    StrangeSchur   = std::make_unique<
        OneFlavourSchurCloverRationalActionMP<WilsonImplR, WilsonImplF>>(
        *StrangeFermOp, *StrangeFermOpF, StrangeRBGridF.get(), strange_rat, 50);
    StrangeLogDetAd = std::make_unique<DTXQCDQCDActionAdapter>(*StrangeLogDet);
    StrangeSchurAd  = std::make_unique<DTXQCDQCDActionAdapter>(*StrangeSchur);
    std::cout << GridLogMessage
              << "[DTXQCD] Nf=2+1: spectator strange ADDED (MP double+single, "
                 "mass_strange=" << mass_strange << " csw=" << csw_ << " rat["
              << strange_rat.lo << "," << strange_rat.hi << "] deg="
              << strange_rat.degree << ")" << std::endl;
  } else {
    std::cout << GridLogMessage
              << "[DTXQCD] Nf=2 (no strange; set ADD_STRANGE=1 for Nf=2+1 to "
                 "match the chroma reference)" << std::endl;
  }

  // ---- action levels.  Level 1 (innermost) bundles the fermion-bilinear
  //      monomial(s) that need the finest dt; gauge gets a 2x multiplier and
  //      aux a 4x multiplier mirroring the TXQCD production hierarchy.  In the
  //      FULL branch L1 holds only the rational PF (no separate LogDet).
  typedef Representations<EmptyRep<DTXQCDField>> Reps;
  ActionLevel<DTXQCDField, Reps> L1(1);
  L1.push_back(&PFAction);
  if (!use_full_pf) L1.push_back(&LogDet);
  if (add_strange) {
    L1.push_back(StrangeLogDetAd.get());   // QCD det(Mee) -- no aux dependence
    L1.push_back(StrangeSchurAd.get());    // QCD det(Mpc) -- no aux dependence
  }
  ActionLevel<DTXQCDField, Reps> L2(TXQCDProduction::detail::env_int("GAUGE_MULT", 2));
  L2.push_back(&GaugeAction);
  ActionLevel<DTXQCDField, Reps> L3(TXQCDProduction::detail::env_int("AUX_MULT",  4));
  L3.push_back(&AuxAction);
  ActionSet<DTXQCDField, Reps> Aset;
  Aset.push_back(L1);
  Aset.push_back(L2);
  Aset.push_back(L3);

  // ---- integrator ----
  IntegratorParameters MD;
  MD.name    = "MinimumNorm2";  // canonical for 16^3x48 (MDSTEPS=10 TRAJL=sqrt2/4)
  if (const char *env = std::getenv("INTEGRATOR"); env && *env) MD.name = env;
  MD.MDsteps = 20;
  if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) MD.MDsteps = std::atoi(ms);
  MD.trajL   = std::sqrt(2.0);
  if (const char *tl = std::getenv("TRAJL"); tl && *tl) MD.trajL = std::atof(tl);
  std::cout << GridLogMessage
            << "INTEGRATOR=" << MD.name << " MDsteps=" << MD.MDsteps
            << " trajL=" << MD.trajL << std::endl;

  // ---- HMC parameters ----
  // NoMetropolis force-accepts the thermalization trajectories (none left once
  // start_traj >= n_therm).  Trajectories run = target total - no_metrop warmup
  // - already-done, guarded non-negative (mirrors Test_dtxqcd_2pt_gencfgs).
  int no_metrop = (start_traj < n_therm) ? (n_therm - start_traj) : 0;
  if (const char *nm = std::getenv("NO_METROP"); nm && *nm)
    no_metrop = std::atoi(nm);
  int traj_to_run = total_traj - no_metrop - start_traj;
  if (traj_to_run < 0) traj_to_run = 0;
  HMCparameters HMCp;
  HMCp.StartTrajectory     = start_traj;
  HMCp.Trajectories        = traj_to_run;
  HMCp.NoMetropolisUntil   = no_metrop;
  HMCp.MetropolisTest      = true;
  HMCp.PerformRandomShift  = false;
  HMCp.StartingType        = "ColdStart";
  HMCp.MD = MD;

  // ---- checkpointing ---- (cfg_dir defined above for the resume scan)
  CheckpointerParameters CPp;
  CPp.config_prefix = cfg_dir + "/ckpoint_lat";
  CPp.rng_prefix    = cfg_dir + "/ckpoint_rng";
  CPp.saveInterval  = meas_skip;
  CPp.format        = "IEEE64BIG";
  DTXQCDCheckpointer ckpt(CPp);

  // ---- signed-Pfaffian + aux diagnostics observer (the sign-reweighting
  //      observable: per-traj gamma5.M48 low eigenvalues -> n_neg -> signPf,
  //      plus aux VEVs / Tr M^-1 / per-action Fdt / aux correlators).  Writes
  //      hmc_diagnostics.<traj>.h5 consumed by analyze_sign_reweighting.py.
  const int n_vev_noise = TXQCDProduction::detail::env_int("VEV_NOISE", 4);
  // Force/Fdt-norm reporting list: include LogDet only when it is an active
  // monomial (EO branch).  The signPf observable is independent of this list.
  std::vector<Action<DTXQCDField> *> diag_actions =
      use_full_pf ? std::vector<Action<DTXQCDField> *>{PFActionPtr, &AuxAction, &GaugeAction}
                  : std::vector<Action<DTXQCDField> *>{PFActionPtr, &LogDet, &AuxAction, &GaugeAction};
  DtxqcdDiagnostics diag(cfg_dir + "/hmc_diagnostics", meas_skip,
                         diag_actions,
                         Grid, RBGrid, pRNG, mass, csw_, lam, n_vev_noise);
  std::vector<HmcObservable<DTXQCDField> *> Obs = {&ckpt, &diag};

  // ---- smearing + integrator dispatch (2x2 runtime selection).
  //   STOUT_NSMEAR>0 (params.h default 1): stout-chained
  //   DTXQCDSmearedConfiguration -- gauge through stout, aux pass-through, with
  //   is_smeared on the gauge/fermion/LogDet actions (required for the
  //   chroma-matched csw=1.249 point).  STOUT_NSMEAR=0: unsmeared (csw=0 scout).
  //   INTEGRATOR picks ForceGradient vs MinimumNorm2 (both compiled).
  const int n_stout = stout_nsmear_inv;
  std::cout << GridLogMessage << "Smearing: STOUT_NSMEAR=" << n_stout
            << "  STOUT_RHO=" << stout_rho_inv << std::endl;
  if (n_stout > 0) {
    Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
    DTXQCDSmearedConfiguration Smear(&Grid, (unsigned int)n_stout, Stout);
    Smear.set_Field(U);
    PFAction.is_smeared    = true;
    if (!use_full_pf) LogDet.is_smeared = true;  // LogDet inactive in FULL branch
    GaugeAction.is_smeared = true;
    if (add_strange) {                            // strange acts on smeared gauge
      StrangeLogDetAd->is_smeared = true;
      StrangeSchurAd->is_smeared  = true;
    }
    if (MD.name == "ForceGradient")
      RunDtxqcdHMC<ForceGradient>(Grid, MD, Aset, Smear, HMCp, sRNG, pRNG, Obs, U);
    else
      RunDtxqcdHMC<MinimumNorm2>(Grid, MD, Aset, Smear, HMCp, sRNG, pRNG, Obs, U);
  } else {
    NoSmearing<DTXQCDCompositeImpl> Smear;
    Smear.set_Field(U);
    if (MD.name == "ForceGradient")
      RunDtxqcdHMC<ForceGradient>(Grid, MD, Aset, Smear, HMCp, sRNG, pRNG, Obs, U);
    else
      RunDtxqcdHMC<MinimumNorm2>(Grid, MD, Aset, Smear, HMCp, sRNG, pRNG, Obs, U);
  }

  std::cout << GridLogMessage
            << "DTXQCD gauge generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
