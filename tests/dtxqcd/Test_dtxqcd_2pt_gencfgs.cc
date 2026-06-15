// Step 1 of the DTXQCD 2pt Fierz-equivalence suite: generate DTXQCD
// configurations on 4^3 x 8.  Mirror of Test_txqcd_2pt_gencfgs.cc with the
// action roster updated to the diquark-tensor variant.  The QCD reference
// ensemble (configs_2pt_qcd_nf2) is shared with the TXQCD test, so it is
// NOT regenerated here -- run Test_txqcd_2pt_gencfgs first (or after) to
// produce the QCD half.
//
// Action structure (matches TXQCD's Wilson Fierz test):
//   L1 (inner, dt fine):   DTXQCDWilsonCloverRationalEOAction (csw=0 = Wilson)
//                        + DTXQCDLogDetCloverEOAction (csw=0)
//                        + DTXQCDAuxiliaryFieldGaussianAction
//   L2 (outer, mult=4):    Wilson plaquette via DTXQCDGaugeActionAdapter
//
// Integrator: ForceGradient, MDsteps=10, trajL=0.5, eps=0.05.  This is the
// same setting the TXQCD Wilson Fierz test runs at and a reasonable
// starting point for DTXQCD -- the per-pole force magnitudes in the
// 1/4-root RHMC differ from TXQCD's 1/2-root only at the residue
// distribution level, and the doubled fermion's per-site Mooee block is
// what the cached EO operator already handles.  Tune from here if dH
// drifts.

#include "Test_dtxqcd_2pt_utils.h"
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/gauge/WilsonGaugeAction.h>

using namespace DtxqcdTest2pt;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  // Env knob N_PROD overrides the production-trajectory count (n_prod);
  // similarly N_THERM overrides n_therm.  Lets us launch long-running streams
  // (e.g. 2000 configs) without recompiling.
  int n_therm_run = n_therm;
  int n_prod_run  = n_prod;
  if (const char *t = std::getenv("N_THERM"); t && *t) n_therm_run = std::atoi(t);
  if (const char *p = std::getenv("N_PROD");  p && *p) n_prod_run  = std::atoi(p);
  int total_traj = n_therm_run + n_prod_run;

  // ==================== DTXQCD ====================
  // Skip-existing-cfgs check uses the COMPILED defaults of n_therm/n_prod and
  // misses our intent to extend. Bypass it when N_PROD env var is set so a
  // launch with N_PROD=1980 can resume from existing cfgs and run further.
  bool force_continue = std::getenv("N_PROD") != nullptr;
  if (!force_continue && dtxqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "DTXQCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating DTXQCD configs (" << total_traj
              << " trajectories)..." << std::endl;
    mkdir_p(dtxqcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_dtxqcd_checkpoint();

    // Rational bracket tuned to Test_dtxqcd_spectrum measurements on
    // 4^3 x 8, mass = 0.3, csw = 0, weak gauge + thermal aux at
    // AUX_FLUCT_LAMBDA = 10 (see DTXQCDCompositeImpl::FillAuxFields):
    //
    //   lambda_min(Mpc^dag Mpc) ~ 0.11
    //   lambda_max(Mpc^dag Mpc) ~ 37
    //
    // bracket lo = 0.05 (well below lambda_min) and hi = 80 (1.5x
    // measured lambda_max + headroom for HMC drift) keeps the shifted
    // multi-shift CG well-conditioned.  AUX_FLUCT_LAMBDA = 10 is exported
    // below before the cold-start init.
    RealD cg_tol = 1e-8;
    // RAT_LO / RAT_HI / RAT_DEGREE env knobs let us tighten the rational
    // bracket to reduce 4th-order ForceGradient remainder.  Defaults match
    // the original EO-tuned values (broad bracket, high degree).
    RealD rat_lo     = 0.05;
    RealD rat_hi     = 80.0;
    int   rat_degree = 12;
    if (const char *v = std::getenv("RAT_LO");     v && *v) rat_lo     = std::atof(v);
    if (const char *v = std::getenv("RAT_HI");     v && *v) rat_hi     = std::atof(v);
    if (const char *v = std::getenv("RAT_DEGREE"); v && *v) rat_degree = std::atoi(v);
    std::cout << GridLogMessage << "DTXQCD rational: lo=" << rat_lo
              << " hi=" << rat_hi << " degree=" << rat_degree << std::endl;
    OneFlavourRationalParams rat_params(
        /*lo=*/rat_lo, /*hi=*/rat_hi,
        /*MaxIter=*/cg_max, /*tolerance=*/cg_tol,
        /*degree=*/rat_degree, /*precision=*/64,
        /*BoundsCheckFreq=*/100,
        /*mdtolerance=*/1e-6);

    RealD lambda_run = lambda;
    if (const char *l = std::getenv("LAMBDA"); l && *l) {
      lambda_run = std::atof(l);
    }
    std::cout << GridLogMessage << "DTXQCD lambda = " << lambda_run << std::endl;
    RealD mass_run = mass;
    if (const char *m = std::getenv("MASS"); m && *m) {
      mass_run = std::atof(m);
    }
    std::cout << GridLogMessage << "DTXQCD mass = " << mass_run << std::endl;
    RealD beta_run = beta;
    if (const char *b = std::getenv("BETA"); b && *b) {
      beta_run = std::atof(b);
    }
    std::cout << GridLogMessage << "DTXQCD beta = " << beta_run << std::endl;
    DTXQCDGaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta_run);
    DTXQCDAuxiliaryFieldGaussianAction           AuxAction(lambda_run);
    // csw=0 => Wilson (no clover term).  The rational/LogDet code paths
    // skip the clover assembly when csw == 0.
    // USE_FULL_PF=1 swaps the EO Schur 1/4-root + LogDet pair for a single
    // non-EO 1/4-root pseudofermion on the full doubled M^dag M.  The
    // 2026-06-09 PSD diagnostic (Test_dtxqcd_psd_check) showed the EO Schur
    // Mpc explodes by 10^3-10^5 above aux_std ~ 0.5 while the full M stays
    // mild; the production HMC breakdowns at lambda=3 are EO-only.  Non-EO
    // bypasses the cliff at the cost of slower per-CG multishift.
    bool use_full_pf = false;
    if (const char *u = std::getenv("USE_FULL_PF"); u && *u) {
      use_full_pf = (std::atoi(u) != 0);
    }
    std::cout << GridLogMessage
              << "DTXQCD pseudofermion = "
              << (use_full_pf ? "FULL (non-EO)" : "EO Schur")
              << std::endl;

    DTXQCDLogDetCloverEOAction                   LogDet(Grid, RBGrid, mass_run, 0.0);
    DTXQCDWilsonCloverRationalEOAction
        PF(Grid, RBGrid, mass_run, rat_params, 0.0);
    DTXQCDWilsonCloverRationalFullAction
        PF_full(Grid, RBGrid, mass_run, rat_params, 0.0);

    // Three-level hierarchy.  TXQCD's Wilson Fierz test bundles AuxGaussian
    // into L1 with PF and LogDet -- a smoke run at that structure showed
    // catastrophic feedback on DTXQCD: the 1/4-root RHMC and the doubled
    // LogDet dump force into the aux slots, that pumps aux momentum, that
    // grows the aux fields, which in turn blows up AuxGaussian's
    // lambda^2 * aux force (observed Force_max ~ 4e4, Fdt_max ~ 1400 by
    // mid-trajectory on cold start).  Separating AuxGaussian onto its own
    // outer level with a x4 multiplier gives it its own time scale and
    // lets the gauge sub-integrator (also x4) carry the fast-varying
    // contributions.  Same structure gen_dtxqcd_cfgs uses.
    // Diagnostic knobs for bias hunt: DISABLE_PF=1 drops PF/LogDet, DISABLE_AUX=1
    // drops AuxGaussian.  Use both to test pure-gauge HMC on DTXQCDField (the
    // gauge adapter zeros aux force; a clean dH at trajectory scale tells us
    // the integrator + composite impl are sound and the bias is elsewhere).
    bool disable_pf  = std::getenv("DISABLE_PF")  && std::atoi(std::getenv("DISABLE_PF"))  != 0;
    bool disable_aux = std::getenv("DISABLE_AUX") && std::atoi(std::getenv("DISABLE_AUX")) != 0;
    if (disable_pf)  std::cout << GridLogMessage << "DTXQCD DISABLE_PF=1"  << std::endl;
    if (disable_aux) std::cout << GridLogMessage << "DTXQCD DISABLE_AUX=1" << std::endl;

    typedef Representations<EmptyRep<DTXQCDField>> Reps;
    ActionLevel<DTXQCDField, Reps> L1(1);
    if (!disable_pf) {
      if (use_full_pf) {
        L1.push_back(&PF_full);     // single rational on full M^dag M;
                                    // LogDet folded in (no companion).
      } else {
        L1.push_back(&PF);
        L1.push_back(&LogDet);
      }
    }
    ActionLevel<DTXQCDField, Reps> L2(2);
    L2.push_back(&GaugeAction);
    int aux_mult = 4;
    if (const char *m = std::getenv("AUX_MULT"); m && *m) {
      aux_mult = std::atoi(m);
    }
    std::cout << GridLogMessage << "DTXQCD aux multiplier = " << aux_mult << std::endl;
    ActionLevel<DTXQCDField, Reps> L3(aux_mult);
    if (!disable_aux) L3.push_back(&AuxAction);
    ActionSet<DTXQCDField, Reps> Aset;
    if (!disable_pf) Aset.push_back(L1);
    Aset.push_back(L2);
    if (!disable_aux) Aset.push_back(L3);

    IntegratorParameters MD;
    MD.name    = "ForceGradient";
    // MDSTEPS env: scaling experiment for the LogDet instability.  At
    // MDsteps = 10 + trajL = 0.1 (eps = 0.01), LogDet Fdt_max bounces
    // between sub-1 (stable) and 3000+ (CG failed, force trash) within
    // the first 3 MD substeps because per-site M_ee_48 sites cross
    // near-singular eigenvalues.  Halving eps via doubled MDsteps tells
    // us whether the instability is purely a step-size issue (stays
    // bounded longer with finer eps) or a per-trajectory cliff that
    // refinement can't reach (LogDet blows up at the same point in
    // configuration-space regardless of step size).
    MD.MDsteps = 10;
    if (const char *ms = std::getenv("MDSTEPS"); ms && *ms) {
      MD.MDsteps = std::atoi(ms);
    }
    MD.trajL   = 0.1;
    if (const char *tl = std::getenv("TRAJL"); tl && *tl) {
      MD.trajL = std::atof(tl);
    }
    std::cout << GridLogMessage << "Integrator: MDsteps=" << MD.MDsteps
              << " trajL=" << MD.trajL
              << " eps=" << (MD.trajL / MD.MDsteps) << std::endl;

    DTXQCDField U(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming DTXQCD from checkpoint at traj "
                << latest << std::endl;
      LoadDtxqcdConfig(U, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      // Thermal aux init at the spectrum-safe fluctuation width.  Without
      // AUX_FLUCT_LAMBDA the env-knob defaults to lambda = 3 -> outlier
      // aux sites near the Pfaffian sign boundary; FLUCT = 10 keeps the
      // doubled M well-conditioned on the cold gauge while HMC evolves
      // the aux toward the physical 1/lambda width.
      if (std::getenv("AUX_FLUCT_LAMBDA") == nullptr) setenv("AUX_FLUCT_LAMBDA", "10.0", 0);
      // AUX_INIT / AUX_INIT_AUTO: shift σ diagonal and s to the SD saddle
      // ⟨σ^{ij}_{ab}⟩_diag = ⟨s⟩ = Σ/λ² where Σ = ⟨Tr M^{-1}⟩/V is the
      // chiral condensate.  Skips the slow trace-mode equilibration at
      // small λ (relaxation rate λ²/Nf Nc ⇒ hundreds of trajectories for
      // λ < 1).  Mirrors TXQCD AUX_INIT pattern in gen_txqcd_cfgs.cc.
      //   AUX_INIT=value  → Σ set explicitly
      //   AUX_INIT_AUTO=1 → measure Σ = vev_trminv on the weak-field gauge
      //                     via Hutchinson on the action's Wilson operator
      //                     (no stout, mass_run, csw=0).
      // Default: no shift (Σ=0), preserves prior behavior for streams that
      // don't set either knob.
      RealD Sigma_init = 0.0;
      bool sigma_auto = false;
      if (const char *si = std::getenv("AUX_INIT"); si && *si) {
        Sigma_init = std::atof(si);
      } else if (const char *sa = std::getenv("AUX_INIT_AUTO");
                 sa && std::atoi(sa) != 0) {
        sigma_auto = true;
      }
      // Step 1: weak-field gauge first (so we can measure on it).
      DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, /*wf=*/0.1);
      if (sigma_auto) {
        // Measure Σ = ⟨Tr M^{-1}⟩/(2V) via Hutchinson on the unsmeared
        // weak-field gauge using the action's Wilson operator (csw=0,
        // periodic BC).  Use a separate RNG so the main pRNG state used
        // for aux generation is unchanged from a non-AUX path.
        WilsonImplParams impl_p;  // default: all-periodic
        typedef WilsonFermion<WilsonImplR> MeasFermOp;
        MeasFermOp Dw(U.U, Grid, RBGrid, mass_run, impl_p);
        MdagMLinearOperator<MeasFermOp, LatticeFermion> HermOp(Dw);
        ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
        const RealD V = (RealD)Grid.gSites();
        GridParallelRNG noisePRNG(&Grid);
        noisePRNG.SeedFixedIntegers({110, 120, 130, 140, 150});
        const int n_noise = 8;
        RealD acc = 0.0;
        for (int h = 0; h < n_noise; ++h) {
          LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
          gaussian(noisePRNG, eta);
          Dw.Mdag(eta, b);
          x = Zero();
          CG(HermOp, b, x);
          acc += innerProduct(eta, x).real() / (2.0 * V);
        }
        Sigma_init = acc / n_noise;
        std::cout << GridLogMessage
                  << "[AUX_INIT_AUTO] Σ = vev_trminv = " << Sigma_init
                  << "  → ⟨s⟩ = N_F·Σ/(2λ²) = "
                  << (DtxqcdNf * Sigma_init / (2.0 * lambda_run * lambda_run))
                  << "  ⟨Tr σ⟩ = 0 (traceless by construction)" << std::endl;
      } else if (Sigma_init != 0.0) {
        std::cout << GridLogMessage
                  << "[AUX_INIT] Σ = " << Sigma_init
                  << "  → ⟨s⟩ = N_F·Σ/(2λ²) = "
                  << (DtxqcdNf * Sigma_init / (2.0 * lambda_run * lambda_run))
                  << "  ⟨Tr σ⟩ = 0 (traceless by construction)" << std::endl;
      }
      // Step 2: fill aux with Gaussian + saddle shift.
      DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, Sigma_init);
      // Step 2b (AUX_INIT_AUTO only): bisect on g(Σ) = Σ - Σ_DTXQCD(Σ)
      // to find the self-consistent saddle satisfying
      // ⟨s⟩* = Nf·Σ_DTXQCD(⟨s⟩*)/λ², i.e., Σ_target = Σ_measured_on_op_with_that_aux.
      // Plain Picard iteration (Σ_{n+1} = Σ_DTXQCD(Σ_n)) is unstable at small
      // λ because |dΣ_DTXQCD/d⟨s⟩| · Nf/λ² > 1 (the map is anti-monotone with
      // slope > 1).  Bisection is robust:
      //   - lo: Σ=0 (no aux shift) → Σ_DTXQCD ≈ Σ_bare > 0, so g(0) < 0
      //   - hi: Σ=Σ_bare (the conventional saddle) → Σ_DTXQCD << Σ_bare, g(hi) > 0
      //   - midpoint bracketing converges in ~log2(Σ_bare/tol) iterations.
      auto measure_sigma_dtxqcd = [&](RealD Sigma_at) -> RealD {
        pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
        DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, Sigma_at);
        DTXQCDWilsonCloverFermionEO Dw_dtxqcd(U.U, Grid, RBGrid, mass_run, 0.0,
                                               U.sigma, U.pi, U.d, U.n, U.s, U.p);
        const RealD V = (RealD)Grid.gSites();
        GridParallelRNG noisePRNG(&Grid);
        noisePRNG.SeedFixedIntegers({400, 410, 420, 430, 440});
        const int n_noise_iter = 4;
        RealD acc = 0.0;
        for (int h = 0; h < n_noise_iter; ++h) {
          DTXQCDFermionDoubled eta(&Grid), b(&Grid), x(&Grid);
          for (int a = 0; a < DtxqcdNf; ++a) {
            gaussian(noisePRNG, eta.upper.f[a]);
            gaussian(noisePRNG, eta.lower.f[a]);
          }
          Dw_dtxqcd.Mdag(eta, b);
          DTXQCDFermionDoubled r(&Grid), p(&Grid), Ap(&Grid), tmp(&Grid);
          x = Zero();
          for (int a = 0; a < DtxqcdNf; ++a) {
            r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
            p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
          }
          RealD r2 = norm2(r), b2 = norm2(b);
          RealD cg_tol2 = 1e-12 * b2;
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
        // Per-quark Σ: full doubled trace ÷ (2·N_F).  Matches plain-Wilson
        // convention at aux=0 (verified Test_dtxqcd_trminv_zeroaux ratio = 4
        // = 2·N_F for N_F=2).
        return (acc / n_noise_iter) / (2.0 * DtxqcdNf);
      };
      if (sigma_auto) {
        int aux_iter_max = 15;
        if (const char *m = std::getenv("AUX_INIT_MAX_ITER"); m && *m)
          aux_iter_max = std::atoi(m);
        RealD aux_iter_tol = 1e-2;  // 1% precision on Σ
        if (const char *t = std::getenv("AUX_INIT_TOL"); t && *t)
          aux_iter_tol = std::atof(t);
        RealD Sigma_lo = 0.0;
        RealD Sigma_hi = Sigma_init;  // initial estimate from Σ_bare
        // Verify bracket signs.
        RealD sigma_dtxqcd_lo = measure_sigma_dtxqcd(Sigma_lo);
        RealD g_lo = Sigma_lo - sigma_dtxqcd_lo;
        RealD sigma_dtxqcd_hi = measure_sigma_dtxqcd(Sigma_hi);
        RealD g_hi = Sigma_hi - sigma_dtxqcd_hi;
        std::cout << GridLogMessage
                  << "[AUX_INIT bracket] g(0)=" << g_lo
                  << "  g(" << Sigma_hi << ")=" << g_hi << std::endl;
        if (g_lo * g_hi > 0.0) {
          std::cout << GridLogMessage
                    << "[AUX_INIT_AUTO] bracket has same sign — fall back to Σ_bare/2"
                    << std::endl;
          Sigma_init = Sigma_hi / 2.0;
        } else {
          // Bisection.
          for (int it = 0; it < aux_iter_max; ++it) {
            RealD Sigma_mid = 0.5 * (Sigma_lo + Sigma_hi);
            RealD sigma_dtxqcd_mid = measure_sigma_dtxqcd(Sigma_mid);
            RealD g_mid = Sigma_mid - sigma_dtxqcd_mid;
            std::cout << GridLogMessage
                      << "[AUX_INIT iter " << it << "] Σ_mid = " << Sigma_mid
                      << "  Σ_DTXQCD = " << sigma_dtxqcd_mid
                      << "  g = " << g_mid << std::endl;
            if (g_mid * g_lo < 0.0) {
              Sigma_hi = Sigma_mid;
              g_hi = g_mid;
            } else {
              Sigma_lo = Sigma_mid;
              g_lo = g_mid;
            }
            if ((Sigma_hi - Sigma_lo) / std::max(0.5 * (Sigma_hi + Sigma_lo), 1e-30)
                < aux_iter_tol) {
              break;
            }
          }
          Sigma_init = 0.5 * (Sigma_lo + Sigma_hi);
        }
        // Final fill with the converged Σ.
        pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
        DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda_run, Sigma_init);
        std::cout << GridLogMessage
                  << "[AUX_INIT_AUTO converged] Σ* = " << Sigma_init
                  << "  → ⟨s⟩* = Nf·Σ*/(2λ²) = "
                  << (DtxqcdNf * Sigma_init / (2.0 * lambda_run * lambda_run))
                  << std::endl;
      }
      if (const char *z = std::getenv("ZERO_DN_INIT"); z && std::atoi(z) != 0) {
        std::cout << GridLogMessage << "Zeroing d, n diquark fields at init" << std::endl;
        U.d = Zero();
        U.n = Zero();
      }
      if (const char *z = std::getenv("ZERO_ALL_AUX"); z && std::atoi(z) != 0) {
        std::cout << GridLogMessage << "Zeroing ALL aux fields at init (cold)" << std::endl;
        U.sigma = Zero();
        U.pi    = Zero();
        U.d     = Zero();
        U.n     = Zero();
        U.s     = Zero();
        U.p     = Zero();
      }
    }

    int no_metrop = (start_traj < n_therm_run) ? (n_therm_run - start_traj) : 0;
    if (const char *nm = std::getenv("NO_METROP"); nm && *nm) {
      no_metrop = std::atoi(nm);
    }
    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - no_metrop - start_traj;
    HMCp.NoMetropolisUntil   = no_metrop;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<DTXQCDCompositeImpl> Smear;
    typedef ForceGradient<DTXQCDCompositeImpl,
                          NoSmearing<DTXQCDCompositeImpl>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(U);

    CheckpointerParameters CPp;
    CPp.config_prefix = dtxqcd_cfg_dir() + "/ckpoint_lat";
    CPp.rng_prefix    = dtxqcd_cfg_dir() + "/ckpoint_rng";
    CPp.saveInterval  = meas_skip;
    CPp.format        = "IEEE64BIG";
    DTXQCDCheckpointer ckpt(CPp);

    DtxqcdDiagnostics diag(dtxqcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
        {"PseudoFermion", &PF},
        {"LogDet",        &LogDet},
        {"AuxGaussian",   &AuxAction},
        {"Gauge",         &GaugeAction}
    }, Grid, RBGrid, pRNG, mass_run, lambda_run, n_vev_noise);

    std::vector<HmcObservable<DTXQCDField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, U);
    HMC.evolve();
  }

  // ==================== QCD (Nf=2 Wilson) — shared with TXQCD test ====================
  // The QCD reference ensemble is generated by Test_txqcd_2pt_gencfgs in
  // configs_2pt_qcd_nf2/.  Re-generate it here if it's missing; otherwise
  // skip so a single sequential run of {txqcd_gencfgs, dtxqcd_gencfgs}
  // doesn't produce two copies.
  if (TxqcdTest2pt::qcd_configs_exist()) {
    std::cout << GridLogMessage
              << "QCD configs already exist, skipping generation." << std::endl;
  } else {
    std::cout << GridLogMessage
              << "Generating QCD Nf=2 configs (" << total_traj
              << " trajectories)..." << std::endl;
    mkdir_p(TxqcdTest2pt::qcd_cfg_dir());

    GridSerialRNG   sRNG;
    GridParallelRNG pRNG(&Grid);

    int start_traj = 0;
    int latest = latest_qcd_checkpoint();

    LatticeGaugeField Umu(&Grid);
    if (latest > 0) {
      std::cout << GridLogMessage << "Resuming QCD from checkpoint at traj "
                << latest << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, latest);
      start_traj = latest;
    } else {
      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      SU<Nc>::ColdConfiguration(Umu);
    }

    WilsonFermionD FermOp(Umu, Grid, RBGrid, mass);
    ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
    TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(FermOp, CG, CG);
    Nf2.is_smeared = false;

    WilsonGaugeActionR GaugeAction(beta);

    typedef Representations<EmptyRep<LatticeGaugeField>> Reps;
    ActionLevel<LatticeGaugeField, Reps> L1(1);
    L1.push_back(&Nf2);
    ActionLevel<LatticeGaugeField, Reps> L2(4);
    L2.push_back(&GaugeAction);
    ActionSet<LatticeGaugeField, Reps> Aset;
    Aset.push_back(L1);
    Aset.push_back(L2);

    IntegratorParameters MD;
    MD.name    = "ForceGradient";
    MD.MDsteps = 10;
    MD.trajL   = 0.5;

    int no_metrop = (start_traj < n_therm_run) ? (n_therm_run - start_traj) : 0;
    if (const char *nm = std::getenv("NO_METROP"); nm && *nm) {
      no_metrop = std::atoi(nm);
    }
    HMCparameters HMCp;
    HMCp.StartTrajectory     = start_traj;
    HMCp.Trajectories        = total_traj - no_metrop - start_traj;
    HMCp.NoMetropolisUntil   = no_metrop;
    HMCp.MetropolisTest      = true;
    HMCp.PerformRandomShift  = false;
    HMCp.StartingType        = "ColdStart";
    HMCp.MD = MD;

    NoSmearing<PeriodicGimplR> Smear;
    typedef ForceGradient<PeriodicGimplR,
                          NoSmearing<PeriodicGimplR>, Reps> IntT;
    IntT MDyn(&Grid, MD, Aset, Smear);
    Smear.set_Field(Umu);

    TxqcdTest2pt::QcdCheckpointer ckpt;
    ckpt.cfg_prefix    = TxqcdTest2pt::qcd_cfg_dir() + "/ckpoint_lat";
    ckpt.rng_prefix    = TxqcdTest2pt::qcd_cfg_dir() + "/ckpoint_rng";
    ckpt.save_interval = meas_skip;

    TxqcdTest2pt::QcdDiagnostics diag(
        TxqcdTest2pt::qcd_cfg_dir() + "/hmc_diagnostics", meas_skip, {
            {"Nf2",   &Nf2},
            {"Gauge", &GaugeAction}
        }, Grid, RBGrid, pRNG, mass, csw, n_vev_noise);

    std::vector<HmcObservable<LatticeGaugeField> *> Obs = {&ckpt, &diag};
    HybridMonteCarlo<IntT> HMC(HMCp, MDyn, sRNG, pRNG, Obs, Umu);
    HMC.evolve();
  }

  std::cout << GridLogMessage << "DTXQCD 2pt config generation complete." << std::endl;
  Grid_finalize();
  return 0;
}
