#pragma once
// Shared auto-scale of the Remez endpoints (`param.lo`, `param.hi`) for the
// DTXQCD doubled-operator rational actions, EO and non-EO alike.  Direct port
// of TXQCDRemezAutoScale.h to the DTXQCD doubled fermion / Pfaffian 1/4-root.
//
// Motivation (the bug this hardens against):
//   The DTXQCD generators hardcode RHMC_LO=0.1 / RHMC_HI=64.  When the doubled
//   M48 spectrum drifts out of that fixed window (intermediate/large lambda, or
//   csw/mass that pushes lambda_max(M48^dag M48) well above 64 or lambda_min
//   well below 0.1), two failure modes appear together:
//     - the per-shift multishift-CG tolerances effectively go absurdly tight
//       (CG stalls / never converges), and
//     - the Remez approximation of x^{-1/4} is constructed on an interval that
//       no longer covers the true spectrum, so it over-extrapolates and emits
//       enormous PF forces -> dH blowup.
//   This is exactly the failure TXQCD just fixed with auto-scaled Remez from
//   Lanczos spectral bounds; this header brings the same hardening to DTXQCD.
//
// Physics difference vs. TXQCD: DTXQCD's rational is the Pfaffian 1/4-root.
// The actions hold x^{-1/4} (PowerNegQuarter, used by S() + deriv() multishift)
// and x^{+1/8} (PowerEighth, used by refresh()).  So DtxqcdRebuildRemez
// generates approximations for 1/4 and 1/8 (NOT TXQCD's 1/2, 1/4 set), and
// Init's exactly the two MultiShiftFunctions the DTXQCD actions actually use.
//
// The Lanczos MUST run on the SAME operator the multishift CG inverts so the
// bounds match the shifts:
//   - EO action:  multishift CG inverts (Mpc^dag Mpc + sigma_k) via DTXQCDMpcOp
//                 (M = Mpc, Mdag = MpcDag).  Lanczos on Mpc^dag Mpc, RB-Odd.
//   - Full action: multishift CG inverts (M^dag M + sigma_k) via DTXQCDMOp.
//                 Lanczos on M^dag M, full Cartesian grid.
// Both operators expose the same M(in,out)/Mdag(in,out) interface on
// DTXQCDFermionDoubled, so a single Lanczos template covers both; the only
// difference is the grid and (for EO) the Odd checkerboard on the fields.
//
// Env gates (all default off / safe values, opt-in via slurm script):
//   RAT_AUTO_HI=1            master switch: enable lo+hi auto-scale (default OFF)
//   RAT_HI_SAFETY=1.5        hi_new = safety * lambda_max_est
//   RAT_PI_STEPS=30          (kept for parity; unused by the Lanczos path)
//   RAT_AUTO_LO_MODE=...     off | lanczos | heuristic  (default lanczos)
//   RAT_LO_SAFETY=0.5        lo_new = safety * lambda_min_est  (lanczos mode)
//   RAT_KAPPA_CAP=500        lo_new = hi / kappa_cap          (heuristic mode)
//   RAT_LANCZOS_NM=30        Lanczos Krylov dimension
//
// IMPORTANT (DTXQCD-specific):  the WHOLE auto-scale block is gated on the
// master switch `enabled` (RAT_AUTO_HI).  With RAT_AUTO_HI unset the Lanczos is
// not run at all and no rebuild happens, so behaviour is byte-identical to the
// pre-autoscale code (the Remez built once in the constructor on [param.lo,
// param.hi]).  This differs from the TXQCD helper, where the lo-side Lanczos
// runs even with the master switch off.
//
// nvcc-compat: the Lanczos uses Eigen only for the real symmetric tridiagonal
// solve (Eigen::MatrixXd).  All ComplexD live in Grid types (innerProduct /
// norm2 on DTXQCDFermionDoubled).  No peekSite inside a thread_for.

#include <Grid/Grid.h>
#include <Grid/Grid_Eigen_Dense.h>
#include <Grid/Eigen/Eigenvalues>
#include <Grid/algorithms/approx/Remez.h>
#include <Grid/algorithms/approx/MultiShiftFunction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>

NAMESPACE_BEGIN(Grid);

// Lo-end auto-scale mode.
//   Off       : keep current lo verbatim
//   Lanczos   : plain Lanczos on M^dag M (no Chebyshev filter) — gives both
//               lambda_max and lambda_min in one short pass.  lo = safety*lmin.
//   Heuristic : free; lo = lambda_max / kappa_cap (assumes operator kappa < cap)
enum class DtxqcdAutoLoMode { Off, Lanczos, Heuristic };

struct DtxqcdRemezAutoScaleParams {
  // ---- master switch / hi side ----------------------------------------
  bool  enabled    = false;   // RAT_AUTO_HI: master switch for the whole block
  RealD hi_safety  = 1.5;     // RAT_HI_SAFETY
  int   pi_steps   = 30;      // RAT_PI_STEPS (parity only; Lanczos path used)
  // ---- lo side --------------------------------------------------------
  DtxqcdAutoLoMode auto_lo_mode = DtxqcdAutoLoMode::Lanczos;  // RAT_AUTO_LO_MODE
  RealD lo_safety   = 0.5;    // RAT_LO_SAFETY (Lanczos: lo = safety * lambda_min)
  RealD kappa_cap   = 500.0;  // RAT_KAPPA_CAP (Heuristic: lo = lambda_max / kappa)
  int   lanczos_nm  = 30;     // RAT_LANCZOS_NM (Lanczos Krylov dim)
  RealD rebuild_hyst = 2.0;   // RAT_REBUILD_HYST: don't rebuild merely to tighten
                              // a bound unless it is > this factor wasteful
                              // (covers the few-% Lanczos noise; coverage
                              // failures always rebuild regardless).
  // ---- tracked across rebuilds ----------------------------------------
  RealD current_hi  = 0.0;
  RealD current_lo  = 0.0;

  static DtxqcdRemezAutoScaleParams FromEnv(RealD initial_lo, RealD initial_hi) {
    DtxqcdRemezAutoScaleParams p;
    p.current_lo = initial_lo;
    p.current_hi = initial_hi;
    if (const char *e = std::getenv("RAT_AUTO_HI");   e && *e) p.enabled    = std::atoi(e) != 0;
    if (const char *e = std::getenv("RAT_HI_SAFETY"); e && *e) p.hi_safety  = std::atof(e);
    if (const char *e = std::getenv("RAT_PI_STEPS");  e && *e) p.pi_steps   = std::atoi(e);
    if (const char *e = std::getenv("RAT_AUTO_LO_MODE"); e && *e) {
      std::string s(e);
      if      (s == "off")  p.auto_lo_mode = DtxqcdAutoLoMode::Off;
      else if (s == "heur" || s == "heuristic")
                            p.auto_lo_mode = DtxqcdAutoLoMode::Heuristic;
      else                  p.auto_lo_mode = DtxqcdAutoLoMode::Lanczos;
    }
    if (const char *e = std::getenv("RAT_LO_SAFETY");  e && *e) p.lo_safety   = std::atof(e);
    if (const char *e = std::getenv("RAT_KAPPA_CAP");  e && *e) p.kappa_cap   = std::atof(e);
    if (const char *e = std::getenv("RAT_LANCZOS_NM"); e && *e) p.lanczos_nm  = std::atoi(e);
    if (const char *e = std::getenv("RAT_REBUILD_HYST"); e && *e) p.rebuild_hyst = std::atof(e);
    return p;
  }
};

// Rebuild the two Remez approximations the DTXQCD rational actions use:
//   PowerNegQuarter = x^{-1/4}   (S() + deriv() multishift)
//   PowerEighth     = x^{+1/8}   (refresh())
// on the interval [lo, hi].  Mirrors the constructor's AlgRemez block so the
// rebuilt rationals are identical in structure to the ones built at startup.
inline void DtxqcdRebuildRemez(RealD lo, RealD hi, int degree, RealD precision,
                               RealD tolerance, const char *label,
                               MultiShiftFunction &PowerNegQuarter,
                               MultiShiftFunction &PowerEighth) {
  AlgRemez remez(lo, hi, precision);
  std::cout << GridLogMessage << "[" << label
            << "] Remez degree " << degree << " on [" << lo << ", " << hi
            << "] for x^(-1/4)" << std::endl;
  remez.generateApprox(degree, 1, 4);
  PowerNegQuarter.Init(remez, tolerance, true);
  std::cout << GridLogMessage << "[" << label
            << "] Remez degree " << degree << " on [" << lo << ", " << hi
            << "] for x^(+1/8)" << std::endl;
  remez.generateApprox(degree, 1, 8);
  PowerEighth.Init(remez, tolerance, false);
}

// Plain Lanczos (no Chebyshev filter) on M^dag M for the DTXQCD doubled
// operator.  Returns BOTH extremal eigenvalues of M^dag M simultaneously.
// Direct analogue of TXQCDLanczosMinMax_Tx, on DTXQCDFermionDoubled.
//
// `Mop` is any operator with M(in,out) / Mdag(in,out) on DTXQCDFermionDoubled:
//   - DTXQCDMOp  (full-volume M; M^dag M)        -> grid = full Cartesian, cb<0
//   - DTXQCDMpcOp (Schur Mpc; Mpc^dag Mpc)       -> grid = RB grid,  cb = Odd
// `cb` < 0 means full-volume (no checkerboard set); cb = Even/Odd sets the
// checkerboard on every working vector so the RB-grid Schur path is correct.
template <class Op>
inline void DtxqcdLanczosMinMax(Op &Mop, GridBase *grid,
                                GridParallelRNG &pRNG, int Nm, int cb,
                                RealD &lmin, RealD &lmax) {
  std::vector<DTXQCDFermionDoubled> V;
  V.reserve(Nm);
  for (int i = 0; i < Nm; ++i) V.emplace_back(grid);
  std::vector<RealD> alpha(Nm, 0.0), beta(Nm, 0.0);
  DTXQCDFermionDoubled w(grid), tmp(grid);

  auto setcb = [cb](DTXQCDFermionDoubled &x) {
    if (cb < 0) return;
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a].Checkerboard() = cb;
      x.lower.f[a].Checkerboard() = cb;
    }
  };

  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, V[0].upper.f[a]);
    gaussian(pRNG, V[0].lower.f[a]);
  }
  setcb(V[0]);
  for (int i = 1; i < Nm; ++i) setcb(V[i]);
  setcb(w);
  setcb(tmp);

  RealD n0 = std::sqrt(norm2(V[0]));
  if (n0 == 0.0) { lmin = lmax = 0.0; return; }
  for (int a = 0; a < DtxqcdNf; ++a) {
    V[0].upper.f[a] = (1.0 / n0) * V[0].upper.f[a];
    V[0].lower.f[a] = (1.0 / n0) * V[0].lower.f[a];
  }

  int k_done = 0;
  for (int k = 0; k < Nm; ++k) {
    Mop.M(V[k], tmp);
    Mop.Mdag(tmp, w);
    if (k > 0)
      for (int a = 0; a < DtxqcdNf; ++a) {
        w.upper.f[a] = w.upper.f[a] + (-beta[k-1]) * V[k-1].upper.f[a];
        w.lower.f[a] = w.lower.f[a] + (-beta[k-1]) * V[k-1].lower.f[a];
      }
    alpha[k] = real(innerProduct(V[k], w));
    for (int a = 0; a < DtxqcdNf; ++a) {
      w.upper.f[a] = w.upper.f[a] + (-alpha[k]) * V[k].upper.f[a];
      w.lower.f[a] = w.lower.f[a] + (-alpha[k]) * V[k].lower.f[a];
    }
    // Full re-orthogonalization (two passes for stability).
    for (int pass = 0; pass < 2; ++pass) {
      for (int j = 0; j <= k; ++j) {
        ComplexD c = innerProduct(V[j], w);
        for (int a = 0; a < DtxqcdNf; ++a) {
          w.upper.f[a] = w.upper.f[a] + (-c) * V[j].upper.f[a];
          w.lower.f[a] = w.lower.f[a] + (-c) * V[j].lower.f[a];
        }
      }
    }
    beta[k] = std::sqrt(norm2(w));
    k_done = k + 1;
    if (beta[k] < 1e-12) break;
    if (k + 1 < Nm) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        V[k+1].upper.f[a] = (1.0 / beta[k]) * w.upper.f[a];
        V[k+1].lower.f[a] = (1.0 / beta[k]) * w.lower.f[a];
      }
    }
  }

  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k_done, k_done);
  for (int i = 0; i < k_done; ++i) {
    T(i, i) = alpha[i];
    if (i + 1 < k_done) { T(i+1, i) = beta[i]; T(i, i+1) = beta[i]; }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  lmin = es.eigenvalues()(0);
  lmax = es.eigenvalues()(k_done - 1);
}

// Shared refresh-time auto-scale step used by both DTXQCD rational actions.
//
// If params.enabled (RAT_AUTO_HI): run DtxqcdLanczosMinMax on the action's own
// multishift operator, then move [current_lo, current_hi] so the Remez interval
// COVERS the safety-scaled spectrum [lo_safety*lambda_min, hi_safety*lambda_max],
// raising OR lowering each bound as needed.  Coverage failures (lo > lambda_min
// or hi < lambda_max) always trigger a rebuild; a bound that merely became
// wastefully wide is only retightened when it is > rebuild_hyst off, so the
// few-% Lanczos noise doesn't cause a rebuild every refresh.  Returns true (and
// writes new_lo/new_hi) iff a rebuild is needed.  With params.enabled == false
// this returns false without touching the operator or RNG -- byte-identical to
// the pre-autoscale path.
//
// NOTE (2026-06-22 fix): the original code only ever RAISED lo (never lowered
// it), so when lambda_min fell below the initial lo (light quark mass -> small
// lambda_min, e.g. lambda_min=0.077 < lo=0.1 at the 16^3x48 production point)
// the rational x^{-1/4} was evaluated outside its valid range on the lowest
// eigenmodes, under-valuing the pseudofermion weight there and over-weighting
// rough/low-plaquette configs -> a biased gauge ensemble (plaq ~0.43 vs the
// QCD 0.513).  The cover-both-ways logic below closes that edge case.
template <class Op>
inline bool DtxqcdRefreshAutoScale(DtxqcdRemezAutoScaleParams &params,
                                   Op &Mop, GridBase *grid,
                                   GridParallelRNG &pRNG, int cb,
                                   const char *label,
                                   RealD &new_lo, RealD &new_hi) {
  new_lo = params.current_lo;
  new_hi = params.current_hi;
  if (!params.enabled) return false;

  RealD lmin = 0.0, lmax = 0.0;
  DtxqcdLanczosMinMax(Mop, grid, pRNG, params.lanczos_nm, cb, lmin, lmax);
  std::cout << GridLogMessage << "[" << label
            << "] auto-scale Lanczos (Nm=" << params.lanczos_nm
            << "): lambda_min_est=" << lmin << ", lambda_max_est=" << lmax
            << "  current=[" << params.current_lo << ", " << params.current_hi
            << "]" << std::endl;

  bool need_rebuild = false;
  const RealD hyst = params.rebuild_hyst;

  // ---- hi side: cover lambda_max with margin hi_safety (>1).  Coverage
  //      requires new_hi >= lambda_max; the target sits at hi_safety*lambda_max.
  //      Rebuild on a coverage failure (hi < lambda_max) OR when hi is more than
  //      `hyst` wastefully high, raising OR lowering to the target. ----
  RealD target_hi = params.hi_safety * lmax;
  if (new_hi < lmax || new_hi > target_hi * hyst) {
    std::cout << GridLogMessage << "[" << label
              << "] auto-scale rat.hi: " << new_hi << " -> " << target_hi
              << "  (lambda_max=" << lmax << ", safety=" << params.hi_safety
              << (new_hi < lmax ? ", COVERAGE FAIL" : ", retighten") << ")"
              << std::endl;
    new_hi = target_hi;
    need_rebuild = true;
  }

  // ---- lo side: cover lambda_min with margin lo_safety (<1).  Coverage
  //      requires new_lo <= lambda_min; the target sits at lo_safety*lambda_min.
  //      Rebuild on a coverage failure (lo > lambda_min -- the edge case the
  //      old raise-only code missed) OR when lo is more than `hyst` wastefully
  //      low, raising OR lowering to the target. ----
  if (params.auto_lo_mode == DtxqcdAutoLoMode::Lanczos) {
    RealD target_lo = params.lo_safety * lmin;
    if (new_lo > lmin || new_lo < target_lo / hyst) {
      std::cout << GridLogMessage << "[" << label
                << "] auto-scale rat.lo (lanczos): " << new_lo << " -> " << target_lo
                << "  (lambda_min=" << lmin << ", safety=" << params.lo_safety
                << (new_lo > lmin ? ", COVERAGE FAIL" : ", retighten") << ")"
                << std::endl;
      new_lo = target_lo;
      need_rebuild = true;
    }
  } else if (params.auto_lo_mode == DtxqcdAutoLoMode::Heuristic && new_hi > 0.0) {
    // Heuristic: bound the condition number, lo = hi/kappa_cap; track it both
    // ways with hysteresis (no measured lambda_min to compare against).
    RealD target_lo = new_hi / params.kappa_cap;
    if (new_lo < target_lo / hyst || new_lo > target_lo * hyst) {
      std::cout << GridLogMessage << "[" << label
                << "] auto-scale rat.lo (heur): " << new_lo << " -> " << target_lo
                << "  (hi/kappa_cap)" << std::endl;
      new_lo = target_lo;
      need_rebuild = true;
    }
  }

  return need_rebuild;
}

NAMESPACE_END(Grid);
