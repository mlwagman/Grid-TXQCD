// Shared Fierz-check helpers for the DTXQCD free-field unit tests.
//
//   Σ_DTXQCD = Tr[M48^{-1}] / (V · 2·N_F)   (per-quark, /(2V) Hutchinson convention)
//   Σ_W       = Tr[D_W^{-1}] / V             (plain Wilson at the same U)
//
// All operators use APBC time (chroma physics convention, matching
// DTXQCDMeooeDoubled::DefaultImplParams() and TXQCD's analogous default).
//
// Mirror of tests/txqcd/Test_txqcd_fierz_check_utils.h — same structure
// adapted to the doubled DTXQCD operator (M48 = block-diag M_upper +
// M_lower, both 24-dim per site → 48 per site for both blocks).

#pragma once

#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/utils/GaugeGroup.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/parallelIO/NerscIO.h>

NAMESPACE_BEGIN(Grid);

// Initialize the frozen gauge field. Aux fields are assumed already
// initialized by the caller (DTXQCDCompositeImpl::ColdConfiguration +
// any seed sequence).
//
// GAUGE_INIT env knob:
//   "cold"        — U_μ(x) = I (default)
//   "tepid"       — Grid TepidConfiguration (small Lie-random amp=0.01)
//   "tepid:AMP"   — explicit amplitude
//   "hot"         — full SU(3) random
//   "nersc:PATH"  — read NERSC-format config
inline RealD DtxqcdInitFrozenGauge(GridParallelRNG &pRNG, DTXQCDField &U) {
  const char *env = std::getenv("GAUGE_INIT");
  std::string mode = (env && *env) ? env : "cold";
  if (mode == "cold") {
    U.U = 1.0;
  } else if (mode == "hot") {
    SU<Nc>::HotConfiguration(pRNG, U.U);
  } else if (mode.rfind("tepid", 0) == 0) {
    RealD amp = 0.01;
    if (mode.size() > 5 && mode[5] == ':') amp = std::atof(mode.c_str() + 6);
    typedef LatticeColourMatrix LCM;
    LCM Umu(U.U.Grid());
    for (int mu = 0; mu < Nd; ++mu) {
      SU<Nc>::LieRandomize(pRNG, Umu, amp);
      PokeIndex<LorentzIndex>(U.U, Umu, mu);
    }
    std::cout << GridLogMessage << "DtxqcdInitFrozenGauge: tepid amp=" << amp << std::endl;
  } else if (mode.rfind("nersc:", 0) == 0) {
    std::string path = mode.substr(6);
    FieldMetaData header;
    NerscIO::readConfiguration(U.U, header, path);
    std::cout << GridLogMessage
              << "DtxqcdInitFrozenGauge: loaded NERSC " << path << std::endl;
  } else {
    std::cerr << "DtxqcdInitFrozenGauge: unknown GAUGE_INIT='" << mode
              << "'.  Falling back to cold." << std::endl;
    U.U = 1.0;
  }
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage
            << "DtxqcdInitFrozenGauge: mode=" << mode << " plaq=" << plaq
            << " (1.0 = cold, < 1 = perturbed)" << std::endl;
  return plaq;
}

inline WilsonImplR::ImplParams DtxqcdFierzApbcImplParams() {
  WilsonImplR::ImplParams p;
  p.boundary_phases.resize(Nd, 1.0);
  p.boundary_phases[Nd - 1] = -1.0;
  return p;
}

// Σ_W reference: Tr[D_W^{-1}] / V on plain Wilson (csw=0) or WilsonClover
// (csw!=0) at the given U, APBC time.  Per-quark trace.
template <class Op>
inline RealD DtxqcdFierzWilsonLikeTrminv_(Op &Dw, GridCartesian &Grid_,
                                           GridParallelRNG &prng,
                                           int n_noise, RealD cg_tol) {
  MdagMLinearOperator<Op, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, 30000);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid_), b(&Grid_), x(&Grid_);
    gaussian(prng, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / n_noise;
}

inline RealD DtxqcdFierzPlainWilsonTrminv(LatticeGaugeField &U,
                                           GridCartesian &Grid_,
                                           GridRedBlackCartesian &RBGrid,
                                           GridParallelRNG &prng,
                                           RealD mass, RealD csw,
                                           int n_noise, RealD cg_tol) {
  auto impl_p = DtxqcdFierzApbcImplParams();
  if (csw == 0.0) {
    WilsonFermionD Dw(U, Grid_, RBGrid, mass, impl_p);
    return DtxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  } else {
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(U, Grid_, RBGrid, mass, csw, csw,
            WilsonAnisotropyCoefficients(), impl_p);
    return DtxqcdFierzWilsonLikeTrminv_(Dw, Grid_, prng, n_noise, cg_tol);
  }
}

// Σ_DTXQCD: stochastic Tr[M48^{-1}] / (V · 2·N_F) on the doubled DTXQCD
// operator (covers both upper and lower blocks).  Per-quark normalization
// → matches DtxqcdFierzPlainWilsonTrminv at aux=0.
inline RealD DtxqcdFierzOpTrminv(DTXQCDField &U,
                                  GridCartesian &Grid_,
                                  GridRedBlackCartesian &RBGrid,
                                  GridParallelRNG &prng,
                                  RealD mass, RealD csw,
                                  int n_noise, RealD cg_tol) {
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const int cg_max_iter = 30000;
  for (int h = 0; h < n_noise; ++h) {
    DTXQCDFermionDoubled eta(&Grid_), b(&Grid_), x(&Grid_);
    DTXQCDFermionDoubled r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(prng, eta.upper.f[a]);
      gaussian(prng, eta.lower.f[a]);
    }
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < DtxqcdNf; ++a) {
      r.upper.f[a] = b.upper.f[a]; r.lower.f[a] = b.lower.f[a];
      p.upper.f[a] = b.upper.f[a]; p.lower.f[a] = b.lower.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max_iter; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  // Both upper + lower blocks → 2·Nf flavors total.
  return acc / (n_noise * 2.0 * (RealD)DtxqcdNf);
}

// Self-consistent aux seed via bisection.  Solves
//   g(Σ) = Σ - Σ_DTXQCD(Σ) = 0
// where Σ_DTXQCD(Σ) is the per-quark trace of M_DTXQCD^{-1} with aux
// filled at ⟨s⟩ = 2·Nf·Σ/λ².  Bisection bracket: [0, Σ_bare].  Mirrors
// the AUX_INIT_AUTO logic in Test_dtxqcd_2pt_gencfgs.cc.
//
// Returns the converged Σ.  The caller is responsible for the final
// FillAuxFields call (which this routine also performs as a side effect).
inline RealD DtxqcdSelfConsistentAuxInit(GridParallelRNG &pRNG,
                                          GridCartesian &Grid_,
                                          GridRedBlackCartesian &RBGrid,
                                          DTXQCDField &U, RealD lambda,
                                          RealD mass, RealD csw,
                                          RealD Sigma_bare, int aux_max_iter,
                                          RealD aux_tol) {
  auto measure_sigma_dtxqcd = [&](RealD Sigma_at) -> RealD {
    pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_at);
    GridParallelRNG noisePRNG(&Grid_);
    noisePRNG.SeedFixedIntegers({400, 410, 420, 430, 440});
    const int n_noise_iter = 4;
    return DtxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG, mass, csw,
                                n_noise_iter, /*cg_tol=*/1e-8);
  };
  RealD Sigma_lo = 0.0;
  RealD Sigma_hi = Sigma_bare;
  RealD sigma_dtxqcd_lo = measure_sigma_dtxqcd(Sigma_lo);
  RealD g_lo = Sigma_lo - sigma_dtxqcd_lo;
  RealD sigma_dtxqcd_hi = measure_sigma_dtxqcd(Sigma_hi);
  RealD g_hi = Sigma_hi - sigma_dtxqcd_hi;
  std::cout << GridLogMessage
            << "[AUX_INIT bracket] g(0)=" << g_lo
            << "  g(" << Sigma_hi << ")=" << g_hi << std::endl;
  RealD Sigma_star = Sigma_bare;
  if (g_lo * g_hi > 0.0) {
    std::cout << GridLogMessage
              << "[AUX_INIT_AUTO] bracket has same sign — fall back to Σ_bare/2"
              << std::endl;
    Sigma_star = Sigma_hi / 2.0;
  } else {
    for (int it = 0; it < aux_max_iter; ++it) {
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
      if ((Sigma_hi - Sigma_lo) /
              std::max(0.5 * (Sigma_hi + Sigma_lo), 1e-30) < aux_tol) {
        break;
      }
    }
    Sigma_star = 0.5 * (Sigma_lo + Sigma_hi);
  }
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_star);
  std::cout << GridLogMessage
            << "[AUX_INIT_AUTO converged] Σ* = " << Sigma_star
            << "  → ⟨s⟩* = 2·Nf·Σ*/λ² = "
            << (2.0 * DtxqcdNf * Sigma_star / (lambda * lambda)) << std::endl;
  return Sigma_star;
}

struct DtxqcdFierzCheckResult {
  RealD sigma_dtx;
  RealD sigma_w;
  RealD ratio;
  RealD dev;
  bool pass;
};

inline DtxqcdFierzCheckResult DtxqcdFierzCheck(DTXQCDField &U,
                                                 GridCartesian &Grid_,
                                                 GridRedBlackCartesian &RBGrid,
                                                 RealD mass, RealD csw,
                                                 int n_noise, RealD cg_tol,
                                                 RealD pass_tol,
                                                 const std::string &test_name) {
  RealD V = (RealD)Grid_.gSites();
  auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
  auto tr_pi    = TensorRemove(sum(trace(U.pi)));
  auto tr_s     = TensorRemove(sum(trace(U.s)));
  auto tr_p     = TensorRemove(sum(trace(U.p)));
  auto tr_d     = TensorRemove(sum(trace(U.d)));
  auto tr_n     = TensorRemove(sum(trace(U.n)));
  std::cout << GridLogMessage << std::endl
            << "===== DTXQCD Fierz check on final HMC state =====" << std::endl;
  std::cout << GridLogMessage << "Aux trace VEVs:"
            << "\n   ⟨Tr σ⟩ = " << (tr_sigma.real()/V) << " + i·" << (tr_sigma.imag()/V)
            << "\n   ⟨Tr π⟩ = " << (tr_pi.real()/V)    << " + i·" << (tr_pi.imag()/V)
            << "\n   ⟨Tr s⟩ = " << (tr_s.real()/V)     << " + i·" << (tr_s.imag()/V)
            << "\n   ⟨Tr p⟩ = " << (tr_p.real()/V)     << " + i·" << (tr_p.imag()/V)
            << "\n   ⟨Tr d⟩ = " << (tr_d.real()/V)     << " + i·" << (tr_d.imag()/V)
            << "\n   ⟨Tr n⟩ = " << (tr_n.real()/V)     << " + i·" << (tr_n.imag()/V)
            << std::endl;
  std::cout << GridLogMessage << "Aux ‖·‖²/V:"
            << "  ‖σ‖²=" << (norm2(U.sigma) / V)
            << "  ‖π‖²=" << (norm2(U.pi)    / V)
            << "  ‖s‖²=" << (norm2(U.s)     / V)
            << "  ‖p‖²=" << (norm2(U.p)     / V)
            << "  ‖d‖²=" << (norm2(U.d)     / V)
            << "  ‖n‖²=" << (norm2(U.n)     / V)
            << std::endl;

  GridParallelRNG noisePRNG(&Grid_);
  noisePRNG.SeedFixedIntegers({1001, 1002, 1003, 1004, 1005});
  RealD sigma_dtx = DtxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG,
                                         mass, csw, n_noise, cg_tol);
  noisePRNG.SeedFixedIntegers({2001, 2002, 2003, 2004, 2005});
  RealD sigma_w = DtxqcdFierzPlainWilsonTrminv(U.U, Grid_, RBGrid, noisePRNG,
                                                 mass, csw, n_noise, cg_tol);
  RealD ratio = sigma_dtx / sigma_w;
  RealD dev = std::fabs(ratio - 1.0);
  bool pass = (dev < pass_tol);

  std::cout << GridLogMessage << "Σ_DTXQCD (with aux) = " << sigma_dtx << std::endl;
  std::cout << GridLogMessage << "Σ_W      (plain D_W) = " << sigma_w
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage << "ratio Σ_DTXQCD / Σ_W = " << ratio
            << "   |dev| = " << dev
            << "   tol = " << pass_tol << std::endl;
  std::cout << GridLogMessage
            << "[" << test_name << "] " << (pass ? "PASS" : "FAIL") << std::endl;

  return {sigma_dtx, sigma_w, ratio, dev, pass};
}

NAMESPACE_END(Grid);
