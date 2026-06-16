// Shared Fierz-check helpers for the free-field unit tests:
//
//   • Σ_TX = Tr[M_TX^{-1}] / (V · Nf)   (per-quark, /(2V) Hutchinson convention)
//   • Σ_W  = Tr[D_W^{-1}] / V           (plain Wilson at the same U)
//   • do_fierz_check: stochastic ratio + PASS/FAIL on |ratio - 1| < tol
//
// All operators are constructed with APBC time (chroma physics convention,
// matching TXQCDWilsonOp::DefaultImplParams() and
// TXQCDWilsonCloverFermionEO::DefaultImplParams()).
//
// Header so it can be shared between Test_txqcd_freefield_qbarq and
// Test_txqcd_freefield_qbarq_light (and any future variants).

#pragma once

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

NAMESPACE_BEGIN(Grid);

inline WilsonImplR::ImplParams TxqcdFierzApbcImplParams() {
  WilsonImplR::ImplParams p;
  p.boundary_phases.resize(Nd, 1.0);
  p.boundary_phases[Nd - 1] = -1.0;
  return p;
}

// Stochastic Tr[D_W^{-1}] / V on plain Wilson at the given U, APBC time.
inline RealD TxqcdFierzPlainWilsonTrminv(LatticeGaugeField &U,
                                          GridCartesian &Grid_,
                                          GridRedBlackCartesian &RBGrid,
                                          GridParallelRNG &prng,
                                          RealD mass, int n_noise,
                                          RealD cg_tol) {
  auto impl_p = TxqcdFierzApbcImplParams();
  WilsonFermionD Dw(U, Grid_, RBGrid, mass, impl_p);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
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

// Stochastic Tr[M_TX^{-1}] / (V · Nf) on the doubled TXQCD Wilson-clover.
// Per-quark normalization → matches plain_wilson_trminv at aux=0.
inline RealD TxqcdFierzOpTrminv(TXQCDField &U,
                                 GridCartesian &Grid_,
                                 GridRedBlackCartesian &RBGrid,
                                 GridParallelRNG &prng,
                                 RealD mass, RealD csw,
                                 int n_noise, RealD cg_tol) {
  std::array<RealD, TxqcdNf> mass_arr;
  for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass;
  TXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw);
  RealD V = (RealD)Grid_.gSites();
  RealD acc = 0.0;
  const int cg_max_iter = 30000;
  for (int h = 0; h < n_noise; ++h) {
    TXQCDFermionNf eta(&Grid_), b(&Grid_), x(&Grid_);
    TXQCDFermionNf r(&Grid_), p(&Grid_), Ap(&Grid_), tmp(&Grid_);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(prng, eta.f[a]);
    Dw.Mdag(eta, b);
    x = Zero();
    for (int a = 0; a < TxqcdNf; ++a) {
      r.f[a] = b.f[a];
      p.f[a] = b.f[a];
    }
    RealD r2 = norm2(r), b2 = norm2(b);
    RealD tol2 = cg_tol * cg_tol * b2;
    if (b2 < 1e-30) continue;
    for (int k = 0; k < cg_max_iter; ++k) {
      Dw.M(p, tmp);
      Dw.Mdag(tmp, Ap);
      RealD pAp = innerProduct(p, Ap).real();
      RealD alpha = r2 / pAp;
      for (int a = 0; a < TxqcdNf; ++a) {
        x.f[a] = x.f[a] + alpha * p.f[a];
        r.f[a] = r.f[a] - alpha * Ap.f[a];
      }
      RealD r2_new = norm2(r);
      if (r2_new < tol2) { r2 = r2_new; break; }
      RealD beta = r2_new / r2;
      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta * p.f[a];
      r2 = r2_new;
    }
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / (n_noise * (RealD)TxqcdNf);
}

struct TxqcdFierzCheckResult {
  RealD sigma_tx;
  RealD sigma_w;
  RealD ratio;
  RealD dev;
  bool pass;
};

// Run the full Σ_TX vs Σ_W check on the equilibrated state.  Prints all
// diagnostics (aux trace VEVs, ‖·‖²/V per channel, Σ_TX, Σ_W, ratio, PASS/FAIL).
inline TxqcdFierzCheckResult TxqcdFierzCheck(TXQCDField &U,
                                              GridCartesian &Grid_,
                                              GridRedBlackCartesian &RBGrid,
                                              RealD mass, RealD csw,
                                              int n_noise, RealD cg_tol,
                                              RealD pass_tol,
                                              const std::string &test_name) {
  RealD V = (RealD)Grid_.gSites();

  // Aux trace VEVs (real + imag for parity / Hermitian canaries).
  auto tr_sigma = TensorRemove(sum(trace(U.sigma)));
  auto tr_pi    = TensorRemove(sum(trace(U.pi)));
  auto tr_s     = TensorRemove(sum(trace(U.s)));
  auto tr_p     = TensorRemove(sum(trace(U.p)));
  std::cout << GridLogMessage << std::endl
            << "===== Fierz check on final HMC state =====" << std::endl;
  std::cout << GridLogMessage << "Aux trace VEVs:"
            << "\n   ⟨Tr σ⟩ = " << (tr_sigma.real()/V) << " + i·" << (tr_sigma.imag()/V)
            << "\n   ⟨Tr π⟩ = " << (tr_pi.real()/V)    << " + i·" << (tr_pi.imag()/V)
            << "\n   ⟨Tr s⟩ = " << (tr_s.real()/V)     << " + i·" << (tr_s.imag()/V)
            << "\n   ⟨Tr p⟩ = " << (tr_p.real()/V)     << " + i·" << (tr_p.imag()/V)
            << std::endl;
  std::cout << GridLogMessage << "Aux ‖·‖²/V:"
            << "  ‖σ‖²=" << (norm2(U.sigma) / V)
            << "  ‖π‖²=" << (norm2(U.pi)    / V)
            << "  ‖s‖²=" << (norm2(U.s)     / V)
            << "  ‖p‖²=" << (norm2(U.p)     / V)
            << "  ‖t‖²=" << (norm2(U.t)     / V)
            << std::endl;

  // Independent noise streams for σ_TX and σ_W.
  GridParallelRNG noisePRNG(&Grid_);
  noisePRNG.SeedFixedIntegers({1001, 1002, 1003, 1004, 1005});
  RealD sigma_tx = TxqcdFierzOpTrminv(U, Grid_, RBGrid, noisePRNG,
                                       mass, csw, n_noise, cg_tol);
  noisePRNG.SeedFixedIntegers({2001, 2002, 2003, 2004, 2005});
  RealD sigma_w = TxqcdFierzPlainWilsonTrminv(U.U, Grid_, RBGrid, noisePRNG,
                                                mass, n_noise, cg_tol);
  RealD ratio = sigma_tx / sigma_w;
  RealD dev = std::fabs(ratio - 1.0);
  bool pass = (dev < pass_tol);

  std::cout << GridLogMessage << "Σ_TX (with aux)  = " << sigma_tx << std::endl;
  std::cout << GridLogMessage << "Σ_W  (plain D_W) = " << sigma_w
            << "   ← Fierz target" << std::endl;
  std::cout << GridLogMessage << "ratio Σ_TX / Σ_W = " << ratio
            << "   |dev| = " << dev
            << "   tol = " << pass_tol << std::endl;
  std::cout << GridLogMessage
            << "[" << test_name << "] " << (pass ? "PASS" : "FAIL") << std::endl;

  return {sigma_tx, sigma_w, ratio, dev, pass};
}

NAMESPACE_END(Grid);
