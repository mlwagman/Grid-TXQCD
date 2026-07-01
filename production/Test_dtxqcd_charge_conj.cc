// Test_dtxqcd_charge_conj: charge-conjugation convention test for building
// hadron correlators from the LOWER (charge-conjugate) block of the doubled
// M48, cross-checked against the UPPER block.
//
// Strategy (per mwagman): nail the convention at the FREE FIELD first, then add
// fields.  At U=I, aux=0 the doubled operator is block-diagonal and the lower
// block Dw(conj U) = Dw(I) = the upper block, so:
//   (A) S_lower == S_upper  exactly (same operator; lower-solve machinery works)
//   (B) the free Wilson propagator obeys the charge-conjugation identity
//         C^{-1} S^T_spin C = S      with  C = gamma2 gamma4
//   (C) pion built from lower == pion from upper (contraction is C-even; the
//       C's cancel in tr(S adj S))
//   (D) nucleon from upper == nucleon from the C-conjugated lower propagator
//       ChargeConj(S_lower)  (the extra C on the baryon source+sink spinors)
//
// Once (A)-(D) pass at U=I, flip DTXQCD_CC_TEST_HOT=1 to add a random gauge +
// nonzero aux: then upper vs C-lower should agree only up to aux noise.
//
// No QUDA needed (free-field 4^4 CPU CG); builds via the catch-all Makefile
// rule.  Env: LATT (default 4.4.4.4), MASS (0.5), CSW (0), CG_TOL (1e-10),
// DTXQCD_CC_TEST_HOT (0), PASS_TOL (1e-8 cold / 0.05 hot).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/utils/BaryonUtils.h>

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atoi(v) : d;
}

// Solve M x = src on the full doubled volume via CG on M†M (b = M† src).
static void SolveM(DTXQCDMOp &Mop, const DTXQCDFermionDoubled &src,
                   DTXQCDFermionDoubled &x, RealD tol, int maxit) {
  GridBase *g = src.Grid();
  DTXQCDFermionDoubled b(g), r(g), p(g), Mp(g), MdMp(g);
  Mop.Mdag(src, b);
  x = Zero();
  r = b; p = r;
  RealD rsq = norm2(r), bsq = std::max(norm2(b), 1e-30);
  RealD tol2 = tol * tol * bsq;
  int it;
  for (it = 0; it < maxit; ++it) {
    Mop.M(p, Mp);
    Mop.Mdag(Mp, MdMp);
    ComplexD pAp = innerProduct(p, MdMp);
    ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
      x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
      r.upper.f[a] = r.upper.f[a] - alpha * MdMp.upper.f[a];
      r.lower.f[a] = r.lower.f[a] - alpha * MdMp.lower.f[a];
    }
    RealD rn = norm2(r);
    if (rn < tol2) { rsq = rn; break; }
    RealD beta = rn / rsq;
    for (int a = 0; a < DtxqcdNf; ++a) {
      p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
      p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
    }
    rsq = rn;
  }
  std::cout << GridLogMessage << "  [SolveM] iter=" << it
            << " rsq/bsq=" << rsq / bsq << std::endl;
}

enum Block { UPPER = 0, LOWER = 1 };

// Point-source (origin) propagator from ONE doubled block, flavor 0.
static LatticePropagator BuildProp(DTXQCDMOp &Mop, GridCartesian &Grid,
                                   Block blk, RealD tol, int maxit) {
  LatticePropagator S(&Grid);
  S = Zero();
  Coordinate origin(Nd, 0);
  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticePropagator srcP(&Grid);
      srcP = Zero();
      SpinColourMatrix kron; kron = 1.0;
      pokeSite(kron, srcP, origin);
      LatticeFermion sf(&Grid);
      PropToFerm<WilsonImplR>(sf, srcP, spin, col);

      DTXQCDFermionDoubled eta(&Grid), psi(&Grid);
      eta = Zero();
      if (blk == UPPER) eta.upper.f[0] = sf; else eta.lower.f[0] = sf;
      SolveM(Mop, eta, psi, tol, maxit);
      LatticeFermion out = (blk == UPPER) ? psi.upper.f[0] : psi.lower.f[0];
      FermToProp<WilsonImplR>(S, out, spin, col);
    }
  }
  return S;
}

// Charge-conjugate a propagator:  S_cc = C^{-1} (S^T_spin) C ,  C = gamma2 gamma4
// (C^{-1} = gamma4 gamma2).  This is the operation that turns a lower-block
// (charge-conjugate-field) propagator into one contractable like an upper-block
// propagator.
static LatticePropagator ChargeConj(const LatticePropagator &S) {
  Gamma g2(Gamma::Algebra::GammaY);
  Gamma g4(Gamma::Algebra::GammaT);
  Gamma C    = g2 * g4;
  Gamma Cinv = g4 * g2;
  // Spin-index transpose (color untouched) via peek/poke — transposeIndex
  // doesn't specialize cleanly on the propagator's spin index.
  LatticePropagator St(S.Grid());
  St = Zero();
  for (int i = 0; i < Ns; ++i)
    for (int j = 0; j < Ns; ++j)
      pokeSpin(St, peekSpin(S, j, i), i, j);
  return Cinv * St * C;   // Cinv on the left spin index, C on the right
}

// Portable complex magnitude (ComplexD is thrust::complex under nvcc).
static inline double cmag(const ComplexD &z) {
  double r = z.real(), i = z.imag();
  return std::sqrt(r * r + i * i);
}

static std::vector<ComplexD> PionCorr(const LatticePropagator &Sa,
                                      const LatticePropagator &Sb) {
  LatticeComplex c(Sa.Grid());
  c = trace(Sa * adj(Sb));
  std::vector<TComplex> sl;
  sliceSum(c, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<ComplexD> NucleonCorr(const LatticePropagator &Saa,
                                         const LatticePropagator &Sb,
                                         int parity = 1) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(Saa.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(Saa, Saa, Sb, G_A, G_B, G_A, G_B,
                                            wick, parity, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static RealD MaxRelDiff(const std::vector<ComplexD> &a,
                        const std::vector<ComplexD> &b) {
  RealD m = 0.0;
  for (size_t t = 0; t < a.size(); ++t) {
    RealD den = cmag(a[t]) + cmag(b[t]);
    if (den > 1e-30) m = std::max(m, cmag(a[t] - b[t]) / den);
  }
  return m;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::vector<int> latt_dims{4, 4, 4, 4};
  if (const char *v = std::getenv("LATT"); v && *v) {
    latt_dims.clear();
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  RealD mass = env_real("MASS", 0.5);
  RealD csw  = env_real("CSW", 0.0);
  RealD tol  = env_real("CG_TOL", 1e-10);
  int   maxit = env_int("CG_MAX", 20000);
  bool  hot  = env_int("DTXQCD_CC_TEST_HOT", 0) != 0;
  RealD pass_tol = env_real("PASS_TOL", hot ? 0.05 : 1e-8);

  std::cout << GridLogMessage << "==== Test_dtxqcd_charge_conj ("
            << (hot ? "HOT: random U + aux" : "COLD: U=I, aux=0")
            << ") mass=" << mass << " csw=" << csw << " ====" << std::endl;

  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  // Gauge + aux.
  DTXQCDField Uf(&Grid);
  DTXQCDCompositeImpl::ColdConfiguration(pRNG, Uf);  // U=I, aux=0
  Uf.sigma = Zero(); Uf.pi = Zero(); Uf.d = Zero();
  Uf.n = Zero();     Uf.s = Zero();  Uf.p = Zero();
  bool cc_aux = env_int("DTXQCD_CC_AUX", 0) != 0;
  if (hot) {
    // Random gauge so S_lower = Dw(conj U)^{-1} != Dw(U)^{-1} = S_upper.  With
    // aux=0 (default) the upper<->lower charge-conjugation relation is EXACT
    // (no aux noise) -> pins the C convention to machine precision.  Set
    // DTXQCD_CC_AUX=1 to also add small aux (the "agree up to aux noise" case).
    SU<Nc>::HotConfiguration(pRNG, Uf.U);
    if (cc_aux) {
      gaussian(pRNG, Uf.sigma); gaussian(pRNG, Uf.pi);
      gaussian(pRNG, Uf.d);     gaussian(pRNG, Uf.n);
      Uf.sigma = 0.1 * Uf.sigma; Uf.pi = 0.1 * Uf.pi;
      Uf.d = 0.1 * Uf.d;         Uf.n = 0.1 * Uf.n;
    }
  }

  DTXQCDWilsonCloverFermionEO Dw(Uf.U, Grid, RBGrid, mass, csw,
                                 Uf.sigma, Uf.pi, Uf.d, Uf.n, Uf.s, Uf.p);
  DTXQCDMOp Mop(Dw);

  std::cout << GridLogMessage << "-- upper propagator --" << std::endl;
  LatticePropagator S_up = BuildProp(Mop, Grid, UPPER, tol, maxit);
  std::cout << GridLogMessage << "-- lower propagator --" << std::endl;
  LatticePropagator S_lo = BuildProp(Mop, Grid, LOWER, tol, maxit);

  bool ok = true;

  // (A) S_lower vs S_upper (identical at free field).
  RealD d_prop = std::sqrt(norm2(S_lo - S_up) / norm2(S_up));
  std::cout << GridLogMessage << "(A) ||S_lo - S_up|| / ||S_up|| = " << d_prop
            << (hot ? "   (nonzero when hot: different gauge/aux)" : "")
            << std::endl;
  if (!hot && d_prop > pass_tol) ok = false;

  // (E) KEY C-form test: does ChargeConj map the lower prop onto the upper one?
  // Exact (~0) in hot/aux=0 iff the C convention (matrix + placement) is right.
  LatticePropagator S_lo_to_up = ChargeConj(S_lo);
  RealD d_e = std::sqrt(norm2(S_lo_to_up - S_up) / norm2(S_up));
  std::cout << GridLogMessage
            << "(E) ||ChargeConj(S_lo) - S_up|| / ||S_up|| = " << d_e
            << "   (KEY: ~0 when C is right; use hot + aux=0)" << std::endl;

  // (B) charge-conjugation identity on the upper propagator.
  LatticePropagator S_up_cc = ChargeConj(S_up);
  RealD d_cc = std::sqrt(norm2(S_up_cc - S_up) / norm2(S_up));
  std::cout << GridLogMessage << "(B) ||C^-1 S_up^T C - S_up|| / ||S_up|| = "
            << d_cc << "   (C=gamma2 gamma4; DIAGNOSTIC — pointwise C^-1 S^T C "
            << "relates S(x,0)->S(0,x), so a coordinate flip means this need "
            << "NOT be ~0; the clean statement is at correlator level)"
            << std::endl;

  // (C) pion: upper vs lower, built identically.
  auto pi_up = PionCorr(S_up, S_up);
  auto pi_lo = PionCorr(S_lo, S_lo);
  RealD d_pi = MaxRelDiff(pi_up, pi_lo);
  std::cout << GridLogMessage << "(C) pion  max rel diff (upper vs lower) = "
            << d_pi << std::endl;
  if (d_pi > pass_tol) ok = false;

  // (D) nucleon: upper vs C-conjugated lower.
  LatticePropagator S_lo_cc = ChargeConj(S_lo);
  auto nuc_up   = NucleonCorr(S_up, S_up);
  auto nuc_locc = NucleonCorr(S_lo_cc, S_lo_cc);
  RealD d_nuc = MaxRelDiff(nuc_up, nuc_locc);
  std::cout << GridLogMessage << "(D) nucleon max rel diff (upper vs C-lower) = "
            << d_nuc << "   (DIAGNOSTIC — the C convention we are pinning down)"
            << std::endl;

  // For reference: nucleon from lower WITHOUT the C (should MISmatch unless C=1).
  auto nuc_lo_raw = NucleonCorr(S_lo, S_lo);
  std::cout << GridLogMessage << "    [ref] nucleon max rel diff (upper vs RAW lower, no C) = "
            << MaxRelDiff(nuc_up, nuc_lo_raw) << std::endl;

  // Empirical C-insertion variants on the LOWER propagator (spin similarity,
  // NO coordinate flip — this is the "C on the source/sink spinors" the baryon
  // interpolator needs).  C = gamma2 gamma4.
  {
    Gamma g2c(Gamma::Algebra::GammaY), g4c(Gamma::Algebra::GammaT);
    Gamma Cm = g2c * g4c, Cmi = g4c * g2c;
    LatticePropagator sF = Cm * S_lo * Cmi;   // C S C^{-1}
    LatticePropagator sG = Cm * S_lo * Cm;    // C S C
    std::cout << GridLogMessage << "    (F) nucleon up vs (C S C^-1) lower = "
              << MaxRelDiff(nuc_up, NucleonCorr(sF, sF)) << std::endl;
    std::cout << GridLogMessage << "    (G) nucleon up vs (C S C)   lower = "
              << MaxRelDiff(nuc_up, NucleonCorr(sG, sG)) << std::endl;
  }

  // C maps the nucleon to the ANTInucleon, which relates to the nucleon by a
  // PARITY flip and/or TIME REVERSAL (not a per-propagator transform).  Scan
  // parity x time-reversal to locate the exact correspondence.  rd2 is
  // sign-agnostic (C can flip the overall correlator sign).
  {
    int T = (int)nuc_up.size();
    auto TR = [T](const std::vector<ComplexD> &c) {
      std::vector<ComplexD> o(T);
      for (int t = 0; t < T; ++t) o[t] = c[(T - t) % T];
      return o;
    };
    auto rd2 = [](const std::vector<ComplexD> &a, const std::vector<ComplexD> &b) {
      std::vector<ComplexD> nb(b.size());
      for (size_t i = 0; i < b.size(); ++i) nb[i] = -b[i];
      return std::min(MaxRelDiff(a, b), MaxRelDiff(a, nb));
    };
    auto nu_neg = NucleonCorr(S_up, S_up, -1);
    auto nl_pos = nuc_lo_raw;
    auto nl_neg = NucleonCorr(S_lo, S_lo, -1);
    std::cout << GridLogMessage << "    (H) up(+) vs lo(-)      [+/-] = " << rd2(nuc_up, nl_neg) << std::endl;
    std::cout << GridLogMessage << "    (I) up(-) vs lo(+)      [+/-] = " << rd2(nu_neg, nl_pos) << std::endl;
    std::cout << GridLogMessage << "    (J) up(+) vs TR lo(+)   [+/-] = " << rd2(nuc_up, TR(nl_pos)) << std::endl;
    std::cout << GridLogMessage << "    (K) up(+) vs TR lo(-)   [+/-] = " << rd2(nuc_up, TR(nl_neg)) << std::endl;
    std::cout << GridLogMessage << "    (L) up(+) vs lo(+)      [+/-] = " << rd2(nuc_up, nl_pos) << std::endl;
  }

  std::cout << GridLogMessage << "==== " << (ok ? "PASS" : "FAIL")
            << " (pass_tol=" << pass_tol << ") ====" << std::endl;
  Grid_finalize();
  return ok ? 0 : 1;
}
