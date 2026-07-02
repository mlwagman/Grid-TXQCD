// Test_dtxqcd_aux_force_aggregate
//
// Follow-up to Test_dtxqcd_aux_drive_symmetry (commit 894ea80f) for the
// <Tr[pi]> vs <p> factor-30 VEV asymmetry at lambda=0.1 (see
// HANDOFF_TRPI_ASYMMETRY.md).  The FD probe tested S(U +/- eps) only -- it
// never touched deriv(), and it only exercised the HOST operator.  This test
// closes both gaps:
//
// Probe 1 (u-equivalence of the operator).  pi and p enter M only through
//   the combination u = pi + p*I_6 (gamma5-coupled identically in upper and
//   lower blocks; pi^T = pi for the singlet direction).  Therefore
//     M(pi + c*I, p)  ==  M(pi, p + c)    EXACTLY (same matrix),
//   for every config and every c.  We check this on the host operator
//   (control; the 12-digit FD symmetry already implies it) and -- the point --
//   on whatever multishift backend the env selects for the rational action's
//   S(), i.e. the production Style C QUDA device operator when
//   DTXQCD_MULTISHIFT_QUDA=1.  Any S(U_A) != S(U_B) here means the SAMPLED
//   ACTION distinguishes the pi-singlet from p: that shifts the equilibrium
//   (Metropolis cannot filter an action-value defect) and is THE bug.
//
// Probe 2 (per-action deriv() aggregates).  For each production action, call
//   deriv(U, F) and compare
//     G_trpi = sum_x Re Tr F.pi(x)    vs    G_p = sum_x Re F.p(x)
//   (and sigma/s as the P-even control), plus the FD gradient of that same
//   action's S along the four singlet directions:
//     g_fd = (S(U + eps*dir) - S(U - eps*dir)) / (2 eps)  ==  G_aggregate
//   if deriv() is the true gradient of S.  A per-action ratio != 1 localizes
//   a force bug (Metropolis-safe for the equilibrium, but at lambda=0.1 the
//   u-orthogonal mode w = p - 6*pi0 has tau ~ O(500) trajs, so a force kick
//   during the NO_METROP warmup freezes in and looks like an SSB VEV).
//
// Env knobs to sweep between runs (each is read-once/static in the actions):
//   DTXQCD_RATFORCE_GPU=0|1      rational aux force CPU thread_for vs GPU
//                                kernel (production launcher sets 1!)
//   DTXQCD_MULTISHIFT_QUDA=0|1   host DP multishift vs QUDA Style C (per-call
//                                read; production default 1)
//   DTXQCD_LOGDET_GPU, DTXQCD_LOGDET_S_GPU, DTXQCD_PRECOMPUTE_GPU  (LogDet)
//   SIGMA_INIT (default 3.07), LAMBDA (default 0.1), RHMC_DEG (default 8)
//
// Run (4^4 first; 16^3x48 with --grid 16.16.16.48 SIGMA_INIT=0.221654383):
//   ./Test_dtxqcd_aux_force_aggregate --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDGaugeActionAdapter.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#endif
#include <cstdlib>

using namespace Grid;

// ---------------------------------------------------------------------------
// Singlet-direction perturbation (same as Test_dtxqcd_aux_drive_symmetry).
// ---------------------------------------------------------------------------
enum class Mode { Trsig, s, Trpi, p };

static const char *ModeName(Mode m) {
  switch (m) {
    case Mode::Trsig: return "Trsig";
    case Mode::s:     return "s    ";
    case Mode::Trpi:  return "Trpi ";
    case Mode::p:     return "p    ";
  }
  return "?";
}

static void PerturbSingletMode(const DTXQCDField &base, Mode mode, RealD eps,
                               DTXQCDField &out) {
  out = base;
  if (mode == Mode::s) {
    typedef typename LatticeDtxqcdS::vector_object::scalar_object SObj;
    SObj shift; shift()()() = eps;
    LatticeDtxqcdS s_shift(base.s.Grid()); s_shift = shift;
    out.s = base.s + s_shift;
  } else if (mode == Mode::p) {
    typedef typename LatticeDtxqcdP::vector_object::scalar_object PObj;
    PObj shift; shift()()() = eps;
    LatticeDtxqcdP p_shift(base.p.Grid()); p_shift = shift;
    out.p = base.p + p_shift;
  } else {
    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
    SigSobj shift; shift = Zero();
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int i = 0; i < Nc; ++i)
        shift()(a, a)(i, i) = eps;
    LatticeDtxqcdSigma sig_shift(base.sigma.Grid()); sig_shift = shift;
    if (mode == Mode::Trsig) out.sigma = base.sigma + sig_shift;
    else                     out.pi    = base.pi    + sig_shift;
  }
}

// ---------------------------------------------------------------------------
// Aggregates of a composite force/deriv field along the four singlet dirs.
//   G_trsig = sum_x Tr F.sigma(x),  G_s = sum_x F.s(x),  etc.
// These are exactly the gradients of S along the PerturbSingletMode dirs
// (chain rule: d/d eps sum hits every diagonal entry once).
// ---------------------------------------------------------------------------
struct Agg { ComplexD trsig, s, trpi, p; };

static Agg Aggregates(DTXQCDField &F) {
  Agg a;
  a.trsig = TensorRemove(sum(trace(F.sigma)));
  a.trpi  = TensorRemove(sum(trace(F.pi)));
  a.s     = TensorRemove(sum(F.s));
  a.p     = TensorRemove(sum(F.p));
  return a;
}

static void PrintAgg(const std::string &tag, const Agg &a) {
  std::cout << GridLogMessage << tag
            << "  G_Trsig = " << a.trsig.real() << " (im " << a.trsig.imag() << ")"
            << "  G_s = " << a.s.real() << " (im " << a.s.imag() << ")"
            << std::endl;
  std::cout << GridLogMessage << tag
            << "  G_Trpi  = " << a.trpi.real() << " (im " << a.trpi.imag() << ")"
            << "  G_p = " << a.p.real() << " (im " << a.p.imag() << ")"
            << std::endl;
  std::cout << GridLogMessage << tag
            << "  ratio G_Trsig/G_s = " << (a.trsig.real() / a.s.real())
            << "   ratio G_Trpi/G_p = " << (a.trpi.real() / a.p.real())
            << std::endl;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
#ifdef GRID_HAVE_QUDA
  // Needed before any DTXQCDMpcOpQUDA construction (the Style C multishift
  // backend); idempotent, harmless when DTXQCD_MULTISHIFT_QUDA=0.
  Grid::Quda::initialize();
#endif

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dvec;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) dvec.push_back(std::stoi(tok));
      if (dvec.size() == (size_t)Nd) latt = Coordinate(dvec);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
  GridSerialRNG   sRNG;        sRNG.SeedFixedIntegers({11, 12, 13, 14});

  auto env_real = [](const char *name, RealD dflt) {
    const char *e = std::getenv(name);
    return (e && *e) ? std::atof(e) : dflt;
  };
  auto env_int = [](const char *name, int dflt) {
    const char *e = std::getenv(name);
    return (e && *e) ? std::atoi(e) : dflt;
  };

  const RealD lambda     = env_real("LAMBDA", 0.1);
  const RealD mass       = -0.245;
  const RealD csw        = 1.24930970916466;
  const RealD beta       = 6.1;
  const RealD Sigma_init = env_real("SIGMA_INIT", 3.07);
  const int   rat_deg    = env_int("RHMC_DEG", 8);
  const RealD cg_tol     = env_real("CG_TOL", 1.0e-12);

  DTXQCDField U(&Grid);
  DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, /*wf_scale=*/0.1);
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_init);

  std::cout << GridLogMessage << "config: lambda=" << lambda << " mass=" << mass
            << " csw=" << csw << " Sigma_init=" << Sigma_init
            << " V4=" << Grid.gSites() << " rat_deg=" << rat_deg
            << " cg_tol=" << cg_tol << std::endl;
  std::cout << GridLogMessage << "aux VEVs:  <Trsig>="
            << TensorRemove(sum(trace(U.sigma))).real() / (RealD)Grid.gSites()
            << "  <Trpi>="
            << TensorRemove(sum(trace(U.pi))).real() / (RealD)Grid.gSites()
            << "  <s>=" << TensorRemove(sum(U.s)).real() / (RealD)Grid.gSites()
            << "  <p>=" << TensorRemove(sum(U.p)).real() / (RealD)Grid.gSites()
            << std::endl;

  // =========================================================================
  // Probe 1a: u-equivalence of the HOST operator.
  //   U_A = (pi + c*I, p)   and   U_B = (pi, p + c)  give the SAME M.
  // =========================================================================
  const RealD c_shift = 0.37;
  DTXQCDField UA(&Grid), UB(&Grid);
  PerturbSingletMode(U, Mode::Trpi, c_shift, UA);
  PerturbSingletMode(U, Mode::p,    c_shift, UB);

  {
    DTXQCDFermionDoubled v(&Grid), MA(&Grid), MB(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, v.upper.f[a]);
      gaussian(pRNG, v.lower.f[a]);
    }
    auto apply_host_M = [&](DTXQCDField &Ux, DTXQCDFermionDoubled &out) {
      DTXQCDWilsonCloverFermionEO Dw(Ux.U, Grid, RBGrid, mass, csw,
                                     Ux.sigma, Ux.pi, Ux.d, Ux.n, Ux.s, Ux.p);
      DTXQCDMOp Mop(Dw);
      Mop.M(v, out);
    };
    apply_host_M(UA, MA);
    apply_host_M(UB, MB);
    RealD n2A = 0.0, n2diff = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion du(&Grid), dl(&Grid);
      du = MA.upper.f[a] - MB.upper.f[a];
      dl = MA.lower.f[a] - MB.lower.f[a];
      n2diff += norm2(du) + norm2(dl);
      n2A    += norm2(MA.upper.f[a]) + norm2(MA.lower.f[a]);
    }
    std::cout << GridLogMessage
              << "[u-equiv host M]  |M(pi+cI,p)v - M(pi,p+c)v| / |Mv| = "
              << std::scientific << std::setprecision(3)
              << std::sqrt(n2diff / n2A)
              << "   (expect ~1e-16; anything larger = host operator breaks "
                 "pi/p singlet identity)" << std::endl;
  }

  // =========================================================================
  // Actions (mirror the non-EO production stack of the lambda=0.1 chain).
  // =========================================================================
  const RealD u0 = 1.0;
  DTXQCDGaugeActionAdapter<PlaqPlusRectangleAction<PeriodicGimplR>>
      GaugeAction(beta, -beta / (20.0 * u0 * u0));
  DTXQCDAuxiliaryFieldGaussianAction AuxAction(lambda);
  DTXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass, csw);
  OneFlavourRationalParams rp(/*lo=*/0.1, /*hi=*/64.0, /*MaxIter=*/10000,
                              /*tolerance=*/cg_tol, /*degree=*/rat_deg,
                              /*precision=*/50, /*BoundsCheckFreq=*/0,
                              /*mdtolerance=*/cg_tol);
  DTXQCDWilsonCloverRationalFullAction Rational(Grid, RBGrid, mass, rp, csw);

  // refresh() first (rational needs Phi; also triggers deferred Remez build).
  GaugeAction.refresh(U, sRNG, pRNG);
  AuxAction.refresh(U, sRNG, pRNG);
  LogDet.refresh(U, sRNG, pRNG);
  Rational.refresh(U, sRNG, pRNG);

  // =========================================================================
  // Probe 1b: u-equivalence of each M-derived action's S (with Phi held
  // fixed).  S(U_A) == S(U_B) required.  The rational S goes through the
  // env-selected multishift backend -- with DTXQCD_MULTISHIFT_QUDA=1 this
  // probes the PRODUCTION Style C device operator.
  // =========================================================================
  std::cout << GridLogMessage << "==== Probe 1b: S(pi+cI) vs S(p+c), c = "
            << c_shift << " ====" << std::endl;
  auto u_equiv_S = [&](Action<DTXQCDField> &A, const char *name,
                       RealD noise_floor) {
    RealD SA = A.S(UA);
    RealD SB = A.S(UB);
    std::cout << GridLogMessage << "[u-equiv S] " << name
              << ":  S_A = " << std::setprecision(15) << SA
              << "  S_B = " << SB
              << "  S_A - S_B = " << std::scientific << std::setprecision(3)
              << (SA - SB) << "  rel = " << ((SA - SB) / SA)
              << "  (must be < ~" << noise_floor << ")" << std::endl;
    return SA - SB;
  };
  u_equiv_S(LogDet,   "LogDetCloverEO", 1e-10);
  u_equiv_S(Rational, "RationalFull  ", 1e-8);
  {
    // AuxGaussian S is NOT u-symmetric by design; print observed vs analytic.
    RealD SA = AuxAction.S(UA), SB = AuxAction.S(UB);
    RealD trpi_sum = TensorRemove(sum(trace(U.pi))).real();
    RealD p_sum    = TensorRemove(sum(U.p)).real();
    RealD V4 = (RealD)Grid.gSites();
    RealD expect = 0.5 * lambda * lambda *
                   ((2.0 * c_shift * trpi_sum + 6.0 * c_shift * c_shift * V4) -
                    (2.0 * c_shift * p_sum + c_shift * c_shift * V4));
    std::cout << GridLogMessage << "[u-equiv S] AuxGaussian    :  S_A - S_B = "
              << std::scientific << std::setprecision(6) << (SA - SB)
              << "  analytic = " << expect
              << "  (difference by design; ratio = "
              << ((SA - SB) / expect) << ", must be 1)" << std::endl;
  }

  // =========================================================================
  // Probe 2: per-action deriv() aggregates + FD-vs-deriv gradient check.
  // =========================================================================
  std::cout << GridLogMessage << "==== Probe 2: deriv() aggregates ===="
            << std::endl;
  auto probe_action = [&](Action<DTXQCDField> &A, const char *name, RealD eps,
                          bool do_fd) {
    std::cout << GridLogMessage << "---- [" << name << "] ----" << std::endl;
    DTXQCDField F(&Grid);
    A.deriv(U, F);
    Agg raw = Aggregates(F);
    PrintAgg(std::string("[") + name + " raw ]", raw);
    DTXQCDField Fp = DTXQCDCompositeImpl::projectForce(F);
    Agg proj = Aggregates(Fp);
    PrintAgg(std::string("[") + name + " proj]", proj);

    if (do_fd) {
      auto fd_grad = [&](Mode m) {
        DTXQCDField Up(&Grid), Um(&Grid);
        PerturbSingletMode(U, m,  eps, Up);
        PerturbSingletMode(U, m, -eps, Um);
        RealD Sp = A.S(Up), Sm = A.S(Um);
        return (Sp - Sm) / (2.0 * eps);
      };
      RealD g_trsig = fd_grad(Mode::Trsig);
      RealD g_s     = fd_grad(Mode::s);
      RealD g_trpi  = fd_grad(Mode::Trpi);
      RealD g_p     = fd_grad(Mode::p);
      std::cout << GridLogMessage << "[" << name << " FD  ]"
                << "  g_Trsig = " << g_trsig << "  g_s = " << g_s
                << "  g_Trpi = " << g_trpi << "  g_p = " << g_p
                << "   (eps = " << eps << ")" << std::endl;
      std::cout << GridLogMessage << "[" << name << " D/FD]"
                << "  Trsig: " << (raw.trsig.real() / g_trsig)
                << "  s: " << (raw.s.real() / g_s)
                << "  Trpi: " << (raw.trpi.real() / g_trpi)
                << "  p: " << (raw.p.real() / g_p)
                << "   (all four must be 1.000...)" << std::endl;
    }
  };

  probe_action(GaugeAction, "GaugeAction ", 1e-4, /*do_fd=*/false);
  probe_action(AuxAction,   "AuxGaussian ", 1e-4, /*do_fd=*/true);
  probe_action(LogDet,      "LogDetEO    ", 1e-4, /*do_fd=*/true);
  probe_action(Rational,    "RationalFull", 1e-3, /*do_fd=*/true);

  std::cout << GridLogMessage << "==== verdicts ====" << std::endl;
  std::cout << GridLogMessage
            << "u-equiv violations  -> ACTION-level pi/p asymmetry (shifts "
               "equilibrium; Metropolis cannot filter it)." << std::endl;
  std::cout << GridLogMessage
            << "D/FD ratio != 1     -> deriv() is not the gradient of S for "
               "that action/slot (force-only bug; freezes into the soft w "
               "mode at small lambda via the NO_METROP warmup)." << std::endl;
  std::cout << GridLogMessage
            << "Sweep DTXQCD_RATFORCE_GPU=0|1 and DTXQCD_MULTISHIFT_QUDA=0|1 "
               "to compare CPU vs production-GPU paths." << std::endl;

  Grid_finalize();
  return 0;
}
