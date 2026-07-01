// Test_dtxqcd_aux_drive_symmetry
//
// Diagnostic for the observed <Tr[π]> ≠ <p> asymmetry in low-λ production
// chains (see project_dtxqcd_low_lambda_isosinglet_condensate).  My
// derivation from the aux Gaussian action and fermion couplings predicts
//
//    <Tr[π]> = 6·<π₀>              (observable is 6× the amplitude)
//    equilibrium in the π₀ mode:  6λ² π₀ = f_π₀   →  π₀ = f_π₀ / (6λ²)
//    equilibrium in the p  mode:  λ²  p   = f_p    →  p   = f_p  / λ²
//    ⇒  <Tr[π]> / <p>   =   f_π₀ / f_p
//
// where f_π₀ and f_p are the fermion-induced forces on the π_singlet
// direction (π → π + ε I_6) and on p respectively.  If the code treats the
// paired modes symmetrically then f_π₀ = f_p and <Tr[π]> = <p>.  Observation
// shows a ~31× ratio at λ=0.1, so this test measures f_π₀/f_p (and the
// σ/s analogue) on a random gauge + aux config to see which side of the
// equation the asymmetry lives on.
//
// Method:  finite difference on the LogDet-EO action
//   S(U + ε·direction) − S(U − ε·direction)  →  −2ε·f_direction + O(ε³)
// perturbing along four modes:
//   (Tr[σ]-direction)  σ → σ + ε I_6
//   (Tr[π]-direction)  π → π + ε I_6
//   (s)                s → s + ε
//   (p)                p → p + ε
//
// Reports the raw forces and the two symmetry ratios:
//   R_scalar        = f_Trσ / f_s        (should equal 1 if code symmetric,
//                                          and observation says σ/s ≈ same VEV)
//   R_pseudoscalar  = f_Trπ / f_p        (would need to equal ~31 to explain
//                                          the observed <Trπ>/<p> ratio)
//
// LogDet-EO is what enters the EO-Schur RHMC action; the low-λ chains ran
// non-EO (rational full), so LogDet only partially represents the drive.
// But: if the SYMMETRY breaks in LogDet, we know the code has a bug; if
// LogDet is symmetric, the asymmetry must be in the pseudofermion piece
// and we'd need to iterate on the diagnostic to probe there.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <cstdlib>

using namespace Grid;

// Direct single-shift CG for  (M† M) X = Phi  on the full-volume doubled
// operator.  No preconditioning; fine for 4^4 diagnostics.  Returns iteration
// count consumed.
static int SolveMdagM(DTXQCDMOp &Mop, const DTXQCDFermionDoubled &Phi,
                      DTXQCDFermionDoubled &X, RealD tol, int max_iter = 5000) {
  X = Zero();
  DTXQCDFermionDoubled r(Phi.Grid()), p(Phi.Grid());
  DTXQCDFermionDoubled Ap(Phi.Grid()), tmp(Phi.Grid());
  for (int a = 0; a < DtxqcdNf; ++a) {
    r.upper.f[a] = Phi.upper.f[a];  r.lower.f[a] = Phi.lower.f[a];
    p.upper.f[a] = Phi.upper.f[a];  p.lower.f[a] = Phi.lower.f[a];
  }
  RealD r2 = norm2(r), b2 = r2;
  RealD tol2 = tol * tol * b2;
  int k = 0;
  for (; k < max_iter; ++k) {
    Mop.M(p, tmp);
    Mop.Mdag(tmp, Ap);
    RealD pAp = innerProduct(p, Ap).real();
    RealD alpha = r2 / pAp;
    for (int a = 0; a < DtxqcdNf; ++a) {
      X.upper.f[a] = X.upper.f[a] + alpha * p.upper.f[a];
      X.lower.f[a] = X.lower.f[a] + alpha * p.lower.f[a];
      r.upper.f[a] = r.upper.f[a] - alpha * Ap.upper.f[a];
      r.lower.f[a] = r.lower.f[a] - alpha * Ap.lower.f[a];
    }
    RealD r2_new = norm2(r);
    if (r2_new < tol2) { ++k; break; }
    RealD beta = r2_new / r2;
    for (int a = 0; a < DtxqcdNf; ++a) {
      p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
      p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
    }
    r2 = r2_new;
  }
  return k;
}

// S_pf(U) = Phi† (M(U)† M(U))⁻¹ Phi via CG on the doubled Wilson-Clover op.
static RealD PseudofermionAction(const DTXQCDField &U,
                                  RealD mass, RealD csw,
                                  GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                                  const DTXQCDFermionDoubled &Phi,
                                  RealD tol = 1e-10) {
  DTXQCDField &Umut = const_cast<DTXQCDField &>(U);
  DTXQCDWilsonCloverFermionEO Dw(Umut.U, Grid, RBGrid, mass, csw,
                                  Umut.sigma, Umut.pi, Umut.d, Umut.n,
                                  Umut.s, Umut.p);
  DTXQCDMOp Mop(Dw);
  DTXQCDFermionDoubled X(&Grid);
  SolveMdagM(Mop, Phi, X, tol);
  return innerProduct(Phi, X).real();
}

// Build a fresh DTXQCDField equal to `base` except perturbed by +eps along a
// named singlet direction.
enum class Mode { Trsig, s, Trpi, p };

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
  } else if (mode == Mode::Trsig || mode == Mode::Trpi) {
    // Add eps · I_6 to the 6×6 color-flavor matrix at every site.
    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSobj;
    SigSobj shift; shift = Zero();
    for (int a = 0; a < DtxqcdNf; ++a)
      for (int i = 0; i < Nc; ++i)
        shift()(a, a)(i, i) = eps;
    LatticeDtxqcdSigma sig_shift(base.sigma.Grid()); sig_shift = shift;
    if (mode == Mode::Trsig) out.sigma = base.sigma + sig_shift;
    else /* Trpi */          out.pi    = base.pi    + sig_shift;
  }
}

static const char *ModeName(Mode m) {
  switch (m) {
    case Mode::Trsig: return "Tr[σ]-dir (σ → σ + ε·I_6)";
    case Mode::s:     return "s        (s → s + ε)";
    case Mode::Trpi:  return "Tr[π]-dir (π → π + ε·I_6)";
    case Mode::p:     return "p        (p → p + ε)";
  }
  return "?";
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  // Match the production driver: pRNG.SeedFixedIntegers({6,7,8,9,10}) is the
  // final seed reset before the final FillAuxFields (line 268 of
  // production/gen_dtxqcd_cfgs.cc).  Result: the aux state we compute here is
  // BIT-IDENTICAL to what HMC actually starts from — but only at the same
  // lattice size + MPI decomposition as production (RNG stream is per-site).
  GridParallelRNG pRNG(&Grid); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  // Reference config: EXACTLY the recipe the λ=0.1 chain uses to seed
  // the first trajectory.  ThermalAuxConfiguration = weak-field gauge
  // (wf_scale=0.1, chroma WEAK_FIELD default) + FillAuxFields at physical λ.
  // AUX_INIT (Σ_init) matches what the driver's AUX_INIT_AUTO Picard produces
  // on that config; use 3.07 as a placeholder — the equilibrium symmetry
  // question is about the SHAPE of the aux Gaussian around the σ shift, not
  // its absolute value.  We also probe with Σ_init=0 for a P-conservation-
  // exact reference where the fermion drive on the pseudoscalar channel should
  // be zero by symmetry — any nonzero f_Trπ or f_p there would be a bug.
  const RealD lambda      = 0.1;
  const RealD mass        = -0.245;
  const RealD csw         = 1.24930970916466;
  const RealD Sigma_init  = (std::getenv("SIGMA_INIT") ? std::atof(std::getenv("SIGMA_INIT"))
                                                       : 3.07);

  DTXQCDField U(&Grid);
  DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, /*wf_scale=*/0.1);
  DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma_init);

  // Report the reference-config aux VEVs so we know where we're probing from.
  auto aux_avg = [&](const auto &field) {
    return TensorRemove(sum(field)).real() / (RealD)Grid.gSites();
  };
  std::cout << GridLogMessage
            << "ref-config aux averages:  <Trsig>-like from σ diagonals: "
            << TensorRemove(sum(trace(U.sigma))).real() / (RealD)Grid.gSites()
            << "  <Trpi>: "
            << TensorRemove(sum(trace(U.pi))).real() / (RealD)Grid.gSites()
            << "  <s>: " << aux_avg(U.s) << "  <p>: " << aux_avg(U.p)
            << std::endl;
  std::cout << GridLogMessage << "(if code is symmetric, LogDet + PF force on Tr[π] must equal force on p)"
            << std::endl;

  DTXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass, csw);
  LogDet.refresh(U, /*sRNG unused*/*(GridSerialRNG *)nullptr, pRNG);

  auto S = [&](const DTXQCDField &Ux) { return LogDet.S(const_cast<DTXQCDField &>(Ux)); };

  const RealD S0 = S(U);
  std::cout << GridLogMessage << "reference S_logdet(U) = " << std::scientific
            << std::setprecision(12) << S0 << std::endl;
  std::cout << GridLogMessage << "lambda=" << lambda << " mass=" << mass
            << " csw=" << csw << " Σ_init=" << Sigma_init
            << " V4=" << Grid.gSites() << std::endl;

  // eps chosen small enough that FD gives clean linear response, large enough
  // to be well above LogDet numerical noise (~1e-10 relative).
  const RealD eps = 1e-4;
  std::cout << GridLogMessage << "FD eps = " << eps << std::endl;

  auto fd_force = [&](Mode m) {
    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbSingletMode(U, m,  eps, Up);
    PerturbSingletMode(U, m, -eps, Um);
    RealD Sp = S(Up), Sm = S(Um);
    // Force = -∂S/∂direction ≈ -(Sp - Sm)/(2 eps)
    RealD f = -(Sp - Sm) / (2.0 * eps);
    std::cout << GridLogMessage
              << "  " << ModeName(m)
              << ":  S(+ε) − S(−ε) = " << (Sp - Sm)
              << "   f = " << f << std::endl;
    return f;
  };

  std::cout << GridLogMessage << "---- singlet-mode fermion forces ----" << std::endl;
  RealD f_Trsig = fd_force(Mode::Trsig);
  RealD f_s     = fd_force(Mode::s);
  RealD f_Trpi  = fd_force(Mode::Trpi);
  RealD f_p     = fd_force(Mode::p);

  // Symmetry ratios.  If the code treats paired modes identically, these are 1.
  // Deviations directly quantify the fermion-loop asymmetry between the
  // (Tr[σ]/6 + s) singlet and the (Tr[π]/6 + p) singlet directions.
  std::cout << GridLogMessage << "---- symmetry ratios ----" << std::endl;
  std::cout << GridLogMessage
            << "  R_scalar        = f_Trσ / f_s   = " << (f_Trsig / f_s)
            << "     (observed <Trσ>/<s> ≈ 1.001)" << std::endl;
  std::cout << GridLogMessage
            << "  R_pseudoscalar  = f_Trπ / f_p   = " << (f_Trpi / f_p)
            << "     (observed <Trπ>/<p> ≈ 31 at λ=0.1)" << std::endl;

  // Predicted equilibrium ratio from LogDet drive alone:
  //   <Trπ>_eq / <p>_eq = f_Trπ / f_p
  std::cout << GridLogMessage
            << "  <Trπ>/<p> predicted from LogDet drive alone = " << (f_Trpi / f_p)
            << std::endl;

  // ----------------------------------------------------------------------
  // Pseudofermion probe: S_pf(U) = Phi† (M† M)⁻¹ Phi (single-shift proxy for
  // the rational (M†M)^{-1/4} action that drives the non-EO chains).  For a
  // fair estimator, draw Phi = M(U_ref) · η with η Gaussian — that puts Phi
  // in the "equilibrium distribution" for S_pf.
  // ----------------------------------------------------------------------
  std::cout << GridLogMessage << std::endl
            << GridLogMessage << "---- pseudofermion Φ†(M†M)⁻¹Φ probe ----"
            << std::endl;

  // Set up reference Mop and draw Phi = M · η.
  DTXQCDField Umut(&Grid); Umut = U;
  DTXQCDWilsonCloverFermionEO Dw_ref(Umut.U, Grid, RBGrid, mass, csw,
                                      Umut.sigma, Umut.pi, Umut.d, Umut.n,
                                      Umut.s, Umut.p);
  DTXQCDMOp Mop_ref(Dw_ref);
  DTXQCDFermionDoubled eta(&Grid), Phi(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, eta.upper.f[a]);
    gaussian(pRNG, eta.lower.f[a]);
  }
  Mop_ref.M(eta, Phi);
  const RealD S_pf_0 = PseudofermionAction(U, mass, csw, Grid, RBGrid, Phi);
  std::cout << GridLogMessage
            << "reference S_pf(U) = " << std::scientific << std::setprecision(12)
            << S_pf_0 << std::endl;

  const RealD eps_pf = 1e-3;   // larger step since CG tolerance sets a
                               // ~1e-10 noise floor on S_pf
  std::cout << GridLogMessage << "FD eps_pf = " << eps_pf << std::endl;

  auto fd_pf_force = [&](Mode m) {
    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbSingletMode(U, m,  eps_pf, Up);
    PerturbSingletMode(U, m, -eps_pf, Um);
    RealD Sp = PseudofermionAction(Up, mass, csw, Grid, RBGrid, Phi);
    RealD Sm = PseudofermionAction(Um, mass, csw, Grid, RBGrid, Phi);
    RealD f = -(Sp - Sm) / (2.0 * eps_pf);
    std::cout << GridLogMessage
              << "  " << ModeName(m)
              << ":  S(+ε) − S(−ε) = " << (Sp - Sm)
              << "   f_pf = " << f << std::endl;
    return f;
  };

  RealD f_pf_Trsig = fd_pf_force(Mode::Trsig);
  RealD f_pf_s     = fd_pf_force(Mode::s);
  RealD f_pf_Trpi  = fd_pf_force(Mode::Trpi);
  RealD f_pf_p     = fd_pf_force(Mode::p);

  std::cout << GridLogMessage << "---- pseudofermion symmetry ratios ----"
            << std::endl;
  std::cout << GridLogMessage
            << "  R_pf_scalar        = f_pf_Trσ / f_pf_s   = "
            << (f_pf_Trsig / f_pf_s) << std::endl;
  std::cout << GridLogMessage
            << "  R_pf_pseudoscalar  = f_pf_Trπ / f_pf_p   = "
            << (f_pf_Trpi / f_pf_p)
            << "     (need ≈ 31 to explain the observed VEV asymmetry)"
            << std::endl;

  // Combined force (LogDet + PF) is what drives the FULL non-EO PF action's
  // equilibrium in Style-C production.  Report combined ratios for completeness.
  std::cout << GridLogMessage
            << "  R_total_pseudoscalar (LogDet+PF) = "
            << ((f_Trpi + f_pf_Trpi) / (f_p + f_pf_p)) << std::endl;

  Grid_finalize();
  return 0;
}
