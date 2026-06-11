// Test_dtxqcd_wilson_clover_fermion_eo: validate the DTXQCDWilsonCloverFermionEO
// wrapper class against the individually validated operator pieces.
//
// Five end-to-end checks (csw = 1.25, random U / aux / FS built from U):
//   1. Wrapper.M matches the inline "Wilson + Delta + cross + clover"
//      assembly used in Test_dtxqcd_gamma5_herm_full (regression).
//   2. Wrapper.M satisfies gamma_5 M gamma_5 = M^dag, i.e. Wrapper.Mdag
//      matches conj-transpose acting via the gamma_5 identity.
//   3. Wrapper.Mooee matches DtxqcdApplyMooeeDoubled on full-volume input
//      (the wrapper only forwards; this is a regression for the optional
//       clover threading).
//   4. Wrapper.Meooe matches DTXQCDMeooeDoubled at the CB level.
//   5. Mooee * MooeeInv = I on a random fermion (validates the per-site
//      48x48 LU solve path).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_wilson_clover_fermion_eo --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({401, 402, 403, 404});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Random gauge + aux ----------
  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d(&Grid);
  LatticeDtxqcdN     n(&Grid);
  LatticeDtxqcdS     s(&Grid);
  LatticeDtxqcdP     p(&Grid);
  DtxqcdHermitianCFGaussian(pRNG, sigma);
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdRealScalarGaussian(pRNG, s);
  DtxqcdRealScalarGaussian(pRNG, p);
  DtxqcdHermitianCFGaussian(pRNG, d);
  DtxqcdHermitianCFGaussian(pRNG, n);

  const RealD mass = 0.4;
  const RealD csw  = 1.25;

  // Construct the wrapper.
  DTXQCDWilsonCloverFermionEO M_wrap(U, Grid, RBGrid, mass, csw,
                                    sigma, pi, d, n, s, p);

  // ---------- Random doubled fermion ----------
  DTXQCDFermionDoubled v(&Grid), w(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v.upper.f[a]);
    gaussian(pRNG, v.lower.f[a]);
    gaussian(pRNG, w.upper.f[a]);
    gaussian(pRNG, w.lower.f[a]);
  }

  // ---------- 1.  Wrapper.M matches explicit assembly ----------
  {
    DTXQCDFermionDoubled out_wrap(&Grid), out_ref(&Grid);
    M_wrap.M(v, out_wrap);

    // Explicit inline assembly: Wilson + Delta + cross + clover, using the
    // same FS that the wrapper built.
    DTXQCDMeooeDoubled meooe_ref(U, Grid, RBGrid, mass);
    for (int a = 0; a < DtxqcdNf; ++a) {
      meooe_ref.UpperWilson().M(v.upper.f[a], out_ref.upper.f[a]);
      meooe_ref.LowerWilson().M(v.lower.f[a], out_ref.lower.f[a]);
    }
    DTXQCDFermionNf delta_u(&Grid), delta_l(&Grid);
    DtxqcdApplyDeltaDiag     (sigma, pi, s, p, v.upper, delta_u);
    DtxqcdApplyDeltaDiagLower(sigma, pi, s, p, v.lower, delta_l);
    DTXQCDFermionNf cross_u(&Grid), cross_l(&Grid);
    DtxqcdApplyDnCross(d, n, v.lower, cross_u);
    DtxqcdApplyDnCross(d, n, v.upper, cross_l);
    DTXQCDFermionNf clov_u(&Grid), clov_l(&Grid);
    DtxqcdApplyCloverUpper(csw, M_wrap.FieldStrength(), v.upper, clov_u);
    DtxqcdApplyCloverLower(csw, M_wrap.FieldStrength(), v.lower, clov_l);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_ref.upper.f[a] = out_ref.upper.f[a] + delta_u.f[a] + cross_u.f[a]
                         + clov_u.f[a];
      out_ref.lower.f[a] = out_ref.lower.f[a] + delta_l.f[a] + cross_l.f[a]
                         + clov_l.f[a];
    }

    DTXQCDFermionDoubled diff(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      diff.upper.f[a] = out_wrap.upper.f[a] - out_ref.upper.f[a];
      diff.lower.f[a] = out_wrap.lower.f[a] - out_ref.lower.f[a];
    }
    RealD rel = std::sqrt(norm2(diff)
                          / std::max(norm2(out_ref), 1e-30));
    check("Wrapper.M vs explicit assembly (rel)", rel, 1e-13);
  }

  // ---------- 2.  Wrapper gamma_5 Hermiticity via Mdag ----------
  {
    DTXQCDFermionDoubled Mv(&Grid), Mdag_w(&Grid);
    M_wrap.M(v, Mv);
    M_wrap.Mdag(w, Mdag_w);
    // <Mdag w, v> should equal <w, M v>.
    ComplexD A = innerProduct(Mdag_w, v);
    ComplexD B = innerProduct(w, Mv);
    RealD resid = std::abs(A - B);
    RealD ref = std::max({std::abs(A), std::abs(B), 1.0});
    std::cout << GridLogMessage << "<Mdag w, v> = " << A
              << "  <w, M v> = " << B << std::endl;
    check("Wrapper.Mdag identity |<Mdag w, v> - <w, M v>| (rel)",
          resid / ref, 1e-12);
  }

  // ---------- 3.  Wrapper.Mooee matches DtxqcdApplyMooeeDoubled ----------
  {
    DTXQCDFermionDoubled out_wrap(&Grid), out_ref(&Grid);
    M_wrap.Mooee(v, out_wrap);
    DtxqcdApplyMooeeDoubled(mass, sigma, pi, d, n, s, p,
                            v.upper, v.lower, out_ref.upper, out_ref.lower,
                            csw, &M_wrap.FieldStrength());

    DTXQCDFermionDoubled diff(&Grid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      diff.upper.f[a] = out_wrap.upper.f[a] - out_ref.upper.f[a];
      diff.lower.f[a] = out_wrap.lower.f[a] - out_ref.lower.f[a];
    }
    RealD rel = std::sqrt(norm2(diff)
                          / std::max(norm2(out_ref), 1e-30));
    check("Wrapper.Mooee vs DtxqcdApplyMooeeDoubled (rel)", rel, 1e-14);
  }

  // ---------- 4.  Wrapper.Meooe matches DTXQCDMeooeDoubled at CB ----------
  {
    // Build CB-restricted input from the full-volume v.
    DTXQCDFermionDoubled v_e(&RBGrid), out_wrap(&RBGrid), out_ref(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(Even, v_e.upper.f[a], v.upper.f[a]);
      pickCheckerboard(Even, v_e.lower.f[a], v.lower.f[a]);
      out_wrap.upper.f[a].Checkerboard() = Odd;
      out_wrap.lower.f[a].Checkerboard() = Odd;
      out_ref.upper.f[a].Checkerboard()  = Odd;
      out_ref.lower.f[a].Checkerboard()  = Odd;
    }
    M_wrap.Meooe(v_e, out_wrap);

    DTXQCDMeooeDoubled meooe_ref(U, Grid, RBGrid, mass);
    meooe_ref.Meooe(v_e.upper, v_e.lower, out_ref.upper, out_ref.lower);

    RealD worst_rel = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion diff_u(&RBGrid), diff_l(&RBGrid);
      diff_u.Checkerboard() = Odd;
      diff_l.Checkerboard() = Odd;
      diff_u = out_wrap.upper.f[a] - out_ref.upper.f[a];
      diff_l = out_wrap.lower.f[a] - out_ref.lower.f[a];
      RealD ru = std::sqrt(norm2(diff_u))
               / std::max(std::sqrt(norm2(out_ref.upper.f[a])), 1e-30);
      RealD rl = std::sqrt(norm2(diff_l))
               / std::max(std::sqrt(norm2(out_ref.lower.f[a])), 1e-30);
      worst_rel = std::max({worst_rel, ru, rl});
    }
    check("Wrapper.Meooe vs DTXQCDMeooeDoubled (rel)", worst_rel, 1e-14);
  }

  // ---------- 5.  Mooee * MooeeInv = I ----------
  // MooeeInv requires a CB-lattice input (post-Phase-4 EO refactor); peek
  // even half of the random full-volume v into a CB doubled fermion and
  // round-trip there.  Reuses the outer RBGrid -- the EO wrapper stores a
  // pointer-typed reference to its rbgrid_, so a fresh local
  // GridRedBlackCartesian here would fail the assert against the wrapper's
  // stored pointer.
  {
    DTXQCDFermionDoubled v_e(&RBGrid), inv_v(&RBGrid), back(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(Even, v_e.upper.f[a], v.upper.f[a]);
      pickCheckerboard(Even, v_e.lower.f[a], v.lower.f[a]);
    }
    M_wrap.MooeeInv(v_e, inv_v);
    M_wrap.Mooee(inv_v, back);

    DTXQCDFermionDoubled diff(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      diff.upper.f[a] = back.upper.f[a] - v_e.upper.f[a];
      diff.lower.f[a] = back.lower.f[a] - v_e.lower.f[a];
    }
    RealD rel = std::sqrt(norm2(diff)
                          / std::max(norm2(v_e), 1e-30));
    check("Mooee * MooeeInv = I (rel)", rel, 1e-10);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
