// Test_dtxqcd_gamma5_herm_full: end-to-end gamma_5-Hermiticity check on the
// full doubled M operator (mass + clover + Delta_diag + d,n cross + Wilson
// hopping per block), with random U, aux fields, and clover field strength.
//
// With the Cstar M_22 = C^T D^T C construction (and same for X^A), each block
// satisfies gamma_5 (block) gamma_5 = (block)^dag individually:
//   - Upper:    gamma_5 (D_qcd + Delta_diag) gamma_5 = (D_qcd + Delta_diag)^dag
//   - Lower:    gamma_5 (C^T D_qcd^T C + Delta_diag_lower) gamma_5 = same^dag
//                (using gamma_5 commutes with C and the lower-block Cstar
//                 transposes cancel the Hermitian-conjugate)
//   - Off-diag: gamma_5 (2 d gamma_5 + 2 n) gamma_5 = 2 d gamma_5 + 2 n (Hermitian)
//
// So the doubled operator satisfies just Gamma_5 D Gamma_5 = D^dag with
// Gamma_5 = gamma_5 (x) I_2 (identical gamma_5 on upper and lower).  No
// extra C wrapping needed (the paper's Eq. 314 "C gamma_5 D gamma_5 C = D^dag"
// was for the old construction with the missing D^T transpose).
//
// Verification: for any random doubled fermions v, w,
//   <w, gamma_5 M gamma_5 v> = <M w, v>   (using inner product duality
//                                          <w, M^dag v> = <M w, v>)
// equivalently
//   <gamma_5 w, M gamma_5 v> = <M w, v>   (gamma_5 self-adjoint)
//
// Run: ./tests/dtxqcd/Test_dtxqcd_gamma5_herm_full --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>

using namespace Grid;

// Apply the full doubled M to a doubled fermion at the full volume:
//   out_upper = WilsonFermion(U).M(in_upper)  // mass + hopping per flavor
//             + Delta_diag(in_upper)
//             + (2 d gamma_5 + 2 n) in_lower
//             + (csw && FS ? -(csw/2) F sigma in_upper : 0)
//   out_lower = WilsonFermion(U*).M(in_lower)  // mass + Cstar hopping
//             + Delta_diag_lower(in_lower)
//             + (2 d gamma_5 + 2 n) in_upper
//             + (csw && FS ? +(csw/2) F^T sigma in_lower : 0)
static void ApplyMDoubled(
    DTXQCDMeooeDoubled &meooe,
    const LatticeDtxqcdSigma &sigma,
    const LatticeDtxqcdPi &pi,
    const LatticeDtxqcdT &t,
    const LatticeDtxqcdD &d,
    const LatticeDtxqcdN &n,
    double csw,
    const std::vector<LatticeColourMatrix> *FS,
    const DTXQCDFermionNf &in_upper,
    const DTXQCDFermionNf &in_lower,
    DTXQCDFermionNf &out_upper,
    DTXQCDFermionNf &out_lower) {
  GridBase *grid = in_upper.Grid();

  // WilsonFermion::M applies mass + hopping per flavor.
  for (int a = 0; a < DtxqcdNf; ++a) {
    meooe.UpperWilson().M(in_upper.f[a], out_upper.f[a]);
    meooe.LowerWilson().M(in_lower.f[a], out_lower.f[a]);
  }

  // Delta_diag insertion: upper uses standard, lower uses Cstar (tensor sign-flipped).
  DTXQCDFermionNf delta_upper(grid), delta_lower(grid);
  DtxqcdApplyDeltaDiag(sigma, pi, t, in_upper, delta_upper);
  DtxqcdApplyDeltaDiagLower(sigma, pi, t, in_lower, delta_lower);

  // Off-diagonal d, n cross term (couples upper and lower).
  DTXQCDFermionNf cross_upper(grid), cross_lower(grid);
  DtxqcdApplyDnCross(d, n, in_lower, cross_upper);
  DtxqcdApplyDnCross(d, n, in_upper, cross_lower);

  // Combine the pieces so far.
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_upper.f[a] = out_upper.f[a] + delta_upper.f[a] + cross_upper.f[a];
    out_lower.f[a] = out_lower.f[a] + delta_lower.f[a] + cross_lower.f[a];
  }

  // Optional clover.
  if (csw != 0.0 && FS != nullptr) {
    DTXQCDFermionNf clov_upper(grid), clov_lower(grid);
    DtxqcdApplyCloverUpper(csw, *FS, in_upper, clov_upper);
    DtxqcdApplyCloverLower(csw, *FS, in_lower, clov_lower);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out_upper.f[a] = out_upper.f[a] + clov_upper.f[a];
      out_lower.f[a] = out_lower.f[a] + clov_lower.f[a];
    }
  }
}

static void ApplyGamma5(const DTXQCDFermionNf &in, DTXQCDFermionNf &out) {
  Gamma g5(Gamma::Algebra::Gamma5);
  for (int a = 0; a < DtxqcdNf; ++a) out.f[a] = g5 * in.f[a];
}

static ComplexD DoubledInner(const DTXQCDFermionNf &au, const DTXQCDFermionNf &al,
                             const DTXQCDFermionNf &bu, const DTXQCDFermionNf &bl) {
  return innerProduct(au, bu) + innerProduct(al, bl);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({301, 302, 303, 304});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Build random gauge, aux, FS ----------
  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdT     t(&Grid);
  LatticeDtxqcdD     d(&Grid);
  LatticeDtxqcdN     n(&Grid);
  DtxqcdRealGaussian(pRNG, sigma);
  DtxqcdRealGaussian(pRNG, pi);
  DtxqcdGaussianAntisymTensor(pRNG, t);
  DtxqcdHermitianGaussian(pRNG, d);
  DtxqcdHermitianGaussian(pRNG, n);

  std::vector<LatticeColourMatrix> FS;
  FS.reserve(6);
  for (int k = 0; k < 6; ++k) {
    LatticeColourMatrix X(&Grid);
    gaussian(pRNG, X);
    LatticeColourMatrix F(&Grid);
    F = X - adj(X);
    FS.push_back(std::move(F));
  }

  const RealD mass = 0.4;
  const RealD csw  = 1.25;

  DTXQCDMeooeDoubled meooe(U, Grid, RBGrid, mass);

  // ---------- Random doubled fermion test pair ----------
  DTXQCDFermionNf v_u(&Grid), v_l(&Grid), w_u(&Grid), w_l(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, v_u.f[a]);
    gaussian(pRNG, v_l.f[a]);
    gaussian(pRNG, w_u.f[a]);
    gaussian(pRNG, w_l.f[a]);
  }

  // Case A: with clover off (csw = 0) — isolates the Mooee + Meooe + cross pieces.
  {
    // gamma_5 v
    DTXQCDFermionNf g5_v_u(&Grid), g5_v_l(&Grid);
    ApplyGamma5(v_u, g5_v_u);
    ApplyGamma5(v_l, g5_v_l);
    // M (gamma_5 v)
    DTXQCDFermionNf M_g5_v_u(&Grid), M_g5_v_l(&Grid);
    ApplyMDoubled(meooe, sigma, pi, t, d, n, 0.0, nullptr,
                  g5_v_u, g5_v_l, M_g5_v_u, M_g5_v_l);
    // gamma_5 (M (gamma_5 v)) = gamma_5 M gamma_5 v
    DTXQCDFermionNf g5_M_g5_v_u(&Grid), g5_M_g5_v_l(&Grid);
    ApplyGamma5(M_g5_v_u, g5_M_g5_v_u);
    ApplyGamma5(M_g5_v_l, g5_M_g5_v_l);
    // M w
    DTXQCDFermionNf M_w_u(&Grid), M_w_l(&Grid);
    ApplyMDoubled(meooe, sigma, pi, t, d, n, 0.0, nullptr,
                  w_u, w_l, M_w_u, M_w_l);

    // LHS = <w, gamma_5 M gamma_5 v>
    ComplexD LHS = DoubledInner(w_u, w_l, g5_M_g5_v_u, g5_M_g5_v_l);
    // RHS = <M w, v>     (= <w, M^dag v>)
    ComplexD RHS = DoubledInner(M_w_u, M_w_l, v_u, v_l);
    RealD resid = std::abs(LHS - RHS);
    RealD ref   = std::max({std::abs(LHS), std::abs(RHS), 1.0});
    std::cout << GridLogMessage
              << "csw=0:  <w, g5 M g5 v> = " << LHS
              << "  <M w, v> = " << RHS << std::endl;
    check("csw=0:  gamma_5 M gamma_5 = M^dag (rel)", resid / ref, 1e-12);
  }

  // Case B: with clover on (csw = 1.25) — full operator including clover.
  {
    DTXQCDFermionNf g5_v_u(&Grid), g5_v_l(&Grid);
    ApplyGamma5(v_u, g5_v_u);
    ApplyGamma5(v_l, g5_v_l);
    DTXQCDFermionNf M_g5_v_u(&Grid), M_g5_v_l(&Grid);
    ApplyMDoubled(meooe, sigma, pi, t, d, n, csw, &FS,
                  g5_v_u, g5_v_l, M_g5_v_u, M_g5_v_l);
    DTXQCDFermionNf g5_M_g5_v_u(&Grid), g5_M_g5_v_l(&Grid);
    ApplyGamma5(M_g5_v_u, g5_M_g5_v_u);
    ApplyGamma5(M_g5_v_l, g5_M_g5_v_l);
    DTXQCDFermionNf M_w_u(&Grid), M_w_l(&Grid);
    ApplyMDoubled(meooe, sigma, pi, t, d, n, csw, &FS,
                  w_u, w_l, M_w_u, M_w_l);

    ComplexD LHS = DoubledInner(w_u, w_l, g5_M_g5_v_u, g5_M_g5_v_l);
    ComplexD RHS = DoubledInner(M_w_u, M_w_l, v_u, v_l);
    RealD resid = std::abs(LHS - RHS);
    RealD ref   = std::max({std::abs(LHS), std::abs(RHS), 1.0});
    std::cout << GridLogMessage
              << "csw=1.25:  <w, g5 M g5 v> = " << LHS
              << "  <M w, v> = " << RHS << std::endl;
    check("csw=1.25: gamma_5 M gamma_5 = M^dag (rel)", resid / ref, 1e-12);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
