// Test_dtxqcd_logdet_aux_force: finite-difference force test for the
// aux-field derivatives of DTXQCDLogDetCloverEOAction.
//
// For a random perturbation direction Y (across all aux fields), the
// directional derivative
//     <dS/dU, Y>    (analytic, from deriv())
// must match the symmetric finite difference
//     (S(U + h Y) - S(U - h Y)) / (2 h)   (numerical)
// to O(h^2).  With h ~ 1e-3 we expect ~8-digit agreement on a 4^4 lattice.
//
// Gauge force is masked off (Y.U = 0) in this test since DTXQCDLogDetCloverEOAction
// has its gauge clover deriv still TODO.  Aux forces (sigma^A, pi^A, t^A,
// d, n) are validated here.
//
// Two runs: csw = 0 (no clover) and csw = 1.25 (with clover, but only aux
// FD is checked; clover-via-U gauge force deferred).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_logdet_aux_force --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h>

using namespace Grid;

// Inner product matching the directional-derivative convention used by the
// analytic deriv(): Sigma_x Sigma_{indep DOFs} dS/dDOF[x] * Y[x].
//
// sigma^A, pi^A, t^A_{mu,nu}: real-valued DOFs stored as iVector<vComplex,3>
// (with imag=0 by construction).  localInnerProduct gives Sum conj(F) Y
// which on real Y reduces to Re(F)*Re(Y) -- matches directional derivative.
//
// d, n: HERMITIAN complex color matrices.  Treating d_{ij} as independent
// complex (Wirtinger), directional derivative = Sum F_{ij} Y_{ij}_full =
// v2 AuxInnerReal: CF Hermitian fields use Re Tr(F * Y^T) (natural-Wirtinger);
// scalar singlets use plain localInnerProduct (real DOF in real part).
static RealD AuxInnerReal(const DTXQCDField &A, const DTXQCDField &B) {
  RealD r = 0.0;
  r += TensorRemove(sum(trace(A.sigma * adj(B.sigma)))).real();
  r += TensorRemove(sum(trace(A.pi    * adj(B.pi))))   .real();
  r += TensorRemove(sum(trace(A.d     * adj(B.d))))    .real();
  r += TensorRemove(sum(trace(A.n     * adj(B.n))))    .real();
  r += TensorRemove(sum(localInnerProduct(A.s, B.s))).real();
  r += TensorRemove(sum(localInnerProduct(A.p, B.p))).real();
  return r;
}

// Build perturbed U' = U + scale * Y on aux components only (U.U unchanged).
static void PerturbAux(const DTXQCDField &U, const DTXQCDField &Y, RealD scale,
                       DTXQCDField &out) {
  out.U     = U.U;
  out.sigma = U.sigma + scale * Y.sigma;
  out.pi    = U.pi    + scale * Y.pi;
  out.d     = U.d     + scale * Y.d;
  out.n     = U.n     + scale * Y.n;
  out.s     = U.s     + scale * Y.s;
  out.p     = U.p     + scale * Y.p;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt = GridDefaultLatt();              // honor --grid (e.g. 16.16.16.48)
  if (latt.size() == 0) latt = Coordinate(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({501, 502, 503, 504});

  int exitcode = 0;
  auto check = [&](const char *name, RealD num, RealD ana, RealD tol) {
    RealD rel = std::abs(num - ana)
              / std::max({std::abs(num), std::abs(ana), 1.0});
    bool ok = (rel < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " :  numeric = " << num << "  analytic = " << ana
              << "  rel = " << rel << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Random U + aux + Y direction ----------
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  DtxqcdHermitianCFGaussian(pRNG, U.sigma);
  DtxqcdHermitianCFGaussian(pRNG, U.pi);
  DtxqcdHermitianCFGaussian(pRNG, U.d);
  DtxqcdHermitianCFGaussian(pRNG, U.n);
  DtxqcdRealScalarGaussian(pRNG, U.s);
  DtxqcdRealScalarGaussian(pRNG, U.p);

  // Random aux-only perturbation direction Y.
  DTXQCDField Y(&Grid);
  Y.U = Zero();
  DtxqcdHermitianCFGaussian(pRNG, Y.sigma);
  DtxqcdHermitianCFGaussian(pRNG, Y.pi);
  DtxqcdHermitianCFGaussian(pRNG, Y.d);
  DtxqcdHermitianCFGaussian(pRNG, Y.n);
  DtxqcdRealScalarGaussian(pRNG, Y.s);
  DtxqcdRealScalarGaussian(pRNG, Y.p);

  // Apply env-gated alternative projections (must match the HMC subspace
  // that the force kernel was derived for; otherwise FD samples a
  // different manifold than the analytic force gradient).
  // Under DN_COMPLEX_SYMMETRIC: σ/π real-symm; d/n re-generated as truly
  // complex-symmetric (raw Gaussian + complex-symm proj — keeps imag DOFs).
  // Under SIGMA_PI_HERMITIAN_ONLY: σ/π stay Hermitian; d/n still truly
  // complex-symmetric.
  if (DtxqcdDnComplexSymmetric()) {
    if (!DtxqcdSigmaPiHermitianOnly()) {
      DtxqcdRealSymmetricCFInPlace(U.sigma);
      DtxqcdRealSymmetricCFInPlace(U.pi);
      DtxqcdRealSymmetricCFInPlace(Y.sigma);
      DtxqcdRealSymmetricCFInPlace(Y.pi);
    }
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, U.n);
    DtxqcdComplexSymmetricCFGaussian(pRNG, Y.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, Y.n);
  }

  // AUX_SCALE: shrink the (variance-1) aux toward the production fluctuation
  // (~1/lambda).  Large random aux makes M_ee near-singular at large volume,
  // which inflates the aux-force FD; AUX_SCALE<1 isolates a real force-formula
  // bug from that conditioning artifact.
  if (const char *v = std::getenv("AUX_SCALE"); v && *v) {
    RealD as = std::atof(v);
    U.sigma = as * U.sigma; U.pi = as * U.pi; U.d = as * U.d;
    U.n = as * U.n;         U.s = as * U.s;   U.p = as * U.p;
  }

  const RealD mass = 0.4;
  // FD step: 1e-6 is the v1 value.  The Mooee +4 normalization bug fix
  // grew |S| ~ 35x and |dS/dh| in the all-aux directions accordingly, so
  // h^2 truncation noise crossed the 1e-6 tol on the csw=1.25 all-aux
  // direction.  1e-7 keeps truncation in the noise floor (CG-free LogDet
  // means no lower limit from solver tol).
  const RealD h    = 5e-8;

  // ---------- csw = 0 -- per-piece breakdown ----------
  DTXQCDLogDetCloverEOAction action(Grid, RBGrid, mass, /*csw=*/0.0);
  DTXQCDField dSdU(&Grid);
  action.deriv(U, dSdU);

  auto check_piece = [&](const char *name, std::function<void(DTXQCDField&)> zero_others) {
    DTXQCDField Y_piece(&Grid);
    Y_piece = Zero();
    Y_piece.sigma = Y.sigma;  Y_piece.pi = Y.pi;
    Y_piece.d     = Y.d;      Y_piece.n  = Y.n;
    Y_piece.s     = Y.s;      Y_piece.p  = Y.p;
    Y_piece.U = Zero();
    zero_others(Y_piece);

    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbAux(U, Y_piece, +h, Up);
    PerturbAux(U, Y_piece, -h, Um);
    RealD num = (action.S(Up) - action.S(Um)) / (2.0 * h);
    RealD ana = AuxInnerReal(dSdU, Y_piece);
    check(name, num, ana, 1e-6);
  };

  check_piece("csw=0 sigma-only", [](DTXQCDField &P) {
    P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 pi-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 d-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.n = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 n-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.s = Zero(); P.p = Zero();
  });
  check_piece("csw=0 s-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.p = Zero();
  });
  check_piece("csw=0 p-only", [](DTXQCDField &P) {
    P.sigma = Zero(); P.pi = Zero(); P.d = Zero(); P.n = Zero(); P.s = Zero();
  });
  check_piece("csw=0 all-aux", [](DTXQCDField &) {});

  // ---------- csw = 1.25 all-aux ----------
  DTXQCDLogDetCloverEOAction action2(Grid, RBGrid, mass, /*csw=*/1.25);
  DTXQCDField dSdU2(&Grid);
  action2.deriv(U, dSdU2);
  {
    DTXQCDField Up(&Grid), Um(&Grid);
    PerturbAux(U, Y, +h, Up);
    PerturbAux(U, Y, -h, Um);
    RealD num = (action2.S(Up) - action2.S(Um)) / (2.0 * h);
    RealD ana = AuxInnerReal(dSdU2, Y);
    check("csw=1.25 all-aux FD vs analytic", num, ana, 1e-6);
  }

  // ---------- csw = 1.25 gauge clover force FD ----------
  // Perturb U_mu -> exp(h * E_mu) U_mu with E_mu in the su(N) Lie algebra.
  // deriv() returns gauge force in Convention A (half gradient), so the FD
  // relationship is dS/dh = -2 * Re Tr(E * F_A) summed over Lorentz.
  // (Mirrors TXQCDLogDetCloverEOAction's gauge-FD pattern.)
  {
    std::array<LatticeColourMatrix, 4> Emu{
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
        LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid)};
    for (int mu = 0; mu < Nd; ++mu)
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

    RealD ana = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU2.U, mu);
      ana += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    ana *= -2.0;

    LatticeGaugeField Usaved = U.U;

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu  = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], h, 12);
      PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
    }
    RealD Sp = action2.S(U);

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu  = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], -h, 12);
      PokeIndex<LorentzIndex>(U.U, expE * Umu, mu);
    }
    RealD Sm = action2.S(U);

    U.U = Usaved;

    RealD fd = (Sp - Sm) / (2.0 * h);
    check("csw=1.25 gauge clover FD vs analytic", fd, ana, 1e-6);

    RealD nU = std::sqrt(norm2(dSdU2.U));
    std::cout << GridLogMessage << "|dSdU.U| = " << nU
              << " (clover gauge force; should be nonzero)" << std::endl;
    if (nU < 1e-10) {
      std::cout << GridLogError << "[FAIL] gauge force vanishes" << std::endl;
      exitcode = 1;
    }
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
