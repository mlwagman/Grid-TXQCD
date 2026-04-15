// Test_aux_gaussian: Phase 4b acceptance gate for the TXQCD auxiliary-field
// machinery. Three checks:
//
//   1. Gaussian moments. HotConfiguration samples sigma, pi, s, p, t from
//      unit-variance Hermitian/antisym-Hermitian measure. Analytic moments:
//        <Tr X^2>/site = N^2  (N = Nf for sigma,pi; N = Nc for s,p)
//        <sum_{mu<nu} Tr t_{mu,nu}^2>/site = (Nd*(Nd-1)/2) * Nc^2.
//      Derivation: Hermitian NxN matrix has N real diagonal DOFs with
//      variance 1 and N(N-1)/2 complex off-diagonal DOFs with real and imag
//      parts each of variance 1/2. Expanding Tr(X^2) = sum_i X_ii^2 +
//      sum_{i!=j} |X_ij|^2 gives N*1 + 2*(N(N-1)/2)*1 = N^2.
//
//   2. S and deriv consistency for AuxiliaryFieldGaussianAction. With S =
//      (lambda^2/2) * (sum of Hermitian and tensor square norms), the
//      gradient wrt an independent real DOF c_k in X is lambda^2 * c_k, so
//      a perturbation X -> X + h*Y gives dS = lambda^2 * h * sum_k c_k(X)
//      c_k(Y). When Y = X that is exactly 2*S(X)/h. Numerically we check
//      the central finite difference matches <deriv(U), Y> where the inner
//      product is the component-wise sum of trace(X Y) etc. matching the
//      derivative's metric.
//
//   3. Leapfrog reversibility. Evolve (U, P) forward by N MD steps and then
//      reverse P -> -P and evolve another N steps. Should return to the
//      starting (U, -P) within FP tolerance. Exercises the composite
//      update_P / update_U paths end to end on a trivial action.

#include <Grid/Grid.h>

using namespace Grid;

// Real-valued "inner product" used for the finite-difference force check.
// Matches the metric implied by (lambda^2/2) Tr X^2: each aux lattice
// contributes Re Tr(A*B) summed over sites, with the tensor piece summed
// over mu<nu to mirror TensorFieldSquareNorm.
template <class LatticeMat>
RealD HermInner(LatticeMat &A, LatticeMat &B) {
  return TensorRemove(sum(trace(A * B))).real();
}

RealD TensorInner(LatticeTField &A, LatticeTField &B) {
  autoView(Av, A, CpuRead);
  autoView(Bv, B, CpuRead);
  GridBase *grid = A.Grid();
  RealD total = 0.0;
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        auto Am = Av[ss]()(mu, nu);
        auto Bm = Bv[ss]()(mu, nu);
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            total += real(Reduce(Am(i, j) * Bm(j, i)));
      }
    }
  });
  grid->GlobalSum(total);
  return total;
}

RealD CompositeInner(TXQCDField &A, TXQCDField &B) {
  return HermInner(A.sigma, B.sigma) + HermInner(A.pi, B.pi)
       + HermInner(A.s, B.s) + HermInner(A.p, B.p)
       + TensorInner(A.t, B.t);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  GridSerialRNG sRNG;
  sRNG.SeedFixedIntegers({5, 6, 7, 8});

  RealD vol = Grid.gSites();

  int exitcode = 0;
  auto check = [&](const char *name, RealD measured, RealD expected,
                   RealD tol) {
    RealD rel = std::abs(measured - expected) /
                std::max(std::abs(expected), 1.0);
    std::cout << GridLogMessage << "[check] " << name << " : measured "
              << measured << " expected " << expected << " rel " << rel
              << (rel < tol ? "  PASS" : "  FAIL") << std::endl;
    if (rel >= tol) exitcode = 1;
  };

  // ---------- 1. Gaussian moments of HotConfiguration ----------
  //
  // Average <Tr X^2>/site over Nsamp independent HotConfigurations.
  const int Nsamp = 64;
  RealD mean_sigma = 0, mean_pi = 0, mean_s = 0, mean_p = 0, mean_t = 0;
  for (int k = 0; k < Nsamp; ++k) {
    TXQCDField U(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    mean_sigma += HermitianFieldSquareNorm(U.sigma);
    mean_pi    += HermitianFieldSquareNorm(U.pi);
    mean_s     += HermitianFieldSquareNorm(U.s);
    mean_p     += HermitianFieldSquareNorm(U.p);
    mean_t     += TensorFieldSquareNorm(U.t);
  }
  mean_sigma /= (Nsamp * vol);
  mean_pi    /= (Nsamp * vol);
  mean_s     /= (Nsamp * vol);
  mean_p     /= (Nsamp * vol);
  mean_t     /= (Nsamp * vol);

  const int Nf = TxqcdNf;
  const RealD expected_flavor = RealD(Nf * Nf);
  const RealD expected_color  = RealD(Nc * Nc);
  const RealD expected_tensor = RealD((Nd * (Nd - 1) / 2) * Nc * Nc);
  // Moments converge at 1/sqrt(Nsamp * #DOF * vol). 15% gives headroom.
  RealD tol_moment = 0.15;
  check("<Tr sigma^2>/site", mean_sigma, expected_flavor, tol_moment);
  check("<Tr pi^2>/site",    mean_pi,    expected_flavor, tol_moment);
  check("<Tr s^2>/site",     mean_s,     expected_color,  tol_moment);
  check("<Tr p^2>/site",     mean_p,     expected_color,  tol_moment);
  check("<Tr t^2>/site",     mean_t,     expected_tensor, tol_moment);

  // ---------- 2. Finite-difference check of S and deriv ----------
  //
  // For S = (lambda^2/2) * |U|^2 with |.|^2 as TXQCDCompositeImpl norm,
  //   S(U + h*Y) - S(U - h*Y) = 2 h * <dS/dU, Y> + O(h^3)
  // where <., .> is the metric induced by the same norm. We compute the
  // LHS explicitly and the RHS via AuxGaussianAction::deriv and compare.
  {
    RealD lambda = 1.3;
    AuxiliaryFieldGaussianAction Saux(lambda);

    TXQCDField U(&Grid), Y(&Grid), Up(&Grid), Um(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    TXQCDCompositeImpl::HotConfiguration(pRNG, Y);
    // Only aux slots matter; deriv zeroes gauge. Zero the gauge perturbation
    // to keep the inner-product exact.
    Y.U = Zero();
    U.U = Zero();

    RealD h = 1e-3;
    Up = U; Up.sigma = U.sigma + h*Y.sigma; Up.pi = U.pi + h*Y.pi;
    Up.s = U.s + h*Y.s; Up.p = U.p + h*Y.p; Up.t = U.t + h*Y.t;
    Um = U; Um.sigma = U.sigma - h*Y.sigma; Um.pi = U.pi - h*Y.pi;
    Um.s = U.s - h*Y.s; Um.p = U.p - h*Y.p; Um.t = U.t - h*Y.t;

    RealD num_deriv = (Saux.S(Up) - Saux.S(Um)) / (2.0 * h);
    Saux.deriv(U, dS);
    RealD ana_deriv = CompositeInner(dS, Y);
    check("AuxGaussianAction deriv vs FD", num_deriv, ana_deriv, 1e-4);
  }

  // ---------- 3. Leapfrog reversibility on the aux Gaussian action ----------
  //
  // H(U,P) = |P|^2/2 + S_aux(U). Update P half-step, U full-step, P half-step
  // per MD step. Evolve N steps; reverse momentum; evolve N steps; should
  // return to (U0, -P0) up to FP error. Check per-component L2 residual.
  {
    RealD lambda = 0.7;
    AuxiliaryFieldGaussianAction Saux(lambda);

    TXQCDField U(&Grid), U0(&Grid), P(&Grid), P0(&Grid), F(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    U.U = Zero();                 // no gauge dynamics
    TXQCDCompositeImpl::generate_momenta(P, sRNG, pRNG);
    P.U = Zero();
    U0 = U; P0 = P;

    int Nmd = 20;
    RealD dt = 0.05;

    auto step_P = [&](RealD ep) {
      Saux.deriv(U, F);
      F = TXQCDCompositeImpl::projectForce(F);
      P.sigma = P.sigma - F.sigma * ep;
      P.pi    = P.pi    - F.pi    * ep;
      P.s     = P.s     - F.s     * ep;
      P.p     = P.p     - F.p     * ep;
      P.t     = P.t     - F.t     * ep;
    };
    auto step_U = [&](RealD ep) {
      U.sigma = U.sigma + P.sigma * ep;
      U.pi    = U.pi    + P.pi    * ep;
      U.s     = U.s     + P.s     * ep;
      U.p     = U.p     + P.p     * ep;
      U.t     = U.t     + P.t     * ep;
    };
    auto leapfrog = [&](int N) {
      for (int i = 0; i < N; ++i) {
        step_P(0.5 * dt);
        step_U(dt);
        step_P(0.5 * dt);
      }
    };

    RealD H0 = 0.5 * TXQCDCompositeImpl::FieldSquareNorm(P) + Saux.S(U);
    leapfrog(Nmd);
    RealD H1 = 0.5 * TXQCDCompositeImpl::FieldSquareNorm(P) + Saux.S(U);
    // Reverse.
    P.sigma = -P.sigma; P.pi = -P.pi; P.s = -P.s; P.p = -P.p; P.t = -P.t;
    leapfrog(Nmd);
    // After reverse P should match -P0, U should match U0.
    TXQCDField dU(&Grid), dP(&Grid);
    dU.sigma = U.sigma - U0.sigma;
    dU.pi    = U.pi    - U0.pi;
    dU.s     = U.s     - U0.s;
    dU.p     = U.p     - U0.p;
    dU.t     = U.t     - U0.t;
    dU.U     = Zero();
    dP.sigma = P.sigma + P0.sigma;
    dP.pi    = P.pi    + P0.pi;
    dP.s     = P.s     + P0.s;
    dP.p     = P.p     + P0.p;
    dP.t     = P.t     + P0.t;
    dP.U     = Zero();
    RealD resU = std::sqrt(TXQCDCompositeImpl::FieldSquareNorm(dU));
    RealD resP = std::sqrt(TXQCDCompositeImpl::FieldSquareNorm(dP));
    std::cout << GridLogMessage
              << "Leapfrog reversibility: |dU| = " << resU
              << "  |dP| = " << resP
              << "  dH_fwd = " << (H1 - H0) << std::endl;
    if (resU > 1e-8 || resP > 1e-8) {
      std::cout << GridLogError
                << "Reversibility residual exceeds 1e-8 tolerance" << std::endl;
      exitcode = 1;
    }
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
