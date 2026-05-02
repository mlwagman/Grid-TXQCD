// Verify that the mu parameter is exactly a field redefinition.
//
// Under sigma' = mu * sigma (and similarly for pi, s, p, t), the TXQCD action
// S = ψ̄ (M_W + mu*Delta(aux)) ψ + (lambda^2/2) Tr(aux^2)
// transforms as
//   S_aux = (lambda^2/2) Tr(sigma^2) = (lambda^2/(2*mu^2)) Tr(sigma'^2)
//          = (lambda'^2/2) Tr(sigma'^2)            with lambda' = lambda/mu
//   S_pf  = ψ̄ (M_W + mu*Delta(aux)) ψ = ψ̄ (M_W + Delta(aux')) ψ
// So (lambda, mu) is exactly equivalent to (lambda*mu, mu) <-> (lambda, 1)
// under aux -> aux*mu (or, dually, (lambda*mu, mu, aux/mu) === (lambda, 1, aux)).
//
// This test computes the total TXQCD action at two parameterisations of the
// SAME physical theory and verifies they agree to machine precision:
//   Setup A:  lambda_a = L,   mu_a = 1,   aux_a = aux_baseline
//   Setup B:  lambda_b = M*L, mu_b = M,   aux_b = aux_baseline / M
// for an arbitrary scaling M.  S_aux and S_pf must each match in setups A
// and B; the gauge action is independent of aux/mu so always matches.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetEOAction.h>
#include <Grid/qcd/action/txqcd/AuxGaussianAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG sRNG;
  pRNG.SeedFixedIntegers({11, 12, 13, 14});
  sRNG.SeedFixedIntegers({11, 12, 13, 14});

  // Reference (baseline) field configuration.
  TXQCDField U_A(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U_A.U);
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG, U_A.sigma); U_A.sigma = aux_scale * U_A.sigma;
  HermitianGaussian(pRNG, U_A.pi);    U_A.pi    = aux_scale * U_A.pi;
  HermitianGaussian(pRNG, U_A.s);     U_A.s     = aux_scale * U_A.s;
  HermitianGaussian(pRNG, U_A.p);     U_A.p     = aux_scale * U_A.p;
  GaussianAntisymTensor(pRNG, U_A.t); U_A.t     = aux_scale * U_A.t;

  // Test parameters.
  const RealD mass    = 0.3;
  const RealD lambda  = 4.0;     // baseline lambda
  const RealD M       = 2.0;     // mu rescale factor
  const RealD lambda_B = lambda * M;  // = 8.0
  const RealD mu_A    = 1.0;
  const RealD mu_B    = M;

  // ---- Aux Gaussian action ----
  // S_aux(lambda) = (lambda^2/2) sum [Tr sigma^2 + Tr pi^2 + Tr s^2 + Tr p^2
  //                                 + sum_{mu<nu} Tr t_{mu,nu}^2]
  AuxiliaryFieldGaussianAction aux_A(lambda);
  AuxiliaryFieldGaussianAction aux_B(lambda_B);

  // Setup B aux: aux_B = aux_A / M
  TXQCDField U_B(&Grid);
  U_B.U = U_A.U;
  U_B.sigma = (1.0 / M) * U_A.sigma;
  U_B.pi    = (1.0 / M) * U_A.pi;
  U_B.s     = (1.0 / M) * U_A.s;
  U_B.p     = (1.0 / M) * U_A.p;
  U_B.t     = (1.0 / M) * U_A.t;

  RealD S_aux_A = aux_A.S(U_A);
  RealD S_aux_B = aux_B.S(U_B);

  std::cout << GridLogMessage << "[A] lambda=" << lambda << " mu=" << mu_A
            << "  S_aux=" << S_aux_A << std::endl;
  std::cout << GridLogMessage << "[B] lambda=" << lambda_B << " mu=" << mu_B
            << "  S_aux=" << S_aux_B << std::endl;

  RealD rel_aux = std::abs(S_aux_A - S_aux_B) / std::max(std::abs(S_aux_A), 1.0);
  bool aux_match = rel_aux < 1e-12;
  std::cout << GridLogMessage << "[aux equiv] |dS|/|S| = " << rel_aux
            << (aux_match ? "  PASS" : "  FAIL") << std::endl;

  // ---- Pseudofermion action ----
  // M_TXQCD = M_W + mu * Delta(aux). Under aux_B = aux_A/M and mu_B = M*mu_A,
  //   M_B = M_W + M*1 * Delta(aux_A/M) = M_W + Delta(aux_A) = M_A.
  // So a fixed pseudofermion Phi gives the same S_pf in both setups.

  // Build operator A (lambda doesn't enter operator; mu does).
  OneFlavourRationalParams rat;
  rat.lo = 1e-3; rat.hi = 30.0; rat.precision = 64;
  rat.degree = 12; rat.tolerance = 1e-12; rat.MaxIter = 50000;
  rat.mdtolerance = 1e-12;
  TXQCDWilsonRationalEOAction action_A(Grid, RBGrid, mass, rat, mu_A);
  TXQCDWilsonRationalEOAction action_B(Grid, RBGrid, mass, rat, mu_B);

  // Heatbath setup A: produces a Phi.  Then COPY this Phi to action_B for
  // a deterministic comparison (same fermion source so any S difference is
  // operator-induced).
  action_A.refresh(U_A, sRNG, pRNG);
  action_B.PseudoFermion() = action_A.PseudoFermion();

  RealD S_pf_A = action_A.S(U_A);
  RealD S_pf_B = action_B.S(U_B);

  std::cout << GridLogMessage << "[A] S_pf=" << S_pf_A << std::endl;
  std::cout << GridLogMessage << "[B] S_pf=" << S_pf_B << std::endl;

  RealD rel_pf = std::abs(S_pf_A - S_pf_B) / std::max(std::abs(S_pf_A), 1.0);
  bool pf_match = rel_pf < 1e-9;  // CG tolerance is the floor here
  std::cout << GridLogMessage << "[pf equiv] |dS|/|S| = " << rel_pf
            << (pf_match ? "  PASS" : "  FAIL") << std::endl;

  // ---- LogDet action ----
  TXQCDLogDetEOAction logdet_A(Grid, RBGrid, mass, mu_A);
  TXQCDLogDetEOAction logdet_B(Grid, RBGrid, mass, mu_B);

  RealD S_ld_A = logdet_A.S(U_A);
  RealD S_ld_B = logdet_B.S(U_B);

  std::cout << GridLogMessage << "[A] S_logdet=" << S_ld_A << std::endl;
  std::cout << GridLogMessage << "[B] S_logdet=" << S_ld_B << std::endl;

  RealD rel_ld = std::abs(S_ld_A - S_ld_B) / std::max(std::abs(S_ld_A), 1.0);
  bool ld_match = rel_ld < 1e-12;
  std::cout << GridLogMessage << "[logdet equiv] |dS|/|S| = " << rel_ld
            << (ld_match ? "  PASS" : "  FAIL") << std::endl;

  bool all_pass = aux_match && pf_match && ld_match;
  std::cout << GridLogMessage
            << (all_pass
                ? "ALL CHECKS PASSED: mu is a field redefinition; (lambda, mu) === (lambda*mu, mu*1) under aux -> aux/mu."
                : "SOME CHECKS FAILED: investigate.")
            << std::endl;

  Grid_finalize();
  return all_pass ? 0 : 1;
}
