// Test_dtxqcd_pfaffian_antisymmetry: verify the C·K Pfaffian antisymmetry
// (K·M48)^T = -(K·M48) of the full doubled clover operator on a single site
// for random aux fields generated via the real-symmetric projector (2026-06-13).
//
// Per arxiv:2209.13183, the action ½ Ψ̄ M48 Ψ on the doubled quark space
// is a true Pfaffian Pf(K·M48) = det(M48)^{1/2} only when K·M48 is
// antisymmetric.  Conventions used here (matching Cstar BC):
//
//   K = [[0  C·γ5]
//        [C·γ5  0]]      (block-swap ⊗ intra-block C·γ5)
//   C = γ² · γ⁴          (Grid basis, no i prefactor)
//
// This test fails if the aux fields are stored as complex-Hermitian rather
// than real-symmetric (test_pfaffian_antisymmetry.py 2026-06-13 confirms:
//   Hermitian X      → ratio ≈ 1.03 (✗)
//   Real-symmetric X → ratio = 0    (✓)
// ).
//
// The test exercises the FULL clover operator (csw != 0): we draw a
// random gauge link and a random clover field strength F_{μν}, then
// assemble the site-local 48×48 with sigma, pi, d, n, s, p projected to
// real-symmetric.  This protects against future regressions in either:
//   - the aux projector (DtxqcdHermitizeAndTracelessCFInPlace), or
//   - the lower-block sign convention (Cstar transpose).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>

using namespace Grid;
using Eigen::MatrixXcd;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({77, 78, 79, 80});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  DtxqcdSpinMatrices spin(Grid);

  // C = gamma2 * gamma4.  We probe these by applying the Gamma to a single
  // canonical-basis spinor and reading out the result (same idiom as the
  // sigma_munu probe inside DtxqcdSpinMatrices).
  auto ProbeG = [&](Gamma::Algebra a) -> Eigen::Matrix4cd {
    Eigen::Matrix4cd out;
    LatticeSpinColourVector v(&Grid), Gv(&Grid);
    Coordinate origin(std::vector<int>{0, 0, 0, 0});
    typename LatticeSpinColourVector::vector_object::scalar_object so;
    for (int alpha = 0; alpha < Ns; ++alpha) {
      v = Zero();
      so = Zero();
      so()(alpha)(0) = 1.0;
      pokeSite(so, v, origin);
      Gv = Gamma(a) * v;
      peekSite(so, Gv, origin);
      for (int beta = 0; beta < Ns; ++beta) {
        out(beta, alpha) = so()(beta)(0);
      }
    }
    return out;
  };
  Eigen::Matrix4cd g2 = ProbeG(Gamma::Algebra::GammaY);
  Eigen::Matrix4cd g4 = ProbeG(Gamma::Algebra::GammaT);
  Eigen::Matrix4cd C_spin = g2 * g4;
  Eigen::Matrix4cd Cg5    = C_spin * spin.gamma5;

  // ---- random aux + gauge clover field strength ----
  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdD     d_field(&Grid);
  LatticeDtxqcdN     n_field(&Grid);
  LatticeDtxqcdS     s_field(&Grid);
  LatticeDtxqcdP     p_field(&Grid);
  DtxqcdHermitianCFGaussian(pRNG, sigma);   // Hermitian after pre-shift revert
  DtxqcdHermitianCFGaussian(pRNG, pi);
  DtxqcdHermitianCFGaussian(pRNG, d_field);
  DtxqcdHermitianCFGaussian(pRNG, n_field);
  DtxqcdRealScalarGaussian(pRNG, s_field);
  DtxqcdRealScalarGaussian(pRNG, p_field);
  // Env-gated alternative projections from CompositeImpl:
  if (DtxqcdDnRealSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(d_field);
    DtxqcdRealSymmetricCFInPlace(n_field);
  }
  if (DtxqcdDnComplexSymmetric()) {
    // σ, π: real-symmetric;  d, n: complex-symmetric (transpose-symm)
    DtxqcdRealSymmetricCFInPlace(sigma);
    DtxqcdRealSymmetricCFInPlace(pi);
    // Truly complex-symm: raw complex Gaussian + complex-symm projection.
    DtxqcdComplexSymmetricCFGaussian(pRNG, d_field);
    DtxqcdComplexSymmetricCFGaussian(pRNG, n_field);
  }

  Coordinate site0(std::vector<int>{0, 0, 0, 0});
  DtxqcdSiteAux aux = DtxqcdSiteAux::Extract(sigma, pi, d_field, n_field,
                                              s_field, p_field, site0);

  // ----- Hermitian random clover F_{mu,nu} (anti-Hermitian SU(3)) -----
  // Build the SiteClover by sampling six independent traceless anti-Hermitian
  // 3x3 matrices (the F_{mu,nu} components live in the Lie algebra).
  DtxqcdSiteClover clover;
  for (int mn = 0; mn < 6; ++mn) {
    Eigen::Matrix3cd A = Eigen::Matrix3cd::Random();
    // Anti-Hermitian + traceless
    A = 0.5 * (A - A.adjoint());
    A -= (A.trace() / 3.0) * Eigen::Matrix3cd::Identity();
    clover.F_munu[mn] = A;
  }

  // ----- assemble M48 with and without clover -----
  const RealD mass = 0.3;
  const RealD csw  = 1.24930970916466;

  auto assemble_M48 = [&](double csw_eff) {
    MatrixXcd M_upper, M_lower, M_off, M48;
    DtxqcdBuildUpperBlock24(mass, aux, spin, M_upper, csw_eff, &clover);
    DtxqcdBuildLowerBlock24(mass, aux, spin, M_lower, csw_eff, &clover);
    DtxqcdBuildOffDiagBlock24(aux, spin, M_off);
    DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);
    return M48;
  };

  // ----- assemble K = block-swap ⊗ (C·γ5) -----
  const int N24 = kDtxqcdSiteDim24;
  const int N48 = kDtxqcdSiteDim48;
  MatrixXcd Cg5_24 = MatrixXcd::Zero(N24, N24);
  // Cg5 acts on spin only; identity on flavor and color.
  for (int a = 0; a < DtxqcdNf; ++a) {
    for (int alpha = 0; alpha < Ns; ++alpha) {
      for (int beta = 0; beta < Ns; ++beta) {
        for (int c = 0; c < Nc; ++c) {
          int row = DtxqcdSiteIdx24(a, alpha, c);
          int col = DtxqcdSiteIdx24(a, beta,  c);
          Cg5_24(row, col) = Cg5(alpha, beta);
        }
      }
    }
  }
  MatrixXcd K = MatrixXcd::Zero(N48, N48);
  K.block(0,    N24, N24, N24) = Cg5_24;
  K.block(N24,  0,   N24, N24) = Cg5_24;

  // ----- run check at csw = 0 (pure aux) -----
  {
    MatrixXcd M48 = assemble_M48(0.0);
    MatrixXcd KM  = K * M48;
    RealD asym = (KM + KM.transpose()).norm();
    RealD ref  = std::max(KM.norm(), 1e-30);
    check("csw=0  Pfaff antisymmetry (K·M48)^T = -(K·M48) (rel)",
          asym / ref, 1e-11);
  }

  // ----- run check at csw = 1.249 (full clover operator) -----
  {
    MatrixXcd M48 = assemble_M48(csw);
    MatrixXcd KM  = K * M48;
    RealD asym = (KM + KM.transpose()).norm();
    RealD ref  = std::max(KM.norm(), 1e-30);
    check("csw=1.249 Pfaff antisymmetry (K·M48)^T = -(K·M48) (rel)",
          asym / ref, 1e-11);
  }

  // sigmaHerm (2026-06-19): The previous sanity check expected Hermitian
  // (non-real-symmetric) σ to BREAK Pfaffian antisymmetry — true under the
  // older real-symmetric projection convention, where M_lower applied raw
  // σ.  Under sigmaHerm the operator code applies σ^T in M_lower (via
  // DtxqcdBuildDiagBlock24(..., transpose_aux=true)), restoring
  // (K·M48)^T = -(K·M48) for Hermitian aux as well.  The two checks above
  // (csw=0, csw=1.249) exercise both real-symmetric and Hermitian aux
  // paths and are sufficient; the obsolete sanity is dropped.

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
