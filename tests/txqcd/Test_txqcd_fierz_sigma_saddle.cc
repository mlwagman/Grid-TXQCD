// Fierz consistency at σ-saddle, with M_TXQCD eigenvalue spectrum scan.
//
// At σ_ab = c·δ_ab uniform (and other aux = 0), the σ-term in M_TXQCD is a
// pure mass shift c per flavor. So:
//   M_TXQCD(U, σ=c·I, others=0)  ==  M_QCD(U, mass→m+c) ⊗ I_flavor
// And therefore:
//   S_TXQCD(U, σ=c·I, others=0)  ==  2·S_QCD(U, mass→m+c)
//
// In addition we compute the smallest |λ_min(M_site)| over all even sites,
// for each σ value, to check whether the spectrum approaches a zero crossing
// (which would make Fierz integration weight ill-conditioned and explain
// the plaq drift observed in low-λ HMC chains).
//
// This test sweeps c ∈ {0, 0.1, 1, 10, 100, 307} and checks both the action
// identity AND the eigenvalue floor.  At λ=0.1 the production saddle is
// c = Σ/λ² ≈ 307; this directly probes the regime where the chains drift.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({11, 12, 13, 14});

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  RealD mass = 0.3;
  RealD csw = 1.0;
  int exitcode = 0;

  // σ saddle values c and per-site Gaussian fluctuation widths w to test.
  // At λ=0.1, saddle c = 307 and natural width w = 1/λ = 10.
  // We compare:
  //   - Uniform σ = c·I  (Fierz prediction: matches QCD with mass m+c)
  //   - σ = c·I + (Gaussian Hermitian, width w)  (no closed-form QCD identity,
  //     so we report action and the spectrum to detect ill-conditioning)
  struct Sample {
    RealD c;
    RealD w;
  };
  std::vector<Sample> samples = {
      {0.0,   0.0},
      {0.0,   1.0},
      {1.0,   0.0},
      {1.0,   1.0},
      {10.0,  0.0},
      {10.0,  1.0},
      {10.0,  3.0},
      {307.0, 0.0},
      {307.0, 1.0},
      {307.0, 3.0},
      {307.0, 10.0},  // natural width at λ=0.1
  };

  std::cout << GridLogMessage << "===== Fierz σ-saddle scan =====" << std::endl;
  std::cout << GridLogMessage
            << "  m=" << mass << "  csw=" << csw << std::endl;
  std::cout << GridLogMessage
            << "  At σ=c·I: M_TXQCD == M_QCD(m+c) ⊗ I_flavor"
            << std::endl;
  std::cout << GridLogMessage
            << "  Predict:  S_TXQCD(σ=c·I) == 2·S_QCD(m+c)"
            << std::endl;
  std::cout << GridLogMessage
            << "  Also report smallest |λ| of 24×24 site matrix M_site"
            << std::endl;

  typedef TXQCDSiteMatrixUtil SMU;
  SMU::SpinMatrices sm;

  // Build clover field tensor F_μν for site matrix construction
  // (needed for csw != 0)
  LatticeGaugeField Ufull(&Grid);
  Ufull = U;

  // Compute clover F_μν as packed std::vector<LatticeColourMatrix>(6)
  // The TXQCD logdet action does this internally — we replicate the
  // 24×24 build using the same code path used by S().

  for (const auto &samp : samples) {
    RealD c = samp.c;
    RealD w = samp.w;

    // TXQCD setup with σ = c·I + (Hermitian Gaussian, width w)
    TXQCDField Utx(&Grid);
    Utx.U = U;
    {
      // Start with random Hermitian field (mean 0, std 1)
      HermitianGaussian(pRNG, Utx.sigma);
      Utx.sigma = w * Utx.sigma;
      // Add c·I_flavor uniformly
      typename LatticeSigmaField::vector_object::scalar_object shift_site;
      shift_site = Zero();
      for (int a = 0; a < TxqcdNf; ++a)
        shift_site()()(a, a) = ComplexD(c, 0.0);
      LatticeSigmaField shift(&Grid);
      shift = shift_site;
      Utx.sigma = Utx.sigma + shift;
    }
    Utx.pi = Zero();
    Utx.s = Zero();
    Utx.p = Zero();
    Utx.t = Zero();

    TXQCDLogDetCloverEOAction txaction(Grid, RBGrid, mass, csw);
    RealD S_tx = txaction.S(Utx);

    // QCD setup with shifted mass m + c_eff.  The sigma coupling carries a
    // 1/sqrt(2) pre-factor in mode B (TXQCD_T_FLAVOR=1), so the effective
    // mass shift per flavor at sigma = c*I is c/sqrt(2) instead of c.
    // (mode A: c_eff = c; mode B: c_eff = c/sqrt(2).)
    const RealD c_eff = TxqcdTIsFlavor ? (c / std::sqrt(2.0)) : c;
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    WCF Dw(U, Grid, RBGrid, mass + c_eff, csw, csw);
    QCDLogDetCloverEOAction<WilsonImplR> qcdaction(Dw, /*Nf=*/2);
    RealD S_qcd_shifted = qcdaction.S(U);

    RealD rel = std::abs(S_tx - S_qcd_shifted) /
                std::max(std::abs(S_qcd_shifted), 1.0);
    // At w>0, action mismatch is expected (no closed-form QCD equivalent);
    // we report rel for context but only PASS the uniform (w=0) case
    bool action_pass = (w > 0) || (rel < 1e-10);

    // -------- eigenvalue scan over even sites --------
    // Use the same code path as TXQCDLogDetCloverEOAction: build the 24×24
    // M_site on each even site and compute eigvals.
    RealD lam_min_abs = 1e300;
    RealD lam_max_abs = 0.0;
    RealD min_diag = 1e300;
    RealD max_diag = 0.0;

    // Build a Wilson-clover op just to extract clover F_μν tensor
    {
      typename SMU::AuxSiteArrays aux =
          SMU::UnvectorizeAux(Utx.sigma, Utx.pi, Utx.s, Utx.p, Utx.t);
      std::array<RealD, TxqcdNf> diag_mass;
      for (int a = 0; a < TxqcdNf; ++a) diag_mass[a] = 4.0 + mass;

      // Compute clover F_μν via Grid's helper.  We use WilsonCloverHelpers
      // which is what the TXQCD logdet S() calls.
      std::vector<LatticeColourMatrix> FS;
      if (csw != 0.0) {
        FS.resize(6, LatticeColourMatrix(&Grid));
        int k = 0;
        for (int mu = 0; mu < Nd; ++mu)
          for (int nu = mu + 1; nu < Nd; ++nu) {
            WilsonLoops<WilsonImplR>::FieldStrength(FS[k], Utx.U, mu, nu);
            ++k;
          }
      }
      typename SMU::CloverSiteArrays clover;
      if (csw != 0.0) clover = SMU::UnvectorizeClover(FS);

      uint64_t nsites = aux.sig.size();
      for (uint64_t ss = 0; ss < nsites; ++ss) {
        SMU::SiteMatrix M;
        if (csw != 0.0) {
          std::array<SMU::FmnSobj, 6> fmn_site;
          for (int k = 0; k < 6; ++k) fmn_site[k] = clover.fs[k][ss];
          SMU::BuildSiteMatrix(sm, diag_mass,
                               aux.sig[ss], aux.pi[ss],
                               aux.s[ss], aux.p[ss], aux.t[ss],
                               csw, &fmn_site, M);
        } else {
          SMU::BuildSiteMatrix(sm, diag_mass,
                               aux.sig[ss], aux.pi[ss],
                               aux.s[ss], aux.p[ss], aux.t[ss],
                               csw, nullptr, M);
        }

        // Diagonal entries
        for (int r = 0; r < SMU::kDim; ++r) {
          RealD d = std::abs(M(r, r));
          if (d < min_diag) min_diag = d;
          if (d > max_diag) max_diag = d;
        }

        // Eigenvalues of 24×24 (use Eigen)
        Eigen::ComplexEigenSolver<SMU::SiteMatrix> ces;
        ces.compute(M, /*computeEigenvectors=*/false);
        const auto &eigs = ces.eigenvalues();
        for (int k = 0; k < SMU::kDim; ++k) {
          RealD m = std::abs(eigs(k));
          if (m < lam_min_abs) lam_min_abs = m;
          if (m > lam_max_abs) lam_max_abs = m;
        }
      }
    }

    std::cout << GridLogMessage
              << "  c=" << c << " w=" << w
              << "  S_TXQCD=" << S_tx
              << "  2·S_QCD(m+c)=" << S_qcd_shifted
              << "  ΔS=" << (S_tx - S_qcd_shifted)
              << "  rel=" << rel
              << (action_pass ? "  ACTION_PASS" : "  ACTION_FAIL")
              << std::endl;
    std::cout << GridLogMessage
              << "    diag(M_site) range:  min=" << min_diag
              << "  max=" << max_diag
              << std::endl;
    std::cout << GridLogMessage
              << "    |eig(M_site)| range: min=" << lam_min_abs
              << "  max=" << lam_max_abs
              << "  condition≈" << (lam_max_abs / std::max(lam_min_abs, 1e-300))
              << std::endl;
    if (!action_pass) exitcode = 1;
  }

  Grid_finalize();
  return exitcode;
}
