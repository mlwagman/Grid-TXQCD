// AUX_INIT sanity check for the kinetic-action FFT filter.
//
// At Z=0 the σ field after FillAuxFields has covariance ⟨σ(x)σ(y)⟩ = δ_{xy}/λ²
// (local Gaussian).  After applying TXQCDKineticFilter (FFT filter h(k) =
// sqrt(λ²/(λ²+Z·k̂²))) we should have ⟨σ(x)σ(y)⟩ = (1/V) Σ_k e^{ik(x-y)} ·
// 1/(λ²+Z·k̂²).
//
// Three diagnostics:
//   * <σ_aa>/site averaged over space = Σ/λ² (filter preserves k=0 mode)
//   * Var(σ_aa - mean) / site = (1/V) Σ_{k≠0} 1/(λ²+Z·k̂²) per real DOF
//   * Per-momentum |σ(k)|² averaged over many draws = 1/(λ²+Z·k̂²) ·
//     (N_f real DOFs counting the σ_00 diagonal)
//
// Compares Z=0 vs Z=10 to verify the filter is doing what AuxKineticAction
// wants for the joint quadratic+kinetic equilibrium.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({100, 101, 102, 103});

  const RealD lambda = 7.0;
  // VEV from notes: Σ ≈ vev_trminv ≈ 3.07 at light-quark mass.  We use the
  // notes value as a stand-in (k=0 mode is preserved by the filter so the
  // exact Σ doesn't affect the spectrum test).
  const RealD Sigma  = 3.07;
  const int Ntrial = 1024;
  const RealD V = Grid.gSites();

  auto measure = [&](RealD Z_sigma) {
    // Run Ntrial independent draws, accumulate per-trial statistics.
    RealD sum_norm2 = 0, sum_norm2_sq = 0;
    RealD sum_mean = 0, sum_mean_sq = 0;
    // Per-momentum |σ_aa(k)|² accumulator at four representative momenta:
    //   k1 = (1,0,0,0)·2π/L   k̂² = 4 sin²(π/L)
    //   k2 = (2,0,0,0)·2π/L   k̂² = 4 sin²(2π/L) = 4
    //   k3 = (1,1,0,0)·2π/L   k̂² = 8 sin²(π/L)
    //   k4 = (2,2,2,2)·2π/L   k̂² = 16
    std::vector<Coordinate> kpts;
    kpts.push_back(Coordinate(std::vector<int>{1,0,0,0}));
    kpts.push_back(Coordinate(std::vector<int>{2,0,0,0}));
    kpts.push_back(Coordinate(std::vector<int>{1,1,0,0}));
    kpts.push_back(Coordinate(std::vector<int>{2,2,2,2}));
    std::vector<RealD> sum_sk2(kpts.size(), 0.0);

    for (int t = 0; t < Ntrial; ++t) {
      TXQCDField U(&Grid);
      U.U = Zero();
      TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma);

      // Set env vars and apply the filter.  setenv() is per-process and lives
      // through the rest of this trial; we overwrite each Z to be explicit.
      setenv("SIGMA_KINETIC_Z", std::to_string(Z_sigma).c_str(), 1);
      setenv("PI_KINETIC_Z",    "0", 1);
      setenv("S_KINETIC_Z",     "0", 1);
      setenv("P_KINETIC_Z",     "0", 1);
      setenv("T_KINETIC_Z",     "0", 1);
      TXQCDKineticFilter::ApplyFromEnv(U, lambda, Sigma);

      // norm2(σ)/V (global).
      RealD n2 = HermitianFieldSquareNorm(U.sigma) / V;
      sum_norm2    += n2;
      sum_norm2_sq += n2 * n2;

      // Spatial mean of σ_00 (trace over flavor element 0,0).
      LatticeComplex sigma_00(&Grid);
      {
        autoView(sv, U.sigma, CpuRead);
        autoView(s00, sigma_00, CpuWrite);
        thread_for(o, Grid.oSites(), {
          auto v = sv[o]()()(0,0);  // σ at flavor (0,0)
          s00[o]()()() = v;
        });
      }
      ComplexD mean_s00 = TensorRemove(sum(sigma_00)) / V;
      sum_mean    += real(mean_s00);
      sum_mean_sq += real(mean_s00) * real(mean_s00);

      // FFT and probe a few momenta.
      FFT theFFT(&Grid);
      LatticeComplex s00_k(&Grid);
      theFFT.FFT_all_dim(s00_k, sigma_00, FFT::forward);
      for (size_t ki = 0; ki < kpts.size(); ++ki) {
        // Extract |σ_00(k)|² at this fixed momentum.  Build a delta-mask and
        // sum |σ_k · mask|² — cheap and bit-deterministic.
        LatticeComplex mask(&Grid);
        mask = Zero();
        Coordinate gc(Nd, 0);
        for (int d = 0; d < Nd; ++d) gc[d] = kpts[ki][d];
        // pokeSite needs scalar_object — build a 1.0 site and poke it in.
        typename LatticeComplex::vector_object::scalar_object one;
        one()()() = ComplexD(1.0, 0.0);
        pokeSite(one, mask, gc);
        ComplexD val = TensorRemove(sum(mask * s00_k));
        sum_sk2[ki] += val.real()*val.real() + val.imag()*val.imag();
      }
    }
    RealD mean_n2 = sum_norm2 / Ntrial;
    RealD sd_n2   = std::sqrt(sum_norm2_sq / Ntrial - mean_n2 * mean_n2);
    RealD mean_m  = sum_mean / Ntrial;
    RealD sd_m    = std::sqrt(sum_mean_sq / Ntrial - mean_m * mean_m);

    std::cout << GridLogMessage << "----  Z_sigma = " << Z_sigma << "  ----" << std::endl;
    std::cout << GridLogMessage << "  <norm2(σ)>/V  = " << mean_n2
              << " ± " << sd_n2 / std::sqrt(Ntrial) << std::endl;
    std::cout << GridLogMessage << "  <σ_00 spatial avg>  = " << mean_m
              << " ± " << sd_m / std::sqrt(Ntrial)
              << "   (expect " << Sigma / (lambda * lambda) << ")" << std::endl;
    for (size_t ki = 0; ki < kpts.size(); ++ki) {
      RealD khat2 = 0.0;
      for (int d = 0; d < Nd; ++d) {
        RealD th = M_PI * (RealD)kpts[ki][d] / 4.0;
        khat2 += 4.0 * std::sin(th) * std::sin(th);
      }
      RealD prediction = V / (lambda * lambda + Z_sigma * khat2);
      RealD measured = sum_sk2[ki] / Ntrial;
      std::cout << GridLogMessage << "  k=("
                << kpts[ki][0] << "," << kpts[ki][1] << "," << kpts[ki][2]
                << "," << kpts[ki][3] << ")  k̂²=" << khat2
                << "   |σ_00(k)|² (meas) = " << measured
                << "   (pred V/(λ²+Z·k̂²) = " << prediction
                << ", ratio=" << measured / prediction << ")" << std::endl;
    }
  };

  measure(0.0);
  measure(1.0);
  measure(10.0);

  Grid_finalize();
  return 0;
}
