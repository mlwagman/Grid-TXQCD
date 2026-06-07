#pragma once
// Helper to convert a locally-Gaussian aux field draw into a kinetic-
// augmented Gaussian draw at fixed (λ, Z).  Used both at measurement time
// (meas_helper.h, after FillAuxFields) and HMC init time (gen_txqcd_cfgs*.cc,
// same place) so the σ/π/s/p/t distributions start at the Fierz-correct
// equilibrium for the kinetic action — no extra thermalization needed.
//
// The per-Fourier-mode rescaling
//
//   h(k) = sqrt(λ²/(λ² + Z·k̂²)),    k̂² = Σ_μ 4 sin²(k_μ/2)
//
// is applied to (field − mean), then mean is added back.  Result variance
// per mode = 1/(λ² + Z·k̂²) — matches the joint quadratic+kinetic equilibrium.
//
// Each helper variant takes the field and its per-site mean (zero for
// π/p/t).  Z = 0 is a no-op (early return).

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/algorithms/FFT.h>

NAMESPACE_BEGIN(Grid);

namespace TXQCDKineticFilter {

  // Apply the kinetic filter to a single aux field with a given per-site
  // (constant) mean.  Templated over the field type so it covers σ-flavor,
  // s-color, and antisym-tensor t.
  template <class FieldT, class SiteMean>
  inline void Apply(FieldT &field, const SiteMean &site_mean,
                    RealD lambda, RealD Z) {
    if (Z == 0.0) return;
    GridCartesian *gridc = (GridCartesian *)field.Grid();
    FieldT mean_lat(gridc);
    mean_lat = site_mean;
    FieldT delta = field - mean_lat;
    FFT theFFT(gridc);
    FieldT delta_k(gridc);
    theFFT.FFT_all_dim(delta_k, delta, FFT::forward);
    Coordinate Ldim = gridc->_fdimensions;
    RealD lam2 = lambda * lambda;
    {
      autoView(dkv, delta_k, CpuWrite);
      thread_for(o, gridc->oSites(), {
        typedef typename FieldT::vector_object::scalar_object Sobj;
        const int Nsimd = gridc->iSites();
        ExtractBuffer<Sobj> buf(Nsimd);
        extract<typename FieldT::vector_object, Sobj>(dkv[o], buf);
        for (int lane = 0; lane < Nsimd; ++lane) {
          Coordinate gcoor;
          gridc->RankIndexToGlobalCoor(gridc->ThisRank(), o, lane, gcoor);
          RealD khat2 = 0.0;
          for (int d = 0; d < Nd; ++d) {
            RealD theta = M_PI * (RealD)gcoor[d] / (RealD)Ldim[d];
            RealD s = std::sin(theta);
            khat2 += 4.0 * s * s;
          }
          RealD h = std::sqrt(lam2 / (lam2 + Z * khat2));
          buf[lane] = h * buf[lane];
        }
        merge<typename FieldT::vector_object, Sobj>(dkv[o], buf);
      });
    }
    theFFT.FFT_all_dim(delta, delta_k, FFT::backward);
    field = mean_lat + delta;
  }

  // All-fields convenience: pull mean values for σ and s from Σ/λ²
  // formulas, take π/p/t mean as zero.  Used after FillAuxFields anywhere
  // an HMC or measurement wants the kinetic-correct initial distribution.
  inline void ApplyAll(TXQCDField &U, RealD lambda, RealD Sigma,
                       RealD Z_sigma, RealD Z_pi, RealD Z_s, RealD Z_p,
                       RealD Z_t) {
    if (Z_sigma != 0.0) {
      typename LatticeSigmaField::vector_object::scalar_object sigma_mean;
      sigma_mean = Zero();
      RealD sigma_diag = Sigma / (lambda * lambda);
      for (int a = 0; a < TxqcdNf; ++a) sigma_mean()()(a, a) = sigma_diag;
      Apply(U.sigma, sigma_mean, lambda, Z_sigma);
    }
    if (Z_pi != 0.0) {
      typename LatticePiField::vector_object::scalar_object pi_zero;
      pi_zero = Zero();
      Apply(U.pi, pi_zero, lambda, Z_pi);
    }
    if (Z_s != 0.0) {
      typename LatticeSFieldC::vector_object::scalar_object s_mean;
      s_mean = Zero();
      RealD s_diag = static_cast<RealD>(TxqcdNf) * Sigma /
                     (std::sqrt(2.0) * static_cast<RealD>(Nc) * lambda * lambda);
      for (int i = 0; i < Nc; ++i) s_mean()()(i, i) = s_diag;
      Apply(U.s, s_mean, lambda, Z_s);
    }
    if (Z_p != 0.0) {
      typename LatticePFieldC::vector_object::scalar_object p_zero;
      p_zero = Zero();
      Apply(U.p, p_zero, lambda, Z_p);
    }
    if (Z_t != 0.0) {
      typename LatticeTField::vector_object::scalar_object t_zero;
      t_zero = Zero();
      Apply(U.t, t_zero, lambda, Z_t);
    }
  }

  // Convenience: read the 5 env vars (SIGMA_KINETIC_Z, PI_KINETIC_Z,
  // S_KINETIC_Z, P_KINETIC_Z, T_KINETIC_Z) and call ApplyAll.  Returns true
  // if any Z was applied (useful for the caller's log line).
  inline bool ApplyFromEnv(TXQCDField &U, RealD lambda, RealD Sigma) {
    RealD Zs = 0, Zpi = 0, Zsf = 0, Zp = 0, Zt = 0;
    if (const char *e = std::getenv("SIGMA_KINETIC_Z"); e && *e) Zs  = std::atof(e);
    if (const char *e = std::getenv("PI_KINETIC_Z");    e && *e) Zpi = std::atof(e);
    if (const char *e = std::getenv("S_KINETIC_Z");     e && *e) Zsf = std::atof(e);
    if (const char *e = std::getenv("P_KINETIC_Z");     e && *e) Zp  = std::atof(e);
    if (const char *e = std::getenv("T_KINETIC_Z");     e && *e) Zt  = std::atof(e);
    bool any = (Zs!=0.0) || (Zpi!=0.0) || (Zsf!=0.0) || (Zp!=0.0) || (Zt!=0.0);
    if (any) {
      std::cout << GridLogMessage
                << "[TXQCDKineticFilter] applying h(k)=sqrt(λ²/(λ²+Z·k̂²)) with "
                << "Z_σ=" << Zs << " Z_π=" << Zpi << " Z_s=" << Zsf
                << " Z_p=" << Zp << " Z_t=" << Zt << std::endl;
      ApplyAll(U, lambda, Sigma, Zs, Zpi, Zsf, Zp, Zt);
    }
    return any;
  }

}  // namespace TXQCDKineticFilter

NAMESPACE_END(Grid);
