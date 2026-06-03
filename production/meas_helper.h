#pragma once
// Shared helpers for production measurement binaries.
//
//   load_qcd_gauge(traj, Grid, RBGrid, Umu, sRNG, pRNG)
//   load_txqcd_field(traj, Grid, RBGrid, U, sRNG, pRNG, seed_offset)
//
// Both honour IMPORT_CFG=<path> (chroma LIME or NERSC; auto-detected).  When
// IMPORT_CFG is set we deterministically seed the RNGs (instead of reading the
// per-traj checkpoint) — used for chroma cross-checks where there is no
// matching Grid RNG checkpoint on disk.
//
// load_txqcd_field additionally auto-initializes the aux fields when the gauge
// is imported, mirroring gen_txqcd_cfgs_2plus1.cc's AUX_INIT_AUTO path:
//
//   AUX_INIT=<Σ>          manual override
//   AUX_INIT_ITERATIONS=N optional self-consistent refinement (post-fill)
//   else                  auto-measure Σ = vev_trminv = Tr[M⁻¹]/(2V) on the
//                         stout-smeared imported gauge, antiperiodic time BC
//
// Equilibrium relations driving FillAuxFields (corrected 2026-05-17; the old
// "Σ≡vev_trminv/2=Tr/(4V)" was WRONG — confirmed on dynamical streams 0.2%):
//   Σ        = vev_trminv  (single-flavor QCD Tr[M⁻¹]/(2V))
//   <σ_aa>   = Σ / λ²            (per-flavor diagonal component)
//   <s_ii>   = (N_f/(√2·N_c)) · Σ / λ²
//   vev_sigma diag = N_f·<σ_aa>  (trivial N_f flavor trace)
//
// See the TXQCD aux init convention memory and gen_txqcd_cfgs*.cc.

#include "params.h"
#include <Grid/parallelIO/IldgIO.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/smearing/StoutSmearing.h>
#include <Grid/qcd/smearing/GaugeConfiguration.h>
#include <Grid/algorithms/FFT.h>
#include <cstdio>
#include <cstring>

namespace TXQCDProduction {

// Detect NERSC ("BEGIN_HEADER") vs LIME by sniffing the first 12 bytes.
inline void read_gauge_any(const std::string &path, LatticeGaugeField &U) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  char magic[16] = {0};
  if (fp) { std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
  FieldMetaData header;
  if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(U, header, path);
  } else {
    IldgReader IR;
    IR.open(path);
    IR.readConfiguration(U, header);
    IR.close();
  }
}

// QCD measurement gauge loader.
//   IMPORT_CFG=<path>   → load that file directly; seed RNGs deterministically.
//   else                → NERSC ckpoint_lat.<traj>, NERSC ckpoint_rng.<traj>.
inline void load_qcd_gauge(int traj, LatticeGaugeField &Umu,
                            GridSerialRNG &sRNG, GridParallelRNG &pRNG) {
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "[load_qcd_gauge] IMPORT_CFG=" << ic
              << std::endl;
    read_gauge_any(std::string(ic), Umu);
    // No matching RNG checkpoint exists for an imported cfg; seed deterministically.
    sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
    pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
  } else {
    std::string cf = qcd_cfg_dir() + "/ckpoint_lat." + std::to_string(traj);
    std::string rf = qcd_cfg_dir() + "/ckpoint_rng." + std::to_string(traj);
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rf);
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cf);
  }
}

// Auto-measure Σ on the stout-smeared gauge with antiperiodic time BC, using
// the standard QCD Wilson-clover operator (Δ doesn't enter yet at first-fill
// time).  Mirrors gen_txqcd_cfgs_2plus1.cc:[IMPORT_CFG+AUX_AUTO] block.
inline RealD txqcd_measure_sigma_static(GridCartesian &Grid,
                                         GridRedBlackCartesian &RBGrid,
                                         const LatticeGaugeField &U_gauge,
                                         GridParallelRNG &noisePRNG) {
  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv, Stout);
  SmearMeas.set_Field(const_cast<LatticeGaugeField &>(U_gauge));
  LatticeGaugeField Usm = SmearMeas.get_SmearedU();
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> MFO;
  MFO Dw(Usm, Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<MFO, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(1e-8, cg_max);
  RealD V = (RealD)Grid.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_vev_noise; ++h) {
    LatticeFermion eta(&Grid), b(&Grid), x(&Grid);
    gaussian(noisePRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return (acc / n_vev_noise);  // Σ = vev_trminv = Tr[M⁻¹]/(2V) (no /2 — corrected 2026-05-17)
}

// TXQCD measurement field loader.
//   IMPORT_CFG=<path>  → load gauge from file, auto-init aux (AUX_INIT, else
//                        auto-measure on stout-smeared gauge).  Optional
//                        AUX_INIT_ITERATIONS for self-consistent refinement.
//   else               → standard per-traj checkpoint (gauge + aux sidecar).
inline void load_txqcd_field(int traj,
                              GridCartesian &Grid,
                              GridRedBlackCartesian &RBGrid,
                              TXQCDField &U,
                              GridSerialRNG &sRNG,
                              GridParallelRNG &pRNG,
                              int seed_offset = 0) {
  if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "[load_txqcd_field] IMPORT_CFG=" << ic
              << std::endl;
    read_gauge_any(std::string(ic), U.U);
    // Aux sidecar doesn't exist for imported cfgs — seed RNGs deterministically
    // (matches gen_txqcd_cfgs_2plus1.cc's IMPORT_CFG branch).
    sRNG.SeedFixedIntegers({1 + seed_offset, 2 + seed_offset, 3 + seed_offset,
                            4 + seed_offset, 5 + seed_offset});
    pRNG.SeedFixedIntegers({6 + seed_offset, 7 + seed_offset, 8 + seed_offset,
                            9 + seed_offset, 10 + seed_offset});
    // Σ priority: AUX_INIT env override, else auto-measure Tr[M⁻¹] on the
    // stout-smeared imported gauge.
    RealD Sigma = 0.0;
    if (const char *si = std::getenv("AUX_INIT"); si && *si) {
      Sigma = std::atof(si);
      std::cout << GridLogMessage << "[IMPORT_CFG+AUX_INIT] Σ=" << Sigma
                << " (manual override)" << std::endl;
    } else {
      GridParallelRNG noisePRNG(&Grid);
      noisePRNG.SeedFixedIntegers(
          {seed_offset + 11, seed_offset + 12, seed_offset + 13,
           seed_offset + 14, seed_offset + 15});
      Sigma = txqcd_measure_sigma_static(Grid, RBGrid, U.U, noisePRNG);
      std::cout << GridLogMessage << "[IMPORT_CFG+AUX_AUTO] Σ=" << Sigma
                << "  → <σ_aa>=" << Sigma / (lambda * lambda) << std::endl;
    }
    TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma);
    // AUX_KINETIC_Z_SIGMA / AUX_KINETIC_Z_S: post-hoc kinetic-action filter.
    // The drawn σ/s have local-Gaussian variance 1/λ².  Map to the
    // kinetic-action variance 1/(λ²+Z·k̂²) by:
    //   1. subtract the per-site mean (so the filter only touches fluct);
    //   2. FFT to k-space;
    //   3. multiply each mode by h(k) = sqrt(λ²/(λ²+Z·k̂²));
    //   4. inverse FFT;
    //   5. add the mean back.
    // Result: σ-VEV unchanged (Σ/λ²), σ-fluctuations damped at high k̂²
    // exactly as the kinetic action would in HMC.  Single-shot — does not
    // need MCMC; valid because the local prior is gaussian + linear filter
    // → still gaussian with the target covariance.
    auto apply_kinetic = [&](auto &field, const auto &site_mean, RealD Z) {
      typedef typename std::remove_reference_t<decltype(field)> FieldT;
      FieldT mean_lat(field.Grid());
      mean_lat = site_mean;
      FieldT delta = field - mean_lat;
      // FFT_all_dim forward.
      GridCartesian *gridc = (GridCartesian *)field.Grid();
      FFT theFFT(gridc);
      FieldT delta_k(gridc);
      theFFT.FFT_all_dim(delta_k, delta, FFT::forward);
      // Filter in-place via accelerator_for.
      Coordinate Ldim = gridc->_fdimensions;
      RealD lam2 = lambda * lambda;
      {
        autoView(dkv, delta_k, CpuWrite);
        // Compute h(k) at each site from its global coordinate.  Use site
        // index → global coord helper.  For multi-SIMD, broadcast scalar.
        thread_for(o, gridc->oSites(), {
          // Walk through SIMD lanes; build a per-lane filter and apply.
          typedef typename FieldT::vector_object::scalar_object Sobj;
          typedef typename Sobj::scalar_type Stype;
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
      }  // release autoView lock before next FFT writes to delta_k
      // Inverse FFT.  Grid's FFT_dim backward already includes 1/G per dim
      // (see Grid/algorithms/FFT.h line ~321), so forward+backward already
      // round-trips to the input — no extra normalization needed.
      theFFT.FFT_all_dim(delta, delta_k, FFT::backward);
      field = mean_lat + delta;
    };
    if (const char *zs = std::getenv("AUX_KINETIC_Z_SIGMA"); zs && *zs) {
      RealD Z = std::atof(zs);
      std::cout << GridLogMessage << "[AUX_KINETIC_Z_SIGMA] applying kinetic "
                << "filter to σ with Z = " << Z
                << " (h(k)=sqrt(λ²/(λ²+Z·k̂²)))" << std::endl;
      TxqcdSiteSigma sigma_mean;
      sigma_mean = Zero();
      RealD sigma_diag = Sigma / (lambda * lambda);
      for (int a = 0; a < TxqcdNf; ++a) sigma_mean()()(a, a) = sigma_diag;
      apply_kinetic(U.sigma, sigma_mean, Z);
    }
    if (const char *zs = std::getenv("AUX_KINETIC_Z_S"); zs && *zs) {
      RealD Z = std::atof(zs);
      std::cout << GridLogMessage << "[AUX_KINETIC_Z_S] applying kinetic "
                << "filter to s with Z = " << Z << std::endl;
      TxqcdSiteS s_mean;
      s_mean = Zero();
      RealD s_diag = static_cast<RealD>(TxqcdNf) * Sigma /
                     (std::sqrt(2.0) * static_cast<RealD>(Nc) * lambda * lambda);
      for (int i = 0; i < Nc; ++i) s_mean()()(i, i) = s_diag;
      apply_kinetic(U.s, s_mean, Z);
    }
    // π, p, t have zero mean — just filter with h(k), no mean to add back.
    if (const char *zs = std::getenv("AUX_KINETIC_Z_PI"); zs && *zs) {
      RealD Z = std::atof(zs);
      std::cout << GridLogMessage << "[AUX_KINETIC_Z_PI] applying kinetic "
                << "filter to π with Z = " << Z << std::endl;
      typename LatticePiField::vector_object::scalar_object pi_zero;
      pi_zero = Zero();
      apply_kinetic(U.pi, pi_zero, Z);
    }
    if (const char *zs = std::getenv("AUX_KINETIC_Z_P"); zs && *zs) {
      RealD Z = std::atof(zs);
      std::cout << GridLogMessage << "[AUX_KINETIC_Z_P] applying kinetic "
                << "filter to p with Z = " << Z << std::endl;
      typename LatticePFieldC::vector_object::scalar_object p_zero;
      p_zero = Zero();
      apply_kinetic(U.p, p_zero, Z);
    }
    if (const char *zs = std::getenv("AUX_KINETIC_Z_T"); zs && *zs) {
      RealD Z = std::atof(zs);
      std::cout << GridLogMessage << "[AUX_KINETIC_Z_T] applying kinetic "
                << "filter to t with Z = " << Z << std::endl;
      typename LatticeTField::vector_object::scalar_object t_zero;
      t_zero = Zero();
      apply_kinetic(U.t, t_zero, Z);
    }
    // AUX_VAR_FRAC=<f>: scale the fluctuation about the mean by sqrt(f).
    // f<1 reduces variance, f=0 → frozen mean.  Preserves the per-site mean
    // exactly.  Used for the "mean-field + small noise" probe.  Applied
    // BEFORE the FROZEN_MEAN reset so FROZEN_MEAN takes precedence.
    if (const char *vf = std::getenv("AUX_VAR_FRAC"); vf && *vf) {
      RealD f = std::atof(vf);
      RealD scale = std::sqrt(std::max(f, 0.0));
      std::cout << GridLogMessage << "[AUX_VAR_FRAC] scaling aux fluctuations by sqrt("
                << f << ") = " << scale << " (mean preserved)" << std::endl;
      // σ_new = σ_mean + scale·(σ_old − σ_mean).
      // σ mean (per-flavor diagonal):
      TxqcdSiteSigma sigma_mean;
      sigma_mean = Zero();
      RealD sigma_diag = Sigma / (lambda * lambda);
      for (int a = 0; a < TxqcdNf; ++a) sigma_mean()()(a, a) = sigma_diag;
      LatticeSigmaField sigma_mean_lat(U.sigma.Grid());
      sigma_mean_lat = sigma_mean;
      U.sigma = sigma_mean_lat + scale * (U.sigma - sigma_mean_lat);
      // s mean (per-color diagonal):
      TxqcdSiteS s_mean;
      s_mean = Zero();
      RealD s_diag = static_cast<RealD>(TxqcdNf) * Sigma /
                     (std::sqrt(2.0) * static_cast<RealD>(Nc) * lambda * lambda);
      for (int i = 0; i < Nc; ++i) s_mean()()(i, i) = s_diag;
      LatticeSFieldC s_mean_lat(U.s.Grid());
      s_mean_lat = s_mean;
      U.s = s_mean_lat + scale * (U.s - s_mean_lat);
      // π, p, t have zero mean → just scale.
      U.pi = scale * U.pi;
      U.p  = scale * U.p;
      U.t  = scale * U.t;
    }
    // AUX_FROZEN_MEAN=1: replace the drawn (gaussian + mean) aux fields with
    // their site-constant MEAN values only — no fluctuations.  Used to test
    // the mean-field picture (σ as a static effective mass shift).
    if (std::getenv("AUX_FROZEN_MEAN")) {
      std::cout << GridLogMessage
                << "[AUX_FROZEN_MEAN] setting σ=Σ/λ²·I, s=Σ/(√2·N_c·λ²)·I, "
                << "π=p=t=0 at every site (no fluctuations)" << std::endl;
      U.pi = Zero();
      U.p  = Zero();
      U.t  = Zero();
      // σ: flavor-diagonal constant Σ/λ².
      TxqcdSiteSigma sigma_mean;
      sigma_mean = Zero();
      RealD sigma_diag = Sigma / (lambda * lambda);
      for (int a = 0; a < TxqcdNf; ++a) sigma_mean()()(a, a) = sigma_diag;
      U.sigma = sigma_mean;
      // s: color-diagonal constant N_f·Σ/(√2·N_c·λ²).
      TxqcdSiteS s_mean;
      s_mean = Zero();
      RealD s_diag = static_cast<RealD>(TxqcdNf) * Sigma /
                     (std::sqrt(2.0) * static_cast<RealD>(Nc) * lambda * lambda);
      for (int i = 0; i < Nc; ++i) s_mean()()(i, i) = s_diag;
      U.s = s_mean;
    }

    // Optional self-consistent refinement using the FULL TXQCD operator.
    if (const char *ni = std::getenv("AUX_INIT_ITERATIONS"); ni && *ni) {
      int n_iter = std::atoi(ni);
      Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
      SmearedConfiguration<PeriodicGimplR> SmearMeas(&Grid, stout_nsmear_inv,
                                                      Stout);
      SmearMeas.set_Field(U.U);
      LatticeGaugeField Usm = SmearMeas.get_SmearedU();
      WilsonImplParams impl_p_meas;
      impl_p_meas.boundary_phases.resize(Nd, 1.0);
      impl_p_meas.boundary_phases[Nd - 1] = -1.0;
      RealD V = (RealD)Grid.gSites();
      GridParallelRNG noisePRNG(&Grid);
      noisePRNG.SeedFixedIntegers(
          {seed_offset + 21, seed_offset + 22, seed_offset + 23,
           seed_offset + 24, seed_offset + 25});
      for (int it = 0; it < n_iter; ++it) {
        std::array<RealD, TxqcdNf> mass_arr;
        for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
        TXQCDWilsonCloverOp Mop(Usm, Grid, RBGrid, mass_arr,
                                 U.sigma, U.pi, U.s, U.p, U.t, csw,
                                 impl_p_meas);
        RealD acc = 0.0;
        for (int h = 0; h < n_vev_noise; ++h) {
          TXQCDFermionNf eta(&Grid), b(&Grid), x(&Grid);
          for (int a = 0; a < TxqcdNf; ++a) gaussian(noisePRNG, eta.f[a]);
          Mop.Mdag(eta, b);
          x = Zero();
          // Inline single-shift CG on M_TXQCD†·M_TXQCD.
          TXQCDFermionNf r(&Grid), p(&Grid), Mp(&Grid), MdMp(&Grid);
          r = b; p = r;
          RealD rsq = norm2(r);
          RealD bsq = std::max(norm2(b), 1e-30);
          RealD tol2 = 1e-16 * bsq;
          for (int cg_it = 0; cg_it < cg_max; ++cg_it) {
            Mop.M(p, Mp); Mop.Mdag(Mp, MdMp);
            ComplexD pAp = innerProduct(p, MdMp);
            ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
            for (int a = 0; a < TxqcdNf; ++a) x.f[a] = x.f[a] + alpha * p.f[a];
            for (int a = 0; a < TxqcdNf; ++a) r.f[a] = r.f[a] - alpha * MdMp.f[a];
            RealD rsq_new = norm2(r);
            if (rsq_new < tol2) break;
            RealD beta_cg = rsq_new / rsq;
            for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
            rsq = rsq_new;
          }
          acc += innerProduct(eta, x).real() / (2.0 * (RealD)TxqcdNf * V);
        }
        RealD Sigma2 = (acc / n_vev_noise);  // Σ = vev_trminv (no /2)
        std::cout << GridLogMessage << "[AUX_ITER " << (it + 1) << "/" << n_iter
                  << "] Σ=" << Sigma2
                  << "  → <σ_aa>=" << Sigma2 / (lambda * lambda) << std::endl;
        TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, Sigma2);
      }
    }
  } else {
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  txqcd_cfg_dir() + "/ckpoint_lat",
                                  txqcd_cfg_dir() + "/ckpoint_rng", traj);
  }
}

}  // namespace TXQCDProduction
