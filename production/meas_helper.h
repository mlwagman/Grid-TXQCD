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
