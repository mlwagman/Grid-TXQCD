// Measure connected pion correlator C_π(t) on a deterministic weak-field
// gauge for QCD (no aux) and TXQCD at multiple λ values (AUX_INIT_AUTO σ).
//
// Probes whether the connected pion gets heavier at small λ as predicted
// by the m_eff = m + λ·σ_VEV picture (with σ_VEV = Σ/λ²).  At equilibrium
// the disconnected (σ-mediated) diagram should compensate via Fierz so
// the TOTAL pion mass is λ-independent — this binary computes only the
// CONNECTED piece.
//
// Compile: production Makefile builds with -DTXQCD_Nf=3 (matches gen_txqcd_cfgs).
//
// Output: pion correlator C(t) for QCD + each λ in stdout, machine-readable.
#include "params.h"
#include "quda_helper.h"
#include <cstdio>
#include <cstring>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <Grid/qcd/smearing/StoutSmearing.h>

using namespace TXQCDProduction;

// Auto-measure Σ on stout-smeared weak-field gauge with antiperiodic time BC.
// (Lifted from gen_txqcd_cfgs.cc AUX_INIT_AUTO path.)
static RealD AutoMeasureSigma(GridCartesian &Grid, GridRedBlackCartesian &RBGrid,
                              GridParallelRNG &noisePRNG,
                              const LatticeGaugeField &Usm,
                              int n_noise = 4) {
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(const_cast<LatticeGaugeField&>(Usm), Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  Grid::QudaPropSolver<WCF> solver(Dw, HermOp, Usm, mass_light, csw, 1e-8, cg_max);
  RealD V = (RealD)Grid.gSites();
  RealD acc = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    LatticeFermion eta(&Grid), x(&Grid);
    gaussian(noisePRNG, eta);
    solver.solve(eta, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  RealD vev_trminv = acc / n_noise;
  return vev_trminv / 2.0;  // Σ = vev_trminv / 2
}

// Pion correlator from light-light propagator: C(t) = Σ_x Tr[γ5 P(x,t) γ5 P†(x,t)]
// = Σ_x Tr[P(x,t) · P†(x,t)] (γ5-Hermiticity reduces to a norm-like sum).
static std::vector<RealD> PionCorrelator(const LatticePropagator &S) {
  LatticeComplex corr(S.Grid());
  corr = trace(S * adj(S));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

// Connected nucleon (proton, "uud") correlator using BaryonUtils contraction.
// Identical conventions to meas_conn_qcd.cc.
static std::vector<ComplexD> NucleonCorrelator(const LatticePropagator &S) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(S, S, S, G_A, G_B, G_A, G_B,
                                            wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Build a point source of identity in spin-color at origin.
static void PointSource(LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  Coordinate origin(Nd, 0);
  pokeSite(kron, src, origin);
}

// QCD path: invert plain Wilson-Clover (no aux fields) for a propagator.
static LatticePropagator QcdPropagator(const LatticeGaugeField &Usm,
                                        GridCartesian &Grid,
                                        GridRedBlackCartesian &RBGrid,
                                        const LatticePropagator &src,
                                        RealD tol) {
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(const_cast<LatticeGaugeField&>(Usm), Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  Grid::QudaPropSolver<WCF> solver(Dw, HermOp, Usm, mass_light, csw, tol, cg_max);

  LatticePropagator prop(&Grid); prop = Zero();
  // Solve column-by-column over (spin, color).
  for (int s = 0; s < Ns; ++s) {
    for (int c = 0; c < Nc; ++c) {
      LatticeFermion psi(&Grid), x(&Grid);
      psi = Zero();
      PropToFerm<WilsonImplR>(psi, src, s, c);
      solver.solve(psi, x);
      FermToProp<WilsonImplR>(prop, x, s, c);
    }
  }
  return prop;
}

// TXQCD path: solve Mop·x = source on TXQCDFermionNf, extract flavor-0
// (light) propagator.  Uses an inline simple CG (multishift not needed for
// a single solve at a single mass).
static void TxqcdCG(TXQCDWilsonCloverOp &Mop, const TXQCDFermionNf &b,
                    TXQCDFermionNf &x, RealD tol, int maxit) {
  GridBase *g = b.Grid();
  TXQCDFermionNf r(g), p(g), Mp(g), MdMp(g);
  x = Zero();
  r = b;
  p = r;
  RealD rsq = norm2(r);
  RealD bsq = std::max(norm2(b), 1e-30);
  RealD tol2 = tol * tol * bsq;
  for (int it = 0; it < maxit; ++it) {
    Mop.M(p, Mp);
    Mop.Mdag(Mp, MdMp);
    ComplexD pAp = innerProduct(p, MdMp);
    ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
    axpy(x, alpha, p);
    axpy(r, -alpha, MdMp);
    RealD rsq_new = norm2(r);
    if (rsq_new < tol2) {
      std::cout << GridLogMessage << "    [TXQCD CG] converged iter=" << it+1
                << " resid²=" << rsq_new << std::endl;
      return;
    }
    RealD beta_cg = rsq_new / rsq;
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
    rsq = rsq_new;
  }
  std::cout << GridLogMessage << "    [TXQCD CG] DID NOT CONVERGE in "
            << maxit << " iters, resid²=" << rsq << std::endl;
}

static LatticePropagator TxqcdLightPropagator(LatticeGaugeField &Usm,
                                              TXQCDField &U,
                                              GridCartesian &Grid,
                                              GridRedBlackCartesian &RBGrid,
                                              const LatticePropagator &src,
                                              RealD tol) {
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  // Build per-flavor mass array.  For Nf=3 the third slot holds the strange
  // mass (== mass_light on the cl3_16_48_b6p1_m0p2450 ensemble); for Nf=2
  // the second slot is also light (Nf=2+1 setup uses Nf=1 QCD strange
  // separately and has no strange in the TXQCD op).
  std::array<RealD, TxqcdNf> mass_arr;
  for (int a = 0; a < TxqcdNf; ++a) mass_arr[a] = mass_light;
  if (TxqcdNf >= 3) mass_arr[TxqcdNf - 1] = mass_strange;
  TXQCDWilsonCloverOp Mop(Usm, Grid, RBGrid, mass_arr,
                           U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);

  LatticePropagator prop(&Grid); prop = Zero();
  for (int s = 0; s < Ns; ++s) {
    for (int c = 0; c < Nc; ++c) {
      TXQCDFermionNf psi(&Grid), b(&Grid), x(&Grid);
      // Source on flavor 0 (light); other flavors zero.
      for (int a = 0; a < TxqcdNf; ++a) psi.f[a] = Zero();
      PropToFerm<WilsonImplR>(psi.f[0], src, s, c);
      // Mdag·source
      Mop.Mdag(psi, b);
      x = Zero();
      TxqcdCG(Mop, b, x, tol, cg_max);
      // Extract flavor-0 component as the "light" propagator entry.
      FermToProp<WilsonImplR>(prop, x.f[0], s, c);
    }
  }
  return prop;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Use --grid 16.16.16.48 from CLI; fall back to a small lattice if not set.
  Coordinate latt = GridDefaultLatt();
  if (latt.size() != 4 || latt[0] <= 0) latt = Coordinate({16, 16, 16, 48});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  std::cout << GridLogMessage << "Lattice: " << latt[0] << "."
            << latt[1] << "." << latt[2] << "." << latt[3] << std::endl;
  std::cout << GridLogMessage << "mass_light=" << mass_light
            << " mass_strange=" << mass_strange
            << " csw=" << csw << std::endl;

  // Deterministic seed so QCD + all TXQCD λ runs see the SAME weak-field gauge.
  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({42, 43, 44, 45, 46});
  pRNG.SeedFixedIntegers({142, 143, 144, 145, 146});

  // Gauge field source priority:
  //   1. LOAD_TXQCD_CKPT="cfg_dir:traj"  → load gauge+aux from TXQCD HMC chain.
  //                                        The aux fields are taken from disk;
  //                                        FillAuxFields is skipped for this run.
  //   2. IMPORT_CFG=<path>               → load gauge from NERSC/LIME, fill aux
  //                                        via FillAuxFields(λ, Σ_auto).
  //   3. (default)                       → generate weak-field gauge.
  TXQCDField U(&Grid);
  bool aux_loaded_from_ckpt = false;
  if (const char *lc = std::getenv("LOAD_TXQCD_CKPT"); lc && *lc) {
    std::string spec(lc);
    auto colon = spec.find(':');
    if (colon == std::string::npos) {
      std::cerr << "LOAD_TXQCD_CKPT must be \"cfg_dir:traj\"" << std::endl;
      Grid_finalize(); return 1;
    }
    std::string cfg_dir = spec.substr(0, colon);
    int traj = std::atoi(spec.substr(colon + 1).c_str());
    std::cout << GridLogMessage << "LOAD_TXQCD_CKPT cfg_dir=" << cfg_dir
              << " traj=" << traj << std::endl;
    TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
    aux_loaded_from_ckpt = true;
  } else if (const char *ic = std::getenv("IMPORT_CFG"); ic && *ic) {
    std::cout << GridLogMessage << "IMPORT_CFG=" << ic << std::endl;
    FILE *f = std::fopen(ic, "rb");
    char magic[16] = {0};
    if (f) { std::fread(magic, 1, sizeof(magic), f); std::fclose(f); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GS;
      NerscIO::readConfiguration<GS>(U.U, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(U.U, header);
      IR.close();
    }
  } else {
    TXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
  }
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);
  std::cout << GridLogMessage << "gauge plaq=" << plaq << std::endl;

  // Smear once and measure Σ via Tr[(M†M)^{-1}].
  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&Grid, stout_nsmear_inv, Stout);
  Smear.set_Field(U.U);
  LatticeGaugeField Usm = Smear.get_SmearedU();

  GridParallelRNG noisePRNG(&Grid);
  noisePRNG.SeedFixedIntegers({1042, 1043, 1044, 1045, 1046});
  std::cout << GridLogMessage << "[AutoMeasureSigma] computing..." << std::endl;
  RealD Sigma = AutoMeasureSigma(Grid, RBGrid, noisePRNG, Usm, 4);
  std::cout << GridLogMessage << "Σ (auto) = " << Sigma << std::endl;

  // Build point source at origin.
  LatticePropagator src(&Grid);
  PointSource(src);

  // QCD baseline: plain Wilson-Clover.
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "==== QCD (no aux fields) ====" << std::endl;
  LatticePropagator P_qcd = QcdPropagator(Usm, Grid, RBGrid, src, 1e-8);
  std::vector<RealD> C_qcd = PionCorrelator(P_qcd);
  std::vector<ComplexD> N_qcd = NucleonCorrelator(P_qcd);

  // For each λ, fill aux fields at AUX_INIT_AUTO equilibrium and measure.
  std::vector<RealD> lambdas;
  if (const char *l = std::getenv("LAMBDAS"); l && *l) {
    std::stringstream ss(l);
    RealD x; while (ss >> x) lambdas.push_back(x);
  } else {
    lambdas = {4, 5, 6, 8, 12, 18};
  }

  std::map<RealD, std::vector<RealD>>    C_tx;
  std::map<RealD, std::vector<ComplexD>> N_tx;
  for (RealD lam : lambdas) {
    std::cout << GridLogMessage << std::endl;
    std::cout << GridLogMessage << "==== TXQCD λ=" << lam << " ====" << std::endl;
    if (aux_loaded_from_ckpt) {
      // Aux fields already on U from TXQCDCheckpointer::ReadConfig — keep them.
      RealD vs = TensorRemove(sum(trace(U.sigma))).real() / Grid.gSites();
      std::cout << GridLogMessage << "  using aux from ckpt (vev_sigma="
                << vs << ")" << std::endl;
    } else {
      // Re-seed pRNG identically per λ so aux-field initialization is the
      // same RNG state for fair comparison.
      GridParallelRNG pRNG_lam(&Grid);
      pRNG_lam.SeedFixedIntegers({500 + (int)(10*lam), 501 + (int)(10*lam),
                                   502 + (int)(10*lam), 503 + (int)(10*lam),
                                   504 + (int)(10*lam)});
      TXQCDCompositeImpl::FillAuxFields(pRNG_lam, U, lam, Sigma);
      RealD vs = TensorRemove(sum(trace(U.sigma))).real() / Grid.gSites();
      std::cout << GridLogMessage << "  vev_sigma = " << vs
                << " (expected 3·Σ/λ² = " << 3*Sigma/(lam*lam) << ")" << std::endl;
    }

    LatticePropagator P_tx = TxqcdLightPropagator(Usm, U, Grid, RBGrid, src, 1e-8);
    C_tx[lam] = PionCorrelator(P_tx);
    N_tx[lam] = NucleonCorrelator(P_tx);
  }

  // Output table: t  C_QCD  C_λ=4  C_λ=5  ...
  std::cout << std::endl;
  std::cout << "# === pion correlator on weak-field gauge ===" << std::endl;
  std::cout << "# Σ = " << Sigma << std::endl;
  std::cout << "# t  C_QCD";
  for (RealD lam : lambdas) std::cout << "  C_λ=" << lam;
  std::cout << std::endl;
  int T = (int)C_qcd.size();
  for (int t = 0; t < T; ++t) {
    std::cout << t << "  " << std::scientific << std::setprecision(8) << C_qcd[t];
    for (RealD lam : lambdas) std::cout << "  " << C_tx[lam][t];
    std::cout << std::defaultfloat << std::endl;
  }

  // Ratios: log(C_λ / C_QCD) at each t.
  std::cout << std::endl;
  std::cout << "# === log(C_pi_λ / C_pi_QCD) — slope = m_pi_QCD - m_pi_λ ===" << std::endl;
  std::cout << "# t";
  for (RealD lam : lambdas) std::cout << "  λ=" << lam;
  std::cout << std::endl;
  for (int t = 0; t < T; ++t) {
    std::cout << t;
    for (RealD lam : lambdas) {
      RealD r = (C_qcd[t] != 0 && C_tx[lam][t] != 0)
                ? std::log(std::abs(C_tx[lam][t] / C_qcd[t])) : 0.0;
      std::cout << "  " << std::scientific << std::setprecision(4) << r;
    }
    std::cout << std::defaultfloat << std::endl;
  }

  // Nucleon table.
  std::cout << std::endl;
  std::cout << "# === nucleon (uud) correlator real part ===" << std::endl;
  std::cout << "# t  N_QCD";
  for (RealD lam : lambdas) std::cout << "  N_λ=" << lam;
  std::cout << std::endl;
  for (int t = 0; t < T; ++t) {
    std::cout << t << "  " << std::scientific << std::setprecision(8) << N_qcd[t].real();
    for (RealD lam : lambdas) std::cout << "  " << N_tx[lam][t].real();
    std::cout << std::defaultfloat << std::endl;
  }

  std::cout << std::endl;
  std::cout << "# === log(N_λ / N_QCD) — slope = m_N_QCD - m_N_λ ===" << std::endl;
  std::cout << "# t";
  for (RealD lam : lambdas) std::cout << "  λ=" << lam;
  std::cout << std::endl;
  for (int t = 0; t < T; ++t) {
    std::cout << t;
    for (RealD lam : lambdas) {
      RealD nq = N_qcd[t].real();
      RealD nl = N_tx[lam][t].real();
      RealD r = (nq != 0.0 && nl != 0.0 && nq * nl > 0.0)
                ? std::log(std::abs(nl / nq)) : 0.0;
      std::cout << "  " << std::scientific << std::setprecision(4) << r;
    }
    std::cout << std::defaultfloat << std::endl;
  }

  Grid_finalize();
  return 0;
}
