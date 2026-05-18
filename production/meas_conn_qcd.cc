#include "params.h"
#include "quda_helper.h"
#include "meas_helper.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/CovariantSmearing.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

static std::vector<LatticeColourMatrix>
ExtractLinks(const LatticeGaugeField &U) {
  std::vector<LatticeColourMatrix> Umu(Nd, U.Grid());
  for (int mu = 0; mu < Nd; ++mu)
    Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
  return Umu;
}

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

// Returns {C_pos(t), C_neg(t)}: nucleon correlator with +parity and
// -parity projectors.  Forward-backward avg in analysis:
//   C_avg(t) = (C_pos(t) - C_neg((T - t) mod T)) / 2
static std::pair<std::vector<ComplexD>, std::vector<ComplexD>>
NucleonCorrelatorPosNeg(const LatticePropagator &S) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  auto contract_proj = [&](int parity) -> std::vector<ComplexD> {
    LatticeComplex Cn(S.Grid());
    BaryonUtils<WilsonImplR>::ContractBaryons(
        S, S, S, G_A, G_B, G_A, G_B, wick, parity, Cn);
    std::vector<TComplex> sl;
    sliceSum(Cn, sl, Nd - 1);
    std::vector<ComplexD> out(sl.size());
    for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
    return out;
  };
  auto pos = contract_proj(+1);
  auto neg = contract_proj(-1);
  return {pos, neg};
}

// Time-reversed (charge-conjugated) quark propagator, chroma's barhqlq recipe:
//   q_TR = -(γ₅γ₄) · S · (γ₅γ₄)
// γ₅γ₄ = γ₁γ₂γ₃ = chroma's Gamma(7).  No time-reflection of the propagator;
// the time-reflection is applied in the OUTPUT INDEX MAPPING with per-t_eff
// APBC signs.  Gives ~√2 noise reduction at plateau when contracted +parity
// and combined with forward via chroma t_eff mapping (see analysis loop).
static LatticePropagator BuildTimeReversedProp(const LatticePropagator &S) {
  Gamma g5(Gamma::Algebra::Gamma5);
  Gamma g4(Gamma::Algebra::GammaT);
  return -(g5 * g4 * S * g5 * g4);
}

// +parity nucleon correlator from a single propagator (used for both forward
// S and time-reversed S_TR).  Returns lattice-time array.
static std::vector<ComplexD> NucleonCorrelatorPos(const LatticePropagator &S) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(
      S, S, S, G_A, G_B, G_A, G_B, wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<Coordinate> SourceGrid(const Coordinate &latt) {
  Coordinate origin = src_grid_origin();
  const int sx = space_src_per_dim_runtime();
  const int st = time_src_per_dim_runtime();
  std::vector<Coordinate> sites;
  for (int ix = 0; ix < sx; ++ix)
    for (int iy = 0; iy < sx; ++iy)
      for (int iz = 0; iz < sx; ++iz)
        for (int it = 0; it < st; ++it) {
          Coordinate s(Nd);
          s[0] = (origin[0] + ix * latt[0] / sx) % latt[0];
          s[1] = (origin[1] + iy * latt[1] / sx) % latt[1];
          s[2] = (origin[2] + iz * latt[2] / sx) % latt[2];
          s[3] = (origin[3] + it * latt[3] / st) % latt[3];
          sites.push_back(s);
        }
  return sites;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_conn_qcd <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  Coordinate latt = lattice_size();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  mkdir_p(qcd_data_dir());

  LatticeGaugeField Umu(&Grid);
  load_qcd_gauge(traj, Umu, sRNG, pRNG);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(Umu);
  LatticeGaugeField U_inv = SmearInv.get_SmearedU();

  // Stout smearing for source/sink Gaussian smearing
  Smear_Stout<PeriodicGimplR> StoutSrc(stout_rho_src);
  SmearedConfiguration<PeriodicGimplR> SmearSrc(&Grid, stout_nsmear_src, StoutSrc);
  SmearSrc.set_Field(Umu);
  LatticeGaugeField U_src = SmearSrc.get_SmearedU();
  auto U_src_links = ExtractLinks(U_src);

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(U_inv, Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  Grid::QudaPropSolver<WCF> solver(Dw, HermOp, U_inv, mass_light, csw, cg_tol, cg_max);

  int T = latt[Nd - 1];
  auto sources = SourceGrid(latt);
  int nsrc = (int)sources.size();

  std::vector<std::vector<RealD>>    all_pion;
  std::vector<std::vector<ComplexD>> all_nucleon;     // C_pos(t) lattice frame
  std::vector<std::vector<ComplexD>> all_nucleon_neg; // C_neg(t) for legacy FB
  // Chroma-style time-reversed correlator (gated by QCD_TIME_REVERSED=1).
  bool use_time_reversed = std::getenv("QCD_TIME_REVERSED") != nullptr;
  std::vector<std::vector<ComplexD>> all_nucleon_tr;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn QCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S(&Grid);
    S = Zero();

    bool use_multi = std::getenv("QCD_MULTISRC") != nullptr;

    if (use_multi) {
      // M.3: build all 12 sources, batch-invert via QUDA invertMultiSrcQuda.
      const int Nrhs = Ns * Nc;
      std::vector<LatticeFermion> sfs, xs;
      sfs.reserve(Nrhs); xs.reserve(Nrhs);
      for (int j = 0; j < Nrhs; ++j) {
        sfs.emplace_back(&Grid);
        xs.emplace_back(&Grid);
      }
      int j = 0;
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col, ++j) {
          LatticePropagator srcP(&Grid); srcP = Zero();
          SpinColourMatrix kron; kron = 1.0;
          pokeSite(kron, srcP, src);
          PropToFerm<WilsonImplR>(sfs[j], srcP, spin, col);
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sfs[j],
                                                           gauss_width, gauss_niter, Nd - 1);
        }
      }
      solver.solve_multi(sfs, xs);
      j = 0;
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col, ++j) {
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, xs[j],
                                                           gauss_width, gauss_niter, Nd - 1);
          FermToProp<WilsonImplR>(S, xs[j], spin, col);
        }
      }
    } else {
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col) {
          LatticePropagator srcP(&Grid);
          srcP = Zero();
          SpinColourMatrix kron;
          kron = 1.0;
          pokeSite(kron, srcP, src);

          LatticeFermion sf(&Grid), x(&Grid);
          PropToFerm<WilsonImplR>(sf, srcP, spin, col);

          // Gaussian smear source
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sf,
                                                           gauss_width, gauss_niter, Nd - 1);

          x = Zero();
          solver.solve(sf, x);

          // Gaussian smear sink
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, x,
                                                           gauss_width, gauss_niter, Nd - 1);

          FermToProp<WilsonImplR>(S, x, spin, col);
        }
      }
    }

    all_pion.push_back(PionCorrelator(S));
    auto pn = NucleonCorrelatorPosNeg(S);
    all_nucleon.push_back(std::move(pn.first));
    all_nucleon_neg.push_back(std::move(pn.second));
    if (use_time_reversed) {
      // Chroma-style TR'd propagator → +parity nucleon contraction.
      LatticePropagator S_TR = BuildTimeReversedProp(S);
      all_nucleon_tr.push_back(NucleonCorrelatorPos(S_TR));
    }
  }

  // Per-source SHIFT to source-relative time, then FB-average.  Doing FB on
  // lattice-time source-averaged correlators is wrong for nsrc>1.
  bool apbc = true;
  auto shift_real = [&](const std::vector<std::vector<RealD>> &raw)
      -> std::vector<std::vector<RealD>> {
    std::vector<std::vector<RealD>> out(nsrc, std::vector<RealD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = (t_eff + ts) % T;
        out[i][t_eff] = raw[i][t_lat];
      }
    }
    return out;
  };
  auto shift_fwd_apbc = [&](const std::vector<std::vector<ComplexD>> &raw)
      -> std::vector<std::vector<ComplexD>> {
    std::vector<std::vector<ComplexD>> out(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = (t_eff + ts) % T;
        double sign_fwd = (apbc && (t_eff + ts) >= T) ? -1.0 : +1.0;
        out[i][t_eff] = sign_fwd * raw[i][t_lat];
      }
    }
    return out;
  };
  // Chroma-style backward shift for time-reversed correlator: t_eff =
  // (T - t_lat + t_s) mod T, sign_back = -1 if (t_eff - t_s) > 0 && APBC.
  auto shift_tr_chroma = [&](const std::vector<std::vector<ComplexD>> &raw)
      -> std::vector<std::vector<ComplexD>> {
    std::vector<std::vector<ComplexD>> out(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i) {
      int ts = sources[i][Nd - 1];
      for (int t_eff = 0; t_eff < T; ++t_eff) {
        int t_lat = ((T - t_eff + ts) % T + T) % T;
        double sign_back = (apbc && (t_eff - ts) > 0) ? -1.0 : +1.0;
        out[i][t_eff] = sign_back * raw[i][t_lat];
      }
    }
    return out;
  };
  auto pion_s     = shift_real(all_pion);
  auto nucl_pos_s = shift_fwd_apbc(all_nucleon);
  auto nucl_neg_s = shift_fwd_apbc(all_nucleon_neg);

  // Chroma-style FB averaging (gives ~√2 noise reduction at plateau).
  std::vector<std::vector<ComplexD>> all_nucl_tr_s, all_nucl_fb_chroma;
  if (use_time_reversed) {
    all_nucl_tr_s = shift_tr_chroma(all_nucleon_tr);
    all_nucl_fb_chroma.assign(nsrc, std::vector<ComplexD>(T, 0.0));
    for (int i = 0; i < nsrc; ++i)
      for (int t = 0; t < T; ++t)
        all_nucl_fb_chroma[i][t] = 0.5 * (nucl_pos_s[i][t] + all_nucl_tr_s[i][t]);
  }

  std::vector<RealD> pion_avg(T, 0.0);
  std::vector<ComplexD> nucl_avg(T, 0.0);
  std::vector<ComplexD> nucl_neg_avg(T, 0.0);
  std::vector<ComplexD> nucl_fbavg(T, 0.0);                 // legacy (Grid parity)
  std::vector<ComplexD> nucl_fb_chroma_avg(T, 0.0);         // chroma-TR FB
  for (int i = 0; i < nsrc; ++i)
    for (int t = 0; t < T; ++t) {
      pion_avg[t] += pion_s[i][t] / nsrc;
      nucl_avg[t] += nucl_pos_s[i][t] / (double)nsrc;
      nucl_neg_avg[t] += nucl_neg_s[i][t] / (double)nsrc;
      if (use_time_reversed)
        nucl_fb_chroma_avg[t] += all_nucl_fb_chroma[i][t] / (double)nsrc;
    }
  for (int t = 0; t < T; ++t) {
    int trev = (T - t) % T;
    nucl_fbavg[t] = 0.5 * (nucl_avg[t] - nucl_neg_avg[trev]);
  }

  std::string outfile = qcd_data_dir() + "/conn_qcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    // All averaged correlators are in SOURCE-RELATIVE TIME with APBC sign.
    write(wr, "pion_conn", pion_avg);
    write(wr, "nucleon", nucl_avg);
    write(wr, "nucleon_neg", nucl_neg_avg);
    write(wr, "nucleon_fbavg", nucl_fbavg);                  // legacy (Grid parity)
    if (use_time_reversed)
      write(wr, "nucleon_fb_chroma", nucl_fb_chroma_avg);    // ~√2 noise reduction
    // Per-source SHIFTED with APBC sign (index 0 = each source's t_src).
    write(wr, "pion_per_src", pion_s);
    write(wr, "nucleon_per_src", nucl_pos_s);
    write(wr, "nucleon_neg_per_src", nucl_neg_s);
    if (use_time_reversed) {
      write(wr, "nucleon_tr_per_src", all_nucl_tr_s);
      write(wr, "nucleon_fb_chroma_per_src", all_nucl_fb_chroma);
    }
    // Lattice-time raw per-src for back-compat / debugging.
    write(wr, "nucleon_per_src_lat", all_nucleon);
    write(wr, "nucleon_neg_per_src_lat", all_nucleon_neg);
    if (use_time_reversed)
      write(wr, "nucleon_tr_per_src_lat", all_nucleon_tr);
    write(wr, "traj", traj);
    write(wr, "plaq", WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu));
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
#ifdef GRID_HAVE_QUDA
  Grid::Quda::finalize();
#endif
  Grid_finalize();
  return 0;
}
