// meas_conn_dtxqcd: connected pion + nucleon correlators on existing DTXQCD
// configs.  Sibling of meas_conn_txqcd.cc and meas_conn_qcd.cc.
//
// Uses plain Wilson Dirac on the DTXQCD gauge (no aux fields in the inverter)
// -- the goal is to see whether the gauge ensemble has been shifted by aux
// backreaction in a way that affects standard QCD observables (effective
// pion mass, nucleon mass).  For the doubled DTXQCD operator with full aux
// insertions, use a different binary; this one isolates the GAUGE-only diff.
//
// Mirrors meas_conn_qcd.cc structure (point sources on a multi-src grid,
// optional time-reversed nucleon, FB-averaging) but loads gauges from a
// DTXQCDCheckpointer-format checkpoint (gauge + aux + RNG sidecars).
//
// Usage:
//   ./meas_conn_dtxqcd <traj> [--grid LxLyLzLt] [--mpi mxmymzmt]
//
// Env knobs:
//   CFG_DIR  : override default cfgs/dtxqcd_<lambda> input directory
//   DATA_DIR : override default meas_2pt/dtxqcd_<lambda> output directory
//   MEAS_SPACE_SRC, MEAS_TIME_SRC : source grid density per dim (see params.h)
//   QCD_TIME_REVERSED=1 : also save the chroma-style time-reversed FB nucleon

#include "params.h"
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

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

// Returns {C_pos(t), C_neg(t)} -- nucleon +parity and -parity projectors.
// Forward-backward average in analysis:
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

// Plain Wilson point propagator on the DTXQCD gauge (aux ignored).
static void WilsonPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                            RealD m, GridCartesian &Grid,
                            GridRedBlackCartesian &RBGrid,
                            const Coordinate &src, RealD tol, int maxit) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, m);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);

  LatticePropagator srcP(&Grid);
  srcP = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, srcP, src);
  S = Zero();
  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticeFermion sf(&Grid), b(&Grid), x(&Grid);
      PropToFerm<WilsonImplR>(sf, srcP, spin, col);
      Dw.Mdag(sf, b);
      x = Zero();
      CG(HermOp, b, x);
      FermToProp<WilsonImplR>(S, x, spin, col);
    }
  }
}

static std::vector<Coordinate> SourceGrid(const Coordinate &latt, int traj) {
  Coordinate origin = src_grid_origin(traj);
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
    std::cerr << "Usage: meas_conn_dtxqcd <traj>" << std::endl;
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

  std::string cfg_dir = dtxqcd_cfg_dir();
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = std::string(d);
  std::string data_dir = dtxqcd_data_dir();
  if (const char *d = std::getenv("DATA_DIR"); d && *d) data_dir = std::string(d);
  mkdir_p(data_dir);

  const RealD lam = lambda_dtxqcd;
  const RealD m   = mass_light_dtxqcd;
  const RealD tol = TXQCDProduction::detail::env_real("CG_TOL", 1e-10);
  const int   cgmax = TXQCDProduction::detail::env_int("CG_MAX", 100000);

  int T = latt[Nd - 1];
  std::cout << GridLogMessage << "======== meas_conn_dtxqcd ========" << std::endl;
  std::cout << GridLogMessage << "  traj    = " << traj << std::endl;
  std::cout << GridLogMessage << "  lambda  = " << lam << std::endl;
  std::cout << GridLogMessage << "  mass    = " << m << std::endl;
  std::cout << GridLogMessage << "  lattice = " << latt[0] << "x" << latt[1]
            << "x" << latt[2] << "x" << latt[3] << "  T=" << T << std::endl;
  std::cout << GridLogMessage << "  cfg dir = " << cfg_dir << std::endl;
  std::cout << GridLogMessage << "  out dir = " << data_dir << std::endl;
  std::cout << GridLogMessage << "===================================" << std::endl;

  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);

  auto sources = SourceGrid(latt, traj);
  int nsrc = (int)sources.size();
  std::cout << GridLogMessage << "  nsrc    = " << nsrc << std::endl;

  std::vector<std::vector<RealD>>    all_pion;
  std::vector<std::vector<ComplexD>> all_nucl_pos, all_nucl_neg;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn DTXQCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S(&Grid);
    WilsonPointProp(S, U.U, m, Grid, RBGrid, src, tol, cgmax);
    all_pion.push_back(PionCorrelator(S));
    auto pn = NucleonCorrelatorPosNeg(S);
    all_nucl_pos.push_back(std::move(pn.first));
    all_nucl_neg.push_back(std::move(pn.second));
  }

  // Per-source SHIFT to source-relative time + APBC sign for nucleon.
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
  auto pion_s     = shift_real(all_pion);
  auto nucl_pos_s = shift_fwd_apbc(all_nucl_pos);
  auto nucl_neg_s = shift_fwd_apbc(all_nucl_neg);

  std::vector<RealD>    pion_avg(T, 0.0);
  std::vector<ComplexD> nucl_avg(T, 0.0), nucl_neg_avg(T, 0.0);
  std::vector<ComplexD> nucl_fbavg(T, 0.0);
  for (int i = 0; i < nsrc; ++i)
    for (int t = 0; t < T; ++t) {
      pion_avg[t]     += pion_s[i][t] / nsrc;
      nucl_avg[t]     += nucl_pos_s[i][t] / (double)nsrc;
      nucl_neg_avg[t] += nucl_neg_s[i][t] / (double)nsrc;
    }
  for (int t = 0; t < T; ++t) {
    int trev = (T - t) % T;
    nucl_fbavg[t] = 0.5 * (nucl_avg[t] - nucl_neg_avg[trev]);
  }

  std::string outfile = data_dir + "/conn_dtxqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    // SOURCE-RELATIVE TIME, APBC sign applied.
    write(wr, "pion_conn",     pion_avg);
    write(wr, "nucleon",       nucl_avg);
    write(wr, "nucleon_neg",   nucl_neg_avg);
    write(wr, "nucleon_fbavg", nucl_fbavg);
    write(wr, "pion_per_src",    pion_s);
    write(wr, "nucleon_per_src", nucl_pos_s);
    write(wr, "nucleon_neg_per_src", nucl_neg_s);
    write(wr, "traj",   traj);
    write(wr, "lambda", lam);
    write(wr, "mass",   m);
    write(wr, "plaq",   plaq);
    Coordinate s = src_grid_origin(traj);
    std::vector<int> sv(Nd);
    for (int d = 0; d < Nd; ++d) sv[d] = s[d];
    write(wr, "src_shift", sv);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
