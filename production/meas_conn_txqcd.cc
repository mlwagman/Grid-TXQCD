#include "params.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/CovariantSmearing.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

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
  int it;
  for (it = 0; it < maxit; ++it) {
    Mop.M(p, Mp);
    Mop.Mdag(Mp, MdMp);
    ComplexD pAp = innerProduct(p, MdMp);
    ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
    axpy(x, alpha, p);
    axpy(r, -alpha, MdMp);
    RealD rsq_new = norm2(r);
    if (rsq_new < tol2) break;
    RealD beta_cg = rsq_new / rsq;
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
    rsq = rsq_new;
  }
}

static std::vector<LatticeColourMatrix>
ExtractLinks(const LatticeGaugeField &U) {
  std::vector<LatticeColourMatrix> Umu(Nd, U.Grid());
  for (int mu = 0; mu < Nd; ++mu)
    Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
  return Umu;
}

static std::vector<RealD> PionCorrelator(const LatticePropagator &S_d,
                                         const LatticePropagator &S_u) {
  LatticeComplex corr(S_d.Grid());
  corr = trace(S_d * adj(S_u));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

static std::vector<ComplexD> NucleonCorrelator(const LatticePropagator &S_u,
                                               const LatticePropagator &S_d) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S_u.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(S_u, S_u, S_d, G_A, G_B, G_A, G_B,
                                            wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<Coordinate> SourceGrid(const Coordinate &latt) {
  Coordinate origin = src_grid_origin();
  std::vector<Coordinate> sites;
  for (int ix = 0; ix < src_per_dim; ++ix)
    for (int iy = 0; iy < src_per_dim; ++iy)
      for (int iz = 0; iz < src_per_dim; ++iz)
        for (int it = 0; it < src_per_dim; ++it) {
          Coordinate s(Nd);
          s[0] = (origin[0] + ix * latt[0] / src_per_dim) % latt[0];
          s[1] = (origin[1] + iy * latt[1] / src_per_dim) % latt[1];
          s[2] = (origin[2] + iz * latt[2] / src_per_dim) % latt[2];
          s[3] = (origin[3] + it * latt[3] / src_per_dim) % latt[3];
          sites.push_back(s);
        }
  return sites;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_conn_txqcd <traj>" << std::endl;
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

  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  mkdir_p(data_dir());

  TXQCDField U(&Grid);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                txqcd_cfg_dir() + "/ckpoint_lat",
                                txqcd_cfg_dir() + "/ckpoint_rng", traj);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(U.U);
  LatticeGaugeField U_inv = SmearInv.get_SmearedU();

  // Stout smearing for source/sink Gaussian smearing
  Smear_Stout<PeriodicGimplR> StoutSrc(stout_rho_src);
  SmearedConfiguration<PeriodicGimplR> SmearSrc(&Grid, stout_nsmear_src, StoutSrc);
  SmearSrc.set_Field(U.U);
  LatticeGaugeField U_src = SmearSrc.get_SmearedU();
  auto U_src_links = ExtractLinks(U_src);

  // Build TXQCD operator on inversion-smeared links
  TXQCDWilsonCloverOp Mop(U_inv, Grid, RBGrid, mass_light,
                           U.sigma, U.pi, U.s, U.p, U.t, csw);

  int T = latt[Nd - 1];
  auto sources = SourceGrid(latt);
  int nsrc = (int)sources.size();

  std::vector<std::vector<RealD>>    all_pion;
  std::vector<std::vector<ComplexD>> all_nucleon;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn TXQCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S_u(&Grid), S_d(&Grid);
    S_u = Zero();
    S_d = Zero();

    for (int flavor = 0; flavor < TxqcdNf; ++flavor) {
      LatticePropagator &Sout = (flavor == 0) ? S_u : S_d;
      for (int spin = 0; spin < Ns; ++spin) {
        for (int col = 0; col < Nc; ++col) {
          // Point source
          LatticePropagator srcP(&Grid);
          srcP = Zero();
          SpinColourMatrix kron;
          kron = 1.0;
          pokeSite(kron, srcP, src);

          LatticeFermion sf(&Grid);
          PropToFerm<WilsonImplR>(sf, srcP, spin, col);

          // Gaussian smear source
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sf,
                                                           gauss_width, gauss_niter, Nd - 1);

          // Solve M†M x = M† src
          TXQCDFermionNf snf(U.Grid()), b(U.Grid()), x(U.Grid());
          snf.f[0] = Zero();
          snf.f[1] = Zero();
          snf.f[flavor] = sf;
          Mop.Mdag(snf, b);
          TxqcdCG(Mop, b, x, cg_tol, cg_max);

          // Gaussian smear sink
          CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, x.f[flavor],
                                                           gauss_width, gauss_niter, Nd - 1);

          FermToProp<WilsonImplR>(Sout, x.f[flavor], spin, col);
        }
      }
    }

    all_pion.push_back(PionCorrelator(S_d, S_u));
    all_nucleon.push_back(NucleonCorrelator(S_u, S_d));
  }

  // Average over sources
  std::vector<RealD> pion_avg(T, 0.0);
  std::vector<ComplexD> nucl_avg(T, 0.0);
  for (int i = 0; i < nsrc; ++i)
    for (int t = 0; t < T; ++t) {
      pion_avg[t] += all_pion[i][t] / nsrc;
      nucl_avg[t] += all_nucleon[i][t] / (double)nsrc;
    }

  std::string outfile = data_dir() + "/conn_txqcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    write(wr, "pion_conn", pion_avg);
    write(wr, "nucleon", nucl_avg);
    write(wr, "pion_per_src", all_pion);
    write(wr, "nucleon_per_src", all_nucleon);
    write(wr, "traj", traj);
    write(wr, "plaq", WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U));
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
