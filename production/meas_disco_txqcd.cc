#include "params.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

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
  for (int it = 0; it < maxit; ++it) {
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

static std::vector<ComplexD>
StochasticLoop_ud(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                  GridParallelRNG &pRNG, int nn) {
  int T = grid->GlobalDimensions()[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);
  std::vector<ComplexD> L(T, 0.0);

  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf src(grid), b(grid), x(grid);
    src.f[0] = Zero();
    gaussian(pRNG, src.f[1]);
    Mop.Mdag(src, b);
    TxqcdCG(Mop, b, x, cg_tol, cg_max);

    LatticeFermion g5x(grid);
    g5x = g5 * x.f[0];
    LatticeComplex lf(grid);
    lf = localInnerProduct(src.f[1], g5x);
    std::vector<TComplex> sl;
    sliceSum(lf, sl, Nd - 1);
    for (int t = 0; t < T; ++t)
      L[t] += TensorRemove(sl[t]) / (2.0 * nn);
  }
  return L;
}

static RealD StochasticTrMinv_TX(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                                 GridParallelRNG &pRNG, int nn) {
  RealD V = (RealD)grid->gSites();
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf eta(grid), b(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, eta.f[a]);
    Mop.Mdag(eta, b);
    TxqcdCG(Mop, b, x, cg_tol, cg_max);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
}

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static RealD StochasticTrMinv_QCD(WCF &Dw, GridBase *grid,
                                  GridParallelRNG &pRNG, int nn) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, cg_max);
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    LatticeFermion eta(grid), b(grid), x(grid);
    gaussian(pRNG, eta);
    Dw.Mdag(eta, b);
    x = Zero();
    CG(HermOp, b, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_disco_txqcd <traj>" << std::endl;
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

  mkdir_p(txqcd_data_dir());

  TXQCDField U(&Grid);
  TXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                txqcd_cfg_dir() + "/ckpoint_lat",
                                txqcd_cfg_dir() + "/ckpoint_rng", traj);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(U.U);
  LatticeGaugeField Usmeared = SmearInv.get_SmearedU();

  TXQCDWilsonCloverOp Mop(Usmeared, Grid, RBGrid, mass_light,
                           U.sigma, U.pi, U.s, U.p, U.t, csw);

  std::cout << GridLogMessage << "[disco TXQCD] traj=" << traj << std::endl;

  auto loop_ud = StochasticLoop_ud(Mop, &Grid, pRNG, n_noise_disco);
  RealD trminv = StochasticTrMinv_TX(Mop, &Grid, pRNG, n_noise_disco);

  // Strange quark VEV (standard QCD operator on smeared links)
  WCF Dw_s(Usmeared, Grid, RBGrid, mass_strange, csw, csw);
  RealD trminv_strange = StochasticTrMinv_QCD(Dw_s, &Grid, pRNG, n_noise_disco);

  // Aux field VEVs
  RealD V = (RealD)Grid.gSites();
  RealD vev_sigma = TensorRemove(sum(trace(U.sigma))).real() / V;
  RealD vev_s = TensorRemove(sum(trace(U.s))).real() / V;

  std::string outfile = txqcd_data_dir() + "/disco_txqcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    write(wr, "loop_ud", loop_ud);
    write(wr, "trminv", trminv);
    write(wr, "trminv_strange", trminv_strange);
    write(wr, "vev_sigma", vev_sigma);
    write(wr, "vev_s", vev_s);
    write(wr, "traj", traj);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
