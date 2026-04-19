// Step (Symanzik): Disconnected pion loops and VEVs using Wilson-Clover.

#include "Test_txqcd_2pt_symanzik_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace TxqcdTest2ptSymanzik;

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
    if (rsq_new < tol2) { rsq = rsq_new; break; }
    RealD beta_cg = rsq_new / rsq;
    for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
    rsq = rsq_new;
  }
}

static std::vector<ComplexD>
StochasticLoop_ud(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                  GridParallelRNG &pRNG, int nn, RealD tol, int maxit) {
  int T = grid->GlobalDimensions()[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);
  std::vector<ComplexD> L(T, 0.0);

  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf src(grid), b(grid), x(grid);
    src.f[0] = Zero();
    gaussian(pRNG, src.f[1]);
    Mop.Mdag(src, b);
    TxqcdCG(Mop, b, x, tol, maxit);

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
                                 GridParallelRNG &pRNG, int nn, RealD tol,
                                 int maxit) {
  RealD V = (RealD)grid->gSites();
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf eta(grid), b(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, eta.f[a]);
    Mop.Mdag(eta, b);
    TxqcdCG(Mop, b, x, tol, maxit);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
}

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
static RealD StochasticTrMinv_QCD(WCF &Dw, GridBase *grid,
                                  GridParallelRNG &pRNG, int nn, RealD tol,
                                  int maxit) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);
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

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);

  int T = latt[Nd - 1];
  auto trajs = meas_trajs();
  mkdir_p(meas_dir());

  std::vector<std::vector<ComplexD>> loop_ud;
  std::vector<RealD> vev_sigma, vev_s, trminv_tx, trminv_qcd;
  std::vector<RealD> trminv_strange_tx, trminv_strange_qcd;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  SmearedConfiguration<PeriodicGimplR> SmearPolicy(&Grid, stout_nsmear, Stout);

  // TXQCD
  {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc] TXQCD symanzik traj=" << traj << std::endl;

      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      LoadTxqcdConfig(U, sRNG, pRNG, traj);

      SmearPolicy.set_Field(U.U);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      GridCartesian *Ug = dynamic_cast<GridCartesian *>(U.Grid());
      GridRedBlackCartesian RB(Ug);
      TXQCDWilsonCloverOp Mop(Usmeared, *Ug, RB, mass, U.sigma, U.pi, U.s,
                               U.p, U.t, csw);

      loop_ud.push_back(StochasticLoop_ud(Mop, &Grid, pRNG, n_noise,
                                           meas_tol, cg_max));

      RealD V = (RealD)Grid.gSites();
      vev_sigma.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
      vev_s.push_back(TensorRemove(sum(trace(U.s))).real() / V);
      trminv_tx.push_back(
          StochasticTrMinv_TX(Mop, &Grid, pRNG, n_noise, meas_tol, cg_max));

      WCF Dw_s(Usmeared, Grid, RBGrid, mass_s, csw, csw);
      trminv_strange_tx.push_back(
          StochasticTrMinv_QCD(Dw_s, &Grid, pRNG, n_noise, meas_tol, cg_max));
    }
  }

  // QCD
  {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc] QCD symanzik traj=" << traj << std::endl;

      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      LoadQcdConfig(Umu, sRNG, pRNG, traj);

      SmearPolicy.set_Field(Umu);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      WCF Dw(Usmeared, Grid, RBGrid, mass, csw, csw);
      trminv_qcd.push_back(
          StochasticTrMinv_QCD(Dw, &Grid, pRNG, n_noise, meas_tol, cg_max));

      WCF Dw_s(Usmeared, Grid, RBGrid, mass_s, csw, csw);
      trminv_strange_qcd.push_back(
          StochasticTrMinv_QCD(Dw_s, &Grid, pRNG, n_noise, meas_tol, cg_max));
    }
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_disc.h5");
    write(wr, "loop_ud", loop_ud);
    write(wr, "vev_sigma", vev_sigma);
    write(wr, "vev_s", vev_s);
    write(wr, "trminv", trminv_tx);
    write(wr, "trminv_strange", trminv_strange_tx);
  }
  {
    Hdf5Writer wr(meas_dir() + "/meas_qcd_disc.h5");
    write(wr, "trminv", trminv_qcd);
    write(wr, "trminv_strange", trminv_strange_qcd);
  }

  std::cout << GridLogMessage << "Disconnected + VEV symanzik measurements written to "
            << meas_dir() << "/*.h5" << std::endl;
  Grid_finalize();
  return 0;
}
