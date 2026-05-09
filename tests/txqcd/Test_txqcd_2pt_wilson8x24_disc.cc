// Step 3: Compute disconnected pion loops and VEVs from stochastic noise sources.
//
// Disconnected pion: per-timeslice off-diagonal loop
//   L_ud(t) = sum_{x in t} Tr_sc[G_{ud}(x,x) gamma_5]
// estimated with Gaussian volume noise. The comparison program forms the
// correlator Disc(dt) = (1/V) sum_{t0} L_ud(t0+dt) conj(L_ud(t0))
// using gamma_5-Hermiticity: L_du(t) = conj(L_ud(t)).
//
// VEVs: <Tr sigma>/V, <Tr s>/V (direct from aux fields),
//        stochastic Re Tr M^{-1}/V (TXQCD and QCD).
//
// Writes: meas_2pt/meas_{txqcd,qcd}_disc.h5

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/serialisation/Hdf5IO.h>

using namespace TxqcdTest2ptWilson8x24;

static void TxqcdCG(TXQCDWilsonOp &Mop, const TXQCDFermionNf &b,
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

// Per-timeslice off-diagonal loop L_ud(t) via Gaussian noise on flavor d.
// Grid's gaussian gives E[|eta|^2]=2 per complex DOF; 1/2 corrects this.
static std::vector<ComplexD>
StochasticLoop_ud(TXQCDWilsonOp &Mop, GridBase *grid, GridParallelRNG &pRNG,
                  int nn, RealD tol, int maxit) {
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

// Stochastic Re Tr M^{-1} / V for TXQCDWilsonOp.
static RealD StochasticTrMinv_TX(TXQCDWilsonOp &Mop, GridBase *grid,
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

// Stochastic Re Tr M_W^{-1} / V for single-flavor Wilson.
static RealD StochasticTrMinv_QCD(WilsonFermionD &Dw, GridBase *grid,
                                  GridParallelRNG &pRNG, int nn, RealD tol,
                                  int maxit) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
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
  std::vector<RealD> polyakov_re_tx, polyakov_im_tx;
  std::vector<RealD> polyakov_re_qcd, polyakov_im_qcd;

  // WHICH={qcd,txqcd,both} env override -- skip TXQCD or QCD if cfgs of
  // the other side don't exist yet (so we can measure QCD Sigma_l first
  // before generating TXQCD with the resulting lambda_opt).
  std::string WHICH = "both";
  if (const char *v = std::getenv("WHICH"); v && *v) WHICH = v;
  bool do_qcd   = (WHICH == "qcd"   || WHICH == "both");
  bool do_txqcd = (WHICH == "txqcd" || WHICH == "both");
  // Auto-skip if cfgs missing.
  if (!txqcd_configs_exist()) do_txqcd = false;
  if (!qcd_configs_exist())   do_qcd   = false;
  std::cout << GridLogMessage << "[disc] WHICH=" << WHICH
            << " do_qcd=" << do_qcd << " do_txqcd=" << do_txqcd << std::endl;

  // TXQCD: loops + VEVs
  if (do_txqcd) {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc] TXQCD traj=" << traj << std::endl;

      sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
      pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});
      LoadTxqcdConfig(U, sRNG, pRNG, traj);

      {
        ComplexD poly = WilsonLoops<PeriodicGimplR>::avgPolyakovLoop(U.U);
        polyakov_re_tx.push_back(poly.real());
        polyakov_im_tx.push_back(poly.imag());
      }

      GridCartesian *Ug = dynamic_cast<GridCartesian *>(U.Grid());
      GridRedBlackCartesian RB(Ug);
      TXQCDWilsonOp Mop(U.U, *Ug, RB, mass_runtime(), U.sigma, U.pi, U.s, U.p, U.t);

      loop_ud.push_back(StochasticLoop_ud(Mop, &Grid, pRNG, n_noise,
                                           meas_tol, cg_max));

      RealD V = (RealD)Grid.gSites();
      vev_sigma.push_back(TensorRemove(sum(trace(U.sigma))).real() / V);
      vev_s.push_back(TensorRemove(sum(trace(U.s))).real() / V);
      trminv_tx.push_back(
          StochasticTrMinv_TX(Mop, &Grid, pRNG, n_noise, meas_tol, cg_max));
    }
  }

  // QCD: stochastic Tr M_W^{-1} / V
  if (do_qcd) {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[disc] QCD traj=" << traj << std::endl;

      sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
      pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});
      LoadQcdConfig(Umu, sRNG, pRNG, traj);

      {
        ComplexD poly = WilsonLoops<PeriodicGimplR>::avgPolyakovLoop(Umu);
        polyakov_re_qcd.push_back(poly.real());
        polyakov_im_qcd.push_back(poly.imag());
      }

      WilsonFermionD Dw(Umu, Grid, RBGrid, mass_runtime());
      trminv_qcd.push_back(
          StochasticTrMinv_QCD(Dw, &Grid, pRNG, n_noise, meas_tol, cg_max));
    }
  }

  if (do_txqcd) {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_disc.h5");
    write(wr, "loop_ud", loop_ud);
    write(wr, "vev_sigma", vev_sigma);
    write(wr, "vev_s", vev_s);
    write(wr, "trminv", trminv_tx);
    write(wr, "polyakov_re", polyakov_re_tx);
    write(wr, "polyakov_im", polyakov_im_tx);
  }
  if (do_qcd) {
    Hdf5Writer wr(meas_dir() + "/meas_qcd_disc.h5");
    write(wr, "trminv", trminv_qcd);
    write(wr, "polyakov_re", polyakov_re_qcd);
    write(wr, "polyakov_im", polyakov_im_qcd);
  }

  std::cout << GridLogMessage << "Disconnected + VEV measurements written to "
            << meas_dir() << "/*.h5" << std::endl;
  Grid_finalize();
  return 0;
}
