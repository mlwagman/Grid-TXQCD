#include "params.h"
#include "quda_helper.h"
#include "quda_txqcd_helper.h"
#include "meas_helper.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

// Phase M.4 multi-RHS Schur-EO solve (copied from meas_conn_txqcd.cc).  Reduces
// then runs a single lockstep CG over N RHS, then reconstructs each x.
// TODO: refactor into a shared header alongside conn.
static void SchurSolveTxqcdMultiRHS(Grid::TXQCDWilsonCloverFermionEO &Meo,
                                     Grid::GridRedBlackCartesian *rb,
                                     const std::vector<Grid::TXQCDFermionNf> &b,
                                     std::vector<Grid::TXQCDFermionNf> &x,
                                     RealD tol, int max_iter) {
  using namespace Grid;
  int N = (int)b.size();
  GRID_ASSERT(N > 0);
  GRID_ASSERT((int)x.size() == N);

  TXQCDCloverSchurOp SchurOp(Meo);

  std::vector<TXQCDFermionNf> src_o(N, TXQCDFermionNf(rb));
  std::vector<TXQCDFermionNf> b_e(N, TXQCDFermionNf(rb));
  std::vector<TXQCDFermionNf> x_o(N, TXQCDFermionNf(rb));
  for (int j = 0; j < N; ++j) {
    TXQCDFermionNf b_o(rb), tmp_e(rb), tmp_o(rb), rhs_o(rb);
    for (int a = 0; a < TxqcdNf; ++a) {
      pickCheckerboard(Even, b_e[j].f[a], b[j].f[a]);
      pickCheckerboard(Odd,  b_o.f[a],   b[j].f[a]);
      tmp_e.f[a].Checkerboard() = Even;
      tmp_o.f[a].Checkerboard() = Odd;
      rhs_o.f[a].Checkerboard() = Odd;
      x_o[j].f[a].Checkerboard() = Odd;
    }
    Meo.MooeeInv(b_e[j], tmp_e);
    Meo.Meooe(tmp_e, tmp_o);
    for (int a = 0; a < TxqcdNf; ++a) rhs_o.f[a] = b_o.f[a] - tmp_o.f[a];
    SchurOp.MpcDag(rhs_o, src_o[j]);
  }

  TXQCDMultiRHSConjugateGradient CG(tol, max_iter);
  CG(SchurOp, src_o, x_o);
  int max_iter_b = 0, max_idx = 0, min_iter_b = max_iter;
  for (int j = 0; j < N; ++j) {
    if (CG.IterationsToComplete[j] > max_iter_b) {
      max_iter_b = CG.IterationsToComplete[j]; max_idx = j;
    }
    if (CG.IterationsToComplete[j] < min_iter_b)
      min_iter_b = CG.IterationsToComplete[j];
  }
  std::cout << GridLogMessage
            << "[disco SchurMultiRHS] N=" << N
            << "  CG iter min=" << min_iter_b << " max=" << max_iter_b
            << "  slowest_RHS=" << max_idx << std::endl;

  for (int j = 0; j < N; ++j) {
    TXQCDFermionNf tmp_e(rb), x_e(rb);
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp_e.f[a].Checkerboard() = Even;
      x_e.f[a].Checkerboard() = Even;
    }
    Meo.Meooe(x_o[j], tmp_e);
    for (int a = 0; a < TxqcdNf; ++a) tmp_e.f[a] = b_e[j].f[a] - tmp_e.f[a];
    Meo.MooeeInv(tmp_e, x_e);

    for (int a = 0; a < TxqcdNf; ++a) {
      setCheckerboard(x[j].f[a], x_e.f[a]);
      setCheckerboard(x[j].f[a], x_o[j].f[a]);
    }
  }
}

// ---- Original single-RHS estimators (used as fallback / for bit-exact ref) --

static std::vector<ComplexD>
StochasticLoop_ud(Grid::QudaTxqcdPropSolver &solver,
                  GridBase *grid, GridParallelRNG &pRNG, int nn) {
  int T = grid->GlobalDimensions()[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);
  std::vector<ComplexD> L(T, 0.0);

  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf src(grid), x(grid);
    src.f[0] = Zero();
    gaussian(pRNG, src.f[1]);
    solver.solve(src, x);

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

static RealD StochasticTrMinv_TX(Grid::QudaTxqcdPropSolver &solver,
                                 GridBase *grid, GridParallelRNG &pRNG, int nn) {
  RealD V = (RealD)grid->gSites();
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    TXQCDFermionNf eta(grid), x(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, eta.f[a]);
    solver.solve(eta, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
}

static RealD StochasticTrMinv_QCD(WCF &Dw, GridBase *grid,
                                  const LatticeGaugeField &Usm, RealD mass,
                                  GridParallelRNG &pRNG, int nn) {
  RealD V = (RealD)grid->gSites();
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  Grid::QudaPropSolver<WCF> solver(Dw, HermOp, Usm, mass, csw, cg_tol, cg_max);
  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    LatticeFermion eta(grid), x(grid);
    gaussian(pRNG, eta);
    solver.solve(eta, x);
    acc += innerProduct(eta, x).real() / (2.0 * V);
  }
  return acc / nn;
}

// ---- Multi-RHS estimators (TXQCD_DISCO_MULTIRHS=1 / QCD_DISCO_MULTISRC=1) ----
//
// Bit-exact noise vectors vs the single-RHS path: same gaussian-draw order on
// the same pRNG.  Solver path differs (Schur-EO multi-RHS CG vs full-volume
// QudaTxqcdPropSolver), so per-estimator values agree to ~CG tolerance only,
// not bit-for-bit.

static std::vector<ComplexD>
StochasticLoop_ud_multirhs(Grid::TXQCDWilsonCloverFermionEO &Meo,
                            Grid::GridRedBlackCartesian *rb,
                            GridBase *grid, GridParallelRNG &pRNG, int nn,
                            RealD tol, int max_iter) {
  int T = grid->GlobalDimensions()[Nd - 1];
  Gamma g5(Gamma::Algebra::Gamma5);
  std::vector<ComplexD> L(T, 0.0);

  std::vector<TXQCDFermionNf> src;  src.reserve(nn);
  std::vector<TXQCDFermionNf> x;    x.reserve(nn);
  for (int h = 0; h < nn; ++h) {
    src.emplace_back(grid);
    x.emplace_back(grid);
    src[h].f[0] = Zero();
    gaussian(pRNG, src[h].f[1]);
  }

  SchurSolveTxqcdMultiRHS(Meo, rb, src, x, tol, max_iter);

  for (int h = 0; h < nn; ++h) {
    LatticeFermion g5x(grid);
    g5x = g5 * x[h].f[0];
    LatticeComplex lf(grid);
    lf = localInnerProduct(src[h].f[1], g5x);
    std::vector<TComplex> sl;
    sliceSum(lf, sl, Nd - 1);
    for (int t = 0; t < T; ++t)
      L[t] += TensorRemove(sl[t]) / (2.0 * nn);
  }
  return L;
}

static RealD
StochasticTrMinv_TX_multirhs(Grid::TXQCDWilsonCloverFermionEO &Meo,
                              Grid::GridRedBlackCartesian *rb,
                              GridBase *grid, GridParallelRNG &pRNG, int nn,
                              RealD tol, int max_iter) {
  RealD V = (RealD)grid->gSites();
  std::vector<TXQCDFermionNf> eta;  eta.reserve(nn);
  std::vector<TXQCDFermionNf> x;    x.reserve(nn);
  for (int h = 0; h < nn; ++h) {
    eta.emplace_back(grid);
    x.emplace_back(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, eta[h].f[a]);
  }

  SchurSolveTxqcdMultiRHS(Meo, rb, eta, x, tol, max_iter);

  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    acc += innerProduct(eta[h], x[h]).real() / (2.0 * V);
  }
  return acc / nn;
}

static RealD
StochasticTrMinv_QCD_multisrc(Grid::QudaPropSolver<WCF> &solver,
                               GridBase *grid, GridParallelRNG &pRNG, int nn) {
  RealD V = (RealD)grid->gSites();
  std::vector<LatticeFermion> eta;  eta.reserve(nn);
  std::vector<LatticeFermion> x;    x.reserve(nn);
  for (int h = 0; h < nn; ++h) {
    eta.emplace_back(grid);
    x.emplace_back(grid);
    gaussian(pRNG, eta[h]);
  }

  solver.solve_multi(eta, x);

  RealD acc = 0.0;
  for (int h = 0; h < nn; ++h) {
    acc += innerProduct(eta[h], x[h]).real() / (2.0 * V);
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
  load_txqcd_field(traj, Grid, RBGrid, U, sRNG, pRNG);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(U.U);
  LatticeGaugeField Usmeared = SmearInv.get_SmearedU();

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;

  // Env-var gates: same convention as meas_conn (presence-of-key enables).
  bool use_disco_multirhs =
      (std::getenv("TXQCD_DISCO_MULTIRHS") != nullptr) &&
      (std::string(std::getenv("TXQCD_DISCO_MULTIRHS")) != "0");
  bool use_disco_multisrc_qcd =
      (std::getenv("QCD_DISCO_MULTISRC") != nullptr) &&
      (std::string(std::getenv("QCD_DISCO_MULTISRC")) != "0");

  // TXQCD operator (full-volume, used by fallback QudaTxqcdPropSolver).
  TXQCDWilsonCloverOp Mop(Usmeared, Grid, RBGrid, mass_light,
                           U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);
  std::array<RealD, TxqcdNf> mass_arr;
  mass_arr.fill(mass_light);

  // EO operator used by multi-RHS path.  Constructed lazily so the
  // single-RHS fallback path doesn't pay setup cost.
  std::unique_ptr<TXQCDWilsonCloverFermionEO> Meo;
  if (use_disco_multirhs) {
    Meo.reset(new TXQCDWilsonCloverFermionEO(
        Usmeared, Grid, RBGrid, mass_light,
        U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p));
    std::cout << GridLogMessage
              << "[meas_disco_txqcd] TXQCD_DISCO_MULTIRHS=1 — Schur-EO multi-RHS ("
              << n_noise_disco << " RHS lockstep)" << std::endl;
  }

  std::cout << GridLogMessage << "[disco TXQCD] traj=" << traj << std::endl;

  std::vector<ComplexD> loop_ud;
  RealD trminv;
  if (use_disco_multirhs) {
    loop_ud = StochasticLoop_ud_multirhs(*Meo, &RBGrid, &Grid, pRNG,
                                          n_noise_disco,
                                          cg_tol_runtime(), cg_max);
    trminv  = StochasticTrMinv_TX_multirhs(*Meo, &RBGrid, &Grid, pRNG,
                                            n_noise_disco,
                                            cg_tol_runtime(), cg_max);
  } else {
    Grid::QudaTxqcdPropSolver tx_solver(Mop, mass_arr, csw, Usmeared,
                                         cg_tol, cg_max);
    loop_ud = StochasticLoop_ud(tx_solver, &Grid, pRNG, n_noise_disco);
    trminv  = StochasticTrMinv_TX(tx_solver, &Grid, pRNG, n_noise_disco);
  }

  // Strange quark VEV (standard QCD operator on smeared links)
  WCF Dw_s(Usmeared, Grid, RBGrid, mass_strange, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
  RealD trminv_strange;
  if (use_disco_multisrc_qcd) {
    MdagMLinearOperator<WCF, LatticeFermion> HermOp_s(Dw_s);
    Grid::QudaPropSolver<WCF> qcd_solver(Dw_s, HermOp_s, Usmeared,
                                          mass_strange, csw,
                                          cg_tol_runtime(), cg_max);
    std::cout << GridLogMessage
              << "[meas_disco_txqcd] QCD_DISCO_MULTISRC=1 — QUDA multi-src ("
              << n_noise_disco << " RHS batched)" << std::endl;
    trminv_strange = StochasticTrMinv_QCD_multisrc(qcd_solver, &Grid, pRNG,
                                                    n_noise_disco);
  } else {
    trminv_strange = StochasticTrMinv_QCD(Dw_s, &Grid, Usmeared, mass_strange,
                                           pRNG, n_noise_disco);
  }

  // Aux field VEVs
  RealD V = (RealD)Grid.gSites();
  RealD vev_sigma = TensorRemove(sum(trace(U.sigma))).real() / V;
  RealD vev_s = TensorRemove(sum(trace(U.s))).real() / V;

  // Sanity-check metadata: plaq on the unsmeared gauge.
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U);

  std::string outfile = txqcd_data_dir() + "/disco_txqcd_" + std::to_string(traj) + ".h5";
  if (Grid.IsBoss()) {
    Hdf5Writer wr(outfile);
    write(wr, "loop_ud", loop_ud);
    write(wr, "trminv", trminv);
    write(wr, "trminv_strange", trminv_strange);
    write(wr, "vev_sigma", vev_sigma);
    write(wr, "vev_s", vev_s);
    write(wr, "traj", traj);
    write(wr, "plaq", plaq);
    write(wr, "mass_light", mass_light);
    write(wr, "mass_strange", mass_strange);
    write(wr, "lambda", lambda);
    write(wr, "n_noise_disco", n_noise_disco);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
#ifdef GRID_HAVE_QUDA
  Grid::Quda::finalize();
#endif
  Grid_finalize();
  return 0;
}
