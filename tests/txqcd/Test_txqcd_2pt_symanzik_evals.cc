// Eigenvalue measurement: smallest eigenvalues of M†M (= λ²(γ₅M))
// for light and strange WilsonClover operators on stout-smeared configs.
// Near-zero eigenvalues signal sign(det(M)) fluctuations requiring reweighting.

#include "Test_txqcd_2pt_symanzik_utils.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedLanczos.h>

using namespace TxqcdTest2ptSymanzik;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static std::vector<RealD>
SmallestEvalsMdagM(WCF &Dw, GridCartesian &Grid, GridParallelRNG &pRNG, int Nev) {
  MdagMLinearOperator<WCF, LatticeFermion> MdagM(Dw);
  int Nk = Nev, Nm = std::max(Nev * 8, 40), Nstop = Nev;
  Chebyshev<LatticeFermion> Cheb(15.0, 65.0, 51);
  FunctionHermOp<LatticeFermion> ChebyOp(Cheb, MdagM);
  PlainHermOp<LatticeFermion>    PlainOp(MdagM);
  ImplicitlyRestartedLanczos<LatticeFermion> IRL(ChebyOp, PlainOp,
                                                  Nstop, Nk, Nm, 1e-8, 10000);
  std::vector<RealD> eval(Nm);
  std::vector<LatticeFermion> evec(Nm, &Grid);
  LatticeFermion src(&Grid);
  gaussian(pRNG, src);
  int Nconv = 0;
  IRL.calc(eval, evec, src, Nconv, false);
  eval.resize(std::min(Nconv, Nev));
  std::sort(eval.begin(), eval.end());
  for (auto &e : eval) e = std::sqrt(std::abs(e));
  return eval;
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

  auto trajs = meas_trajs();
  mkdir_p(meas_dir());

  const int Nev = 3;

  Smear_Stout<PeriodicGimplR> Stout(stout_rho);
  SmearedConfiguration<PeriodicGimplR> SmearPolicy(&Grid, stout_nsmear, Stout);

  std::vector<std::vector<RealD>> evals_light_tx, evals_strange_tx;
  std::vector<std::vector<RealD>> evals_light_qcd, evals_strange_qcd;

  {
    TXQCDField U(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[evals] TXQCD traj=" << traj << std::endl;
      LoadTxqcdConfig(U, sRNG, pRNG, traj);
      SmearPolicy.set_Field(U.U);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      pRNG.SeedFixedIntegers({100, 200, 300, 400, 500});

      WCF Dw_light(Usmeared, Grid, RBGrid, mass, csw, csw);
      auto ev_l = SmallestEvalsMdagM(Dw_light, Grid, pRNG, Nev);
      evals_light_tx.push_back(ev_l);
      std::cout << GridLogMessage << "  light |lambda(g5M)|:";
      for (auto e : ev_l) std::cout << " " << e;
      std::cout << std::endl;

      WCF Dw_strange(Usmeared, Grid, RBGrid, mass_s, csw, csw);
      auto ev_s = SmallestEvalsMdagM(Dw_strange, Grid, pRNG, Nev);
      evals_strange_tx.push_back(ev_s);
      std::cout << GridLogMessage << "  strange |lambda(g5M)|:";
      for (auto e : ev_s) std::cout << " " << e;
      std::cout << std::endl;
    }
  }

  {
    LatticeGaugeField Umu(&Grid);
    for (int traj : trajs) {
      std::cout << GridLogMessage << "[evals] QCD traj=" << traj << std::endl;
      LoadQcdConfig(Umu, sRNG, pRNG, traj);
      SmearPolicy.set_Field(Umu);
      LatticeGaugeField Usmeared = SmearPolicy.get_SmearedU();

      pRNG.SeedFixedIntegers({100, 200, 300, 400, 500});

      WCF Dw_light(Usmeared, Grid, RBGrid, mass, csw, csw);
      auto ev_l = SmallestEvalsMdagM(Dw_light, Grid, pRNG, Nev);
      evals_light_qcd.push_back(ev_l);
      std::cout << GridLogMessage << "  light |lambda(g5M)|:";
      for (auto e : ev_l) std::cout << " " << e;
      std::cout << std::endl;

      WCF Dw_strange(Usmeared, Grid, RBGrid, mass_s, csw, csw);
      auto ev_s = SmallestEvalsMdagM(Dw_strange, Grid, pRNG, Nev);
      evals_strange_qcd.push_back(ev_s);
      std::cout << GridLogMessage << "  strange |lambda(g5M)|:";
      for (auto e : ev_s) std::cout << " " << e;
      std::cout << std::endl;
    }
  }

  {
    Hdf5Writer wr(meas_dir() + "/meas_txqcd_evals.h5");
    write(wr, "evals_light", evals_light_tx);
    write(wr, "evals_strange", evals_strange_tx);
  }
  {
    Hdf5Writer wr(meas_dir() + "/meas_qcd_evals.h5");
    write(wr, "evals_light", evals_light_qcd);
    write(wr, "evals_strange", evals_strange_qcd);
  }

  RealD min_eval = 1e10;
  for (auto &ev : evals_light_tx)
    for (auto e : ev) min_eval = std::min(min_eval, e);
  for (auto &ev : evals_strange_tx)
    for (auto e : ev) min_eval = std::min(min_eval, e);
  for (auto &ev : evals_light_qcd)
    for (auto e : ev) min_eval = std::min(min_eval, e);
  for (auto &ev : evals_strange_qcd)
    for (auto e : ev) min_eval = std::min(min_eval, e);

  std::cout << GridLogMessage << "Eigenvalue measurements written to "
            << meas_dir() << "/*.h5" << std::endl;
  std::cout << GridLogMessage << "Smallest |lambda(g5*M)| = " << min_eval << std::endl;
  if (min_eval < 0.01)
    std::cout << GridLogMessage << "WARNING: near-zero eigenvalue detected — "
              << "potential sign(det(M)) fluctuation!" << std::endl;
  else
    std::cout << GridLogMessage << "All eigenvalues well away from zero." << std::endl;

  Grid_finalize();
  return 0;
}
