// Diagnostic: connected pion on TXQCD gauge configs with sigma/s aux frozen at
// their ensemble-mean diagonals (other aux fields zeroed).  In that limit
// M_TX = M_W + <sigma> I_f + <s> I_c collapses to plain Wilson at shifted bare
// mass, so we just invert plain Wilson on the TXQCD gauge field at
//   m_eff = mass + sigma_mean_pf + s_mean_pc
// and compare m_eff(pion) to (i) the unmodified TXQCD connected pion and
// (ii) the QCD connected pion.  Tests whether the connected-pion shift in
// TXQCD is captured by a pure mean-field aux shift, or whether sigma
// fluctuations contribute substantially.
//
// Defaults to wilson8x24 ensemble at LAMBDA=7.76 measured aux VEVs.  Override
// the shift via FREEZE_DM env var.
//
// Writes: meas_<lam>/meas_freeze_conn.h5

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptWilson8x24;

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, src, site);
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

static void QcdPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                         RealD m, GridCartesian &Grid,
                         GridRedBlackCartesian &RBGrid,
                         const Coordinate &src, RealD tol, int maxit) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, m);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit);

  LatticePropagator srcP(&Grid);
  PointSource(src, srcP);
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
  Coordinate src = src_site();
  mkdir_p(meas_dir());

  if (!txqcd_configs_exist()) {
    std::cout << GridLogMessage
              << "[freeze] TXQCD configs not found; nothing to do."
              << std::endl;
    Grid_finalize();
    return 0;
  }

  // FREEZE_DM env var overrides the default mass shift.  Defaults below match
  // the LAMBDA=7.76 wilson8x24 aux VEVs (sigma_pf=0.0483, s_pc=0.0231).
  RealD dm = 0.0714;
  if (const char *v = std::getenv("FREEZE_DM")) dm = std::atof(v);
  RealD m_eff_bare = mass + dm;

  std::cout << GridLogMessage << "[freeze] LAMBDA=" << lambda_runtime()
            << " mass=" << mass << " dm=" << dm
            << " m_eff_bare=" << m_eff_bare << std::endl;

  std::vector<std::vector<RealD>> pion_frozen;
  std::vector<RealD> plaq;

  TXQCDField U(&Grid);
  for (int traj : trajs) {
    std::cout << GridLogMessage << "[freeze] TXQCD traj=" << traj
              << " (Wilson at m_eff=" << m_eff_bare << ")" << std::endl;
    LoadTxqcdConfig(U, sRNG, pRNG, traj);
    plaq.push_back(WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U));
    LatticePropagator S(&Grid);
    QcdPointProp(S, U.U, m_eff_bare, Grid, RBGrid, src, meas_tol, cg_max);
    pion_frozen.push_back(PionCorrelator(S, S));
  }

  Hdf5Writer wr(meas_dir() + "/meas_freeze_conn.h5");
  write(wr, "pion_conn", pion_frozen);
  write(wr, "plaq", plaq);
  write(wr, "dm", dm);
  write(wr, "m_eff_bare", m_eff_bare);

  std::cout << GridLogMessage << "[freeze] wrote " << meas_dir()
            << "/meas_freeze_conn.h5" << std::endl;
  Grid_finalize();
  return 0;
}
