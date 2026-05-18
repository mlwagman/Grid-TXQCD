// Validate Schur EO single-shift CG produces same answer as full-volume MdagM CG.
//
// Builds both Mop (full-vol) and Meo (EO) with the same gauge + aux fields,
// solves the same source via both, and checks cos(x_eo, x_fv) ≥ 0.99999.
//
// Run:
//   ./test_eo_solver --grid 4.4.4.4 --mpi 1.1.1.1
//
// Should be ~bit-exact since both solve the same M^{-1}·b mathematically.

#include "params.h"
#include "quda_txqcd_helper.h"
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

using namespace Grid;
using namespace TXQCDProduction;

// Same Schur reduce/reconstruct used by meas_conn_txqcd.
static void SchurSolveTxqcd(TXQCDWilsonCloverFermionEO &Meo,
                             GridRedBlackCartesian *rb,
                             const TXQCDFermionNf &b,
                             TXQCDFermionNf &x,
                             RealD tol, int max_iter) {
  TXQCDFermionNf b_e(rb), b_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    pickCheckerboard(Even, b_e.f[a], b.f[a]);
    pickCheckerboard(Odd,  b_o.f[a], b.f[a]);
  }
  TXQCDFermionNf tmp_e(rb), tmp_o(rb), rhs_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    tmp_e.f[a].Checkerboard() = Even;
    tmp_o.f[a].Checkerboard() = Odd;
    rhs_o.f[a].Checkerboard() = Odd;
  }
  Meo.MooeeInv(b_e, tmp_e);
  Meo.Meooe(tmp_e, tmp_o);
  for (int a = 0; a < TxqcdNf; ++a) rhs_o.f[a] = b_o.f[a] - tmp_o.f[a];

  TXQCDCloverSchurOp SchurOp(Meo);
  TXQCDFermionNf src_o(rb), x_o(rb);
  for (int a = 0; a < TxqcdNf; ++a) {
    src_o.f[a].Checkerboard() = Odd;
    x_o.f[a].Checkerboard() = Odd;
  }
  SchurOp.MpcDag(rhs_o, src_o);
  TXQCDConjugateGradient CG(tol, max_iter);
  CG(SchurOp, src_o, x_o);
  std::cout << GridLogMessage << "[EO]  CG iter=" << CG.IterationsToComplete
            << " resid=" << CG.TrueResidual << std::endl;

  Meo.Meooe(x_o, tmp_e);
  for (int a = 0; a < TxqcdNf; ++a) tmp_e.f[a] = b_e.f[a] - tmp_e.f[a];
  TXQCDFermionNf x_e(rb);
  for (int a = 0; a < TxqcdNf; ++a) x_e.f[a].Checkerboard() = Even;
  Meo.MooeeInv(tmp_e, x_e);

  for (int a = 0; a < TxqcdNf; ++a) {
    setCheckerboard(x.f[a], x_e.f[a]);
    setCheckerboard(x.f[a], x_o.f[a]);
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid4(latt, simd, mpi);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridSerialRNG sRNG; GridParallelRNG pRNG(&Grid4);
  sRNG.SeedFixedIntegers({1,2,3,4,5});
  pRNG.SeedFixedIntegers({6,7,8,9,10});

  TXQCDField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  TXQCDCompositeImpl::FillAuxFields(pRNG, U, /*lambda=*/6.0, /*Sigma=*/0.10);

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd-1] = -1.0;
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;

  TXQCDWilsonCloverOp           Mop(U.U, Grid4, RBGrid4, mass,
                                     U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);
  TXQCDWilsonCloverFermionEO    Meo(U.U, Grid4, RBGrid4, mass,
                                     U.sigma, U.pi, U.s, U.p, U.t, csw, impl_p);

  TXQCDFermionNf b(&Grid4);
  for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, b.f[a]);

  RealD tol = 1e-10;
  int max_iter = 50000;

  // Path 1: full-volume MdagM CG via QudaTxqcdPropSolver fallback path.
  std::array<RealD, TxqcdNf> mass_arr; mass_arr.fill(mass);
  TXQCDFermionNf x_fv(&Grid4);
  unsetenv("QUDA_SOLVER");
  {
    QudaTxqcdPropSolver solver(Mop, mass_arr, csw, U.U, tol, max_iter);
    solver.solve(b, x_fv);
  }

  // Path 2: Schur EO.
  TXQCDFermionNf x_eo(&Grid4);
  SchurSolveTxqcd(Meo, &RBGrid4, b, x_eo, tol, max_iter);

  // Compare residuals
  TXQCDFermionNf Mx(&Grid4), r(&Grid4);
  Mop.M(x_fv, Mx);
  for (int a = 0; a < TxqcdNf; ++a) r.f[a] = Mx.f[a] - b.f[a];
  RealD rel_fv = std::sqrt(norm2(r) / std::max(norm2(b), 1e-30));

  Mop.M(x_eo, Mx);
  for (int a = 0; a < TxqcdNf; ++a) r.f[a] = Mx.f[a] - b.f[a];
  RealD rel_eo = std::sqrt(norm2(r) / std::max(norm2(b), 1e-30));

  RealD nfv = norm2(x_fv), neo = norm2(x_eo);
  ComplexD ip = innerProduct(x_fv, x_eo);
  RealD cos_v = real(ip) / std::sqrt(nfv * neo);
  RealD factor = real(ip) / neo;

  std::cout << GridLogMessage << "===== EO vs full-volume validation =====" << std::endl;
  std::cout << GridLogMessage << "  full-vol: |M·x − b|/|b| = " << rel_fv << std::endl;
  std::cout << GridLogMessage << "  EO:       |M·x − b|/|b| = " << rel_eo << std::endl;
  std::cout << GridLogMessage << "  cos(x_fv, x_eo) = " << cos_v << std::endl;
  std::cout << GridLogMessage << "  factor          = " << factor << std::endl;
  bool pass = (cos_v > 0.99999) && (std::abs(factor - 1.0) < 1e-3)
              && (rel_fv < 100*tol) && (rel_eo < 100*tol);
  std::cout << GridLogMessage << "  RESULT: " << (pass ? "PASS" : "FAIL") << std::endl;

  Grid_finalize();
  return pass ? 0 : 1;
}
