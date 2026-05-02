// Phase 3 validation: QudaCloverInverter on a small lattice.
//
// The test asks: given a random source b and a random gauge U, does the
// QUDA-computed x = M^-1·b satisfy Grid's M · x ≈ b?
//
// If yes (relative residual < 1e-8 say), we have full chain agreement:
//   - Field conversion (Phase 2) is correct.
//   - Gauge ordering (lex vs EO, row vs col, dir layout) is correct.
//   - Gamma basis matches between Grid and QUDA.
//   - Clover sign convention matches.
//   - Mass / kappa convention matches.
//
// If the residual norm is comparable to the source norm but the *vector*
// is twisted (i.e. norm(M·x) ≈ norm(b) but M·x ≠ b), it's a similarity
// transform — almost always a gamma-basis mismatch.  We log enough state
// to diagnose.
//
// Run:
//   mpirun -np 1 ./test_quda_inverter --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>

using namespace Grid;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt_size  = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian Grid4(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid4(&Grid4);

  GridParallelRNG pRNG(&Grid4);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  Quda::initialize();

  // -------------------------------------------------------------------------
  // Build a random gauge field and a Grid Wilson-clover operator that
  // matches the QUDA settings: mass, csw, anti-periodic t boundary.
  // -------------------------------------------------------------------------
  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;

  LatticeGaugeField U(&Grid4);
  SU<Nc>::HotConfiguration(pRNG, U);

  // Antiperiodic time BC matches our production ensemble.
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF D(U, Grid4, RBGrid4, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);

  // -------------------------------------------------------------------------
  // Build the QUDA inverter with matching params.
  // -------------------------------------------------------------------------
  // Sweep all 4 QUDA gamma-basis options and report which (if any) makes
  // M_grid·(QUDA inverse)·src ≈ src.  This nails down the basis question
  // empirically rather than guessing from gamma-matrix definitions.
  struct BasisChoice { QudaGammaBasis b; const char *name; };
  std::vector<BasisChoice> bases = {
      {QUDA_DEGRAND_ROSSI_GAMMA_BASIS, "DEGRAND_ROSSI"},
      {QUDA_UKQCD_GAMMA_BASIS,         "UKQCD"},
      {QUDA_CHIRAL_GAMMA_BASIS,        "CHIRAL"},
      {QUDA_DIRAC_PAULI_GAMMA_BASIS,   "DIRAC_PAULI"},
  };

  bool any_pass = false;
  for (auto &bc : bases) {
    QudaCloverParams qp;
    qp.mass = mass;
    qp.csw  = csw;
    qp.anti_periodic_t = true;
    qp.tol = 1e-10;
    qp.max_iter = 5000;
    qp.gamma_basis = bc.b;

    QudaCloverInverter quda_cg(&Grid4, qp);
    quda_cg.SetGauge(U);

    LatticeFermion src(&Grid4), x_quda(&Grid4);
    random(pRNG, src);
    x_quda = Zero();
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(D);

    std::cout << GridLogMessage << "=== Trying gamma_basis = "
              << bc.name << " ===" << std::endl;
    quda_cg(HermOp, src, x_quda);

    LatticeFermion Mx(&Grid4), res(&Grid4);
    D.M(x_quda, Mx);
    res = Mx - src;
    RealD nf_src = norm2(src);
    RealD nf_Mx  = norm2(Mx);
    RealD nf_res = norm2(res);
    RealD rel    = std::sqrt(nf_res / nf_src);
    RealD norm_ratio = std::sqrt(nf_Mx / nf_src);
    std::cout << GridLogMessage << "  iter=" << quda_cg.LastIter()
              << " QUDA-internal-res=" << quda_cg.LastResidual()
              << " M·x/src ratio=" << norm_ratio
              << " rel-resid=" << rel << std::endl;

    bool pass = (rel < 1e-6);
    if (pass) {
      std::cout << GridLogMessage << "  PASS with " << bc.name << std::endl;
      any_pass = true;
    }
  }

  std::cout << GridLogMessage << "==========================================="
            << std::endl;
  std::cout << GridLogMessage << "Phase 3 result: "
            << (any_pass ? "PASS — found matching basis"
                         : "FAIL — no basis matches; check norms above")
            << std::endl;

  Quda::finalize();
  Grid_finalize();
  return any_pass ? 0 : 1;
}
