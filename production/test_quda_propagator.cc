// QUDA vs Grid propagator: real-cfg correctness + speedup on the
// cl3_16_48_b6p1_m0p2450 ensemble.  Solves M·x = b with a point source on
// both Grid's mixed-prec CG and the QudaCloverInverter, compares the
// resulting propagators, reports wallclock for each.
//
// Run:
//   IMPORT_CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime \
//   mpirun -np 1 ./test_quda_propagator --grid 16.16.16.48 --mpi 1.1.1.1
//
// Optional env:
//   CG_TOL=1e-9
//   SKIP_GRID=1     (only run QUDA)
//   SKIP_QUDA=1     (only run Grid)

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <chrono>
#include <cstdio>
#include <cstring>

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

  RealD mass = -0.245;
  RealD csw  = 1.24930970916466;
  RealD cg_tol = 1e-9;
  if (const char *t = std::getenv("CG_TOL"); t && *t) cg_tol = std::atof(t);

  // -------------------------------------------------------------------------
  // Load gauge.  IMPORT_CFG required (this test wants a real chroma cfg).
  // -------------------------------------------------------------------------
  const char *ic = std::getenv("IMPORT_CFG");
  if (!ic || !*ic) {
    std::cout << GridLogMessage << "Set IMPORT_CFG=/path/to/cfg.lime" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::cout << GridLogMessage << "Loading cfg: " << ic << std::endl;
  LatticeGaugeField Umu(&Grid4);
  {
    FILE *f = std::fopen(ic, "rb");
    char magic[16] = {0};
    if (f) { std::fread(magic, 1, sizeof(magic), f); std::fclose(f); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(Umu, header, std::string(ic));
    } else {
      IldgReader IR;
      IR.open(std::string(ic));
      IR.readConfiguration(Umu, header);
      IR.close();
    }
  }
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
  std::cout << GridLogMessage << "Loaded cfg, plaquette = " << plaq << std::endl;

  // -------------------------------------------------------------------------
  // Point source at origin, spin=0, color=0.
  // -------------------------------------------------------------------------
  LatticeFermion src(&Grid4);
  src = Zero();
  {
    typename LatticeFermion::scalar_object s_zero;
    s_zero = Zero();
    s_zero()(0)(0) = ComplexD(1.0, 0.0);
    Coordinate origin({0, 0, 0, 0});
    pokeSite(s_zero, src, origin);
  }
  RealD nf_src = norm2(src);
  std::cout << GridLogMessage << "norm2(point src) = " << nf_src << std::endl;

  // Grid Wilson-clover operator with antiperiodic time BC.
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF D(Umu, Grid4, RBGrid4, mass, csw, csw, WilsonAnisotropyCoefficients(), impl_p);

  bool skip_grid = (std::getenv("SKIP_GRID") != nullptr);
  bool skip_quda = (std::getenv("SKIP_QUDA") != nullptr);

  // -------------------------------------------------------------------------
  // Grid solve: ConjugateGradient on M^†M (full volume).
  // -------------------------------------------------------------------------
  LatticeFermion x_grid(&Grid4); x_grid = Zero();
  double t_grid = 0.0;
  if (!skip_grid) {
    LatticeFermion Mdag_src(&Grid4);
    D.Mdag(src, Mdag_src);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(D);
    ConjugateGradient<LatticeFermion> CG(cg_tol, 50000);
    auto t0 = std::chrono::steady_clock::now();
    CG(HermOp, Mdag_src, x_grid);
    auto t1 = std::chrono::steady_clock::now();
    t_grid = std::chrono::duration<double>(t1 - t0).count();
    std::cout << GridLogMessage << "Grid CG  time = " << t_grid << " s" << std::endl;
  }

  // -------------------------------------------------------------------------
  // QUDA solve.  First call includes kernel tuning; second call is warm
  // and is the per-MD-step cost we'd pay in HMC.
  // -------------------------------------------------------------------------
  LatticeFermion x_quda(&Grid4); x_quda = Zero();
  double t_quda = 0.0, t_quda_warm = 0.0;
  if (!skip_quda) {
    Quda::initialize();
    QudaCloverParams qp;
    qp.mass = mass;
    qp.csw  = csw;
    qp.anti_periodic_t = true;
    qp.tol = cg_tol;
    qp.max_iter = 50000;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    if (const char *s = std::getenv("QUDA_FULL_DP"); s && *s) {
      qp.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
      qp.recon_sloppy = QUDA_RECONSTRUCT_NO;
    }
    QudaCloverInverter quda_cg(&Grid4, qp);
    quda_cg.SetGauge(Umu);

    MdagMLinearOperator<WCF, LatticeFermion> HermOp(D);  // unused by QUDA

    // Cold call: first time through, includes JIT/kernel tuning.
    auto t0 = std::chrono::steady_clock::now();
    quda_cg(HermOp, src, x_quda);
    auto t1 = std::chrono::steady_clock::now();
    t_quda = std::chrono::duration<double>(t1 - t0).count();
    int iter_cold = quda_cg.LastIter();
    std::cout << GridLogMessage << "QUDA CG (cold) time = " << t_quda
              << " s  (iter=" << iter_cold << ")" << std::endl;

    // Warm call: kernel tuning already done.  This is the asymptotic cost.
    LatticeFermion x_quda2(&Grid4); x_quda2 = Zero();
    t0 = std::chrono::steady_clock::now();
    quda_cg(HermOp, src, x_quda2);
    t1 = std::chrono::steady_clock::now();
    t_quda_warm = std::chrono::duration<double>(t1 - t0).count();
    std::cout << GridLogMessage << "QUDA CG (warm) time = " << t_quda_warm
              << " s  (iter=" << quda_cg.LastIter()
              << " QUDA-internal-secs=" << quda_cg.LastSecs() << ")" << std::endl;
  }

  // -------------------------------------------------------------------------
  // Self-consistency: M_grid · x ≈ src on each side.
  // -------------------------------------------------------------------------
  if (!skip_grid) {
    LatticeFermion Mx(&Grid4); D.M(x_grid, Mx);
    RealD r = std::sqrt(norm2(LatticeFermion(Mx - src)) / nf_src);
    std::cout << GridLogMessage << "Grid:  ||M·x − src|| / ||src|| = " << r << std::endl;
  }
  if (!skip_quda) {
    LatticeFermion Mx(&Grid4); D.M(x_quda, Mx);
    RealD r = std::sqrt(norm2(LatticeFermion(Mx - src)) / nf_src);
    std::cout << GridLogMessage << "QUDA:  ||M·x − src|| / ||src|| = " << r << std::endl;
  }

  // -------------------------------------------------------------------------
  // Cross-check: x_grid vs x_quda.
  // -------------------------------------------------------------------------
  if (!skip_grid && !skip_quda) {
    RealD nf_grid = norm2(x_grid);
    RealD nf_diff = norm2(LatticeFermion(x_grid - x_quda));
    RealD rel = std::sqrt(nf_diff / nf_grid);
    std::cout << GridLogMessage << "Cross-check: ||x_grid − x_quda|| / ||x_grid|| = "
              << rel << std::endl;
    if (t_grid > 0 && t_quda > 0) {
      std::cout << GridLogMessage << "Speedup (Grid / QUDA cold) = "
                << t_grid / t_quda << "×" << std::endl;
    }
    if (t_grid > 0 && t_quda_warm > 0) {
      std::cout << GridLogMessage << "Speedup (Grid / QUDA warm) = "
                << t_grid / t_quda_warm << "×" << std::endl;
    }
  }

  if (!skip_quda) Quda::finalize();
  Grid_finalize();
  return 0;
}
