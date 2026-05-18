#include "params.h"
#include "quda_helper.h"
#include "meas_helper.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TXQCDProduction;

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;

static RealD StochasticTrMinv(WCF &Dw, GridBase *grid,
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

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_disco_qcd <traj>" << std::endl;
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

  sRNG.SeedFixedIntegers({11, 12, 13, 14, 15});
  pRNG.SeedFixedIntegers({16, 17, 18, 19, 20});

  mkdir_p(qcd_data_dir());

  LatticeGaugeField Umu(&Grid);
  load_qcd_gauge(traj, Umu, sRNG, pRNG);

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(Umu);
  LatticeGaugeField Usmeared = SmearInv.get_SmearedU();

  std::cout << GridLogMessage << "[disco QCD] traj=" << traj << std::endl;

  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(Usmeared, Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  RealD trminv = StochasticTrMinv(Dw, &Grid, Usmeared, mass_light, pRNG, n_noise_disco);

  WCF Dw_s(Usmeared, Grid, RBGrid, mass_strange, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
  RealD trminv_strange = StochasticTrMinv(Dw_s, &Grid, Usmeared, mass_strange, pRNG, n_noise_disco);

  // Sanity-check metadata: plaq + link_trace on the unsmeared gauge.
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);

  std::string outfile = qcd_data_dir() + "/disco_qcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    write(wr, "trminv", trminv);
    write(wr, "trminv_strange", trminv_strange);
    write(wr, "traj", traj);
    write(wr, "plaq", plaq);
    write(wr, "mass_light", mass_light);
    write(wr, "mass_strange", mass_strange);
    write(wr, "n_noise_disco", n_noise_disco);
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
#ifdef GRID_HAVE_QUDA
  Grid::Quda::finalize();
#endif
  Grid_finalize();
  return 0;
}
