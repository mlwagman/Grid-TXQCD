#include "params.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/CovariantSmearing.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <memory>
#endif

using namespace TXQCDProduction;

static std::vector<LatticeColourMatrix>
ExtractLinks(const LatticeGaugeField &U) {
  std::vector<LatticeColourMatrix> Umu(Nd, U.Grid());
  for (int mu = 0; mu < Nd; ++mu)
    Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
  return Umu;
}

static std::vector<RealD> PionCorrelator(const LatticePropagator &S) {
  LatticeComplex corr(S.Grid());
  corr = trace(S * adj(S));
  std::vector<TComplex> Csl;
  sliceSum(corr, Csl, Nd - 1);
  std::vector<RealD> out(Csl.size());
  for (size_t t = 0; t < Csl.size(); ++t)
    out[t] = TensorRemove(Csl[t]).real();
  return out;
}

static std::vector<ComplexD> NucleonCorrelator(const LatticePropagator &S) {
  Gamma G_A(Gamma::Algebra::Identity);
  Gamma G_B(Gamma::Algebra::SigmaXZ);
  int wick = 0;
  BaryonUtils<WilsonImplR>::WickContractions("uud", "uud", wick);
  LatticeComplex Cn(S.Grid());
  BaryonUtils<WilsonImplR>::ContractBaryons(S, S, S, G_A, G_B, G_A, G_B,
                                            wick, +1, Cn);
  std::vector<TComplex> sl;
  sliceSum(Cn, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

static std::vector<Coordinate> SourceGrid(const Coordinate &latt) {
  Coordinate origin = src_grid_origin();
  std::vector<Coordinate> sites;
  for (int ix = 0; ix < space_src_per_dim; ++ix)
    for (int iy = 0; iy < space_src_per_dim; ++iy)
      for (int iz = 0; iz < space_src_per_dim; ++iz)
        for (int it = 0; it < time_src_per_dim; ++it) {
          Coordinate s(Nd);
          s[0] = (origin[0] + ix * latt[0] / space_src_per_dim) % latt[0];
          s[1] = (origin[1] + iy * latt[1] / space_src_per_dim) % latt[1];
          s[2] = (origin[2] + iz * latt[2] / space_src_per_dim) % latt[2];
          s[3] = (origin[3] + it * latt[3] / time_src_per_dim) % latt[3];
          sites.push_back(s);
        }
  return sites;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: meas_conn_qcd <traj>" << std::endl;
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
  {
    std::string cf = qcd_cfg_dir() + "/ckpoint_lat." + std::to_string(traj);
    std::string rf = qcd_cfg_dir() + "/ckpoint_rng." + std::to_string(traj);
    FieldMetaData header;
    NerscIO::readRNGState(sRNG, pRNG, header, rf);
    typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu, header, cf);
  }

  // Stout smearing for inversions
  Smear_Stout<PeriodicGimplR> StoutInv(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> SmearInv(&Grid, stout_nsmear_inv, StoutInv);
  SmearInv.set_Field(Umu);
  LatticeGaugeField U_inv = SmearInv.get_SmearedU();

  // Stout smearing for source/sink Gaussian smearing
  Smear_Stout<PeriodicGimplR> StoutSrc(stout_rho_src);
  SmearedConfiguration<PeriodicGimplR> SmearSrc(&Grid, stout_nsmear_src, StoutSrc);
  SmearSrc.set_Field(Umu);
  LatticeGaugeField U_src = SmearSrc.get_SmearedU();
  auto U_src_links = ExtractLinks(U_src);

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WilsonImplParams impl_p;
  impl_p.boundary_phases.resize(Nd, 1.0);
  impl_p.boundary_phases[Nd - 1] = -1.0;
  WCF Dw(U_inv, Grid, RBGrid, mass_light, csw, csw,
         WilsonAnisotropyCoefficients(), impl_p);
  MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(cg_tol, cg_max);

#ifdef GRID_HAVE_QUDA
  // Optional QUDA backend: env QUDA_SOLVER=1 swaps the per-source CG for a
  // QudaCloverInverter call.  ~3× speedup on 16³×48 / single A100.
  bool use_quda = std::getenv("QUDA_SOLVER") != nullptr;
  std::unique_ptr<QudaCloverInverter> quda_cg;
  if (use_quda) {
    Quda::initialize();
    QudaCloverParams qp;
    qp.mass = mass_light;
    qp.csw  = csw;
    qp.anti_periodic_t = true;
    qp.tol = cg_tol;
    qp.max_iter = cg_max;
    qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    quda_cg.reset(new QudaCloverInverter(&Grid, qp));
    quda_cg->SetGauge(U_inv);
    std::cout << GridLogMessage << "QUDA_SOLVER active for connected propagator." << std::endl;
  }
#endif

  int T = latt[Nd - 1];
  auto sources = SourceGrid(latt);
  int nsrc = (int)sources.size();

  std::vector<std::vector<RealD>>    all_pion;
  std::vector<std::vector<ComplexD>> all_nucleon;

  for (int isrc = 0; isrc < nsrc; ++isrc) {
    Coordinate &src = sources[isrc];
    std::cout << GridLogMessage << "[conn QCD] traj=" << traj
              << " src=(" << src[0] << "," << src[1] << ","
              << src[2] << "," << src[3] << ")" << std::endl;

    LatticePropagator S(&Grid);
    S = Zero();

    for (int spin = 0; spin < Ns; ++spin) {
      for (int col = 0; col < Nc; ++col) {
        LatticePropagator srcP(&Grid);
        srcP = Zero();
        SpinColourMatrix kron;
        kron = 1.0;
        pokeSite(kron, srcP, src);

        LatticeFermion sf(&Grid), b(&Grid), x(&Grid);
        PropToFerm<WilsonImplR>(sf, srcP, spin, col);

        // Gaussian smear source
        CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, sf,
                                                         gauss_width, gauss_niter, Nd - 1);

        x = Zero();
#ifdef GRID_HAVE_QUDA
        if (use_quda) {
          // QUDA inverts M directly; skip the Mdag pre-multiply.
          (*quda_cg)(HermOp, sf, x);
        } else
#endif
        {
          Dw.Mdag(sf, b);
          CG(HermOp, b, x);
        }

        // Gaussian smear sink
        CovariantSmearing<PeriodicGimplR>::GaussianSmear(U_src_links, x,
                                                         gauss_width, gauss_niter, Nd - 1);

        FermToProp<WilsonImplR>(S, x, spin, col);
      }
    }

    all_pion.push_back(PionCorrelator(S));
    all_nucleon.push_back(NucleonCorrelator(S));
  }

  std::vector<RealD> pion_avg(T, 0.0);
  std::vector<ComplexD> nucl_avg(T, 0.0);
  for (int i = 0; i < nsrc; ++i)
    for (int t = 0; t < T; ++t) {
      pion_avg[t] += all_pion[i][t] / nsrc;
      nucl_avg[t] += all_nucleon[i][t] / (double)nsrc;
    }

  std::string outfile = qcd_data_dir() + "/conn_qcd_" + std::to_string(traj) + ".h5";
  {
    Hdf5Writer wr(outfile);
    write(wr, "pion_conn", pion_avg);
    write(wr, "nucleon", nucl_avg);
    write(wr, "pion_per_src", all_pion);
    write(wr, "nucleon_per_src", all_nucleon);
    write(wr, "traj", traj);
    write(wr, "plaq", WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu));
  }

  std::cout << GridLogMessage << "Written " << outfile << std::endl;
  Grid_finalize();
  return 0;
}
