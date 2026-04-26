// Compute the per-site VEV estimator <Tr M^{-1}>/(2V) used for lambda tuning.
// Replicates the noise-based estimator in gen_qcd_cfgs.cc (QcdDiag), allowing
// evaluation on arbitrary gauge configs — NERSC or ILDG/SciDAC LIME.
//
// Matches production params: stout smear (rho=0.125, n=1) applied before
// building WilsonCloverFermion at mass_light, csw.
//
// Usage:
//   ./compute_vev <cfg>... --grid L.L.L.T --mpi mx.my.mz.mt \
//                [ --n-noise N ] [ --cg-tol T ] [ --seed S ]

#include "params.h"
#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <cstdio>
#include <cstring>

using namespace Grid;
using namespace TXQCDProduction;

enum class CfgFormat { NERSC, LIME_ILDG, UNKNOWN };

static CfgFormat detect_format(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return CfgFormat::UNKNOWN;
  unsigned char buf[16] = {0};
  std::fread(buf, 1, sizeof(buf), f);
  std::fclose(f);
  if (std::memcmp(buf, "BEGIN_HEADER", 12) == 0) return CfgFormat::NERSC;
  if (buf[0] == 0x45 && buf[1] == 0x67 && buf[2] == 0x89 && buf[3] == 0xAB)
    return CfgFormat::LIME_ILDG;
  return CfgFormat::UNKNOWN;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Parse our custom flags + collect filenames.
  std::vector<std::string> files;
  int n_noise = n_vev_noise;   // default from params.h (=8)
  RealD cg_tolerance = 1e-8;
  int rng_seed = 1234567;
  RealD weakfield_scale = 0.0;  // if > 0, generate weak-field cfg in-place
  int weakfield_seed_offset = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--n-noise") { n_noise = std::atoi(argv[++i]); continue; }
    if (a == "--cg-tol")  { cg_tolerance = std::atof(argv[++i]); continue; }
    if (a == "--seed")    { rng_seed = std::atoi(argv[++i]); continue; }
    if (a == "--weakfield") {
      weakfield_scale = std::atof(argv[++i]);
      continue;
    }
    if (a == "--weakfield-seed-offset") {
      weakfield_seed_offset = std::atoi(argv[++i]);
      continue;
    }
    if (a.rfind("--", 0) == 0) { ++i; continue; }  // skip other grid flags + their values
    files.push_back(a);
  }
  if (files.empty() && weakfield_scale == 0.0) {
    std::cerr << "Usage: compute_vev <cfg>... --grid L.L.L.T --mpi mx.my.mz.mt "
                 "[--n-noise N] [--cg-tol T] [--seed S]\n"
                 "       compute_vev --weakfield <scale> --weakfield-seed-offset <K> ...\n";
    Grid_finalize();
    return 1;
  }

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         grid_(latt, simd, mpi);
  GridRedBlackCartesian rbgrid_(&grid_);

  GridParallelRNG pRNG(&grid_);
  pRNG.SeedFixedIntegers({rng_seed, rng_seed + 1, rng_seed + 2,
                          rng_seed + 3, rng_seed + 4});

  Smear_Stout<PeriodicGimplR> Stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> Smear(&grid_, stout_nsmear_inv, Stout);

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  typedef GaugeStatistics<PeriodicGimplR> GS;
  LatticeGaugeField Umu(&grid_);
  RealD V = (RealD)grid_.gSites();

  std::cout << GridLogMessage
            << "compute_vev: mass=" << mass_light << " csw=" << csw
            << " stout rho=" << stout_rho_inv << " n=" << stout_nsmear_inv
            << " n_noise=" << n_noise << " cg_tol=" << cg_tolerance
            << " seed=" << rng_seed << std::endl;

  // If --weakfield given, prepend a synthetic "_weakfield" entry that triggers
  // in-memory generation of a weak-field gauge with the same RNG seed as
  // gen_txqcd_cfgs (seed_offset = LAMBDA*1000).
  if (weakfield_scale > 0.0) {
    files.insert(files.begin(), "_weakfield");
  }
  for (const auto &f : files) {
    FieldMetaData header;
    if (f == "_weakfield") {
      GridParallelRNG wfRNG(&grid_);
      int s = weakfield_seed_offset;
      wfRNG.SeedFixedIntegers({1 + s, 2 + s, 3 + s, 4 + s, 5 + s});
      LatticeColourMatrix Ulink(&grid_);
      for (int mu = 0; mu < Nd; ++mu) {
        SU<Nc>::LieRandomize(wfRNG, Ulink, weakfield_scale);
        PokeIndex<LorentzIndex>(Umu, Ulink, mu);
      }
      std::cout << GridLogMessage
                << "compute_vev: synthesized weak-field gauge with scale="
                << weakfield_scale << " seed_offset=" << weakfield_seed_offset
                << std::endl;
    } else {
      CfgFormat fmt = detect_format(f);
      if (fmt == CfgFormat::NERSC) {
        NerscIO::readConfiguration<GS>(Umu, header, f);
      } else if (fmt == CfgFormat::LIME_ILDG) {
        IldgReader reader;
        reader.open(f);
        reader.readConfiguration(Umu, header);
        reader.close();
      } else {
        std::cerr << "compute_vev: unknown format for " << f << std::endl;
        continue;
      }
    }

    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);

    Smear.set_Field(Umu);
    LatticeGaugeField Usm = Smear.get_SmearedU();
    // Match chroma's <boundary>1 1 1 -1</boundary> — antiperiodic time BC.
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;
    WCF Dw(Usm, grid_, rbgrid_, mass_light, csw, csw,
           WilsonAnisotropyCoefficients(), impl_p);
    MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
    ConjugateGradient<LatticeFermion> CG(cg_tolerance, cg_max);

    RealD acc = 0.0;
    for (int h = 0; h < n_noise; ++h) {
      LatticeFermion eta(&grid_), b(&grid_), x(&grid_);
      gaussian(pRNG, eta);
      Dw.Mdag(eta, b);
      x = Zero();
      CG(HermOp, b, x);
      acc += innerProduct(eta, x).real() / (2.0 * V);
    }
    RealD vev = acc / n_noise;
    std::cout << GridLogMessage << f
              << "\tplaq = " << std::setprecision(10) << plaq
              << "\tvev_trminv = " << vev
              << "  (N_noise=" << n_noise << ")" << std::endl;
  }

  Grid_finalize();
  return 0;
}
