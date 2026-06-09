// PSD diagnostic for TXQCD Wilson-Clover Schur HermOp — clover analog of
// Test_txqcd_mobius_psd_check.  Maps the tensor-condensed phase boundary
// for the 4D clover operator, isolating which aux channel triggers it.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDCloverSchurOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // LATT="Lx.Ly.Lz.Lt" (default 4.4.4.8)
  std::vector<int> dims = {4, 4, 4, 8};
  if (const char *ls = std::getenv("LATT")) {
    dims.clear();
    std::stringstream ss(ls); std::string tok;
    while (std::getline(ss, tok, '.')) dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt4(dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG pRNG4(UGrid); pRNG4.SeedFixedIntegers({7, 8, 9, 10});

  LatticeGaugeField Umu(UGrid);
  const char *gf = std::getenv("GAUGE_FILE");
  const char *gt = std::getenv("GAUGE_TYPE");
  std::string gauge_label;
  if (gf && *gf) {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, std::string(gf));
    gauge_label = std::string("load:") + gf;
  } else {
    bool cold = (gt && std::string(gt) == "cold");
    if (cold) { SU<Nc>::ColdConfiguration(Umu); gauge_label = "cold"; }
    else      { SU<Nc>::HotConfiguration(pRNG4, Umu); gauge_label = "hot"; }
  }

  const char *as_env = std::getenv("AUX_SCALE");
  const RealD aux_scale = (as_env && *as_env) ? std::atof(as_env) : 0.30;
  const char *mass_env = std::getenv("MASS");
  const RealD mass = (mass_env && *mass_env) ? std::atof(mass_env) : -0.245;
  const char *csw_env = std::getenv("CSW");
  const RealD csw = (csw_env && *csw_env) ? std::atof(csw_env) : 1.24930970916466;

  const char *ch_env = std::getenv("CHANNEL");
  std::string channel = ch_env ? std::string(ch_env) : "all";
  auto on = [&](const char *which){
    return channel == "all" || channel == which;
  };
  LatticeSigmaField sigma(UGrid); LatticePiField pi(UGrid);
  LatticeSFieldC s(UGrid);        LatticePFieldC p(UGrid);
  LatticeTField  t(UGrid);
  sigma = Zero(); pi = Zero(); s = Zero(); p = Zero(); t = Zero();
  if (on("sigma")) { HermitianGaussian(pRNG4, sigma); sigma = aux_scale * sigma; }
  if (on("pi"))    { HermitianGaussian(pRNG4, pi);    pi    = aux_scale * pi; }
  if (on("s"))     { HermitianGaussian(pRNG4, s);     s     = aux_scale * s; }
  if (on("p"))     { HermitianGaussian(pRNG4, p);     p     = aux_scale * p; }
  if (on("t"))     { GaussianAntisymTensor(pRNG4, t); t     = aux_scale * t; }

  std::cout << GridLogMessage << "[psd-clover] GAUGE=" << gauge_label
            << " LATT=" << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3]
            << " AUX=" << aux_scale << " mass=" << mass << " csw=" << csw
            << " channel=" << channel << std::endl;

  TXQCDWilsonCloverFermionEO Mop(Umu, *UGrid, *UrbGrid, mass,
                                 sigma, pi, s, p, t, csw);
  TXQCDCloverSchurOp Schur(Mop);

  const int Nsamp = 20;
  RealD min_ratio = 1e300, max_ratio = -1e300;
  int n_negative = 0;
  TXQCDFermionNf x(UrbGrid), Hx(UrbGrid);
  for (int i = 0; i < Nsamp; ++i) {
    LatticeFermion x4(UGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG4, x4);
      x.f[a].Checkerboard() = Odd; pickCheckerboard(Odd, x.f[a], x4);
      Hx.f[a].Checkerboard() = Odd;
    }
    Schur.HermOp(x, Hx);
    RealD nx2 = norm2(x);
    ComplexD ip = innerProduct(x, Hx);
    RealD ratio = real(ip) / nx2;
    if (ratio < 0) ++n_negative;
    if (ratio < min_ratio) min_ratio = ratio;
    if (ratio > max_ratio) max_ratio = ratio;
    std::cout << GridLogMessage << "[psd-clover] sample " << i
              << " <x,Hx>/<x,x>=" << ratio << std::endl;
  }

  std::cout << GridLogMessage
            << "[psd-clover] min_ratio=" << min_ratio
            << " max_ratio=" << max_ratio
            << " n_negative=" << n_negative << "/" << Nsamp << std::endl;
  bool ok = (n_negative == 0 && min_ratio >= 0.0);
  std::cout << GridLogMessage
            << (ok ? "PSD confirmed" : "PSD VIOLATED") << std::endl;

  Grid_finalize();
  return ok ? 0 : 1;
}
