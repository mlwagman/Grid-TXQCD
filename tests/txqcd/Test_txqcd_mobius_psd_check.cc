// PSD diagnostic for the TXQCD Möbius Schur HermOp.
//
// Mpc†Mpc is mathematically PSD.  But if LU MooeeInv has small numerical noise
// and the operator's smallest true eigenvalue is below that noise floor, the
// effective operator becomes indefinite and CG cannot converge.  This test
// samples <x, HermOp(x)> / <x, x> on random x.  A genuinely PSD op gives
// positive ratios across all samples.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  // LATT="Lx.Ly.Lz.Lt" (default 4.4.4.4)
  std::vector<int> dims = {4, 4, 4, 4};
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
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG pRNG4(UGrid); pRNG4.SeedFixedIntegers({7, 8, 9, 10});
  GridParallelRNG pRNG5(FGrid); pRNG5.SeedFixedIntegers({11, 12, 13, 14});

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
  const RealD mass = (mass_env && *mass_env) ? std::atof(mass_env) : 0.05;

  // CHANNEL env selects which aux is nonzero: sigma/pi/s/p/t/all/none.
  // Default = all (all five enabled).
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

  std::cout << GridLogMessage << "[psd] GAUGE=" << gauge_label
            << " LATT=" << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3]
            << " AUX=" << aux_scale << " mass=" << mass
            << " channel=" << channel
            << " M5=1.8 b=1.5 c=0.5" << std::endl;

  RealD M5 = 1.8, b = 1.5, c = 0.5;
  TXQCDMobiusFermionEO Mop(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                           mass, M5, b, c, sigma, pi, s, p, t);
  // Always enable LU here — this isolates the question to the Schur op
  // built around exact-per-block MooeeInv.
  setenv("TXQCD_MOBIUS_LU", "1", 1);
  Mop.BuildLU();
  TXQCDMobiusSchurOp Schur(Mop);

  // NON_EO=1 → use full M†M on FGrid instead of Schur Mpc†Mpc on FrbGrid.
  const char *neo_env = std::getenv("NON_EO");
  bool non_eo = neo_env && std::atoi(neo_env);
  TXQCDMobiusOp Mfull(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                      mass, M5, b, c, sigma, pi, s, p, t);
  std::cout << GridLogMessage << "[psd] op=" << (non_eo ? "M†M (full)"
                                                       : "Mpc†Mpc (EO Schur)")
            << std::endl;

  const int Nsamp = 20;
  RealD min_ratio = 1e300, max_ratio = -1e300;
  RealD min_norm2 = 1e300, max_norm2 = -1e300;
  int n_negative = 0;
  GridBase *xg = non_eo ? (GridBase*)FGrid : (GridBase*)FrbGrid;
  TXQCDFermionNf x(xg), Hx(xg), Mx(xg);
  for (int i = 0; i < Nsamp; ++i) {
    LatticeFermion x5(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG5, x5);
      if (non_eo) { x.f[a] = x5; Hx.f[a] = Zero(); Mx.f[a] = Zero(); }
      else {
        x.f[a].Checkerboard() = Odd; pickCheckerboard(Odd, x.f[a], x5);
        Hx.f[a].Checkerboard() = Odd;
      }
    }
    if (non_eo) { Mfull.M(x, Mx); Mfull.Mdag(Mx, Hx); }
    else        { Schur.HermOp(x, Hx); }
    RealD nx2 = norm2(x);
    ComplexD ip = innerProduct(x, Hx);
    RealD ratio = real(ip) / nx2;
    RealD imag_part = imag(ip);
    if (ratio < 0) ++n_negative;
    if (ratio < min_ratio) min_ratio = ratio;
    if (ratio > max_ratio) max_ratio = ratio;
    if (nx2 < min_norm2) min_norm2 = nx2;
    if (nx2 > max_norm2) max_norm2 = nx2;
    std::cout << GridLogMessage
              << "[psd] sample " << i << " <x,Hx>/<x,x>=" << ratio
              << "  Im<x,Hx>=" << imag_part << std::endl;
  }

  // Inverse iteration on Mee: ||Mee^-1 x||/||x|| → 1/lambda_min(Mee)
  // after a few iterations. Run on Even cb (the cb Mpc inverts internally).
  {
    TXQCDFermionNf y(FrbGrid), z(FrbGrid);
    LatticeFermion y5(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG5, y5);
      y.f[a].Checkerboard() = Even; pickCheckerboard(Even, y.f[a], y5);
      z.f[a].Checkerboard() = Even;
    }
    RealD inv_lmin = 0;
    for (int it = 0; it < 5; ++it) {
      Mop.MooeeInv(y, z);
      RealD nz = std::sqrt(norm2(z));
      RealD ny = std::sqrt(norm2(y));
      inv_lmin = nz / ny;            // bound: ≤ 1/|lambda_min(Mee)|
      RealD inv = 1.0 / nz;
      for (int a = 0; a < TxqcdNf; ++a) y.f[a] = inv * z.f[a];
    }
    std::cout << GridLogMessage
              << "[psd] |1/lambda_min(Mee)| estimate (5-iter inv. iter.) = "
              << inv_lmin
              << "  → |lambda_min(Mee)| ≈ " << 1.0/inv_lmin << std::endl;
  }

  // Power iteration on M†M (non-EO only) to bound true lambda_max.
  if (non_eo) {
    TXQCDFermionNf y(FGrid), z(FGrid), w(FGrid);
    LatticeFermion y5(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) { gaussian(pRNG5, y5); y.f[a] = y5; }
    RealD lmax = 0;
    for (int it = 0; it < 30; ++it) {
      Mfull.M(y, w); Mfull.Mdag(w, z);
      RealD nz = std::sqrt(norm2(z)); lmax = nz / std::sqrt(norm2(y));
      RealD inv = 1.0 / nz;
      for (int a = 0; a < TxqcdNf; ++a) y.f[a] = inv * z.f[a];
    }
    std::cout << GridLogMessage
              << "[psd] lambda_max(M†M) (30-iter power) = " << lmax << std::endl;
  }

  std::cout << GridLogMessage
            << "[psd] min_ratio=" << min_ratio
            << " max_ratio=" << max_ratio
            << " n_negative=" << n_negative << "/" << Nsamp << std::endl;
  int ec = 0;
  bool ok = (n_negative == 0 && min_ratio >= 0.0);
  std::cout << GridLogMessage
            << (ok ? "PSD confirmed (no negative samples)"
                   : "PSD VIOLATED (negative samples detected)") << std::endl;
  if (!ok) ec = 1;

  Grid_finalize();
  return ec;
}
