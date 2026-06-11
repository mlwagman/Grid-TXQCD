// Test_dtxqcd_psd_check: PSD-style spectrum diagnostic for the DTXQCD
// doubled Wilson-Clover operator, comparing the FULL operator M^dag M
// to the EO Schur Mpc^dag Mpc and the per-CB Mooee^dag Mooee.
//
// Motivation: the TXQCD "tensor-condensed phase" at small lambda turned
// out (project_tensor_condensed_phase, 2026-06-09) to be an EO-Schur
// preconditioner artifact -- det(M_full) = det(Mee) * det(Mpc) stays
// smooth while Mee -> 0 and Mpc -> infinity in compensating ways.
// Our DTXQCD HMC breakdown shows the same fingerprint (per-site det(Mee)
// crosses zero, multi-shift CG on Mpc fails).  This test isolates which
// piece is actually pathological at given aux scale.
//
// Env knobs (mirror Test_txqcd_mobius_psd_check):
//   LATT="Lx.Ly.Lz.Lt"            default 4.4.4.8
//   MASS                          default 1.0
//   AUX_SCALE                     default 0.30 (per-channel real std)
//   CSW                           default 1.24930970916466 (production)
//   CHANNEL=sigma|pi|t|d|n|all    default all
//   GAUGE_TYPE=cold|hot           default cold
//   GAUGE_FILE                    optional NERSC config path
//   NSAMP                         default 20
//
// Output per sampler: min/max/n_negative of <x, A x> / <x,x> on random x.
// Run with --grid LxLyLzLt --mpi 1.1.1.1 etc., as for the other psd_check
// tests.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>

using namespace Grid;

namespace {

std::vector<int> ParseLatt(const char *s, std::vector<int> def) {
  if (!s || !*s) return def;
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, '.')) out.push_back(std::atoi(tok.c_str()));
  return out;
}

struct Stats {
  RealD min_ratio = 1e300;
  RealD max_ratio = -1e300;
  RealD geom_log_sum = 0.0;
  int   n_negative = 0;
  int   n = 0;
  void update(RealD r) {
    if (r < min_ratio) min_ratio = r;
    if (r > max_ratio) max_ratio = r;
    if (r < 0) ++n_negative;
    if (r > 0) geom_log_sum += std::log(r);
    ++n;
  }
  RealD geom_mean() const {
    int n_pos = n - n_negative;
    return (n_pos > 0) ? std::exp(geom_log_sum / n_pos) : 0.0;
  }
};

void PrintStats(const std::string &tag, const Stats &s, int nsamp) {
  std::cout << GridLogMessage
            << "[psd-dtxqcd] " << tag
            << "  min=" << s.min_ratio
            << "  max=" << s.max_ratio
            << "  geom_mean=" << s.geom_mean()
            << "  n_neg=" << s.n_negative << "/" << nsamp
            << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  auto dims = ParseLatt(std::getenv("LATT"), {4, 4, 4, 8});
  Coordinate latt4(dims);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG pRNG(UGrid);
  pRNG.SeedFixedIntegers({11, 12, 13, 14});

  // --- Gauge field ---
  LatticeGaugeField Umu(UGrid);
  std::string gauge_label;
  const char *gf = std::getenv("GAUGE_FILE");
  const char *gt = std::getenv("GAUGE_TYPE");
  if (gf && *gf) {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, std::string(gf));
    gauge_label = std::string("load:") + gf;
  } else {
    bool cold = !(gt && std::string(gt) == "hot");
    if (cold) { SU<Nc>::ColdConfiguration(Umu); gauge_label = "cold"; }
    else      { SU<Nc>::HotConfiguration(pRNG, Umu); gauge_label = "hot"; }
  }

  const char *as_env   = std::getenv("AUX_SCALE");
  const char *mass_env = std::getenv("MASS");
  const char *csw_env  = std::getenv("CSW");
  const char *ch_env   = std::getenv("CHANNEL");
  const char *ns_env   = std::getenv("NSAMP");
  const RealD aux_scale = (as_env && *as_env) ? std::atof(as_env) : 0.30;
  const RealD mass      = (mass_env && *mass_env) ? std::atof(mass_env) : 1.0;
  const RealD csw       = (csw_env && *csw_env) ? std::atof(csw_env) : 1.24930970916466;
  const int   nsamp     = (ns_env && *ns_env) ? std::atoi(ns_env) : 20;
  std::string channel   = ch_env ? std::string(ch_env) : "all";
  auto on = [&](const char *which) {
    return channel == "all" || channel == which;
  };

  // --- Aux fields ---
  LatticeDtxqcdSigma sigma(UGrid);
  LatticeDtxqcdPi    pi(UGrid);
  LatticeDtxqcdD     d(UGrid);
  LatticeDtxqcdN     n(UGrid);
  LatticeDtxqcdS     s(UGrid);
  LatticeDtxqcdP     p(UGrid);
  sigma = Zero(); pi = Zero(); d = Zero(); n = Zero(); s = Zero(); p = Zero();
  if (on("sigma")) { DtxqcdHermitianCFGaussian(pRNG, sigma);  sigma = aux_scale * sigma; }
  if (on("pi"))    { DtxqcdHermitianCFGaussian(pRNG, pi);     pi    = aux_scale * pi; }
  if (on("d"))     { DtxqcdHermitianCFGaussian(pRNG, d);      d     = aux_scale * d; }
  if (on("n"))     { DtxqcdHermitianCFGaussian(pRNG, n);      n     = aux_scale * n; }
  if (on("s"))     { DtxqcdRealScalarGaussian(pRNG, s); s = aux_scale * s; }
  if (on("p"))     { DtxqcdRealScalarGaussian(pRNG, p); p = aux_scale * p; }

  std::cout << GridLogMessage
            << "[psd-dtxqcd] GAUGE=" << gauge_label
            << " LATT=" << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3]
            << " AUX=" << aux_scale
            << " mass=" << mass
            << " csw=" << csw
            << " channel=" << channel
            << " Nsamp=" << nsamp << std::endl;

  // --- Operators ---
  DTXQCDWilsonCloverFermionEO Dw(Umu, *UGrid, *UrbGrid, mass, csw,
                                  sigma, pi, d, n, s, p);
  DTXQCDMpcOp Mpc(Dw);

  // --- Samplers ---
  Stats full, mee_even, mee_odd, mpc;

  for (int i = 0; i < nsamp; ++i) {
    // (1) Full M^dag M on full-volume random doubled fermion.
    {
      DTXQCDFermionDoubled x(UGrid), y(UGrid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, x.upper.f[a]);
        gaussian(pRNG, x.lower.f[a]);
      }
      Dw.M(x, y);
      RealD num = norm2(y);     // <x, M^dag M x>
      RealD den = norm2(x);
      RealD r   = num / den;
      full.update(r);
    }

    // (2) Mooee^dag Mooee on each CB (per-site 48x48 block spectrum).
    for (int cb : {Even, Odd}) {
      DTXQCDFermionDoubled x4(UGrid), x(UrbGrid), y(UrbGrid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, x4.upper.f[a]);
        gaussian(pRNG, x4.lower.f[a]);
        pickCheckerboard(cb, x.upper.f[a], x4.upper.f[a]);
        pickCheckerboard(cb, x.lower.f[a], x4.lower.f[a]);
        x.upper.f[a].Checkerboard() = cb;
        x.lower.f[a].Checkerboard() = cb;
        y.upper.f[a].Checkerboard() = cb;
        y.lower.f[a].Checkerboard() = cb;
      }
      Dw.Mooee(x, y);
      RealD num = norm2(y);
      RealD den = norm2(x);
      RealD r   = num / den;
      if (cb == Even) mee_even.update(r);
      else            mee_odd.update(r);
    }

    // (3) Mpc^dag Mpc on odd CB (the actual EO Schur operator the
    //     rational pseudofermion CG inverts).
    {
      DTXQCDFermionDoubled x4(UGrid), x(UrbGrid), y(UrbGrid);
      for (int a = 0; a < DtxqcdNf; ++a) {
        gaussian(pRNG, x4.upper.f[a]);
        gaussian(pRNG, x4.lower.f[a]);
        pickCheckerboard(Odd, x.upper.f[a], x4.upper.f[a]);
        pickCheckerboard(Odd, x.lower.f[a], x4.lower.f[a]);
        x.upper.f[a].Checkerboard() = Odd;
        x.lower.f[a].Checkerboard() = Odd;
        y.upper.f[a].Checkerboard() = Odd;
        y.lower.f[a].Checkerboard() = Odd;
      }
      Mpc.M(x, y);
      RealD num = norm2(y);
      RealD den = norm2(x);
      RealD r   = num / den;
      mpc.update(r);
    }
  }

  PrintStats("M_full^dag M_full ", full,     nsamp);
  PrintStats("Mooee^dag Mooee (e)", mee_even, nsamp);
  PrintStats("Mooee^dag Mooee (o)", mee_odd,  nsamp);
  PrintStats("Mpc^dag   Mpc      ", mpc,     nsamp);

  // Headline cross-comparison.
  RealD ratio_max = mpc.max_ratio / std::max(full.max_ratio, 1e-30);
  std::cout << GridLogMessage
            << "[psd-dtxqcd] max(Mpc) / max(M_full) = " << ratio_max
            << "  (>>1 => EO-only pathology, ~1 => full operator stressed)"
            << std::endl;

  Grid_finalize();
  return 0;
}
