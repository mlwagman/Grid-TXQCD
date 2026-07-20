// Test_dtxqcd_defl_slicedens -- mode-resolved LOCALIZATION diagnostics for
// a cached deformed-EO deflation basis (the wall-tower structure study).
//
// Reads a defl_cache sidecar (DEFL_CACHE_FILE env: explicit path), and for
// each eigenmode k prints:
//   SLICEDENS k=<k> ev=<eval> fu=<upper-grading fraction>
//     rho: <T slice densities rho_k(t), normalized to sum 1>
//   IPR   k=<k> t=<t_peak> ipr=<transverse inverse participation ratio
//     on the peak slice>  (ipr ~ 1/V_x extended ... O(1) localized:
//     the Anderson question -- transverse-localized tower levels imply
//     self-averaging DlogPf config noise; extended implies coherent)
// Offline: kink weight W_k = sum_{|t - t_kink| <= 3} rho_k(t) vs mode
// index maps the tower's localization ladder.
//
// Env: LATT, CFG_DIR, TRAJ, ETA (must match the cache's build strength),
// PROFILE (tent), PROF_T0 (0), DEFL_CACHE_FILE.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#endif
#include "dtxqcd_deform_quda.h"

using namespace Grid;

static int env_int(const char *n, int d) {
  const char *v = std::getenv(n);
  return (v && *v) ? std::atoi(v) : d;
}
static double env_real(const char *n, double d) {
  const char *v = std::getenv(n);
  return (v && *v) ? std::atof(v) : d;
}
static std::string env_str(const char *n, const std::string &d) {
  const char *v = std::getenv(n);
  return (v && *v) ? std::string(v) : d;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::vector<int> latt_dims = {16, 16, 16, 48};
  if (const char *l = std::getenv("LATT")) {
    latt_dims.clear();
    std::string s(l);
    size_t p = 0;
    while (p != std::string::npos) {
      size_t q = s.find('.', p);
      latt_dims.push_back(std::atoi(s.substr(p, q - p).c_str()));
      p = (q == std::string::npos) ? q : q + 1;
    }
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid_(latt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                      GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid_);
  const int T = latt[Nd - 1];
  const int Vx = latt[0] * latt[1] * latt[2];

  RealD mass = env_real("MASS", -0.245);
  RealD csw = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho = env_real("STOUT_RHO", 0.125);
  RealD eta = env_real("ETA", 0.6);
  int traj = env_int("TRAJ", 300);
  const int prof_t0 = env_int("PROF_T0", 0);
  const std::string cachef = env_str("DEFL_CACHE_FILE", "");
  GRID_ASSERT(!cachef.empty() && "set DEFL_CACHE_FILE");
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam5.0000_tla_fromchroma_cw";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid_);
  DTXQCDField U(&Grid_);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG, cfg_dir + "/ckpoint_lat",
                                 cfg_dir + "/ckpoint_rng", traj);
  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid_);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid_, RBGrid, mass, csw, U.sigma,
                                 U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);
  std::unique_ptr<DtxqcdPackedFullHermOp> Hop(
      new DtxqcdPackedFullHermOp(Mop, &Grid_, Even));
  DtxqcdDeformQuda Qdef(*Hop, Grid_, Usm, T);
#ifdef GRID_HAVE_QUDA
  Grid::Quda::initialize();
  std::unique_ptr<DTXQCDMpcOpQUDA_EO> Mq_eo(new DTXQCDMpcOpQUDA_EO(
      Dw, Usm, Grid_, RBGrid, mass, csw, U.sigma, U.pi, U.d, U.n, U.s, U.p));
  std::unique_ptr<DTXQCDMpcOp> Mpc_cpu(new DTXQCDMpcOp(Dw));
  std::unique_ptr<DtxqcdPackedMpcHermOp> HopEO(
      new DtxqcdPackedMpcHermOp(*Mpc_cpu, *Mq_eo, &RBGrid, Odd));
  Qdef.AttachEO(*HopEO, *Mq_eo, RBGrid, Dw);
#endif
  Qdef.SetTentProfile(eta, prof_t0);
  GRID_ASSERT(Qdef.LoadEODeflation(cachef) && "cache load failed");
  const auto &evec = Qdef.EODeflEvecs();
  const auto &eval = Qdef.EODeflEvals();
  const int K = (int)evec.size();
  std::cout << GridLogMessage << "[slicedens] K=" << K << " eta=" << eta
            << " t0=" << prof_t0 << " cache=" << cachef << std::endl;

  typedef DTXQCDFermionDoubled F;
  LatticeInteger tc(&Grid_);
  LatticeCoordinate(tc, Nd - 1);
  for (int k = 0; k < K; ++k) {
    F vrb(&RBGrid), vf(&Grid_);
    DtxqcdUnpackDoubled(evec[k], vrb);
    RealD nu = 0, nl = 0;
    LatticeComplex dens(&Grid_);
    dens = Zero();
    for (int a = 0; a < DtxqcdNf; ++a) {
      vrb.upper.f[a].Checkerboard() = Odd;
      vrb.lower.f[a].Checkerboard() = Odd;
      vf.upper.f[a] = Zero();
      vf.lower.f[a] = Zero();
      setCheckerboard(vf.upper.f[a], vrb.upper.f[a]);
      setCheckerboard(vf.lower.f[a], vrb.lower.f[a]);
      nu += norm2(vf.upper.f[a]);
      nl += norm2(vf.lower.f[a]);
      dens = dens + localInnerProduct(vf.upper.f[a], vf.upper.f[a]);
      dens = dens + localInnerProduct(vf.lower.f[a], vf.lower.f[a]);
    }
    std::vector<TComplex> sums;
    sliceSum(dens, sums, Nd - 1);
    const RealD tot = std::max(nu + nl, 1e-300);
    int tpk = 0;
    RealD rpk = 0;
    std::cout << GridLogMessage << "SLICEDENS k=" << k << " ev="
              << (k < (int)eval.size() ? eval[k] : -1.0)
              << " fu=" << nu / tot << " rho:";
    for (int t = 0; t < T; ++t) {
      const RealD r = TensorRemove(sums[t]).real() / tot;
      std::cout << " " << r;
      if (r > rpk) { rpk = r; tpk = t; }
    }
    std::cout << std::endl;
    // transverse IPR on the peak slice: sum |psi|^4 / (sum |psi|^2)^2
    // over spatial sites (all internal components summed per site).
    LatticeComplex mpk(&Grid_), one(&Grid_);
    one = ComplexD(1.0, 0.0);
    mpk = ComplexD(0.0, 0.0);
    mpk = where(tc == Integer(tpk), one, mpk);
    LatticeComplex d2(&Grid_);
    d2 = mpk * dens;
    LatticeComplex d4(&Grid_);
    d4 = d2 * d2;
    const RealD s2 = real(sum(d2));
    const RealD s4 = real(sum(d4));
    std::cout << GridLogMessage << "IPR k=" << k << " t=" << tpk
              << " ipr=" << s4 / std::max(s2 * s2, 1e-300)
              << " (1/Vx=" << 1.0 / (double)Vx << " extended baseline)"
              << std::endl;
  }
  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
