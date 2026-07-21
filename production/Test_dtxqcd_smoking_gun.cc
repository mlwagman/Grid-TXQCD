// Test_dtxqcd_smoking_gun -- operator-consistency A/B for the tent
// deformation (2026-07-21): four applications of the SAME random odd
// vector, host-Schur-sandwich vs fused-device-EO x SetTentProfile vs
// generic SetProfile(lambda).  The pairwise agreement pattern
// fingerprints the mismatch behind: the discriminator's two-answers
// (7.95e-5 vs 1.99e-5), the meas deflated-solve slowdown, and the
// defl8 estimator bias (+20: exact term via MpcTil, remainder via the
// fused solve).  All four agreeing to 1e-10 = hypothesis dead.
//
// Env: LATT, CFG_DIR, TRAJ, ETA (node strength, e.g. 0.4423), PROF_T0.

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

  RealD mass = env_real("MASS", -0.245);
  RealD csw = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho = env_real("STOUT_RHO", 0.125);
  RealD eta = env_real("ETA", 0.4423);
  int traj = env_int("TRAJ", 300);
  const int prof_t0 = env_int("PROF_T0", 0);
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

  typedef DTXQCDFermionDoubled F;
  // one fixed random odd vector
  F v(&RBGrid);
  GridParallelRNG rrb(&RBGrid);
  rrb.SeedFixedIntegers({71, 72, 73, 74});
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(rrb, v.upper.f[a]);
    gaussian(rrb, v.lower.f[a]);
    v.upper.f[a].Checkerboard() = Odd;
    v.lower.f[a].Checkerboard() = Odd;
  }

  // tent profile lambda (the estimator's convention)
  auto prof = [&](int t) {
    int dt = ((t - prof_t0) % T + T) % T;
    return (double)std::min(dt, T - dt);
  };

  auto n2 = [&](const F &x) { return DtxqcdDeformQuda::fnorm2_(x); };
  auto reldiff = [&](const F &a, const F &b) {
    F d(&RBGrid);
    RealD nd = 0, na = 0;
    for (int c = 0; c < DtxqcdNf; ++c) {
      d.upper.f[c] = a.upper.f[c] - b.upper.f[c];
      d.lower.f[c] = a.lower.f[c] - b.lower.f[c];
    }
    nd = n2(d);
    na = n2(a);
    return std::sqrt(nd / std::max(na, 1e-300));
  };

  // four applications of Mpc~^dag Mpc~ v (the HermOp both the builder
  // IRL and the estimator's Ghat effectively use)
  F t1(&RBGrid), r_host_tent(&RBGrid), r_host_lam(&RBGrid);
  F r_dev_tent(&RBGrid), r_dev_lam(&RBGrid);

  auto HostHH = [&](F &out) {
    Qdef.MpcTil(v, t1);
    Qdef.MpcTilDag(t1, out);
  };
#ifdef GRID_HAVE_QUDA
  auto DevHH = [&](F &out) {
    LatticeDtxqcdPacked pin(&RBGrid), pout(&RBGrid);
    DtxqcdPackDoubled(v, pin);
    pin.Checkerboard() = Odd;
    HopEO->HermOp(pin, pout);
    DtxqcdUnpackDoubled(pout, out);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a].Checkerboard() = Odd;
      out.lower.f[a].Checkerboard() = Odd;
    }
  };
#endif

  Qdef.SetTentProfile(eta, prof_t0);
  HostHH(r_host_tent);
#ifdef GRID_HAVE_QUDA
  DevHH(r_dev_tent);
#endif
  Qdef.SetProfile(eta, [&](int t) { return prof(t); });
  HostHH(r_host_lam);
#ifdef GRID_HAVE_QUDA
  DevHH(r_dev_lam);
#endif

  std::cout << GridLogMessage << "GUN |v|^2=" << n2(v) << std::endl;
  std::cout << GridLogMessage
            << "GUN host: tentAPI-vs-lambdaAPI  rel="
            << reldiff(r_host_tent, r_host_lam) << std::endl;
#ifdef GRID_HAVE_QUDA
  std::cout << GridLogMessage
            << "GUN dev : tentAPI-vs-lambdaAPI  rel="
            << reldiff(r_dev_tent, r_dev_lam) << std::endl;
  std::cout << GridLogMessage
            << "GUN tentAPI: host-vs-dev        rel="
            << reldiff(r_host_tent, r_dev_tent) << std::endl;
  std::cout << GridLogMessage
            << "GUN lambdaAPI: host-vs-dev      rel="
            << reldiff(r_host_lam, r_dev_lam) << std::endl;
#endif
  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
}
