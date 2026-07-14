// probe_deform_spectrum: eigenspectrum of the eta-deformed normal operator
// M~^dag M~ vs eta — WHERE does the e^{3.7 eta} CG hardness come from?
// (One collapsing isolated mode -> continuation guesses / deflation fix it;
// a dense collapsing band -> needs MG-class help.)
//
// For each eta in ETAS (csv): lambda_max via PowerMethod, then the lowest
// NEV eigenvalues via Chebyshev-accelerated IRL on the QUDA-backed M~ (the
// DtxqcdDeformQuda hop-correction operator, identity-gated to 1e-15 against
// the Grid sandwich).  Profile-t0 shifts are lattice translations
// (isospectral), so a single fixed t0 characterizes each eta.
//
// Env: LATT, CFG_DIR, TRAJ (argv[1]), ETAS ("0.2,0.4,0.6,0.8"), NEV (12),
//      IRL_ORD (301), IRL_TOL (1e-4), IRL_LO_FRAC (3e-5: window lo as a
//      fraction of lambda_max — everything below converges fastest),
//      IRL_RESTARTS (20), PROFILE (tent), PROF_T0 (0), MASS, CSW,
//      STOUT_NSMEAR, STOUT_RHO.
//
// Run (1 GPU):  env LATT=16.16.16.48 CFG_DIR=... ETAS=0.2,0.4,0.6,0.8 \
//               ./probe_deform_spectrum 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
//               --shm 512 --shm-mpi 1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDRemezAutoScale.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#endif
#include "dtxqcd_deform_quda.h"

#include <sstream>

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::atoi(v) : d;
}
static std::string env_str(const char *k, const std::string &d) {
  const char *v = std::getenv(k);
  return (v && *v) ? std::string(v) : d;
}

// Shim: DtxqcdDeformQuda's Mtil/MtilDag as the M/Mdag interface that
// DtxqcdLanczosMinMax (the RHMC-autoscale plain Lanczos) expects.
struct DeformOpShim {
  DtxqcdDeformQuda &q;
  void M(const DTXQCDFermionDoubled &in, DTXQCDFermionDoubled &out) {
    q.Mtil(in, out);
  }
  void Mdag(const DTXQCDFermionDoubled &in, DTXQCDFermionDoubled &out) {
    q.MtilDag(in, out);
  }
};

// M~^dag M~ on the packed field (IRL/PowerMethod need one Lattice type).
class PackedDeformNormalOp : public LinearOperatorBase<LatticeDtxqcdPacked> {
 public:
  typedef LatticeDtxqcdPacked Field;
  PackedDeformNormalOp(DtxqcdDeformQuda &q, GridCartesian &g)
      : q_(q), s_(&g), t_(&g), u_(&g) {}
  void HermOp(const Field &in, Field &out) {
    DtxqcdUnpackDoubled(in, s_);
    q_.Mtil(s_, t_);
    q_.MtilDag(t_, u_);
    DtxqcdPackDoubled(u_, out);
    out.Checkerboard() = in.Checkerboard();
  }
  void Op(const Field &in, Field &out) { HermOp(in, out); }
  void AdjOp(const Field &in, Field &out) { HermOp(in, out); }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void OpDiag(const Field &in, Field &out) { GRID_ASSERT(0); }
  void OpDir(const Field &in, Field &out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const Field &in, std::vector<Field> &out) { GRID_ASSERT(0); }

 private:
  DtxqcdDeformQuda &q_;
  DTXQCDFermionDoubled s_, u_;
  DTXQCDFermionDoubled t_;
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  if (argc < 2) {
    std::cerr << "Usage: probe_deform_spectrum <traj>" << std::endl;
    return 1;
  }
  int traj = std::atoi(argv[1]);

  std::vector<int> latt_dims;
  {
    std::string v = env_str("LATT", "16.16.16.48");
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                     GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid);
  const int T = latt[Nd - 1];

  RealD mass = env_real("MASS", -0.245);
  RealD csw = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho = env_real("STOUT_RHO", 0.125);
  const bool tent = env_str("PROFILE", "tent") != "linear";
  const int prof_t0 = env_int("PROF_T0", 0);
  std::string cfg_dir = env_str(
      "CFG_DIR",
      "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw");

  std::vector<double> etas;
  {
    std::stringstream ss(env_str("ETAS", "0.2,0.4,0.6,0.8"));
    std::string tok;
    while (std::getline(ss, tok, ',')) etas.push_back(std::atof(tok.c_str()));
  }
  const int nev = env_int("NEV", 12);
  const int ord = env_int("IRL_ORD", 301);
  const RealD irl_tol = env_real("IRL_TOL", 1e-4);
  const RealD lo_frac = env_real("IRL_LO_FRAC", 3e-5);
  const int restarts = env_int("IRL_RESTARTS", 20);

  std::cout << GridLogMessage << "==== probe_deform_spectrum ====" << std::endl;
  std::cout << GridLogMessage << " traj=" << traj << " profile="
            << (tent ? "tent" : "linear") << " t0=" << prof_t0
            << " nev=" << nev << " ord=" << ord << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG, cfg_dir + "/ckpoint_lat",
                                 cfg_dir + "/ckpoint_rng", traj);

  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw, U.sigma, U.pi,
                                 U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);
#ifdef GRID_HAVE_QUDA
  Grid::Quda::initialize();
  DTXQCDMpcOpQUDA Mq(Usm, Grid, RBGrid, mass, csw, U.sigma, U.pi, U.d, U.n,
                     U.s, U.p);
  Mq.SetAuxStatic(true);
  DtxqcdPackedFullHermOp Hop(Mop, Mq, &Grid, Even);
#else
  DtxqcdPackedFullHermOp Hop(Mop, &Grid, Even);
#endif
  DtxqcdDeformQuda Qdef(Hop, Grid, Usm, T);

  auto prof = [&](int t) {
    const int dt = ((t - prof_t0) % T + T) % T;
    return tent ? (double)std::min(dt, T - dt) : (double)dt;
  };

  PackedDeformNormalOp NOp(Qdef, Grid);
  GridParallelRNG RNG(&Grid);
  RNG.SeedFixedIntegers({314, 159, 265, 358});

  // SPECTRUM_MODE=ritz: the RHMC-autoscale plain Lanczos (DtxqcdLanczosMinMax,
  // no Chebyshev) — lambda_min/max Ritz estimates in ~2*NM applies (~1 s per
  // eta at the native-kernel speed).  Right tool for mildly-conditioned
  // spectra (lambda=5/6); at deep kappa the Ritz lambda_min is an UPPER
  // bound (underconverged), so treat lambda=3/4 ritz numbers as bounds.
  // Default mode "irl" = the Chebyshev-IRL census below.
  const bool ritz = env_str("SPECTRUM_MODE", "irl") == "ritz";
  const int ritz_nm = env_int("NM", 48);
  if (ritz) {
    DeformOpShim shim{Qdef};
    for (double eta : etas) {
      Qdef.SetProfile(eta, prof);
      RealD lmin = 0, lmax = 0;
      DtxqcdLanczosMinMax(shim, (GridBase *)&Grid, RNG, ritz_nm, /*cb=*/-1,
                          lmin, lmax);
      std::cout << GridLogMessage << "[ritz] eta=" << eta
                << " lambda_min~" << lmin << " lambda_max~" << lmax
                << " kappa~" << (lmin > 0 ? lmax / lmin : 0.0) << std::endl;
    }
    std::cout << GridLogMessage << "==== probe_deform_spectrum done ===="
              << std::endl;
    Grid_finalize();
    std::cout.flush();
    std::_Exit(0);
  }

  for (double eta : etas) {
    Qdef.SetProfile(eta, prof);

    LatticeDtxqcdPacked src(&Grid);
    gaussian(RNG, src);
    PowerMethod<LatticeDtxqcdPacked> PM;
    const RealD lmax = PM(NOp, src);
    std::cout << GridLogMessage << "[spectrum] eta=" << eta
              << " lambda_max(MdagM~) ~ " << lmax << std::endl;

    const RealD irl_lo = lo_frac * lmax;
    const RealD irl_hi = 1.5 * lmax;
    const int Nstop = nev;
    const int Nk = nev + 12;
    const int Nm = Nk + std::max(32, Nk);
    Chebyshev<LatticeDtxqcdPacked> cheb(irl_lo, irl_hi, ord);
    FunctionHermOp<LatticeDtxqcdPacked> OpCheby(cheb, NOp);
    PlainHermOp<LatticeDtxqcdPacked> Op(NOp);
    ImplicitlyRestartedLanczos<LatticeDtxqcdPacked> IRL(OpCheby, Op, Nstop, Nk,
                                                        Nm, irl_tol, restarts);
    std::vector<RealD> eval(Nm);
    std::vector<LatticeDtxqcdPacked> evec(Nm, &Grid);
    int Nconv = 0;
    IRL.calc(eval, evec, src, Nconv);
    const int nuse = std::min(Nconv, (int)eval.size());
    std::cout << GridLogMessage << "[spectrum] eta=" << eta
              << " converged=" << nuse << " lambda_max=" << lmax << std::endl;
    for (int i = 0; i < nuse; ++i)
      std::cout << GridLogMessage << "[spectrum] eta=" << eta << " ev[" << i
                << "] = " << eval[i] << "  (kappa_i = " << lmax / eval[i]
                << ")" << std::endl;
  }

  std::cout << GridLogMessage << "==== probe_deform_spectrum done ===="
            << std::endl;
  Grid_finalize();
  std::cout.flush();
  std::_Exit(0);  // skip QUDA teardown (house pattern)
}
