// Valence-mass sweep on a single QCD gauge cfg.
//
// Loads a thermalized QCD config (sea mass set by GAUGE_MASS env, traj by
// TRAJ env) and inverts plain Wilson at a list of progressively lighter
// valence masses to map out where the connected pion mass goes and where
// CG breaks down (exceptional config / Aoki phase intrusion).
//
// Each inversion uses 12 spin-color point sources -> proper pion + nucleon
// correlators.  CG max iter is capped (default 5000); if any CG hits maxit
// without converging the valence mass is flagged in the output and we move
// to the next mass without aborting.
//
// Output: meas_2pt_wilson8x24_valence_sweep_m<gauge>/sweep.h5
//   { valence_masses[N], pion_conn[N, T], nucleon_re[N, T], cg_status[N],
//     max_cg_iter[N], plaquette }
//
// Env vars:
//   GAUGE_MASS=-0.5      sea (gauge-config) mass / cfg dir suffix
//   TRAJ=280             single trajectory # to load
//   VALENCE_MASSES=...   comma-separated list, e.g. "-0.5,-0.55,-0.6,...,-0.85"
//                         default: 13 masses from -0.50 down to -0.86 step -0.03
//   CG_MAX=5000          max CG iters before flagging non-convergence
//   CG_TOL=1e-10         CG tolerance

#include "Test_txqcd_2pt_wilson8x24_utils.h"
#include <Grid/serialisation/Hdf5IO.h>
#include <Grid/qcd/utils/BaryonUtils.h>
#include <Grid/qcd/utils/WilsonLoops.h>

using namespace TxqcdTest2ptWilson8x24;

static void PointSource(const Coordinate &site, LatticePropagator &src) {
  src = Zero();
  SpinColourMatrix kron;
  kron = 1.0;
  pokeSite(kron, src, site);
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

// Plain-Wilson connected propagator at a custom valence mass.
// Returns max CG iter encountered (caps at maxit).  CG_status: 0=ok,
// 1=hit maxit on at least one inversion.
static int QcdPointProp(LatticePropagator &S, LatticeGaugeField &Umu,
                        RealD m_val, GridCartesian &Grid,
                        GridRedBlackCartesian &RBGrid,
                        const Coordinate &src, RealD tol, int maxit,
                        int &max_iter) {
  WilsonFermionD Dw(Umu, Grid, RBGrid, m_val);
  MdagMLinearOperator<WilsonFermionD, LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(tol, maxit, /*err_on_no_conv*/false);

  LatticePropagator srcP(&Grid);
  PointSource(src, srcP);
  S = Zero();
  max_iter = 0;
  int status = 0;
  for (int spin = 0; spin < Ns; ++spin) {
    for (int col = 0; col < Nc; ++col) {
      LatticeFermion sf(&Grid), b(&Grid), x(&Grid);
      PropToFerm<WilsonImplR>(sf, srcP, spin, col);
      Dw.Mdag(sf, b);
      x = Zero();
      CG(HermOp, b, x);
      int iter = CG.IterationsToComplete;
      if (iter <= 0) iter = maxit;
      max_iter = std::max(max_iter, iter);
      if (iter >= maxit) status = 1;
      FermToProp<WilsonImplR>(S, x, spin, col);
    }
  }
  return status;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  // Parse env vars.
  RealD gauge_mass = -0.5;
  if (const char *v = std::getenv("GAUGE_MASS")) gauge_mass = std::atof(v);
  int traj = 280;
  if (const char *v = std::getenv("TRAJ")) traj = std::atoi(v);
  RealD cg_tol = 1e-10;
  if (const char *v = std::getenv("CG_TOL")) cg_tol = std::atof(v);
  int cg_maxiter = 5000;
  if (const char *v = std::getenv("CG_MAX")) cg_maxiter = std::atoi(v);

  std::vector<RealD> valence_masses;
  if (const char *v = std::getenv("VALENCE_MASSES")) {
    std::string s(v);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!tok.empty()) valence_masses.push_back(std::atof(tok.c_str()));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  } else {
    // Default: 13-step sweep from -0.50 to -0.86 in -0.03 steps.
    for (int i = 0; i < 13; ++i)
      valence_masses.push_back(-0.50 - 0.03 * i);
  }
  std::cout << GridLogMessage << "[valence_sweep] gauge_mass=" << gauge_mass
            << " traj=" << traj << " cg_tol=" << cg_tol
            << " cg_maxiter=" << cg_maxiter
            << " n_valence=" << valence_masses.size() << std::endl;

  Coordinate latt = default_latt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  Coordinate src = src_site();

  // Build cfg dir from gauge_mass via the same mass_suffix logic.
  char buf[64];
  std::snprintf(buf, sizeof(buf), "_m%+.4f", gauge_mass);
  std::string gauge_suffix = (std::abs(gauge_mass - mass) < 1e-6) ? "" : std::string(buf);
  std::string gauge_dir = "configs_2pt_qcd_wilson8x24" + gauge_suffix;

  std::cout << GridLogMessage << "[valence_sweep] loading "
            << gauge_dir << "/ckpoint_lat." << traj << std::endl;

  LatticeGaugeField Umu(&Grid);
  TxqcdTest2pt::LoadQcdConfig(Umu, sRNG, pRNG, traj, gauge_dir);
  RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(Umu);
  std::cout << GridLogMessage << "[valence_sweep] plaquette=" << plaq << std::endl;

  // Output containers.
  std::vector<std::vector<RealD>>    pion_all;
  std::vector<std::vector<ComplexD>> nucl_all;
  std::vector<int> cg_status_all;
  std::vector<int> max_iter_all;

  for (size_t i = 0; i < valence_masses.size(); ++i) {
    RealD m_val = valence_masses[i];
    std::cout << GridLogMessage << "[valence_sweep] m_val=" << m_val
              << " (" << (i + 1) << "/" << valence_masses.size() << ")"
              << std::endl;
    LatticePropagator S(&Grid);
    int max_iter = 0;
    int status = QcdPointProp(S, Umu, m_val, Grid, RBGrid, src, cg_tol,
                               cg_maxiter, max_iter);
    auto pion = PionCorrelator(S);
    auto nucl = NucleonCorrelator(S);
    pion_all.push_back(pion);
    nucl_all.push_back(nucl);
    cg_status_all.push_back(status);
    max_iter_all.push_back(max_iter);
    std::cout << GridLogMessage << "  m_val=" << m_val
              << "  C_pi(0)=" << pion[0] << "  C_pi(1)=" << pion[1]
              << "  max_cg_iter=" << max_iter << "  status=" << status
              << std::endl;
  }

  std::string out_dir = "meas_2pt_wilson8x24_valence_sweep" + gauge_suffix;
  mkdir_p(out_dir);
  std::string out_fn = out_dir + "/sweep_traj" + std::to_string(traj) + ".h5";
  Hdf5Writer wr(out_fn);
  write(wr, "valence_masses", valence_masses);
  write(wr, "pion_conn", pion_all);
  write(wr, "nucleon", nucl_all);
  write(wr, "cg_status", cg_status_all);
  write(wr, "max_cg_iter", max_iter_all);
  write(wr, "plaq", plaq);
  write(wr, "gauge_mass", gauge_mass);
  write(wr, "traj", traj);

  std::cout << GridLogMessage << "[valence_sweep] wrote " << out_fn << std::endl;
  Grid_finalize();
  return 0;
}
