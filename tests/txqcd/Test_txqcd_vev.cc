// End-to-end TXQCD correctness: auxiliary-field VEV identities (Phase 4e).
//
// Runs the full TXQCD Wilson HMC and measures auxiliary-field VEVs and
// the stochastic trace Tr(M^{-1}). Verifies the Schwinger-Dyson identities
// that follow from completing the square in the TXQCD Lagrangian.
//
// Because our pseudofermion action uses det(M^dag M) = |det M|^2, each
// TXQCD fermion flavor is effectively DOUBLED, so the SD identity picks
// up a factor of 2 on the Tr M^{-1} side:
//
//   lambda^2 <Tr_f sigma(x)> = 2 <Tr_{f,s,c} M^{-1}(x,x)>
//   lambda^2 <Tr_c s(x)>    = sqrt(2) <Tr M^{-1}(x,x)>
//
// Model-independent consequences (independent of the doubling):
//   <Tr sigma> / <Tr s> = sqrt(2)      (ratio test, no M^{-1} needed)
//   <Tr pi>  = 0                        (parity)
//   <Tr p>   = 0                        (parity)
//
// Lattice 4^4, beta=5.6, lambda=3, mass=0.3. Thermalize 10 trajectories,
// measure 40. Complex Gaussian noise: E[eta^dag A eta] = 2 Tr(A); divide
// the accumulator by 2 to get an unbiased Tr(M^{-1}) estimate.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonPseudoFermionAction.h>
#include <Grid/qcd/action/txqcd/TXQCDCheckpointer.h>
#include <sys/stat.h>

using namespace Grid;

// Configuration-cache helpers: skip HMC if all required checkpoints already
// exist on disk. Checked files are the TXQCDCheckpointer triple
// (<prefix>.<traj>, <prefix>_aux.<traj>, <rng_prefix>.<traj>).
static bool txqcd_file_exists(const std::string &f) {
  struct stat st;
  return stat(f.c_str(), &st) == 0;
}

static bool txqcd_all_cached(const std::string &config_prefix,
                             const std::string &rng_prefix,
                             int first_traj, int last_traj) {
  for (int t = first_traj; t <= last_traj; ++t) {
    std::ostringstream c, a, r;
    c << config_prefix << "." << t;
    a << config_prefix << "_aux." << t;
    r << rng_prefix << "." << t;
    if (!txqcd_file_exists(c.str()) ||
        !txqcd_file_exists(a.str()) ||
        !txqcd_file_exists(r.str())) return false;
  }
  return true;
}

static void txqcd_mkdir_p(const std::string &d) {
  if (!d.empty()) mkdir(d.c_str(), 0755);
}

struct VEVAccumulator : public HmcObservable<TXQCDField> {
  int n_therm;
  int n_noise;
  RealD mass;
  RealD lambda;
  RealD cg_tol;
  int cg_max;

  std::vector<RealD> sigma_tr, s_tr, pi_tr, p_tr, minv_tr;

  VEVAccumulator(int ntherm, RealD mass_, RealD lambda_,
                 int nnoise = 4, RealD cgtol = 1e-10, int cgmax = 10000)
      : n_therm(ntherm), n_noise(nnoise), mass(mass_), lambda(lambda_),
        cg_tol(cgtol), cg_max(cgmax) {}

  void TrajectoryComplete(int traj, TXQCDField &U, GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) override {
    if (traj <= n_therm) return;

    GridBase *grid = U.Grid();
    RealD V = grid->gSites();

    RealD sig = TensorRemove(sum(trace(U.sigma))).real() / V;
    RealD ss  = TensorRemove(sum(trace(U.s))).real() / V;
    RealD pi  = TensorRemove(sum(trace(U.pi))).real() / V;
    RealD pp  = TensorRemove(sum(trace(U.p))).real() / V;
    sigma_tr.push_back(sig);
    s_tr.push_back(ss);
    pi_tr.push_back(pi);
    p_tr.push_back(pp);

    // Stochastic Tr(M^{-1}) via Z2 noise.
    GridCartesian *Ugrid = dynamic_cast<GridCartesian *>(grid);
    GridRedBlackCartesian rbgrid(Ugrid);
    TXQCDWilsonOp Mop(U.U, *Ugrid, rbgrid, mass, U.sigma, U.pi, U.s, U.p, U.t);

    RealD minv_acc = 0.0;
    for (int hit = 0; hit < n_noise; ++hit) {
      TXQCDFermionNf eta(grid), Mdeta(grid), x(grid);
      // Complex Gaussian noise: E[eta_i* eta_j] = 2 delta_{ij}, so
      // E[eta^dag A eta] = 2 Tr(A). We divide the accumulator by 2 below.
      for (int a = 0; a < TxqcdNf; ++a) {
        gaussian(pRNG, eta.f[a]);
      }
      Mop.Mdag(eta, Mdeta);
      x = Zero();
      // CG for MdagM x = Mdag eta → x = M^{-1} eta
      {
        TXQCDFermionNf r(grid), p(grid), Mp(grid), MdMp(grid);
        r = Mdeta; p = r;
        RealD rsq = norm2(r);
        RealD bsq = std::max(norm2(Mdeta), 1e-30);
        RealD tol2 = cg_tol * cg_tol * bsq;
        for (int it = 0; it < cg_max; ++it) {
          Mop.M(p, Mp); Mop.Mdag(Mp, MdMp);
          ComplexD pAp = innerProduct(p, MdMp);
          ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
          axpy(x,  alpha, p);
          axpy(r, -alpha, MdMp);
          RealD rsq_new = norm2(r);
          if (rsq_new < tol2) { rsq = rsq_new; break; }
          RealD beta = rsq_new / rsq;
          for (int aa = 0; aa < TxqcdNf; ++aa)
            p.f[aa] = r.f[aa] + beta * p.f[aa];
          rsq = rsq_new;
        }
      }
      ComplexD dot = innerProduct(eta, x);
      minv_acc += dot.real() / (2.0 * V);  // factor 2 from complex Gaussian norm
    }
    minv_tr.push_back(minv_acc / n_noise);
  }

  static RealD mean(const std::vector<RealD> &v) {
    RealD s = 0; for (auto x : v) s += x; return s / v.size();
  }
  static RealD stderr(const std::vector<RealD> &v) {
    RealD m = mean(v), s2 = 0;
    for (auto x : v) s2 += (x - m) * (x - m);
    return std::sqrt(s2 / (v.size() * (v.size() - 1)));
  }
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  RealD beta   = 5.6;
  RealD lambda = 3.0;
  RealD mass   = 0.3;
  RealD cg_tol = 1e-8;
  int   cg_max = 10000;
  int   n_therm = 10;
  int   n_meas  = 40;

  GaugeActionAdapter<WilsonGaugeActionR> GaugeAction(beta);
  AuxiliaryFieldGaussianAction            AuxAction(lambda);
  TXQCDWilsonPseudoFermionAction          PFAction(Grid, RBGrid, mass, cg_tol, cg_max);

  typedef Representations<EmptyRep<TXQCDField>> TxqcdReps;
  ActionLevel<TXQCDField, TxqcdReps> Level1(1);
  Level1.push_back(&PFAction);
  Level1.push_back(&AuxAction);
  ActionLevel<TXQCDField, TxqcdReps> Level2(4);
  Level2.push_back(&GaugeAction);
  ActionSet<TXQCDField, TxqcdReps> Aset;
  Aset.push_back(Level1);
  Aset.push_back(Level2);

  IntegratorParameters MD;
  MD.name = "LeapFrog"; MD.MDsteps = 80; MD.trajL = 0.5;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory   = 0;
  HMCparams.Trajectories      = n_therm + n_meas;
  HMCparams.NoMetropolisUntil = 0;
  HMCparams.MetropolisTest    = true;
  HMCparams.PerformRandomShift = false;
  HMCparams.StartingType      = "ColdStart";
  HMCparams.MD                = MD;

  NoSmearing<TXQCDCompositeImpl> Smearer;
  typedef LeapFrog<TXQCDCompositeImpl, NoSmearing<TXQCDCompositeImpl>, TxqcdReps> IntegratorT;
  IntegratorT MDynamics(&Grid, MD, Aset, Smearer);

  TXQCDField U(&Grid);
  TXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  Smearer.set_Field(U);

  VEVAccumulator obs(n_therm, mass, lambda, /*n_noise=*/4, /*cg_tol=*/1e-10, cg_max);

  // Config cache: save post-thermalization trajectories to disk; on re-run,
  // skip HMC entirely if every required checkpoint is already present.
  std::string cache_dir    = "configs_vev";
  std::string config_prefix = cache_dir + "/ckpoint_lat";
  std::string rng_prefix    = cache_dir + "/ckpoint_rng";
  txqcd_mkdir_p(cache_dir);

  int first_traj = n_therm + 1;
  int last_traj  = n_therm + n_meas;
  bool cached = txqcd_all_cached(config_prefix, rng_prefix, first_traj, last_traj);

  CheckpointerParameters ckpt_params;
  ckpt_params.config_prefix = config_prefix;
  ckpt_params.rng_prefix    = rng_prefix;
  ckpt_params.saveInterval  = 1;
  ckpt_params.format        = "IEEE64BIG";
  TXQCDCheckpointer ckpt(ckpt_params);

  if (cached) {
    std::cout << GridLogMessage << "VEV test: reusing cached configs from "
              << cache_dir << " (traj " << first_traj << ".." << last_traj
              << "); skipping HMC." << std::endl;
    for (int traj = first_traj; traj <= last_traj; ++traj) {
      ckpt.CheckpointRestore(traj, U, sRNG, pRNG);
      obs.TrajectoryComplete(traj, U, sRNG, pRNG);
    }
  } else {
    std::cout << GridLogMessage << "VEV test: generating configs into "
              << cache_dir << "/" << std::endl;
    std::vector<HmcObservable<TXQCDField> *> Observables = {&obs, &ckpt};
    HybridMonteCarlo<IntegratorT> HMC(HMCparams, MDynamics, sRNG, pRNG,
                                      Observables, U);
    HMC.evolve();
  }

  // ---- report ----
  int exitcode = 0;
  int N = obs.sigma_tr.size();
  std::cout << GridLogMessage << "Measured " << N << " configurations after "
            << n_therm << " thermalization trajectories." << std::endl;

  RealD sig_mean  = VEVAccumulator::mean(obs.sigma_tr);
  RealD sig_err   = VEVAccumulator::stderr(obs.sigma_tr);
  RealD s_mean    = VEVAccumulator::mean(obs.s_tr);
  RealD s_err     = VEVAccumulator::stderr(obs.s_tr);
  RealD pi_mean   = VEVAccumulator::mean(obs.pi_tr);
  RealD pi_err    = VEVAccumulator::stderr(obs.pi_tr);
  RealD p_mean    = VEVAccumulator::mean(obs.p_tr);
  RealD p_err     = VEVAccumulator::stderr(obs.p_tr);
  RealD minv_mean = VEVAccumulator::mean(obs.minv_tr);
  RealD minv_err  = VEVAccumulator::stderr(obs.minv_tr);

  std::cout << GridLogMessage << "<Tr sigma>/V = " << sig_mean
            << " +/- " << sig_err << std::endl;
  std::cout << GridLogMessage << "<Tr s>/V     = " << s_mean
            << " +/- " << s_err << std::endl;
  std::cout << GridLogMessage << "<Tr pi>/V    = " << pi_mean
            << " +/- " << pi_err << std::endl;
  std::cout << GridLogMessage << "<Tr p>/V     = " << p_mean
            << " +/- " << p_err << std::endl;
  std::cout << GridLogMessage << "<Tr M^{-1}>/V = " << minv_mean
            << " +/- " << minv_err << std::endl;

  // Test 1: ratio <Tr sigma> / <Tr s> = sqrt(2)
  {
    RealD ratio = sig_mean / s_mean;
    RealD expected = std::sqrt(2.0);
    RealD ratio_err = std::abs(ratio) *
        std::sqrt((sig_err/sig_mean)*(sig_err/sig_mean) +
                  (s_err/s_mean)*(s_err/s_mean));
    RealD nsigma = std::abs(ratio - expected) / ratio_err;
    bool pass = nsigma < 3.0;
    std::cout << GridLogMessage << "[ratio] <Tr sigma>/<Tr s> = " << ratio
              << " +/- " << ratio_err << " expected " << expected
              << " (" << nsigma << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Test 2: lambda^2 <Tr sigma>/V = 2 <Tr M^{-1}>/V
  //   Factor 2 = flavor doubling from det(M^dag M) pseudofermion.
  {
    RealD lhs = lambda * lambda * sig_mean;
    RealD lhs_err = lambda * lambda * sig_err;
    RealD rhs = 2.0 * minv_mean;
    RealD rhs_err = 2.0 * minv_err;
    RealD diff = lhs - rhs;
    RealD diff_err = std::sqrt(lhs_err*lhs_err + rhs_err*rhs_err);
    RealD nsigma = std::abs(diff) / diff_err;
    bool pass = nsigma < 3.0;
    std::cout << GridLogMessage << "[sigma EOM] lambda^2<Tr sigma>/V = " << lhs
              << " +/- " << lhs_err << "  2<Tr M^{-1}>/V = " << rhs
              << " +/- " << rhs_err << " (" << nsigma << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Test 3: lambda^2 <Tr s>/V = sqrt(2) <Tr M^{-1}>/V
  //   = 2 * (1/sqrt2) * <Tr M^{-1}>, factor 2 from flavor doubling.
  {
    RealD lhs = lambda * lambda * s_mean;
    RealD lhs_err = lambda * lambda * s_err;
    RealD rhs = std::sqrt(2.0) * minv_mean;
    RealD rhs_err = std::sqrt(2.0) * minv_err;
    RealD diff = lhs - rhs;
    RealD diff_err = std::sqrt(lhs_err*lhs_err + rhs_err*rhs_err);
    RealD nsigma = std::abs(diff) / diff_err;
    bool pass = nsigma < 3.0;
    std::cout << GridLogMessage << "[s EOM] lambda^2<Tr s>/V = " << lhs
              << " +/- " << lhs_err << "  sqrt2<Tr M^{-1}>/V = " << rhs
              << " +/- " << rhs_err << " (" << nsigma << " sigma)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Test 4: <Tr pi>/V consistent with zero
  {
    RealD nsigma = (pi_err > 0) ? std::abs(pi_mean) / pi_err : 0.0;
    bool pass = nsigma < 3.0;
    std::cout << GridLogMessage << "[parity pi] <Tr pi>/V = " << pi_mean
              << " +/- " << pi_err << " (" << nsigma << " sigma from zero)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // Test 5: <Tr p>/V consistent with zero
  {
    RealD nsigma = (p_err > 0) ? std::abs(p_mean) / p_err : 0.0;
    bool pass = nsigma < 3.0;
    std::cout << GridLogMessage << "[parity p] <Tr p>/V = " << p_mean
              << " +/- " << p_err << " (" << nsigma << " sigma from zero)"
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
