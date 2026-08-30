#pragma once
// DTXQCDSignHop4t4Observer: sign-sector-hopping heatbath + exact per-traj
// Pfaffian-sign record for SMALL-VOLUME (4^4-class) DTXQCD HMC.
//
// The |Pf| HMC cannot cross Pf=0 (the pseudofermion action diverges), so it
// is trapped in one sign sector and <signPf> under the target measure is
// unmeasurable from a plain chain.  The 2026-07-04 corrected sign-flip scan
// found two aux moves that DO cross sectors while leaving S_aux exactly
// invariant (S_aux is even in every field):
//   s_tslice0:     s -> -s on the t=0 time slice
//   negate_sigma:  sigma -> -sigma globally
// After each trajectory this observer (with probability hop_prob) proposes
// one of them and Metropolis-accepts with
//   A = min(1, |Pf(T(X))| / |Pf(X)|) = min(1, e^{dlad/2}),
//   dlad = log|det M48(T(X))| - log|det M48(X)|,
// computed EXACTLY from the dense gamma5.M48 spectrum (cuSOLVER Zheevd).
// The augmented chain is ergodic across sign sectors, so the per-traj
// signPf = (-1)^(n_neg/2) record (printed every trajectory as [SIGNHOP])
// gives an honest <signPf> under the target measure.
//
// Notes / limits:
// - Dense: dim = 48*V.  4^4 -> 12288 (~40 s/solve, ~2.4 GB buffers).  The
//   constructor asserts a single rank and V <= 4096 sites.
// - The operator is built on the STOUT-SMEARED gauge (n_stout iterations of
//   rho) to match the action when the driver runs is_smeared=true; the aux
//   mutation is applied to the thin-link field (aux passes through smearing
//   unchanged, so the action sees it directly).
// - The chain samples the RATIONAL approx of |Pf|; the hop test uses the
//   exact |Pf|.  The stationary-measure mismatch is O(Remez tolerance).
// - Place BEFORE the checkpointer (post-hop state must be what is saved).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/observables/hmc_observable.h>
#ifdef GRID_CUDA
#include <cusolverDn.h>
#include <cuda_runtime.h>
#endif

NAMESPACE_BEGIN(Grid);

class DTXQCDSignHop4t4Observer : public HmcObservable<DTXQCDField> {
 public:
  DTXQCDSignHop4t4Observer(GridCartesian &Grid_, GridRedBlackCartesian &RB_,
                           RealD mass_, RealD csw_, RealD hop_prob,
                           int n_stout, RealD stout_rho)
      : Grid(Grid_), RBGrid(RB_), mass(mass_), csw(csw_), prob(hop_prob),
        nstout(n_stout), rho(stout_rho),
        V((int)Grid_.gSites()), dim(48 * (int)Grid_.gSites()),
        tcoord(&Grid_) {
    GRID_ASSERT(Grid.gSites() == Grid.lSites());  // dense assembly: 1 rank
    GRID_ASSERT(V <= 4096);                       // 4^4-class volumes only
    H.resize((size_t)dim * dim);
    evals.resize(dim);
    LatticeCoordinate(tcoord, Nd - 1);
#ifdef GRID_CUDA
    GRID_ASSERT(cusolverDnCreate(&cus) == CUSOLVER_STATUS_SUCCESS);
#endif
  }
  // No cusolverDnDestroy: the observer lives until main-scope teardown, by
  // which point the CUDA context can already be gone and the destroy call
  // SIGABRTs at exit (rc=134 in the 1297282 smoke; same teardown issue that
  // makes the measurement binaries _Exit).  Leaking one handle at process
  // exit is harmless.
  ~DTXQCDSignHop4t4Observer() {}

  void TrajectoryComplete(int traj, ConfigurationBase<DTXQCDField> &SmartConfig,
                          GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {
    TrajectoryComplete(traj, SmartConfig.get_U(false), sRNG, pRNG);
  }

  void TrajectoryComplete(int traj, DTXQCDField &U, GridSerialRNG &sRNG,
                          GridParallelRNG & /*pRNG*/) override {
    int n_neg = 0; double lad = 0.0, min_abs = 0.0;
    Spectrum(U, n_neg, lad, min_abs);
    int sgn = ((n_neg / 2) % 2) ? -1 : +1;
    std::cout << GridLogMessage << "[SIGNHOP] traj=" << traj
              << " signPf=" << std::showpos << sgn << std::noshowpos
              << " n_neg=" << n_neg << " logabsdet=" << lad
              << " min_abs_eval=" << min_abs
              << ((n_neg & 1) ? "  (ODD n_neg!)" : "") << std::endl;

    RealD r; random(sRNG, r);
    if (r >= prob) return;
    RealD rmove; random(sRNG, rmove);
    const bool use_sigma = (rmove < 0.5);
    const char *mname = use_sigma ? "negate_sigma" : "s_tslice0";

    DTXQCDField W(&Grid);
    W.U = U.U;
    W.sigma = U.sigma; W.pi = U.pi; W.d = U.d;
    W.n = U.n; W.s = U.s; W.p = U.p;
    if (use_sigma) {
      W.sigma = -W.sigma;
    } else {
      LatticeDtxqcdS neg = -W.s;
      W.s = where(tcoord == Integer(0), neg, W.s);
    }
    int n_neg2 = 0; double lad2 = 0.0, min_abs2 = 0.0;
    Spectrum(W, n_neg2, lad2, min_abs2);
    const int sgn2 = ((n_neg2 / 2) % 2) ? -1 : +1;
    const double dlad = lad2 - lad;
    RealD u; random(sRNG, u);
    const bool accept = (std::log(std::max((double)u, 1e-300)) < 0.5 * dlad);
    if (accept) {
      if (use_sigma) U.sigma = W.sigma;
      else           U.s = W.s;
    }
    std::cout << GridLogMessage << "[SIGNHOP] traj=" << traj
              << " hop=" << mname
              << " eta=" << std::showpos << sgn * sgn2 << std::noshowpos
              << " dlad=" << dlad
              << " accept=" << (accept ? "YES" : "no")
              << (accept ? (((n_neg2 / 2) % 2) ? " signPf_now=-1" : " signPf_now=+1") : "")
              << std::endl;
  }

 private:
  static void Gamma5Doubled(DTXQCDFermionDoubled &x) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.upper.f[a];
      x.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.lower.f[a];
    }
  }

  void SiteCoor(int lex, Coordinate &c) const {
    const Coordinate ld = Grid.GlobalDimensions();
    c.resize(4);
    c[0] = lex % ld[0]; lex /= ld[0];
    c[1] = lex % ld[1]; lex /= ld[1];
    c[2] = lex % ld[2]; lex /= ld[2];
    c[3] = lex;
  }
  int RowIdx(int site, int blk, int a, int sp, int co) const {
    return ((site * 2 + blk) * DtxqcdNf + a) * 12 + sp * 3 + co;
  }

  // Dense gamma5.M48 spectrum of X, operator on the (optionally stout-
  // smeared) gauge.  Fills n_neg / logabsdet / min|eval|.
  void Spectrum(DTXQCDField &X, int &n_neg, double &logabsdet, double &min_abs) {
    typedef typename LatticeFermion::scalar_object FSobj;
    LatticeGaugeField Usm(X.U);
    if (nstout > 0) {
      Smear_Stout<PeriodicGimplR> stout(rho);
      LatticeGaugeField tmp(&Grid);
      for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
    }
    DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw,
                                   X.sigma, X.pi, X.d, X.n, X.s, X.p);
    DTXQCDMOp Mop(Dw);
    DTXQCDFermionDoubled e(&Grid), y(&Grid);
    std::vector<FSobj> col_arr(V);
    for (int site = 0; site < V; ++site) {
      Coordinate c; SiteCoor(site, c);
      for (int blk = 0; blk < 2; ++blk)
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int sp = 0; sp < Ns; ++sp)
            for (int co = 0; co < Nc; ++co) {
              const int j = RowIdx(site, blk, a, sp, co);
              for (int aa = 0; aa < DtxqcdNf; ++aa) {
                e.upper.f[aa] = Zero();
                e.lower.f[aa] = Zero();
              }
              FSobj one; one = Zero();
              one()(sp)(co) = ComplexD(1.0, 0.0);
              if (blk == 0) pokeSite(one, e.upper.f[a], c);
              else          pokeSite(one, e.lower.f[a], c);
              Mop.M(e, y);
              Gamma5Doubled(y);
              for (int blk2 = 0; blk2 < 2; ++blk2)
                for (int a2 = 0; a2 < DtxqcdNf; ++a2) {
                  const LatticeFermion &f =
                      (blk2 == 0) ? y.upper.f[a2] : y.lower.f[a2];
                  unvectorizeToLexOrdArray(col_arr, f);
                  for (int s2 = 0; s2 < V; ++s2)
                    for (int sp2 = 0; sp2 < Ns; ++sp2)
                      for (int co2 = 0; co2 < Nc; ++co2) {
                        const int i = RowIdx(s2, blk2, a2, sp2, co2);
                        auto z = col_arr[s2]()(sp2)(co2);
                        H[(size_t)j * dim + i] =
                            std::complex<double>(z.real(), z.imag());
                      }
                }
            }
    }
#ifdef GRID_CUDA
    cuDoubleComplex *dA; double *dW; int *dInfo;
    GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)dim * dim) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dW, sizeof(double) * dim) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(dA, H.data(), sizeof(cuDoubleComplex) * (size_t)dim * dim,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    int lwork = 0;
    GRID_ASSERT(cusolverDnZheevd_bufferSize(cus, CUSOLVER_EIG_MODE_NOVECTOR,
                CUBLAS_FILL_MODE_LOWER, dim, dA, dim, dW, &lwork)
                == CUSOLVER_STATUS_SUCCESS);
    cuDoubleComplex *dWork;
    GRID_ASSERT(cudaMalloc(&dWork, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
    GRID_ASSERT(cusolverDnZheevd(cus, CUSOLVER_EIG_MODE_NOVECTOR,
                CUBLAS_FILL_MODE_LOWER, dim, dA, dim, dW, dWork, lwork, dInfo)
                == CUSOLVER_STATUS_SUCCESS);
    int info = -1;
    GRID_ASSERT(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    GRID_ASSERT(info == 0);
    GRID_ASSERT(cudaMemcpy(evals.data(), dW, sizeof(double) * dim,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(dA); cudaFree(dW); cudaFree(dInfo); cudaFree(dWork);
#else
    // CPU fallback (2026-08-14, for LOCAL 4^4 signhop scans on non-CUDA
    // builds): Eigen SelfAdjointEigenSolver on the dense Hermitian gamma5.M48
    // (column-major H, dim = 48*V; 4^4 -> 12288 => ~1-3 min per solve).
    {
      Eigen::Map<const Eigen::MatrixXcd> Hm(H.data(), dim, dim);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hm, Eigen::EigenvaluesOnly);
      GRID_ASSERT(es.info() == Eigen::Success);
      for (int i = 0; i < dim; ++i) evals[i] = es.eigenvalues()[i];
    }
#endif
    n_neg = 0; logabsdet = 0.0; min_abs = 1e300;
    for (int i = 0; i < dim; ++i) {
      if (evals[i] < 0) ++n_neg;
      logabsdet += std::log(std::abs(evals[i]));
      min_abs = std::min(min_abs, std::abs(evals[i]));
    }
  }

  GridCartesian &Grid;
  GridRedBlackCartesian &RBGrid;
  RealD mass, csw, prob;
  int nstout; RealD rho;
  int V, dim;
  LatticeInteger tcoord;
  std::vector<std::complex<double>> H;
  std::vector<double> evals;
#ifdef GRID_CUDA
  cusolverDnHandle_t cus;
#endif
};

NAMESPACE_END(Grid);
