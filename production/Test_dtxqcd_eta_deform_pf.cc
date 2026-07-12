// Test_dtxqcd_eta_deform_pf: does the time-graded rescaling of the aux
// off-diagonal couplings preserve det/Pf of M48?  (mwagman 2026-07-12)
//
// Proposed contour deformation (in the spirit of arXiv:2101.12668):
//   d(x,t) -> e^{-eta t} d(x,t),   n(x,t) -> e^{-eta t} n(x,t)      [UR block]
//   d*(x,t) -> e^{+eta t} d*(x,t), n*(x,t) -> e^{+eta t} n*(x,t)    [LL block]
// with all other fields unchanged.  On the deformed contour d* != conj(d),
// so this CANNOT be expressed by scaling the stored d field with real eta
// (the code builds the lower-left block from conj(d), which would scale the
// SAME way, not inversely).  The faithful implementation is at the operator
// level: since the d,n couplings are SITE-LOCAL, the entire (upper,lower)
// sector of the dense M48 is the UR block and (lower,upper) is the LL block:
//   M[(u,x),(l,y)] *= e^{-eta t(x)},   M[(l,x),(u,y)] *= e^{+eta t(x)}.
// S_aux is trivially invariant ((e^{+eta t} d*)(e^{-eta t} d) = d* d) and the
// contour Jacobian cancels between d and d*; the question is whether
// det(M48) (and hence |Pf|, with the sign fixed by continuity in eta) is
// exactly invariant.  Grassmann bookkeeping: absorbing the scalings into
// psi_l -> e^{+eta t} psi_l, psibar_l -> e^{-eta t} psibar_l (unit Jacobian)
// moves the deformation into the LOWER Wilson operator's time hops
// (e^{-eta t} D_W e^{+eta t} = a real-chemical-potential twist + APBC
// boundary factor), so exact invariance is a nontrivial claim about that
// twist cancelling in the doubled structure -- hence this test.
//
// Method (4^4-class volumes, single rank, needs CUDA/cuSOLVER):
//   1. load a production-conditions 4^4 config (stout+csw+aux),
//   2. assemble dense M48 column-by-column (pattern of dtxqcd_sign_hop.h),
//   3. LU (cusolverDnZgetrf) -> log|det| and arg(det),
//   4. repeat on eta-deformed copies (eta = 0.05, 0.1, 0.3),
//   5. CONTROL: UR-only scaling (no LL compensation) at eta=0.1 -- must
//      change det, validating the test's sensitivity.
// PASS: |d log|det|| < 1e-6 and |d arg| < 1e-6 for every compensated eta.
//
// Env: LATT (4.4.4.4), CFG_DIR, TRAJ (250), MASS (-0.245),
//      CSW (1.24930970916466), STOUT_NSMEAR (1), STOUT_RHO (0.125),
//      PASS_TOL (1e-6).

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#ifdef GRID_CUDA
#include <cusolverDn.h>
#include <cuda_runtime.h>
#endif

using namespace Grid;

static RealD env_real(const char *k, RealD d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atof(v) : d;
}
static int env_int(const char *k, int d) {
  const char *v = std::getenv(k); return (v && *v) ? std::atoi(v) : d;
}

#ifdef GRID_CUDA
// log|det| and arg(det) via LU with partial pivoting on the device.
static void DetLU(cusolverDnHandle_t cus, const std::vector<std::complex<double>> &A,
                  int dim, double &logabs, double &phase) {
  cuDoubleComplex *dA; int *dPiv, *dInfo;
  GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)dim * dim) == cudaSuccess);
  GRID_ASSERT(cudaMalloc(&dPiv, sizeof(int) * dim) == cudaSuccess);
  GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
  GRID_ASSERT(cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex) * (size_t)dim * dim,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  int lwork = 0;
  GRID_ASSERT(cusolverDnZgetrf_bufferSize(cus, dim, dim, dA, dim, &lwork)
              == CUSOLVER_STATUS_SUCCESS);
  cuDoubleComplex *dWork;
  GRID_ASSERT(cudaMalloc(&dWork, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
  GRID_ASSERT(cusolverDnZgetrf(cus, dim, dim, dA, dim, dWork, dPiv, dInfo)
              == CUSOLVER_STATUS_SUCCESS);
  int info = -1;
  GRID_ASSERT(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  GRID_ASSERT(info == 0);   // info>0 would mean an exactly singular U
  std::vector<int> piv(dim);
  GRID_ASSERT(cudaMemcpy(piv.data(), dPiv, sizeof(int) * dim,
                         cudaMemcpyDeviceToHost) == cudaSuccess);
  // Diagonal of U: strided copy of A[i*dim+i].
  std::vector<std::complex<double>> diag(dim);
  for (int i = 0; i < dim; ++i) {
    cuDoubleComplex z;
    GRID_ASSERT(cudaMemcpy(&z, dA + (size_t)i * dim + i, sizeof(z),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    diag[i] = std::complex<double>(z.x, z.y);
  }
  int swaps = 0;
  for (int i = 0; i < dim; ++i) if (piv[i] != i + 1) ++swaps;
  logabs = 0.0; phase = (swaps % 2) ? M_PI : 0.0;
  for (int i = 0; i < dim; ++i) {
    logabs += std::log(std::abs(diag[i]));
    phase += std::arg(diag[i]);
  }
  phase = std::remainder(phase, 2.0 * M_PI);
  cudaFree(dA); cudaFree(dPiv); cudaFree(dInfo); cudaFree(dWork);
}
#endif

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
#ifndef GRID_CUDA
  std::cout << "non-CUDA build: no dense LU -- skipping" << std::endl;
  Grid_finalize();
  return 0;
#else
  std::vector<int> latt_dims{4, 4, 4, 4};
  if (const char *v = std::getenv("LATT"); v && *v) {
    latt_dims.clear();
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, '.')) latt_dims.push_back(std::atoi(tok.c_str()));
  }
  Coordinate latt(latt_dims);
  GridCartesian Grid(latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian RBGrid(&Grid);
  GRID_ASSERT(Grid.gSites() == Grid.lSites());   // single rank
  const int V = (int)Grid.gSites();
  GRID_ASSERT(V <= 4096);
  const int dim = 48 * V;
  const int T = latt[Nd - 1];

  RealD mass = env_real("MASS", -0.245);
  RealD csw  = env_real("CSW", 1.24930970916466);
  int nstout = env_int("STOUT_NSMEAR", 1);
  RealD rho  = env_real("STOUT_RHO", 0.125);
  RealD pass_tol = env_real("PASS_TOL", 1e-6);
  int traj   = env_int("TRAJ", 250);
  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam4.0000_4t4_signhop_lam4";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  std::cout << GridLogMessage << "==== Test_dtxqcd_eta_deform_pf ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " mass=" << mass << " csw=" << csw
            << " stout=" << nstout << "x" << rho << " dim=" << dim << std::endl;

  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid);
  DTXQCDField U(&Grid);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);
  LatticeGaugeField Usm(U.U);
  if (nstout > 0) {
    Smear_Stout<PeriodicGimplR> stout(rho);
    LatticeGaugeField tmp(&Grid);
    for (int i = 0; i < nstout; ++i) { stout.smear(tmp, Usm); Usm = tmp; }
  }
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid, RBGrid, mass, csw,
                                 U.sigma, U.pi, U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);

  // ---- dense assembly of M48 (no gamma5), column by column ----
  auto SiteCoor = [&](int lex, Coordinate &c) {
    const Coordinate ld = Grid.GlobalDimensions();
    c.resize(4);
    c[0] = lex % ld[0]; lex /= ld[0];
    c[1] = lex % ld[1]; lex /= ld[1];
    c[2] = lex % ld[2]; lex /= ld[2];
    c[3] = lex;
  };
  auto RowIdx = [&](int site, int blk, int a, int sp, int co) {
    return ((site * 2 + blk) * DtxqcdNf + a) * 12 + sp * 3 + co;
  };
  typedef typename LatticeFermion::scalar_object FSobj;
  std::vector<std::complex<double>> H((size_t)dim * dim);
  {
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
  }
  std::cout << GridLogMessage << "dense M48 assembled" << std::endl;

  // Sanity: the (u,l)/(l,u) sectors must be site-local (all inter-site
  // entries zero) -- the premise of the sector-scaling implementation.
  {
    double offsite = 0.0, onsite = 0.0;
    for (int sj = 0; sj < V; ++sj)
      for (int bj = 0; bj < 2; ++bj)
        for (int k = 0; k < 12 * DtxqcdNf; ++k) {
          const int j = (sj * 2 + bj) * 12 * DtxqcdNf + k;
          for (int si = 0; si < V; ++si)
            for (int bi = 0; bi < 2; ++bi)
              if (bi != bj)
                for (int m = 0; m < 12 * DtxqcdNf; ++m) {
                  const int i = (si * 2 + bi) * 12 * DtxqcdNf + m;
                  double a = std::abs(H[(size_t)j * dim + i]);
                  if (si == sj) onsite += a; else offsite += a;
                }
        }
    std::cout << GridLogMessage << "off-diag sector: onsite weight=" << onsite
              << "  intersite weight=" << offsite
              << "  (intersite MUST be 0)" << std::endl;
    GRID_ASSERT(offsite < 1e-10 * (onsite + 1e-30));
  }

  cusolverDnHandle_t cus;
  GRID_ASSERT(cusolverDnCreate(&cus) == CUSOLVER_STATUS_SUCCESS);
  double lad0, ph0;
  DetLU(cus, H, dim, lad0, ph0);
  std::cout << GridLogMessage << "eta=0: log|det| = " << std::setprecision(15)
            << lad0 << "   arg(det) = " << ph0 << std::endl;

  // eta-deformed copy: rows (site,blk) scaled against column block.
  // uniform=true drops the t-grading (constant e^{-eta} / e^{+eta}): the
  // absorbing psi_l redefinition then commutes with the Wilson hops too, so
  // uniform MUST be exactly invariant -- the discriminator between "the
  // compensation itself is wrong" and "the t-grading twists the time hops
  // (chemical-potential mechanism)".
  // PROFILE env: "linear" (default; e^{-eta t} -- NOTE: carries a large
  // hidden gradient at the t = T-1 -> 0 boundary wrap, i.e. a net
  // charge-winding insertion) or "tent" (mwagman: eta(t) = eta*min(t, T-t),
  // periodic-smooth, reflection-symmetric, zero net winding -- the
  // boundary-correct, non-charge-inserting profile).
  const bool tent = [&] {
    const char *v = std::getenv("PROFILE");
    return v && std::string(v) == "tent";
  }();
  auto prof = [&](int tc) {
    return tent ? (double)std::min(tc, T - tc) : (double)tc;
  };
  auto Deform = [&](double eta, bool compensate_LL, bool uniform) {
    std::vector<std::complex<double>> A(H);
    for (int sj = 0; sj < V; ++sj)
      for (int bj = 0; bj < 2; ++bj)
        for (int k = 0; k < 12 * DtxqcdNf; ++k) {
          const int j = (sj * 2 + bj) * 12 * DtxqcdNf + k;
          for (int si = 0; si < V; ++si) {
            Coordinate c; SiteCoor(si, c);
            const double t = uniform ? 1.0 : prof(c[3]);
            for (int bi = 0; bi < 2; ++bi) {
              double f = 1.0;
              if (bi == 0 && bj == 1) f = std::exp(-eta * t);          // UR
              if (bi == 1 && bj == 0 && compensate_LL)
                f = std::exp(+eta * t);                                 // LL
              if (f == 1.0) continue;
              for (int m = 0; m < 12 * DtxqcdNf; ++m) {
                const int i = (si * 2 + bi) * 12 * DtxqcdNf + m;
                A[(size_t)j * dim + i] *= f;
              }
            }
          }
        }
    return A;
  };

  // ETA_SCAN="0.1,0.2,..." : replace the default t-graded eta list and skip
  // the uniform/site/control arms — the reweighting-range production scan
  // (per-config d log|det| / d arg vs eta; breakdown is set by the CONFIG-TO-
  // CONFIG SPREAD of these, measured by running over an ensemble of TRAJs).
  std::vector<double> scan;
  bool scan_mode = false;
  if (const char *v = std::getenv("ETA_SCAN"); v && *v) {
    scan_mode = true;
    std::stringstream ss(v); std::string tok;
    while (std::getline(ss, tok, ',')) scan.push_back(std::atof(tok.c_str()));
  }
  if (scan_mode) {
    for (double eta : scan) {
      auto A = Deform(eta, /*compensate_LL=*/true, /*uniform=*/false);
      double lad, ph;
      DetLU(cus, A, dim, lad, ph);
      std::cout << GridLogMessage << "SCAN" << (tent ? "-TENT" : "")
                << " traj=" << traj << " eta=" << eta
                << " dlogdet= " << (lad - lad0)
                << " darg= " << std::remainder(ph - ph0, 2.0 * M_PI)
                << std::endl;
    }
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  // CHARGE_CHECK=1: verify the GL(1) charge of the cross-block propagator
  // under the UNIFORM deformation: (M^-1)_lu -> e^{+eta} (M^-1)_lu and
  // (M^-1)_ul -> e^{-eta} (M^-1)_ul (one net LL / UR insertion).  This is
  // the obstruction to the "1-d cross-correlator dial": the anomalous
  // <qbar qbar> source loop carries charge +1, exactly compensating the
  // single d's -1, so the estimator is neutral and the uniform deformation
  // cannot rescale it.
  if (env_int("CHARGE_CHECK", 0) != 0) {
    const double eta = 0.3;
    auto Solve = [&](const std::vector<std::complex<double>> &A,
                     std::vector<std::complex<double>> &x) {
      cuDoubleComplex *dA, *dB; int *dPiv, *dInfo;
      GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)dim * dim) == cudaSuccess);
      GRID_ASSERT(cudaMalloc(&dB, sizeof(cuDoubleComplex) * dim) == cudaSuccess);
      GRID_ASSERT(cudaMalloc(&dPiv, sizeof(int) * dim) == cudaSuccess);
      GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
      GRID_ASSERT(cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex) * (size_t)dim * dim,
                             cudaMemcpyHostToDevice) == cudaSuccess);
      GRID_ASSERT(cudaMemcpy(dB, x.data(), sizeof(cuDoubleComplex) * dim,
                             cudaMemcpyHostToDevice) == cudaSuccess);
      int lwork = 0;
      cusolverDnZgetrf_bufferSize(cus, dim, dim, dA, dim, &lwork);
      cuDoubleComplex *dW;
      GRID_ASSERT(cudaMalloc(&dW, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
      cusolverDnZgetrf(cus, dim, dim, dA, dim, dW, dPiv, dInfo);
      cusolverDnZgetrs(cus, CUBLAS_OP_N, dim, 1, dA, dim, dPiv, dB, dim, dInfo);
      GRID_ASSERT(cudaMemcpy(x.data(), dB, sizeof(cuDoubleComplex) * dim,
                             cudaMemcpyDeviceToHost) == cudaSuccess);
      cudaFree(dA); cudaFree(dB); cudaFree(dPiv); cudaFree(dInfo); cudaFree(dW);
    };
    // Unit source in the UPPER block at site 85, dof 0.
    std::vector<std::complex<double>> e0(dim, 0.0), x0(dim), x1(dim);
    e0[(85 * 2 + 0) * 24 + 0] = 1.0;
    x0 = e0;
    Solve(H, x0);                                        // original M^-1 e
    auto A = Deform(0.0, true, true);                    // copy (eta=0)
    // uniform deformation at eta:
    A = H;
    for (int sj = 0; sj < V; ++sj)
      for (int bj = 0; bj < 2; ++bj)
        for (int k = 0; k < 24; ++k) {
          const int j = (sj * 2 + bj) * 24 + k;
          for (int si = 0; si < V; ++si)
            for (int bi = 0; bi < 2; ++bi) {
              double f = 1.0;
              if (bi == 0 && bj == 1) f = std::exp(-eta);
              if (bi == 1 && bj == 0) f = std::exp(+eta);
              if (f == 1.0) continue;
              for (int m = 0; m < 24; ++m)
                A[(size_t)j * dim + (si * 2 + bi) * 24 + m] *= f;
            }
        }
    x1 = e0;
    Solve(A, x1);                                        // deformed M^-1 e
    // Compare: upper components (uu block) should be UNCHANGED; lower
    // components (lu block) should be scaled by e^{+eta}.
    double uu_dev = 0.0, lu_dev = 0.0, uu_scale = 0.0, lu_scale = 0.0;
    const double ep = std::exp(+eta);
    for (int s = 0; s < V; ++s)
      for (int k = 0; k < 24; ++k) {
        const int iu = (s * 2 + 0) * 24 + k;
        const int il = (s * 2 + 1) * 24 + k;
        uu_dev = std::max(uu_dev, std::abs(x1[iu] - x0[iu]));
        uu_scale = std::max(uu_scale, std::abs(x0[iu]));
        lu_dev = std::max(lu_dev, std::abs(x1[il] - ep * x0[il]));
        lu_scale = std::max(lu_scale, std::abs(x0[il]));
      }
    std::cout << GridLogMessage << "CHARGE_CHECK eta=" << eta
              << ": max|S_uu(def)-S_uu| = " << uu_dev << " (scale " << uu_scale
              << ")   max|S_lu(def)-e^{+eta}S_lu| = " << lu_dev
              << " (scale " << lu_scale << ")" << std::endl;
    std::cout << GridLogMessage << "  => S_uu neutral, S_lu charge +1: "
              << ((uu_dev < 1e-8 * uu_scale && lu_dev < 1e-8 * lu_scale)
                      ? "CONFIRMED" : "VIOLATED") << std::endl;
    Grid_finalize();
    std::fflush(nullptr);
    std::_Exit(0);
  }

  bool ok = true;
  for (bool uniform : {false, true}) {
    for (double eta : {0.05, 0.1, 0.3}) {
      auto A = Deform(eta, /*compensate_LL=*/true, uniform);
      double lad, ph;
      DetLU(cus, A, dim, lad, ph);
      const double dl = lad - lad0;
      const double dp = std::remainder(ph - ph0, 2.0 * M_PI);
      std::cout << GridLogMessage << (uniform ? "UNIFORM eta=" : "eta=") << eta
                << ": d log|det| = " << dl << "   d arg(det) = " << dp
                << ((std::abs(dl) < pass_tol && std::abs(dp) < pass_tol)
                        ? "   INVARIANT" : "   CHANGES") << std::endl;
      // Gate on the UNIFORM arm (exactness claim); the t-graded arm is the
      // measured physics result (expected: eta^2 magnitude + linear phase).
      if (uniform && (std::abs(dl) > pass_tol || std::abs(dp) > pass_tol))
        ok = false;
    }
  }
  // ---- single-site arms (mwagman 2026-07-12): rescale row/col of ONE
  // site's upper-block dofs by e^{+eta}/e^{-eta}.
  //   SIM: scale the FULL row and column (all four sectors) = the true
  //        similarity ((UAU^-1, UB),(CU^-1, D)) => det EXACTLY invariant
  //        (numerical validation of the Schur derivation).
  //   AUX: scale ONLY the aux sectors (UR row-entries, LL col-entries),
  //        leaving the (u,u) Wilson block untouched = the single-site field
  //        deformation.  Differs from SIM by the un-twisting of A's hops at
  //        this site => det CHANGES; its size = the per-site deformation
  //        cost.  This is exactly why the t-graded global deformation is
  //        not exact while the uniform one is (U commutes with A only when
  //        it is constant).
  auto SiteScale = [&](int site0, double eta, bool aux_only) {
    std::vector<std::complex<double>> A(H);
    const double ep = std::exp(+eta), em = std::exp(-eta);
    for (int k = 0; k < 24; ++k) {          // the 24 upper dofs of site0
      const int i0 = (site0 * 2 + 0) * 24 + k;
      for (int j = 0; j < dim; ++j) {       // row i0 *= e^{+eta}
        if (aux_only && ((j / 24) % 2) == 0) continue;   // UR only
        A[(size_t)j * dim + i0] *= ep;
      }
      for (int r = 0; r < dim; ++r) {       // col i0 *= e^{-eta}
        if (aux_only && ((r / 24) % 2) == 0) continue;   // LL only
        A[(size_t)i0 * dim + r] *= em;
      }
    }
    return A;
  };
  for (int site0 : {0, 85}) {   // t=0 corner + a bulk site
    for (bool aux_only : {false, true}) {
      auto A = SiteScale(site0, 0.3, aux_only);
      double lad, ph;
      DetLU(cus, A, dim, lad, ph);
      const double dl = lad - lad0;
      const double dp = std::remainder(ph - ph0, 2.0 * M_PI);
      std::cout << GridLogMessage << (aux_only ? "SITE-AUX" : "SITE-SIM")
                << " site=" << site0 << " eta=0.3: d log|det| = " << dl
                << "   d arg(det) = " << dp
                << ((std::abs(dl) < pass_tol && std::abs(dp) < pass_tol)
                        ? "   INVARIANT" : "   CHANGES") << std::endl;
      // Gate: the similarity arm must be exact.
      if (!aux_only && (std::abs(dl) > pass_tol || std::abs(dp) > pass_tol))
        ok = false;
    }
  }

  // Control: UR-only (no LL compensation) MUST change det.
  {
    auto A = Deform(0.1, /*compensate_LL=*/false, /*uniform=*/false);
    double lad, ph;
    DetLU(cus, A, dim, lad, ph);
    const double dl = lad - lad0;
    std::cout << GridLogMessage << "CONTROL (UR-only, eta=0.1): d log|det| = "
              << dl << "   (must be nonzero)" << std::endl;
    if (std::abs(dl) < 1e-3) {
      std::cout << GridLogError << "control arm shows no sensitivity -- "
                << "test inconclusive" << std::endl;
      ok = false;
    }
  }

  std::cout << GridLogMessage << "==== " << (ok ? "PASS" : "FAIL")
            << " (pass_tol=" << pass_tol << ") ====" << std::endl;
  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(ok ? 0 : 1);
#endif
}
