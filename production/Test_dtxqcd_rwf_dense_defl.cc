// Test_dtxqcd_rwf_dense_defl -- the 4^4 DENSE design study for the DEFLATED
// reweighting trace estimator (HANDOFF_QUDA_DEFORM_HOPWEIGHTS.md section 5c).
//
// The reweighting blocker is Hutchinson VARIANCE: per-config DlogPf noise at
// 24 solves is sigma ~ 20-40 (lambda=5, eta=0.2-0.4, cos) and reweighting
// needs O(1).  Proposed estimator: singular-triplet split at K smallest
// sigma_k of Stil (the odd Schur complement of the chempot-frame Mtil):
//
//   Tr[Stil^-1 dStil] = sum_{k<=K} sigma_k^-1 <u_k| dStil |v_k>    (EXACT)
//                     + E_w <w, Stil^-1 Pperp_u dStil w>           (Hutchinson)
//
// This program decides K BEFORE any production wiring, with zero sampling
// noise: at 4^4 the odd-block operator is N = 48*V/2 = 6144 dense, so we
// assemble Stil and dStil columnwise from the CERTIFIED host operator
// lambdas, LU/SVD them with cuSOLVER, and compute
//   - the EXACT trace (gated against a finite difference of log det Stil
//     at eta +- delta -- validates Stil AND dStil independently),
//   - the exact per-PAIR Hutchinson variance of the deflated remainder for
//     a ladder of K:  Var = ||A'_K||_F^2  with  A_K = Stil^-1 Pperp dStil
//     and A' the grading-antithetic projection (the v -> Gv pair kills
//     grading-odd entries exactly: A'_ij = A_ij iff gamma_i gamma_j = +1),
//   - the exact-term partial sums (their growth vs K is the bias budget
//     for finite-tolerance triplets in production).
//
// Physics note (why this can pay where solve-deflation returned zero): the
// eta=0.8 "deflation is structurally useless" result was about CGNR
// ITERATION COUNTS (mid-band dominated).  The TRACE variance is weighted by
// sigma_k^-2, a different functional of the same spectrum -- the open risk
// is the dense mid-band flattening the curve once K exhausts the resolvable
// outliers.  That is exactly what this program measures.
//
// Env: LATT (4.4.4.4 -- keep it small, dense N^2 = (48V/2)^2), CFG_DIR
// (default the lambda=5 4^4 gate stream), TRAJ (300), ETA (0.4 -- NOTE this
// is the NODE value ep = eta*x_q at which the integrand is evaluated, not
// the endpoint), PROFILE/PROF_T0/PROFILE_W[_SRC] (estimator dispatch, cos
// default here), FD_DELTA (1e-4), KMAX (512), MC_CHECK (32 pairs, 0=off).
//
// Output: [dense-defl] lines -- spectrum head, FD gate, K-ladder table.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/utils/WilsonLoops.h>

#ifdef GRID_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

#include <random>

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

typedef std::complex<double> CD;

#ifdef GRID_CUDA
// log|det| + arg(det) via device LU (pattern: Test_dtxqcd_eta_deform_pf.cc).
static void DetLU(cusolverDnHandle_t cus, const std::vector<CD> &A, int dim,
                  double &logabs, double &phase) {
  cuDoubleComplex *dA;
  int *dPiv, *dInfo;
  GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)dim * dim) == cudaSuccess);
  GRID_ASSERT(cudaMalloc(&dPiv, sizeof(int) * dim) == cudaSuccess);
  GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
  GRID_ASSERT(cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex) * (size_t)dim * dim,
                         cudaMemcpyHostToDevice) == cudaSuccess);
  int lwork = 0;
  GRID_ASSERT(cusolverDnZgetrf_bufferSize(cus, dim, dim, dA, dim, &lwork) ==
              CUSOLVER_STATUS_SUCCESS);
  cuDoubleComplex *dWork;
  GRID_ASSERT(cudaMalloc(&dWork, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
  GRID_ASSERT(cusolverDnZgetrf(cus, dim, dim, dA, dim, dWork, dPiv, dInfo) ==
              CUSOLVER_STATUS_SUCCESS);
  int info = -1;
  GRID_ASSERT(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  GRID_ASSERT(info == 0);
  std::vector<int> piv(dim);
  GRID_ASSERT(cudaMemcpy(piv.data(), dPiv, sizeof(int) * dim,
                         cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<cuDoubleComplex> diag(dim);
  GRID_ASSERT(cudaMemcpy2D(diag.data(), sizeof(cuDoubleComplex), dA,
                           (size_t)(dim + 1) * sizeof(cuDoubleComplex),
                           sizeof(cuDoubleComplex), dim,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
  int swaps = 0;
  for (int i = 0; i < dim; ++i)
    if (piv[i] != i + 1) ++swaps;
  logabs = 0;
  phase = 0;
  for (int i = 0; i < dim; ++i) {
    const CD z(diag[i].x, diag[i].y);
    logabs += std::log(std::abs(z));
    phase += std::arg(z);
  }
  if (swaps & 1) phase += M_PI;
  phase = std::remainder(phase, 2.0 * M_PI);
  cudaFree(dA); cudaFree(dPiv); cudaFree(dInfo); cudaFree(dWork);
}
#endif

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
#ifndef GRID_CUDA
  std::cout << GridLogMessage
            << "[dense-defl] requires GRID_CUDA (cuSOLVER LU/SVD)" << std::endl;
  Grid_finalize();
  return 0;
#else
  std::vector<int> latt_dims = {4, 4, 4, 4};
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
  RealD eta = env_real("ETA", 0.4);
  int traj = env_int("TRAJ", 300);
  const double fd_delta = env_real("FD_DELTA", 1e-4);
  const int kmax = env_int("KMAX", 512);
  const int mc_check = env_int("MC_CHECK", 32);

  const std::string prof_kind = env_str("PROFILE", "cos");
  const bool tent = prof_kind != "linear";
  const bool prof_rounded =
      prof_kind == "tent_rounded" || prof_kind == "tent_rounded2";
  const bool prof_round_src = prof_kind == "tent_rounded2";
  const bool prof_cos = prof_kind == "cos";
  const double prof_w = env_real("PROFILE_W", 4.0);
  const double prof_w_src = env_real("PROFILE_W_SRC", prof_w);
  const int prof_t0 = env_int("PROF_T0", 0);

  std::string cfg_dir = "/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/"
                        "dtxqcd_lam5.0000_4t4_signhop_lam5";
  if (const char *d = std::getenv("CFG_DIR"); d && *d) cfg_dir = d;

  const int V = (int)Grid_.gSites();
  GRID_ASSERT(Grid_.gSites() == Grid_.lSites() && V <= 4096 &&
              "dense study is single-rank, small-volume only");
  const int Nodd = V / 2;
  const int N = 48 * Nodd;

  std::cout << GridLogMessage << "==== Test_dtxqcd_rwf_dense_defl ====" << std::endl;
  std::cout << GridLogMessage << " cfg=" << cfg_dir << " traj=" << traj
            << " eta(node)=" << eta << " profile=" << prof_kind
            << " t0=" << prof_t0 << " N=" << N << std::endl;

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
  DTXQCDWilsonCloverFermionEO Dw(Usm, Grid_, RBGrid, mass, csw, U.sigma, U.pi,
                                 U.d, U.n, U.s, U.p);
  DTXQCDMOp Mop(Dw);

  // ---- profile + weights + chempot-frame Mtil + hop insertion: copied
  // verbatim from Test_dtxqcd_eta_stoch_pf (the certified host reference).
  LatticeInteger tc(&Grid_);
  LatticeCoordinate(tc, Nd - 1);
  auto prof = [&](int t) {
    const int dt = ((t - prof_t0) % T + T) % T;
    if (!tent) return (double)dt;
    const double d = (double)dt;
    if (prof_cos)
      return 0.25 * (double)T * (1.0 - std::cos(2.0 * M_PI * d / (double)T));
    double v = std::min(d, (double)T - d);
    if (!prof_rounded) return v;
    const double a = std::abs(d - 0.5 * T);
    if (a < prof_w) v = 0.5 * T - 0.5 * prof_w - a * a / (2.0 * prof_w);
    if (prof_round_src && prof_w_src > 0) {
      const double b = std::min(d, (double)T - d);
      if (b < prof_w_src) v = 0.5 * prof_w_src + b * b / (2.0 * prof_w_src);
    }
    return v;
  };
  auto WeightField = [&](const auto &f, auto &out, auto fac) {
    out = f;
    for (int t = 0; t < T; ++t) {
      auto scaled = f;
      scaled = ComplexD(fac(t), 0.0) * scaled;
      out = where(tc == Integer(t), scaled, out);
    }
  };
  LatticeDtxqcdD d_wm(&Grid_), d_wp(&Grid_);
  LatticeDtxqcdN n_wm(&Grid_), n_wp(&Grid_);
  auto BuildWeights = [&](double e) {
    WeightField(U.d, d_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.n, n_wm, [&](int t) { return std::exp(-e * prof(t)) - 1.0; });
    WeightField(U.d, d_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
    WeightField(U.n, n_wp, [&](int t) { return std::exp(+e * prof(t)) - 1.0; });
  };
  typedef DTXQCDFermionDoubled F;
  auto AddCross = [&](const LatticeDtxqcdD &dd, const LatticeDtxqcdN &nn,
                      const F &in, F &out, bool lower_to_upper) {
    DTXQCDFermionNf c(&Grid_);
    if (lower_to_upper) {
      DtxqcdApplyDnCross(dd, nn, in.lower, c, /*apply_conj=*/false);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.upper.f[a] = out.upper.f[a] + c.f[a];
    } else {
      DtxqcdApplyDnCross(dd, nn, in.upper, c, /*apply_conj=*/true);
      for (int a = 0; a < DtxqcdNf; ++a)
        out.lower.f[a] = out.lower.f[a] + c.f[a];
    }
  };
  auto Mdef = [&](const F &in, F &out) {
    Mop.M(in, out);
    AddCross(d_wm, n_wm, in, out, true);
    AddCross(d_wp, n_wp, in, out, false);
  };
  LatticeComplex wGu(&Grid_), wGl(&Grid_), wGui(&Grid_), wGli(&Grid_);
  auto BuildGWeights = [&](double e) {
    wGu = ComplexD(1.0, 0.0);
    wGl = ComplexD(1.0, 0.0);
    LatticeComplex one(&Grid_);
    for (int t = 0; t < T; ++t) {
      one = ComplexD(std::exp(-e * prof(t) / 2.0), 0.0);
      wGu = where(tc == Integer(t), one, wGu);
      one = ComplexD(std::exp(+e * prof(t) / 2.0), 0.0);
      wGl = where(tc == Integer(t), one, wGl);
    }
    wGui = ComplexD(1.0, 0.0);
    wGli = ComplexD(1.0, 0.0);
    wGui = wGui / wGu;
    wGli = wGli / wGl;
  };
  auto GApply = [&](F &v, int sign) {
    const LatticeComplex &fu = (sign > 0) ? wGu : wGui;
    const LatticeComplex &fl = (sign > 0) ? wGl : wGli;
    for (int a = 0; a < DtxqcdNf; ++a) {
      v.upper.f[a] = fu * v.upper.f[a];
      v.lower.f[a] = fl * v.lower.f[a];
    }
  };
  auto Mtil = [&](const F &in, F &out) {
    F w(&Grid_);
    w = in;
    GApply(w, +1);
    Mdef(w, out);
    GApply(out, -1);
  };
  // hop-form dMtil/deta (block-diagonal in doubling, cb-CHANGING: pure t-hop)
  Gamma g_t(Gamma::Algebra::GammaT);
  LatticeColourMatrix W3(&Grid_), W7(&Grid_), W3c(&Grid_), W7c(&Grid_);
  {
    LatticeColourMatrix Ut(&Grid_);
    Ut = PeekIndex<LorentzIndex>(Usm, Nd - 1);
    LatticeColourMatrix mUt(&Grid_);
    mUt = ComplexD(-1.0, 0.0) * Ut;
    W3 = where(tc == Integer(T - 1), mUt, Ut);
    W7 = Cshift(LatticeColourMatrix(adj(W3)), Nd - 1, -1);
    W3c = conjugate(W3);
    W7c = Cshift(LatticeColourMatrix(adj(W3c)), Nd - 1, -1);
  }
  LatticeComplex af_u(&Grid_), ab_u(&Grid_), af_l(&Grid_), ab_l(&Grid_);
  auto BuildHopWeights = [&](double e) {
    auto dp = [&](int t) { return prof((t + 1) % T) - prof(t); };
    LatticeComplex one(&Grid_);
    for (int t = 0; t < T; ++t) {
      const double dpf = dp(t);
      const double dpb = dp((t - 1 + T) % T);
      one = ComplexD(-0.5 * dpf * std::exp(-e * dpf / 2.0), 0.0);
      af_u = where(tc == Integer(t), one, af_u);
      one = ComplexD(+0.5 * dpb * std::exp(+e * dpb / 2.0), 0.0);
      ab_u = where(tc == Integer(t), one, ab_u);
      one = ComplexD(+0.5 * dpf * std::exp(+e * dpf / 2.0), 0.0);
      af_l = where(tc == Integer(t), one, af_l);
      one = ComplexD(-0.5 * dpb * std::exp(-e * dpb / 2.0), 0.0);
      ab_l = where(tc == Integer(t), one, ab_l);
    }
  };
  auto dMhopBlock = [&](const LatticeFermion &psi, LatticeFermion &out,
                        const LatticeColourMatrix &Wf,
                        const LatticeColourMatrix &Wb,
                        const LatticeComplex &af, const LatticeComplex &ab) {
    LatticeFermion pp = Cshift(psi, Nd - 1, +1);
    LatticeFermion tf = Wf * pp;
    tf = ComplexD(-0.5, 0.0) * (tf - g_t * tf);
    LatticeFermion pm = Cshift(psi, Nd - 1, -1);
    LatticeFermion tb = Wb * pm;
    tb = ComplexD(-0.5, 0.0) * (tb + g_t * tb);
    out = af * tf + ab * tb;
  };
  auto dMhop = [&](const F &in, F &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      dMhopBlock(in.upper.f[a], out.upper.f[a], W3, W7, af_u, ab_u);
      dMhopBlock(in.lower.f[a], out.lower.f[a], W3c, W7c, af_l, ab_l);
    }
  };

  // ---- odd Schur complement Stil = Moo - Mtil_oe Mee^-1 Mtil_eo and its
  // eta-derivative dStil = -(dM_oe Mee^-1 M_eo + M_oe Mee^-1 dM_eo), built
  // from full-grid applies + cb masks (Mee = Mtil_ee exactly, eta-indep).
  auto MaskCb = [&](F &v, int keep_odd) {
    // keep only the requested checkerboard (pick + zero + restore).
    const int cb = keep_odd ? Odd : Even;
    LatticeFermion h(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(cb, h, v.upper.f[a]);
      v.upper.f[a] = Zero();
      setCheckerboard(v.upper.f[a], h);
      pickCheckerboard(cb, h, v.lower.f[a]);
      v.lower.f[a] = Zero();
      setCheckerboard(v.lower.f[a], h);
    }
  };
  auto MeeInvEven = [&](const F &full_in, F &full_out) {
    // full_out = ZeroExtend( Mee^-1 pick_e(full_in) )
    F he(&RBGrid), hi(&RBGrid);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(Even, he.upper.f[a], full_in.upper.f[a]);
      pickCheckerboard(Even, he.lower.f[a], full_in.lower.f[a]);
    }
    Dw.MooeeInv(he, hi);
    for (int a = 0; a < DtxqcdNf; ++a) {
      full_out.upper.f[a] = Zero();
      full_out.lower.f[a] = Zero();
      setCheckerboard(full_out.upper.f[a], hi.upper.f[a]);
      setCheckerboard(full_out.lower.f[a], hi.lower.f[a]);
    }
  };
  auto Stil = [&](const F &w, F &out) {  // w odd-supported
    F t1(&Grid_), t2(&Grid_), t3(&Grid_);
    Mtil(w, t1);          // odd part = Moo w ; even part = M_eo w
    t2 = t1;
    MaskCb(t2, 1);        // Moo w
    MeeInvEven(t1, t3);   // ext(Mee^-1 M_eo w)
    Mtil(t3, t1);
    MaskCb(t1, 1);        // M_oe Mee^-1 M_eo w
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = t2.upper.f[a] - t1.upper.f[a];
      out.lower.f[a] = t2.lower.f[a] - t1.lower.f[a];
    }
  };
  auto dStil = [&](const F &w, F &out) {  // w odd-supported
    F t1(&Grid_), t2(&Grid_), t3(&Grid_);
    // term 1: dM_oe Mee^-1 M_eo w
    Mtil(w, t1);
    MeeInvEven(t1, t2);
    dMhop(t2, t1);
    MaskCb(t1, 1);
    // term 2: M_oe Mee^-1 dM_eo w
    dMhop(w, t2);
    MeeInvEven(t2, t3);
    Mtil(t3, t2);
    MaskCb(t2, 1);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = ComplexD(-1.0, 0.0) * (t1.upper.f[a] + t2.upper.f[a]);
      out.lower.f[a] = ComplexD(-1.0, 0.0) * (t1.lower.f[a] + t2.lower.f[a]);
    }
  };

  // ---- dense assembly ---------------------------------------------------
  // Column index = odd_site_index*48 + comp, comp = (blk*2+a)*12 + s*3 + c
  // with blk 0=upper 1=lower.  Odd sites enumerated lexicographically.
  std::vector<Coordinate> osites;
  {
    Coordinate c(Nd);
    for (c[3] = 0; c[3] < latt[3]; ++c[3])
      for (c[2] = 0; c[2] < latt[2]; ++c[2])
        for (c[1] = 0; c[1] < latt[1]; ++c[1])
          for (c[0] = 0; c[0] < latt[0]; ++c[0])
            if ((c[0] + c[1] + c[2] + c[3]) % 2 == 1) osites.push_back(c);
  }
  GRID_ASSERT((int)osites.size() == Nodd);

  auto SetUnit = [&](F &v, int col) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      v.upper.f[a] = Zero();
      v.lower.f[a] = Zero();
    }
    const int site = col / 48, comp = col % 48;
    const int blk = comp / 24, a = (comp / 12) % 2, s = (comp % 12) / 3,
              cc = comp % 3;
    typename LatticeFermion::vector_object::scalar_object sp;
    sp = Zero();
    sp()(s)(cc) = ComplexD(1.0, 0.0);
    if (blk == 0)
      pokeSite(sp, v.upper.f[a], osites[site]);
    else
      pokeSite(sp, v.lower.f[a], osites[site]);
  };
  auto ReadColumn = [&](const F &v, std::vector<CD> &M, int col) {
    for (int site = 0; site < Nodd; ++site) {
      for (int blk = 0; blk < 2; ++blk)
        for (int a = 0; a < DtxqcdNf; ++a) {
          auto sp = (blk == 0) ? peekSite(v.upper.f[a], osites[site])
                               : peekSite(v.lower.f[a], osites[site]);
          for (int s = 0; s < Ns; ++s)
            for (int cc = 0; cc < Nc; ++cc) {
              const int row = site * 48 + (blk * 2 + a) * 12 + s * 3 + cc;
              const ComplexD z = sp()(s)(cc);
              M[(size_t)row + (size_t)N * col] = CD(z.real(), z.imag());
            }
        }
    }
  };
  auto Assemble = [&](auto &op, std::vector<CD> &M) {
    M.assign((size_t)N * N, CD(0, 0));
    F ein(&Grid_), eout(&Grid_);
    for (int col = 0; col < N; ++col) {
      SetUnit(ein, col);
      op(ein, eout);
      ReadColumn(eout, M, col);
    }
  };

  cusolverDnHandle_t cus;
  GRID_ASSERT(cusolverDnCreate(&cus) == CUSOLVER_STATUS_SUCCESS);

  // FD gate matrices first (eta +- delta), then the working point.
  std::vector<CD> Smat;
  double ladp = 0, php = 0, ladm = 0, phm = 0;
  BuildWeights(eta + fd_delta);
  BuildGWeights(eta + fd_delta);
  Assemble(Stil, Smat);
  DetLU(cus, Smat, N, ladp, php);
  BuildWeights(eta - fd_delta);
  BuildGWeights(eta - fd_delta);
  Assemble(Stil, Smat);
  DetLU(cus, Smat, N, ladm, phm);

  BuildWeights(eta);
  BuildGWeights(eta);
  BuildHopWeights(eta);
  Assemble(Stil, Smat);
  std::vector<CD> dSmat;
  Assemble(dStil, dSmat);
  std::cout << GridLogMessage << "[dense-defl] Stil + dStil assembled (N="
            << N << ")" << std::endl;

  // ---- exact trace via LU solve X = Stil^-1 dStil -----------------------
  std::vector<CD> X = dSmat;
  {
    cuDoubleComplex *dA, *dB;
    int *dPiv, *dInfo;
    GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)N * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dB, sizeof(cuDoubleComplex) * (size_t)N * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dPiv, sizeof(int) * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(dA, Smat.data(), sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(dB, X.data(), sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    int lwork = 0;
    GRID_ASSERT(cusolverDnZgetrf_bufferSize(cus, N, N, dA, N, &lwork) ==
                CUSOLVER_STATUS_SUCCESS);
    cuDoubleComplex *dW;
    GRID_ASSERT(cudaMalloc(&dW, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
    GRID_ASSERT(cusolverDnZgetrf(cus, N, N, dA, N, dW, dPiv, dInfo) ==
                CUSOLVER_STATUS_SUCCESS);
    int info = -1;
    GRID_ASSERT(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    GRID_ASSERT(info == 0);
    GRID_ASSERT(cusolverDnZgetrs(cus, CUBLAS_OP_N, N, N, dA, N, dPiv, dB, N,
                                 dInfo) == CUSOLVER_STATUS_SUCCESS);
    GRID_ASSERT(cudaMemcpy(X.data(), dB, sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(dA); cudaFree(dB); cudaFree(dPiv); cudaFree(dInfo); cudaFree(dW);
  }
  CD trX(0, 0);
  for (int i = 0; i < N; ++i) trX += X[(size_t)i + (size_t)N * i];
  const CD fd((ladp - ladm) / (2 * fd_delta),
              std::remainder(php - phm, 2.0 * M_PI) / (2 * fd_delta));
  std::cout << GridLogMessage << std::setprecision(12)
            << "[dense-defl] exact Tr[S^-1 dS] = " << trX.real() << " + "
            << trX.imag() << " i" << std::endl;
  std::cout << GridLogMessage << "[dense-defl] FD dlogdet          = "
            << fd.real() << " + " << fd.imag() << " i   |diff|/|Tr| = "
            << std::abs(trX - fd) / std::abs(trX)
            << (std::abs(trX - fd) / std::abs(trX) < 1e-5 ? "   GATE OK"
                                                          : "   ** GATE FAIL **")
            << std::endl;

  // ---- dilution-scheme analysis ------------------------------------------
  // With A dense, the Hutchinson variance of ANY dilution scheme is exact:
  // noise restricted to block b sees only A|_{b x b}, so a full sweep of D
  // blocks (D solves) has Var = sum_b ||A_bb||_F^2.  The equal-solve-budget
  // figure of merit vs plain (1 solve) is  sigma_eff = sqrt(D * sum_b) --
  // compare against sigma_plain = ||A||_F.  Grading-antithetic overlay
  // reported alongside (the production pair).  Blocks are defined from the
  // packed index: comp = (blk*2+a)*12 + s*3 + c, t = osites[site][3].
  auto gamI = [&](int i) { return (i % 48) < 24 ? 1 : -1; };
  auto AnalyzeDilution = [&](const std::vector<CD> &Am, const char *label) {
    struct Scheme {
      const char *name;
      int D;
      std::function<int(int)> id;
    };
    auto siteT = [&](int g) { return osites[g / 48][3]; };
    auto comp = [&](int g) { return g % 48; };
    std::vector<Scheme> schemes = {
        {"t-slice", T, [&](int g) { return siteT(g); }},
        {"spin", 4, [&](int g) { return (comp(g) % 12) / 3; }},
        {"color", 3, [&](int g) { return comp(g) % 3; }},
        {"spin*color", 12, [&](int g) { return comp(g) % 12; }},
        {"grading", 2, [&](int g) { return comp(g) / 24; }},
        {"flavor", 2, [&](int g) { return (comp(g) / 12) % 2; }},
        {"t*spin", 4 * T,
         [&](int g) { return siteT(g) * 4 + (comp(g) % 12) / 3; }},
        {"t*spin*color", 12 * T,
         [&](int g) { return siteT(g) * 12 + comp(g) % 12; }},
    };
    const int NS = (int)schemes.size();
    std::vector<double> acc(NS, 0.0), acc_anti(NS, 0.0);
    double fro = 0, fro_anti = 0;
    std::vector<int8_t> gi(N);
    std::vector<std::vector<int>> bid(NS, std::vector<int>(N));
    for (int i = 0; i < N; ++i) {
      gi[i] = (int8_t)gamI(i);
      for (int s = 0; s < NS; ++s) bid[s][i] = schemes[s].id(i);
    }
#pragma omp parallel
    {
      std::vector<double> lacc(NS, 0.0), lanti(NS, 0.0);
      double lfro = 0, lfro_anti = 0;
#pragma omp for nowait
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          const double m = std::norm(Am[(size_t)i + (size_t)N * j]);
          if (m == 0.0) continue;
          const bool ge = gi[i] * gi[j] > 0;
          lfro += m;
          if (ge) lfro_anti += m;
          for (int s = 0; s < NS; ++s)
            if (bid[s][i] == bid[s][j]) {
              lacc[s] += m;
              if (ge) lanti[s] += m;
            }
        }
#pragma omp critical
      {
        fro += lfro;
        fro_anti += lfro_anti;
        for (int s = 0; s < NS; ++s) {
          acc[s] += lacc[s];
          acc_anti[s] += lanti[s];
        }
      }
    }
    std::cout << GridLogMessage << "[dense-defl] dilution (" << label
              << "): sigma_plain=" << std::scientific << std::setprecision(3)
              << std::sqrt(fro) << " sigma_anti=" << std::sqrt(fro_anti)
              << "  (equal-budget sigma_eff = sqrt(D*sum_b ||A_bb||^2))"
              << std::endl;
    for (int s = 0; s < NS; ++s)
      std::cout << GridLogMessage << "[dense-defl]   " << std::setw(14)
                << schemes[s].name << " D=" << std::setw(3) << schemes[s].D
                << "  sigma_eff=" << std::scientific << std::setprecision(3)
                << std::sqrt(schemes[s].D * acc[s])
                << "  sigma_eff(anti)=" << std::sqrt(schemes[s].D * acc_anti[s])
                << "  capture=" << std::fixed << std::setprecision(4)
                << acc[s] / fro << std::endl;
  };
  AnalyzeDilution(X, "K=0");

  // ---- SVD (full) --------------------------------------------------------
  std::vector<double> sv(N);
  std::vector<CD> Umat((size_t)N * N), VTmat((size_t)N * N);
  {
    cuDoubleComplex *dA, *dU, *dVT, *dW;
    double *dS, *dRW;
    int *dInfo;
    GRID_ASSERT(cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)N * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dU, sizeof(cuDoubleComplex) * (size_t)N * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dVT, sizeof(cuDoubleComplex) * (size_t)N * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dS, sizeof(double) * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dRW, sizeof(double) * N) == cudaSuccess);
    GRID_ASSERT(cudaMalloc(&dInfo, sizeof(int)) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(dA, Smat.data(), sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    int lwork = 0;
    GRID_ASSERT(cusolverDnZgesvd_bufferSize(cus, N, N, &lwork) ==
                CUSOLVER_STATUS_SUCCESS);
    GRID_ASSERT(cudaMalloc(&dW, sizeof(cuDoubleComplex) * (size_t)lwork) == cudaSuccess);
    GRID_ASSERT(cusolverDnZgesvd(cus, 'A', 'A', N, N, dA, N, dS, dU, N, dVT, N,
                                 dW, lwork, dRW, dInfo) ==
                CUSOLVER_STATUS_SUCCESS);
    int info = -1;
    GRID_ASSERT(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    GRID_ASSERT(info == 0 && "SVD did not converge");
    GRID_ASSERT(cudaMemcpy(sv.data(), dS, sizeof(double) * N,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(Umat.data(), dU, sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    GRID_ASSERT(cudaMemcpy(VTmat.data(), dVT, sizeof(cuDoubleComplex) * (size_t)N * N,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(dA); cudaFree(dU); cudaFree(dVT); cudaFree(dS); cudaFree(dRW);
    cudaFree(dW); cudaFree(dInfo);
  }
  std::cout << GridLogMessage << "[dense-defl] sigma_min..(8) =";
  for (int k = 0; k < 8; ++k) std::cout << " " << sv[N - 1 - k];
  std::cout << "  sigma_max = " << sv[0] << std::endl;

  // ---- K ladder ----------------------------------------------------------
  // Work down the tail (smallest sigma first), peeling rank-1 terms off
  // A = X = S^-1 dS:  A_K = X - sum_k (v_k / sigma_k) (u_k^dag dS).
  // Variance per antithetic PAIR = sum over grading-EVEN entries |A_ij|^2
  // (the pair average zeroes grading-odd entries exactly; complex gaussian
  // with E|z_i|^2 = 2 and the estimator's /2 normalization give exactly
  // ||A'||_F^2).  gamma_i = +1 for upper-block components (comp%48 < 24).
  auto gam = [&](int i) { return (i % 48) < 24 ? 1 : -1; };
  std::vector<CD> A = X;
  std::vector<int> ladder = {0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192,
                             256, 384, 512};
  CD exact_cum(0, 0);
  std::cout << GridLogMessage
            << "[dense-defl]   K   sigma_K       exact_cum(re)   "
            << "sigma/pair(anti)   sigma/pair(plain)   remainder(re)"
            << std::endl;
  size_t li = 0;
  std::vector<CD> udS(N), vk(N);
  for (int K = 0; K <= std::min(kmax, N); ++K) {
    if (li < ladder.size() && K == ladder[li]) {
      double fro_anti = 0, fro_plain = 0;
      CD trA(0, 0);
#pragma omp parallel for reduction(+ : fro_anti, fro_plain)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          const double m = std::norm(A[(size_t)i + (size_t)N * j]);
          fro_plain += m;
          if (gam(i) * gam(j) > 0) fro_anti += m;
        }
      for (int i = 0; i < N; ++i) trA += A[(size_t)i + (size_t)N * i];
      std::cout << GridLogMessage << "[dense-defl] " << std::setw(4) << K
                << "  " << std::scientific << std::setprecision(3)
                << (K > 0 ? sv[N - K] : 0.0) << "  " << std::fixed
                << std::setprecision(6) << exact_cum.real() << "  "
                << std::scientific << std::sqrt(fro_anti) << "  "
                << std::sqrt(fro_plain) << "  " << std::fixed << trA.real()
                << std::endl;
      ++li;
      if (li == ladder.size()) break;
    }
    if (K == std::min(kmax, N)) break;
    // peel triplet k = K (0-based from the tail): index n = N-1-K
    const int n = N - 1 - K;
    const double sk = sv[n];
    // u_k = column n of U;  v_k = conj(row n of VT)
    for (int i = 0; i < N; ++i) vk[i] = std::conj(VTmat[(size_t)n + (size_t)N * i]);
    // udS_j = sum_i conj(U_in) dS_ij
#pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      CD acc(0, 0);
      for (int i = 0; i < N; ++i)
        acc += std::conj(Umat[(size_t)i + (size_t)N * n]) *
               dSmat[(size_t)i + (size_t)N * j];
      udS[j] = acc;
    }
    // exact term += sigma^-1 u^dag dS v
    CD uv(0, 0);
    for (int j = 0; j < N; ++j) uv += udS[j] * vk[j];
    exact_cum += uv / sk;
    // A -= (v/sigma) udS
#pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      const CD w = udS[j] / sk;
      if (std::abs(w) == 0.0) continue;
      for (int i = 0; i < N; ++i)
        A[(size_t)i + (size_t)N * j] -= vk[i] * w;
    }
  }

  AnalyzeDilution(A, "final K");

  // ---- optional MC self-check at the final K -----------------------------
  if (mc_check > 0) {
    std::mt19937_64 gen(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    CD m1(0, 0);
    double m2 = 0;
    for (int s = 0; s < mc_check; ++s) {
      std::vector<CD> z(N);
      for (int i = 0; i < N; ++i) z[i] = CD(nd(gen), nd(gen));
      CD e1(0, 0), e2(0, 0);
#pragma omp parallel for
      for (int j = 0; j < N; ++j) {
        CD Az1(0, 0), Az2(0, 0);
        for (int i = 0; i < N; ++i) {
          const CD a = A[(size_t)i + (size_t)N * j];
          Az1 += std::conj(z[i]) * a;
          Az2 += (double)gam(i) * std::conj(z[i]) * a;
        }
#pragma omp critical
        {
          e1 += Az1 * z[j];
          e2 += Az2 * ((double)gam(j) * z[j]);
        }
      }
      const CD est = 0.25 * (e1 + e2);  // pair avg of z^dag A z / 2
      m1 += est;
      m2 += std::norm(est);
    }
    m1 /= (double)mc_check;
    m2 = m2 / mc_check - std::norm(m1);
    std::cout << GridLogMessage << "[dense-defl] MC self-check (" << mc_check
              << " pairs, final K): mean " << m1.real() << " + " << m1.imag()
              << " i   sigma/pair " << std::sqrt(std::max(m2, 0.0))
              << "   (predicts ||A'||_F of the last ladder row)" << std::endl;
  }

  cusolverDnDestroy(cus);
  Grid_finalize();
  std::fflush(nullptr);
  std::_Exit(0);
#endif
}
