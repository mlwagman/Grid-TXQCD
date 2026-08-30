// Test_dtxqcd_sign_aux_draws: Pfaffian-sign statistics of the aux-space
// integral at FIXED gauge field, over independent draws from the aux
// Gaussian prior e^{-S_aux} (width 1/lambda per entry, mean zero).
//
// Purpose (2026-07-01, plaq-deficit investigation): the H-S identity
//   Z(U) = ∫ daux e^{-S_aux(aux)} Pf(K·M48(aux,U))  =  Z_QCD-equiv(U)
// holds EXACTLY at every lambda.  The |Pf| HMC can only reproduce this if
// the sign of Pf is essentially constant over the aux measure at fixed U —
// otherwise the chain is trapped in one sign sector (it cannot cross
// Pf = 0 where its weight vanishes) and equilibrates to a genuinely wrong
// gauge marginal (candidate explanation for plaq = 0.4534 vs QCD 0.5133 at
// lambda <= 0.25).  This test measures, per lambda, over N independent
// prior draws at fixed U:
//   - the low end of the gamma5·M48 spectrum (|lambda|_min: distance to the
//     Pf = 0 surface),
//   - the Kramers-pair-counted sign  s = (-1)^{n_neg/2}  from the EXACT
//     dense spectrum (volume must be small: 4^4 -> dim 12288),
//   - the pair-degeneracy quality (validates the doubling structure).
// A sign that flips across draws at small lambda = the sector-trapping
// mechanism is real.  Sign constant at every lambda = mechanism dead.
//
// The dense eigensolve uses cuSOLVER (Zheevd, eigenvalues only) under
// GRID_CUDA; there is no CPU fallback (run on a GPU node).
//
// Run:
//   ./Test_dtxqcd_sign_aux_draws --grid 4.4.4.4 --mpi 1.1.1.1
// Env: LAMBDAS ("2,1,0.5,0.25,0.1"), NDRAWS (12), MASS (-0.245),
//      CSW (1.24930970916466), GAUGE_KIND (hot|weak|unit, default hot),
//      SIGMA_SHIFT (0.0: prior; >0 adds the sigma/s saddle shift)

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <cstdio>   // GAUGE_FILE magic sniff
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <vector>

#ifdef GRID_CUDA
#include <cusolverDn.h>
#include <cuda_runtime.h>
#endif

using namespace Grid;

static void Gamma5Doubled(DTXQCDFermionDoubled &x) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.upper.f[a];
    x.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.lower.f[a];
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> dvec;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) dvec.push_back(std::stoi(tok));
      if (dvec.size() == (size_t)Nd) latt = Coordinate(dvec);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GRID_ASSERT(mpi[0] * mpi[1] * mpi[2] * mpi[3] == 1);  // dense assembly: 1 rank

  auto env_real = [](const char *n, RealD d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atof(e) : d;
  };
  auto env_int = [](const char *n, int d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atoi(e) : d;
  };
  const RealD mass  = env_real("MASS", -0.245);
  const RealD csw   = env_real("CSW", 1.24930970916466);
  const int   ndraw = env_int("NDRAWS", 12);
  const RealD sigma_shift = env_real("SIGMA_SHIFT", 0.0);
  std::vector<RealD> lambdas{2.0, 1.0, 0.5, 0.25, 0.1};
  if (const char *ls = std::getenv("LAMBDAS")) {
    lambdas.clear();
    std::stringstream ss(ls); std::string tok;
    while (std::getline(ss, tok, ',')) lambdas.push_back(std::atof(tok.c_str()));
  }

  GridParallelRNG pRNG(&Grid); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  DTXQCDField U(&Grid);
  std::string gk = std::getenv("GAUGE_KIND") ? std::getenv("GAUGE_KIND") : "hot";
  if (gk == "unit")       DTXQCDCompositeImpl::ColdConfiguration(pRNG, U);
  else if (gk == "weak")  DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
  else if (gk == "tepid") DTXQCDCompositeImpl::TepidConfiguration(pRNG, U);
  else if (gk == "file") {
    // GAUGE_FILE: equilibrated gauge config (NERSC or, with HAVE_LIME, ILDG).
    const char *gf = std::getenv("GAUGE_FILE");
    GRID_ASSERT(gf && *gf);
    std::FILE *fp = std::fopen(gf, "rb");
    char magic[16] = {0};
    if (fp) { (void)std::fread(magic, 1, sizeof(magic), fp); std::fclose(fp); }
    FieldMetaData header;
    if (std::memcmp(magic, "BEGIN_HEADER", 12) == 0) {
      typedef GaugeStatistics<PeriodicGimplR> GaugeStats;
      NerscIO::readConfiguration<GaugeStats>(U.U, header, std::string(gf));
    } else {
#ifdef HAVE_LIME
      IldgReader IR;
      IR.open(std::string(gf));
      IR.readConfiguration(U.U, header);
      IR.close();
#else
      std::cout << GridLogError << "GAUGE_FILE not NERSC and no HAVE_LIME" << std::endl;
      GRID_ASSERT(0);
#endif
    }
    std::cout << GridLogMessage << "[sign-draws] GAUGE_FILE=" << gf << std::endl;
  }
  else                    SU<Nc>::HotConfiguration(pRNG, U.U);
  // STOUT_NSMEAR (default 0): smear the loaded gauge so the dense M48 sees the
  // production (stout, csw=1.249) operator, not bare rough links.
  {
    const int n_stout = env_int("STOUT_NSMEAR", 0);
    const RealD rho = env_real("STOUT_RHO", 0.125);
    std::cout << GridLogMessage << "[sign-draws] plaq(raw) = "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U)
              << "  STOUT_NSMEAR=" << n_stout << std::endl;
    if (n_stout > 0) {
      Smear_Stout<PeriodicGimplR> stout(rho);
      LatticeGaugeField Usm(&Grid);
      for (int i = 0; i < n_stout; ++i) { stout.smear(Usm, U.U); U.U = Usm; }
      std::cout << GridLogMessage << "[sign-draws] plaq(stout) = "
                << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;
    }
  }
  // SEED (optional): reseed AFTER gauge setup so parallel streams share the
  // gauge (same GAUGE_KIND/FILE) but draw independent aux.  Unset = legacy
  // stream, bit-compatible with the 2026-07-01 runs.
  if (const char *sv = std::getenv("SEED"); sv && *sv) {
    const int sd = std::atoi(sv);
    pRNG.SeedFixedIntegers({sd, sd + 11, sd + 23, sd + 37, sd + 41});
    std::cout << GridLogMessage << "[sign-draws] aux SEED=" << sd << std::endl;
  }
  std::cout << GridLogMessage << "[sign-draws] gauge=" << gk << " mass=" << mass
            << " csw=" << csw << " ndraw=" << ndraw
            << " sigma_shift=" << sigma_shift
            << " V4=" << Grid.gSites() << " dim=" << 48 * Grid.gSites()
            << std::endl;

  const int V   = (int)Grid.gSites();
  const int dim = 48 * V;
  const Coordinate ldims = Grid.GlobalDimensions();

  // lex site index, x fastest (matches unvectorizeToLexOrdArray ordering)
  auto site_coor = [&](int lex, Coordinate &c) {
    c.resize(4);
    c[0] = lex % ldims[0]; lex /= ldims[0];
    c[1] = lex % ldims[1]; lex /= ldims[1];
    c[2] = lex % ldims[2]; lex /= ldims[2];
    c[3] = lex;
  };
  // global row index: ((site*2 + blk)*Nf + a)*12 + spin*3 + color
  auto row_idx = [&](int site, int blk, int a, int sp, int co) {
    return ((site * 2 + blk) * DtxqcdNf + a) * 12 + sp * 3 + co;
  };

  typedef typename LatticeFermion::scalar_object FSobj;

  // Field strength for the per-site clover term (gauge fixed across draws).
  DtxqcdSpinMatrices spin(Grid);
  std::vector<LatticeColourMatrix> FS;
  if (csw != 0.0) {
    FS.reserve(6);
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(&Grid);
        WilsonLoops<WilsonImplR>::FieldStrength(F, U.U, mu, nu);
        FS.push_back(std::move(F));
      }
  }
  // gamma5 sign in the 48-dim site basis (r24 = a*12 + alpha*3 + i, same
  // sign in both 24-blocks -- Gamma5 = diag(g5, g5)).
  auto g5sign48 = [](int r) {
    int alpha = ((r % 24) % 12) / 3;
    return (alpha < 2) ? 1.0 : -1.0;
  };
  // Site-local Pfaffian sign product over the EVEN sublattice:
  //   Pf(K M48) = prod_{x even} Pf(K M_ee(x)) * Pf(Schur_odd)
  // Each 48x48 gamma5.M_ee(x) is Hermitian, doubly degenerate; per-site
  // sign = (-1)^(n_neg/2).  Exactly computable at ANY volume -> if the
  // small-lambda flips live here, production sign tracking is free.
  auto site_sign_product = [&](DTXQCDField &Ux, int &bad_parity) {
    int pair_parity = 0;  // sum of n_neg(x)/2 over even sites, mod 2
    bad_parity = 0;
    Eigen::MatrixXcd M48s;
    Eigen::MatrixXcd Hs(48, 48);
    for (int lex = 0; lex < V; ++lex) {
      Coordinate c; site_coor(lex, c);
      if ((c[0] + c[1] + c[2] + c[3]) % 2) continue;   // even sites only
      DtxqcdSiteAux aux = DtxqcdSiteAux::Extract(Ux.sigma, Ux.pi, Ux.d, Ux.n,
                                                 Ux.s, Ux.p, c);
      Eigen::MatrixXcd Mu, Ml, Mo;
      if (csw != 0.0) {
        DtxqcdSiteClover cl = DtxqcdSiteClover::Extract(FS, c);
        DtxqcdBuildUpperBlock24(mass, aux, spin, Mu, csw, &cl);
        DtxqcdBuildLowerBlock24(mass, aux, spin, Ml, csw, &cl);
      } else {
        DtxqcdBuildUpperBlock24(mass, aux, spin, Mu);
        DtxqcdBuildLowerBlock24(mass, aux, spin, Ml);
      }
      DtxqcdBuildOffDiagBlock24(aux, spin, Mo);
      DtxqcdAssembleDoubled48(Mu, Ml, Mo, M48s);
      for (int r = 0; r < 48; ++r)
        for (int cc = 0; cc < 48; ++cc)
          Hs(r, cc) = g5sign48(r) * M48s(r, cc);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hs,
          Eigen::EigenvaluesOnly);
      int nn = 0;
      for (int k = 0; k < 48; ++k) if (es.eigenvalues()[k] < 0) ++nn;
      if (nn % 2) ++bad_parity;       // should never happen (Kramers pairs)
      pair_parity += nn / 2;
    }
    return (pair_parity % 2) ? -1 : +1;
  };

#ifdef GRID_CUDA
  cusolverDnHandle_t cus;
  GRID_ASSERT(cusolverDnCreate(&cus) == CUSOLVER_STATUS_SUCCESS);
#endif

  std::vector<std::complex<double>> H((size_t)dim * dim);
  std::vector<double> evals(dim);

  // ---- dense spectrum of gamma5.M48(Ux): the assembly + cuSOLVER solve,
  // callable repeatedly per draw (base + flip-move variants). ----
  struct DenseSpec {
    int n_neg; int sgn; double logabsdet, min_abs, pair_split_max, herm_max;
  };
  auto dense_spec = [&](DTXQCDField &Ux) -> DenseSpec {
    DTXQCDWilsonCloverFermionEO Dw(Ux.U, Grid, RBGrid, mass, csw,
                                   Ux.sigma, Ux.pi, Ux.d, Ux.n, Ux.s, Ux.p);
    DTXQCDMOp Mop(Dw);
    DTXQCDFermionDoubled e(&Grid), y(&Grid);
    std::vector<FSobj> col_arr(V);
    for (int site = 0; site < V; ++site) {
      Coordinate c; site_coor(site, c);
      for (int blk = 0; blk < 2; ++blk)
        for (int a = 0; a < DtxqcdNf; ++a)
          for (int sp = 0; sp < Ns; ++sp)
            for (int co = 0; co < Nc; ++co) {
              const int j = row_idx(site, blk, a, sp, co);
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
                        const int i = row_idx(s2, blk2, a2, sp2, co2);
                        auto z = col_arr[s2]()(sp2)(co2);
                        H[(size_t)j * dim + i] =
                            std::complex<double>(z.real(), z.imag());
                      }
                }
            }
    }
    DenseSpec r;
    r.herm_max = 0.0;
    for (unsigned long t = 0; t < 2000; ++t) {
      int i = (int)((1103515245UL * t + 12345UL) % (unsigned long)dim);
      int j = (int)((69069UL * t + 1UL) % (unsigned long)dim);
      auto dlt = H[(size_t)j * dim + i] - std::conj(H[(size_t)i * dim + j]);
      r.herm_max = std::max(r.herm_max, std::abs(dlt));
    }
#ifdef GRID_CUDA
    cuDoubleComplex *dA;
    double *dW; int *dInfo;
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
    {  // CPU fallback (2026-08-14): Eigen dense Hermitian eigenvalues
      Eigen::Map<const Eigen::MatrixXcd> Hm(H.data(), dim, dim);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hm, Eigen::EigenvaluesOnly);
      GRID_ASSERT(es.info() == Eigen::Success);
      evals.resize(dim);
      for (int i = 0; i < dim; ++i) evals[i] = es.eigenvalues()[i];
    }
#endif
    r.n_neg = 0; r.min_abs = 1e300; r.logabsdet = 0.0;
    for (int i = 0; i < dim; ++i) {
      if (evals[i] < 0) ++r.n_neg;
      r.min_abs = std::min(r.min_abs, std::abs(evals[i]));
      r.logabsdet += std::log(std::abs(evals[i]));
    }
    r.pair_split_max = 0.0;
    for (int i = 0; i + 1 < dim; i += 2) {
      double s = std::abs(evals[i] - evals[i + 1]) /
                 std::max(std::abs(evals[i]), 1e-300);
      r.pair_split_max = std::max(r.pair_split_max, s);
    }
    r.sgn = ((r.n_neg / 2) % 2) ? -1 : +1;
    return r;
  };

  // ---- FLIP_MOVES=1: per draw, also test the two verified sector-crossing
  // moves (s_tslice0, negate_sigma; both S_aux-exact) -> per-config eta and
  // Metropolis log-ratio dlad (accept prob = min(1, e^{dlad/2})). ----
  const bool flip_moves = env_int("FLIP_MOVES", 0) != 0;
  LatticeInteger tcoord(&Grid);
  LatticeCoordinate(tcoord, Nd - 1);
  auto mv_s_tslice0 = [&](DTXQCDField &F) {
    LatticeDtxqcdS neg = -F.s;
    F.s = where(tcoord == Integer(0), neg, F.s);
  };
  auto mv_negate_sigma = [&](DTXQCDField &F) { F.sigma = -F.sigma; };

  for (RealD lambda : lambdas) {
    std::cout << GridLogMessage << "======== lambda = " << lambda
              << " ========" << std::endl;
    int nflip = 0, sign_first = 0;
    std::vector<int>    sgn_hist, sites_hist, schur_hist;
    std::vector<double> lad_hist;   // log|det(gamma5 M)| = log|det M| per draw
    // Exact-reweighting instrumentation: with SIGMA_SHIFT>0 the draws come
    // from the SHIFTED proposal q(x) ~ e^{-S_aux(x-mu)} (the aux-init
    // distribution), not the prior.  The exact target weight is then
    //   w = |Pf(x)| e^{-(S_aux(x) - S_aux(x-mu))} = |Pf| e^{-dS}.
    // Print S_aux(x) and dS per draw so the offline analysis can do both the
    // unweighted (aux-init ~ equilibrium) and the exact reweighted estimate.
    DTXQCDAuxiliaryFieldGaussianAction Saux(lambda);
    DTXQCDField Umean(&Grid), Vres(&Grid);
    DTXQCDCompositeImpl::FillAuxFields(pRNG, Umean, lambda, sigma_shift,
                                       /*with_fluct=*/false);  // no RNG use
    for (int d = 0; d < ndraw; ++d) {
      // Independent draw from the (possibly shifted) proposal.
      DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, sigma_shift);
      const double S_x = Saux.S(U);
      Vres.sigma = U.sigma - Umean.sigma;  Vres.pi = U.pi;
      Vres.d = U.d;  Vres.n = U.n;
      Vres.s = U.s - Umean.s;              Vres.p = U.p;
      const double dS = S_x - Saux.S(Vres);

      std::cout << GridLogMessage << "[assemble] start draw " << d << std::endl;
      const DenseSpec base = dense_spec(U);
      const int    n_neg = base.n_neg;
      const double min_abs = base.min_abs, logabsdet = base.logabsdet;
      const double pair_split_max = base.pair_split_max, herm_max = base.herm_max;
      const bool n_neg_even = (n_neg % 2) == 0;
      const int  sgn = base.sgn;

      if (flip_moves) {
        // eta(X) per move: does the move flip signPf on THIS config, and what
        // is the Metropolis log-ratio dlad = log|det(T(X))| - log|det(X)|
        // (S_aux exactly invariant for both moves; accept = min(1, e^{dlad/2})).
        DTXQCDField W(&Grid);
        W.U = U.U;
        for (int m = 0; m < 2; ++m) {
          W.sigma = U.sigma; W.pi = U.pi; W.d = U.d;
          W.n = U.n; W.s = U.s; W.p = U.p;
          const char *mname = m ? "negate_sigma" : "s_tslice0";
          if (m) mv_negate_sigma(W); else mv_s_tslice0(W);
          const DenseSpec r = dense_spec(W);
          std::cout << GridLogMessage << "[lam=" << lambda << " draw " << d
                    << " FLIP " << mname << "]"
                    << "  eta=" << std::showpos << r.sgn * base.sgn
                    << std::noshowpos
                    << "  d_nneg=" << (r.n_neg - base.n_neg)
                    << "  dlad=" << (r.logabsdet - base.logabsdet)
                    << "  min|eval|=" << r.min_abs
                    << std::endl;
        }
      }
      if (d == 0) sign_first = sgn;
      if (sgn != sign_first) ++nflip;
      sgn_hist.push_back(sgn);
      lad_hist.push_back(logabsdet - 2.0 * dS);  // log(|det| e^{-2dS}): |Pf|e^{-dS} = exp(1/2 of this)

      // EO factorization of the sign: full = (even-site-local product) x (Schur).
      int bad_parity = 0;
      const int sgn_sites = site_sign_product(U, bad_parity);
      const int sgn_schur = sgn * sgn_sites;   // signs are +-1
      sites_hist.push_back(sgn_sites);
      schur_hist.push_back(sgn_schur);

      std::cout << GridLogMessage << "[lam=" << lambda << " draw " << d << "]"
                << "  n_neg=" << n_neg << (n_neg_even ? "" : "  (ODD!)")
                << "  sign=(-1)^(n_neg/2)=" << std::showpos << sgn
                << "  sign_sitesEE=" << sgn_sites
                << "  sign_schur=" << sgn_schur << std::noshowpos
                << (bad_parity ? "  (SITE ODD PARITY!)" : "")
                << "  min|eval|=" << min_abs
                << "  logabsdet=" << logabsdet
                << "  S_aux=" << S_x
                << "  dS=" << dS
                << "  pair_split_max=" << pair_split_max
                << "  herm_check=" << herm_max
                << std::endl;
    }
    // |Pf|-weighted <sign> over the prior draws:
    //   <sign>_w = sum_i s_i |Pf_i| / sum_i |Pf_i|,  |Pf_i| = |det M_i|^{1/2}
    // = the per-4^4-patch feasibility number for sign reweighting (compounds
    // ~exponentially in volume).  Max-shifted for overflow safety.
    double lad_max = *std::max_element(lad_hist.begin(), lad_hist.end());
    double wsum = 0.0, swsum = 0.0;
    for (int d = 0; d < ndraw; ++d) {
      double w = std::exp(0.5 * (lad_hist[d] - lad_max));
      wsum  += w;
      swsum += sgn_hist[d] * w;
    }
    int nplus = 0;
    for (int s_ : sgn_hist) if (s_ > 0) ++nplus;
    int npl_sites = 0, npl_schur = 0;
    for (int s_ : sites_hist) if (s_ > 0) ++npl_sites;
    for (int s_ : schur_hist) if (s_ > 0) ++npl_schur;
    std::cout << GridLogMessage << "[lam=" << lambda << " SUMMARY]  draws=" << ndraw
              << "  n_plus=" << nplus << "  n_minus=" << (ndraw - nplus)
              << "  flip_fraction=" << (RealD)nflip / (RealD)ndraw
              << "  <sign>_count=" << (2.0 * nplus - ndraw) / (RealD)ndraw
              << "  <sign>_Pfweighted=" << (swsum / wsum)
              << "  sitesEE_n_plus=" << npl_sites << "/" << ndraw
              << "  schur_n_plus=" << npl_schur << "/" << ndraw
              << std::endl;
  }

#ifdef GRID_CUDA
  cusolverDnDestroy(cus);
#endif
  Grid_finalize();
  return 0;
}
