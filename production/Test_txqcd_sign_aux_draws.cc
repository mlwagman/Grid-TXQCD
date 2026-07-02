// Test_txqcd_sign_aux_draws: does the (naive-coupled) TXQCD Möbius DWF
// operator suppress the aux-driven determinant sign flips that the Wilson
// operator shows?
//
// Context (2026-07-01/02 overnight, plaq-deficit / sign-problem hunt): the
// DTXQCD dense probe showed the Pfaffian sign over aux PRIOR draws is fully
// mixed for lambda <= 3 even on smooth gauge backgrounds (clean only at
// lambda ~ 6).  Mechanism: wide aux (width 1/lambda) drags the Wilson
// operator's unprotected real modes through zero.  Chiral (GW) fermions
// should convert the (sigma, pi) sign structure into rare topology-like
// events -- but the existing TXQCD DWF impl couples Delta NAIVELY (broadcast
// across s-slices, TXQCDMobiusOp), not chirally.  This probe measures, per
// lambda over aux prior draws at fixed gauge:
//   (a) sign det M_wilson-clover (TXQCD 24V-dim dense gamma5.M spectrum)
//   (b) sign det M_mobius(m) and sign det M_mobius(PV=1) (24.V.Ls-dim dense
//       Gamma5 R5 . M spectra; both operators include the SAME naive Delta)
// If (b) flips like (a): naive coupling gives no protection -> the chiral
// coupling is the necessary ingredient for a DTXQCD-DWF port.
// If (b) is clean where (a) flips: even naive DWF helps -> port is cheaper.
//
// TXQCD is undoubled: det real by gamma5(R5)-hermiticity, sign = (-1)^n_neg
// directly (fixed det(Gamma5[R5]) offset drops out of draw-to-draw flips).
//
// Run: ./Test_txqcd_sign_aux_draws --grid 4.4.4.4 --mpi 1.1.1.1
// Env: LAMBDAS ("3,1,0.5,0.1"), NDRAWS (8), LS (4), GAUGE_KIND
//      (tepid|weak|hot|unit, default tepid), MASS_W (-0.245), CSW (1.2493...),
//      MASS_DWF (0.05), M5 (1.8), MOB_B (1.5), MOB_C (0.5), SKIP_DWF (0)

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
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

namespace {

// Dense Hermitian eigenvalues (ascending) via cuSOLVER Zheevd, novector.
void DenseEvals(std::vector<std::complex<double>> &H, int dim,
                std::vector<double> &evals) {
#ifdef GRID_CUDA
  static cusolverDnHandle_t cus = nullptr;
  if (!cus) GRID_ASSERT(cusolverDnCreate(&cus) == CUSOLVER_STATUS_SUCCESS);
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
  evals.resize(dim);
  GRID_ASSERT(cudaMemcpy(evals.data(), dW, sizeof(double) * dim,
                         cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(dA); cudaFree(dW); cudaFree(dInfo); cudaFree(dWork);
#else
  (void)H; (void)dim; (void)evals;
  std::cout << GridLogError << "non-CUDA build: no dense eigensolver" << std::endl;
  abort();
#endif
}

struct SignStats { int n_neg; double min_abs; double logabsdet; };

SignStats AnalyzeEvals(const std::vector<double> &evals) {
  SignStats s{0, 1e300, 0.0};
  for (double e : evals) {
    if (e < 0) ++s.n_neg;
    s.min_abs = std::min(s.min_abs, std::abs(e));
    s.logabsdet += std::log(std::abs(e));
  }
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  auto env_real = [](const char *n, RealD d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atof(e) : d;
  };
  auto env_int = [](const char *n, int d) {
    const char *e = std::getenv(n); return (e && *e) ? std::atoi(e) : d;
  };

  Coordinate latt4(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  const int Ls = env_int("LS", 4);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);
  GRID_ASSERT(mpi[0] * mpi[1] * mpi[2] * mpi[3] == 1);

  const RealD mass_w   = env_real("MASS_W", -0.245);
  const RealD csw      = env_real("CSW", 1.24930970916466);
  const RealD mass_dwf = env_real("MASS_DWF", 0.05);
  const RealD M5       = env_real("M5", 1.8);
  const RealD mob_b    = env_real("MOB_B", 1.5);
  const RealD mob_c    = env_real("MOB_C", 0.5);
  const int   ndraw    = env_int("NDRAWS", 8);
  const int   skip_dwf = env_int("SKIP_DWF", 0);
  std::vector<RealD> lambdas{3.0, 1.0, 0.5, 0.1};
  if (const char *ls_ = std::getenv("LAMBDAS")) {
    lambdas.clear();
    std::stringstream ss(ls_); std::string tok;
    while (std::getline(ss, tok, ',')) lambdas.push_back(std::atof(tok.c_str()));
  }

  GridParallelRNG pRNG(UGrid); pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  TXQCDField U(UGrid);
  std::string gk = std::getenv("GAUGE_KIND") ? std::getenv("GAUGE_KIND") : "tepid";
  if (gk == "unit")       { U.U = ComplexD(1.0, 0.0); }
  else if (gk == "weak")  TXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
  else if (gk == "hot")   SU<Nc>::HotConfiguration(pRNG, U.U);
  else                    PeriodicGimplR::TepidConfiguration(pRNG, U.U);

  const int V = (int)UGrid->gSites();
  const int dim4 = TxqcdNf * 12 * V;
  const int dim5 = dim4 * Ls;
  std::cout << GridLogMessage << "[txqcd-sign] gauge=" << gk
            << " mass_w=" << mass_w << " csw=" << csw
            << " dwf(m=" << mass_dwf << ",M5=" << M5 << ",b=" << mob_b
            << ",c=" << mob_c << ",Ls=" << Ls << ")  ndraw=" << ndraw
            << "  dim4=" << dim4 << "  dim5=" << dim5 << std::endl;

  const Coordinate gd4 = UGrid->GlobalDimensions();
  auto coor4 = [&](int lex, Coordinate &c) {
    c.resize(4);
    c[0] = lex % gd4[0]; lex /= gd4[0];
    c[1] = lex % gd4[1]; lex /= gd4[1];
    c[2] = lex % gd4[2]; lex /= gd4[2];
    c[3] = lex;
  };
  // 5d lex: s fastest (dim 0 of FGrid)
  auto coor5 = [&](int lex, Coordinate &c) {
    c.resize(5);
    c[0] = lex % Ls; lex /= Ls;
    c[1] = lex % gd4[0]; lex /= gd4[0];
    c[2] = lex % gd4[1]; lex /= gd4[1];
    c[3] = lex % gd4[2]; lex /= gd4[2];
    c[4] = lex;
  };

  typedef typename LatticeFermion::scalar_object FSobj;
  std::vector<std::complex<double>> H4((size_t)dim4 * dim4);
  std::vector<std::complex<double>> H5;
  if (!skip_dwf) H5.resize((size_t)dim5 * dim5);
  std::vector<double> evals;

  // Assemble dense gamma5[R5].M for a generic TXQCD op into H (column-major
  // H[j*dim+i]).  nsite = local 4d (or 5d) sites; grid = the fermion grid.
  // apply(e, y) must include the gamma5 (and R5 for 5d) factor.
  auto assemble = [&](auto &&apply, GridCartesian *grid, int nsite5,
                      auto &&mkcoor, std::vector<std::complex<double>> &H,
                      int dim) {
    TXQCDFermionNf e(grid), y(grid);
    std::vector<FSobj> col;
    for (int site = 0; site < nsite5; ++site) {
      Coordinate c; mkcoor(site, c);
      for (int a = 0; a < TxqcdNf; ++a)
        for (int sp = 0; sp < Ns; ++sp)
          for (int co = 0; co < Nc; ++co) {
            const int j = (site * TxqcdNf + a) * 12 + sp * 3 + co;
            for (int aa = 0; aa < TxqcdNf; ++aa) e.f[aa] = Zero();
            FSobj one; one = Zero();
            one()(sp)(co) = ComplexD(1.0, 0.0);
            pokeSite(one, e.f[a], c);
            apply(e, y);
            for (int a2 = 0; a2 < TxqcdNf; ++a2) {
              unvectorizeToLexOrdArray(col, y.f[a2]);
              for (int s2 = 0; s2 < nsite5; ++s2)
                for (int sp2 = 0; sp2 < Ns; ++sp2)
                  for (int co2 = 0; co2 < Nc; ++co2) {
                    const int i = (s2 * TxqcdNf + a2) * 12 + sp2 * 3 + co2;
                    auto z = col[s2]()(sp2)(co2);
                    H[(size_t)j * dim + i] =
                        std::complex<double>(z.real(), z.imag());
                  }
            }
          }
    }
  };

  auto herm_spot = [&](const std::vector<std::complex<double>> &H, int dim) {
    double m = 0.0;
    for (unsigned long t = 0; t < 2000; ++t) {
      int i = (int)((1103515245UL * t + 12345UL) % (unsigned long)dim);
      int j = (int)((69069UL * t + 1UL) % (unsigned long)dim);
      m = std::max(m, std::abs(H[(size_t)j * dim + i] -
                               std::conj(H[(size_t)i * dim + j])));
    }
    return m;
  };

  for (RealD lambda : lambdas) {
    std::cout << GridLogMessage << "======== lambda = " << lambda
              << " ========" << std::endl;
    std::vector<int> sw_hist, sd_hist, spv_hist;
    for (int d = 0; d < ndraw; ++d) {
      TXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda, /*Sigma=*/0.0);

      // ---- (a) Wilson-clover: H4 = gamma5 . M ----
      TXQCDWilsonCloverOp Dwc(U.U, *UGrid, *UrbGrid, mass_w,
                              U.sigma, U.pi, U.s, U.p, U.t, csw);
      auto apply_w = [&](const TXQCDFermionNf &in, TXQCDFermionNf &out) {
        Dwc.M(in, out);
        for (int a = 0; a < TxqcdNf; ++a)
          out.f[a] = Gamma(Gamma::Algebra::Gamma5) * out.f[a];
      };
      assemble(apply_w, UGrid, V, coor4, H4, dim4);
      DenseEvals(H4, dim4, evals);
      SignStats sw = AnalyzeEvals(evals);
      const int sgn_w = (sw.n_neg % 2) ? -1 : +1;
      sw_hist.push_back(sgn_w);
      std::cout << GridLogMessage << "[lam=" << lambda << " draw " << d
                << " WILSON]  n_neg=" << sw.n_neg
                << "  sign=" << std::showpos << sgn_w << std::noshowpos
                << "  min|eval|=" << sw.min_abs
                << "  logabsdet=" << sw.logabsdet
                << "  herm=" << herm_spot(H4, dim4) << std::endl;

      if (skip_dwf) continue;

      // ---- (b) Mobius (naive Delta): H5 = Gamma5 R5 . M, m and PV=1 ----
      auto run_dwf = [&](RealD m, const char *tag, std::vector<int> &hist) {
        TXQCDMobiusOp Dmob(U.U, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                           m, M5, mob_b, mob_c,
                           U.sigma, U.pi, U.s, U.p, U.t);
        LatticeFermion tmp5(FGrid), tmp5b(FGrid);
        LatticeFermion slice(UGrid);
        auto apply_5 = [&](const TXQCDFermionNf &in, TXQCDFermionNf &out) {
          Dmob.M(in, out);
          for (int a = 0; a < TxqcdNf; ++a) {
            // R5: reverse the s slices, then gamma5.
            tmp5 = out.f[a];
            for (int s = 0; s < Ls; ++s) {
              ExtractSlice(slice, tmp5, Ls - 1 - s, 0);
              InsertSlice(slice, out.f[a], s, 0);
            }
            out.f[a] = Gamma(Gamma::Algebra::Gamma5) * out.f[a];
          }
        };
        assemble(apply_5, FGrid, V * Ls, coor5, H5, dim5);
        DenseEvals(H5, dim5, evals);
        SignStats s5 = AnalyzeEvals(evals);
        const int sgn = (s5.n_neg % 2) ? -1 : +1;
        hist.push_back(sgn);
        std::cout << GridLogMessage << "[lam=" << lambda << " draw " << d
                  << " " << tag << "]  n_neg=" << s5.n_neg
                  << "  sign=" << std::showpos << sgn << std::noshowpos
                  << "  min|eval|=" << s5.min_abs
                  << "  logabsdet=" << s5.logabsdet
                  << "  herm=" << herm_spot(H5, dim5) << std::endl;
      };
      run_dwf(mass_dwf, "DWF-m ", sd_hist);
      run_dwf(1.0,      "DWF-PV", spv_hist);
    }

    auto summarize = [&](const char *tag, const std::vector<int> &h) {
      if (h.empty()) return;
      int npl = 0; for (int s_ : h) if (s_ > 0) ++npl;
      std::cout << GridLogMessage << "[lam=" << lambda << " SUMMARY " << tag
                << "]  n_plus=" << npl << "/" << h.size()
                << "  <sign>_count=" << (2.0 * npl - (int)h.size()) / (RealD)h.size()
                << std::endl;
    };
    summarize("WILSON", sw_hist);
    summarize("DWF-m ", sd_hist);
    summarize("DWF-PV", spv_hist);
    if (!sd_hist.empty()) {
      // Effective-operator sign = sign(det M(m))*sign(det M(PV)) per draw.
      int npl = 0;
      for (size_t k = 0; k < sd_hist.size(); ++k)
        if (sd_hist[k] * spv_hist[k] > 0) ++npl;
      std::cout << GridLogMessage << "[lam=" << lambda << " SUMMARY DWF-eff]"
                << "  n_plus=" << npl << "/" << sd_hist.size()
                << "  <sign>_count="
                << (2.0 * npl - (int)sd_hist.size()) / (RealD)sd_hist.size()
                << std::endl;
    }
  }

  Grid_finalize();
  return 0;
}
