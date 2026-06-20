#pragma once
// Phase 0: shared batched 48x48 LU / inverse over the per-site doubled DTXQCD
// matrices via cuBLAS getrfBatched / getriBatched.  Used by both
// DTXQCDWilsonCloverFermionEO (Mooee^{-1} cache) and DTXQCDLogDetCloverEOAction
// (S = -1/2 log|det| and its force).  CUDA-only; callers gate dispatch on
// GRID_CUDA and fall back to the Eigen CPU path otherwise.
//
// Usage (caller fills the forward buffer, then asks for LU or inverse):
//   BatchedInverse48::Ensure(nsites, need_inv);     // size buffers
//   // ... write column-major 48x48 per site into &BatchedInverse48::M_fwd[0]
//   //     (site s at offset s*N2, element (row,col) at col*N + row) ...
//   BatchedInverse48::FactorLU(nsites);             // -> LU in M_fwd  (S_gpu)
//   BatchedInverse48::Invert(nsites);               // -> inverse in M_inv
//
// Buffers are inline static (shared across all instances) to avoid per-ctor
// cudaMalloc churn; they grow monotonically with the largest nsites seen, and
// the getrf/getri pointer arrays are rebuilt only when nsites grows so the
// cached device pointers stay valid.  One shared helper (the two analogous
// TXQCD classes each duplicated their own buffers).

#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>  // kDtxqcdSiteDim48

#ifdef GRID_CUDA
#include <Grid/algorithms/blas/BatchedBlas.h>          // GridBLAS handle
#include <cublas_v2.h>
#endif

NAMESPACE_BEGIN(Grid);

#ifdef GRID_CUDA
namespace DtxqcdBlas {

struct BatchedInverse48 {
  static constexpr int N  = kDtxqcdSiteDim48;  // 48
  static constexpr int N2 = N * N;             // 2304

  // Column-major per-site matrices, packed as [site*N2 + col*N + row].
  inline static deviceVector<ComplexD>  M_fwd;   // getrf overwrites with LU
  inline static deviceVector<ComplexD>  M_inv;   // getri output (Invert only)
  inline static deviceVector<ComplexD *> Amk;    // Amk[i] = &M_fwd[i*N2]
  inline static deviceVector<ComplexD *> Cmk;    // Cmk[i] = &M_inv[i*N2]
  inline static deviceVector<int>        pivots; // nsites * N
  inline static deviceVector<int>        info;   // nsites
  inline static uint64_t cap_nsites{0};
  inline static bool     cap_has_inv{false};

  // Size buffers for nsites; (re)build the getrf/getri pointer arrays when
  // nsites grows or the inverse buffer is first requested.
  static void Ensure(uint64_t nsites, bool need_inv) {
    const uint64_t need = nsites * (uint64_t)N2;
    if (M_fwd.size() < need) M_fwd.resize(need);
    if (need_inv && M_inv.size() < need) M_inv.resize(need);

    const bool grow = (cap_nsites < nsites);
    if (grow) {
      Amk.resize(nsites);
      pivots.resize(nsites * (uint64_t)N);
      info.resize(nsites);
      ComplexD *fwd = &M_fwd[0];
      ComplexD **amk = &Amk[0];
      accelerator_for(i, nsites, 1, { amk[i] = &fwd[i * (uint64_t)N2]; });
    }
    if (need_inv && (grow || !cap_has_inv)) {
      Cmk.resize(nsites);
      ComplexD *inv = &M_inv[0];
      ComplexD **cmk = &Cmk[0];
      accelerator_for(i, nsites, 1, { cmk[i] = &inv[i * (uint64_t)N2]; });
      cap_has_inv = true;
    }
    if (grow) cap_nsites = nsites;
  }

  // getrf only: leaves the LU factors (U on the diagonal) in M_fwd.  S_gpu
  // reads sum_k log|U_kk| over the LU diagonal for log|det|.
  static void FactorLU(uint64_t nsites) {
    Ensure(nsites, /*need_inv=*/false);
    GridBLAS::Init();
    cublasHandle_t h = GridBLAS::gridblasHandle;
    cublasStatus_t st = cublasZgetrfBatched(
        h, N, reinterpret_cast<cuDoubleComplex **>(&Amk[0]), N,
        &pivots[0], &info[0], (int)nsites);
    if (st != CUBLAS_STATUS_SUCCESS) {
      std::cout << GridLogError
                << "[DtxqcdBlas::FactorLU] cuBLAS getrf error " << st
                << std::endl;
      abort();
    }
  }

  // getrf + getri: M_fwd (filled by caller) -> M_inv.  M_fwd is destroyed
  // (overwritten with its LU factors).  Used by the inverse cache + deriv_gpu.
  static void Invert(uint64_t nsites) {
    Ensure(nsites, /*need_inv=*/true);
    GridBLAS::Init();
    cublasHandle_t h = GridBLAS::gridblasHandle;
    cublasStatus_t st1 = cublasZgetrfBatched(
        h, N, reinterpret_cast<cuDoubleComplex **>(&Amk[0]), N,
        &pivots[0], &info[0], (int)nsites);
    cublasStatus_t st2 = cublasZgetriBatched(
        h, N, reinterpret_cast<cuDoubleComplex **>(&Amk[0]), N,
        &pivots[0], reinterpret_cast<cuDoubleComplex **>(&Cmk[0]), N,
        &info[0], (int)nsites);
    if (st1 != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
      std::cout << GridLogError
                << "[DtxqcdBlas::Invert] cuBLAS error getrf=" << st1
                << " getri=" << st2 << std::endl;
      abort();
    }
  }
};

}  // namespace DtxqcdBlas
#endif  // GRID_CUDA

NAMESPACE_END(Grid);
