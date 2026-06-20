#pragma once
// DTXQCD v2 auxiliary-field site and Lattice typedefs.
//
// The v2 derivation (dtxqcd_v2.tex Eqs. 116-128, 297-308) replaces the v1
// roster (Pauli-triplet sigma^A/pi^A/t^A, color-only d/n) with six fields
// whose Fierz-cancellation is sum-of-squares (no negative tensor term):
//
//   sigma^{ij}_{ab}(x)  -- traceless Hermitian 6x6 in color (i,j) x flavor (a,b)
//   pi^{ij}_{ab}(x)     -- same, gamma5-odd
//   d^{ij}_{ab}(x)      -- same, off-diagonal in doubled spinor block
//   n^{ij}_{ab}(x)      -- same, off-diagonal in doubled spinor block
//   s(x)                -- singlet scalar (real)
//   p(x)                -- singlet scalar (real), gamma5-odd
//
// The diagonal piece is X^{ij}_{ab} = sigma + s delta^{ij} delta_{ab}
//                                     + (pi + p delta^{ij} delta_{ab}) gamma5.
// Upper block uses +X; lower block (Cstar-conjugated) uses -X. No tensor.
//
// Storage:
//   sigma, pi, d, n : iScalar<iMatrix<iMatrix<vComplex, Nc>, Nf>>
//                     -- 36 complex = 72 reals/site uncompressed.
//                        After every update DTXQCDCompositeImpl applies
//                        HermitizeAndTracelessInPlace to enforce 35 real
//                        DOFs (Hermitian under combined (i,a)<->(j,b)
//                        transpose + complex conjugation, traceless on
//                        the 6x6 combined index).
//   s, p           : iScalar<iScalar<iScalar<vComplex>>>
//                    -- 1 complex = 2 reals/site uncompressed.
//                       Imaginary part held at zero by RealProjectInPlace.

NAMESPACE_BEGIN(Grid);

// Compile-time flavor count.  Default Nf=2 (u,d).  The 6x6 traceless
// Hermitian structure is independent of basis choice, so the SU(Nf) and
// SU(Nc) generator algebras are not hard-coded here -- only Nf, Nc enter
// the dimensions.
#ifndef DTXQCD_Nf
#define DTXQCD_Nf 2
#endif
static constexpr int DtxqcdNf = DTXQCD_Nf;

// Combined color x flavor dimension (= 6 for Nc=3, Nf=2).
static constexpr int DtxqcdNfNc = DtxqcdNf * Nc;

// ---- Portable complex helpers (CPU std::complex / GPU thrust::complex) ----
// Under GRID_CUDA, Grid's ComplexD is thrust::complex<double>, for which
// std::conj/abs/arg have no overload and which cannot be mixed with Eigen's
// std::complex<double> matrices.  These helpers work on both and are defined
// in this foundational header so every DTXQCD TU (ops, actions, tests) sees
// them.  DtxqcdToStd converts a Grid complex into the std::complex<double>
// expected by the per-site Eigen matrices in DTXQCDSiteMatrix.h.
inline std::complex<double> DtxqcdToStd(const ComplexD &z) {
  return std::complex<double>(z.real(), z.imag());
}
accelerator_inline ComplexD DtxqcdConj(const ComplexD &z) {
  return ComplexD(z.real(), -z.imag());
}
inline double DtxqcdAbs(const ComplexD &z) {
  return std::sqrt(z.real() * z.real() + z.imag() * z.imag());
}
inline double DtxqcdArg(const ComplexD &z) {
  return std::atan2(z.imag(), z.real());
}

// -----------------------------------------------------------------------
// Site tensors
// -----------------------------------------------------------------------

// Color x flavor matrix: outer flavor index (a,b), inner color (i,j).
// Indexing: M.f[a][b](i, j)  -> sigma^{ij}_{ab} when contracted via
// SiteCFMatrix accessor below.
template <class vtype>
using DtxqcdSiteCFMatrix = iScalar<iMatrix<iMatrix<vtype, Nc>, DtxqcdNf>>;

// Singlet scalar (real DOF lives in real part).
template <class vtype>
using DtxqcdSiteScalar = iScalar<iScalar<iScalar<vtype>>>;

// -----------------------------------------------------------------------
// Lattice types
// -----------------------------------------------------------------------

template <class Simd>
using DtxqcdLatticeCFMatrix = Lattice<DtxqcdSiteCFMatrix<Simd>>;

template <class Simd>
using DtxqcdLatticeScalar = Lattice<DtxqcdSiteScalar<Simd>>;

// -----------------------------------------------------------------------
// Default instantiations (vComplex SIMD)
// -----------------------------------------------------------------------

typedef DtxqcdSiteCFMatrix<vComplex>     DtxqcdSiteSigma;
typedef DtxqcdSiteCFMatrix<vComplex>     DtxqcdSitePi;
typedef DtxqcdSiteCFMatrix<vComplex>     DtxqcdSiteD;
typedef DtxqcdSiteCFMatrix<vComplex>     DtxqcdSiteN;
typedef DtxqcdSiteScalar<vComplex>       DtxqcdSiteS;
typedef DtxqcdSiteScalar<vComplex>       DtxqcdSiteP;

typedef DtxqcdLatticeCFMatrix<vComplex>  LatticeDtxqcdSigma;
typedef DtxqcdLatticeCFMatrix<vComplex>  LatticeDtxqcdPi;
typedef DtxqcdLatticeCFMatrix<vComplex>  LatticeDtxqcdD;
typedef DtxqcdLatticeCFMatrix<vComplex>  LatticeDtxqcdN;
typedef DtxqcdLatticeScalar<vComplex>    LatticeDtxqcdS;
typedef DtxqcdLatticeScalar<vComplex>    LatticeDtxqcdP;

NAMESPACE_END(Grid);
