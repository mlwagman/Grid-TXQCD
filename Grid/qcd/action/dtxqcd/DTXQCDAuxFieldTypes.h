#pragma once
// DTXQCD auxiliary-field site and Lattice typedefs.
//
// Five aux fields per site x (dtxqcd.tex Eqs. 122-130, 297):
//   sigma^A(x)        -- 3 real Pauli components (traceless flavor, scalar)
//   pi^A(x)           -- 3 real Pauli components (traceless flavor, gamma5-odd)
//   t^A_{mu nu}(x)    -- 3 real Pauli components per antisymmetric (mu,nu) pair
//   d^{ij}(x)         -- Nc x Nc Hermitian color (off-diagonal in doubled op)
//   n^{ij}(x)         -- Nc x Nc Hermitian color (off-diagonal in doubled op)
//
// Storage:
//   sigma^A, pi^A : iScalar<iScalar<iVector<vComplex, 3>>>
//   t^A_{mu,nu}   : iScalar<iMatrix<iVector<vComplex, 3>, Nd>>
//                   (Nd^2 Lorentz pairs; antisym in (mu,nu), diagonal zero;
//                    invariants enforced by DTXQCDCompositeImpl projectors)
//   d, n          : iScalar<iScalar<iMatrix<vComplex, Nc>>>
//                   (same layout as TXQCD's s, p)
//
// Pauli triplets are stored as vComplex iVectors so Grid's gaussian(),
// arithmetic, norm2, lattice IO, and SIMD vectorization all work unchanged.
// The physical DOF is the REAL part of each component; the imaginary part is
// held at zero by RealProjectInPlace (in DTXQCDCompositeImpl) after every
// generation, projection, force update. This trades a 2x storage overhead and
// the wasted half of each Gaussian sample for full interop with Grid.

NAMESPACE_BEGIN(Grid);

// Compile-time flavor count.  Default Nf=2 (u,d) matches TXQCD and the Pauli
// generator-set assumption below.  Nf>2 needs a different traceless-Hermitian
// basis (SU(Nf) has Nf^2-1 generators); we hard-code 3 for now.
#ifndef DTXQCD_Nf
#define DTXQCD_Nf 2
#endif
static constexpr int DtxqcdNf = DTXQCD_Nf;

// Number of independent Pauli-basis components: tau^A, A = 1..3 for Nf=2.
// = Nf^2 - 1 (the dimension of the SU(Nf) algebra) for general Nf.
static constexpr int DtxqcdNTriplet = DtxqcdNf * DtxqcdNf - 1;

// -----------------------------------------------------------------------
// Site tensors
// -----------------------------------------------------------------------

template <class vtype>
using DtxqcdSiteTriplet = iScalar<iScalar<iVector<vtype, DtxqcdNTriplet>>>;

template <class vtype>
using DtxqcdSiteTensor = iScalar<iMatrix<iVector<vtype, DtxqcdNTriplet>, Nd>>;

template <class vtype>
using DtxqcdSiteColorMatrix = iScalar<iScalar<iMatrix<vtype, Nc>>>;

// -----------------------------------------------------------------------
// Lattice types
// -----------------------------------------------------------------------

template <class Simd>
using DtxqcdLatticeTriplet = Lattice<DtxqcdSiteTriplet<Simd>>;

template <class Simd>
using DtxqcdLatticeTensor = Lattice<DtxqcdSiteTensor<Simd>>;

template <class Simd>
using DtxqcdLatticeColorMatrix = Lattice<DtxqcdSiteColorMatrix<Simd>>;

// -----------------------------------------------------------------------
// Default instantiations (vComplex SIMD)
// -----------------------------------------------------------------------

typedef DtxqcdSiteTriplet<vComplex>     DtxqcdSiteSigma;
typedef DtxqcdSiteTriplet<vComplex>     DtxqcdSitePi;
typedef DtxqcdSiteTensor<vComplex>      DtxqcdSiteT;
typedef DtxqcdSiteColorMatrix<vComplex> DtxqcdSiteD;
typedef DtxqcdSiteColorMatrix<vComplex> DtxqcdSiteN;

typedef DtxqcdLatticeTriplet<vComplex>     LatticeDtxqcdSigma;
typedef DtxqcdLatticeTriplet<vComplex>     LatticeDtxqcdPi;
typedef DtxqcdLatticeTensor<vComplex>      LatticeDtxqcdT;
typedef DtxqcdLatticeColorMatrix<vComplex> LatticeDtxqcdD;
typedef DtxqcdLatticeColorMatrix<vComplex> LatticeDtxqcdN;

NAMESPACE_END(Grid);
