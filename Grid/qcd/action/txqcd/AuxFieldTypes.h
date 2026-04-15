#pragma once
// TXQCD auxiliary-field site and Lattice typedefs.
//
// Five aux fields per site x (notes Eq. 4):
//   sigma_{ab}(x)    -- Nf x Nf Hermitian, flavor
//   pi_{ab}(x)       -- Nf x Nf Hermitian, flavor (gamma5-odd)
//   s^{ij}(x)        -- Nc x Nc Hermitian, color
//   p^{ij}(x)        -- Nc x Nc Hermitian, color (gamma5-odd)
//   t^{ij}_{mu,nu}(x)-- Nc x Nc Hermitian color, antisymmetric in (mu,nu)
//
// Site layout follows ScalarAdjMatrixImplTypes (Grid/qcd/action/scalar/ScalarImpl.h:116):
//   iScalar<iScalar<iMatrix<vtype,N>>>.
// For the tensor field we nest one further iVector<iVector<...,Nd>,Nd> to hold
// the (mu,nu) pair explicitly (antisymmetry is a user-side invariant).

NAMESPACE_BEGIN(Grid);

// Compile-time flavor count for TXQCD. Nf=2 (u,d) is the Phase-3/4 target.
#ifndef TXQCD_Nf
#define TXQCD_Nf 2
#endif
static constexpr int TxqcdNf = TXQCD_Nf;

// -----------------------------------------------------------------------
// Site tensors
// -----------------------------------------------------------------------

template <class vtype, int N>
using TxqcdSiteHermMatrix = iScalar<iScalar<iMatrix<vtype, N>>>;

template <class vtype>
using TxqcdSiteFlavorMatrix = TxqcdSiteHermMatrix<vtype, TxqcdNf>;

template <class vtype>
using TxqcdSiteColorMatrix = TxqcdSiteHermMatrix<vtype, Nc>;

// Antisymmetric Lorentz pair of Nc x Nc color matrices:
//   t(x)[mu][nu]  with t[mu][nu] = -t[nu][mu], diagonal zero.
// We store the full 4x4 block; correctness relies on the TXQCD code only ever
// writing the upper triangle and antisymmetrizing.
template <class vtype>
using TxqcdSiteTensorField =
    iScalar<iMatrix<iMatrix<vtype, Nc>, Nd>>;

// -----------------------------------------------------------------------
// Lattice types (parameterized by SIMD complex vector type)
// -----------------------------------------------------------------------

template <class Simd>
using TxqcdLatticeFlavorMatrix = Lattice<TxqcdSiteFlavorMatrix<Simd>>;

template <class Simd>
using TxqcdLatticeColorMatrix = Lattice<TxqcdSiteColorMatrix<Simd>>;

template <class Simd>
using TxqcdLatticeTensorField = Lattice<TxqcdSiteTensorField<Simd>>;

// -----------------------------------------------------------------------
// Default instantiations (vComplex SIMD)
// -----------------------------------------------------------------------

typedef TxqcdSiteFlavorMatrix<vComplex>   TxqcdSiteSigma;
typedef TxqcdSiteFlavorMatrix<vComplex>   TxqcdSitePi;
typedef TxqcdSiteColorMatrix<vComplex>    TxqcdSiteS;
typedef TxqcdSiteColorMatrix<vComplex>    TxqcdSiteP;
typedef TxqcdSiteTensorField<vComplex>    TxqcdSiteT;

typedef TxqcdLatticeFlavorMatrix<vComplex> LatticeSigmaField;
typedef TxqcdLatticeFlavorMatrix<vComplex> LatticePiField;
typedef TxqcdLatticeColorMatrix<vComplex>  LatticeSFieldC;
typedef TxqcdLatticeColorMatrix<vComplex>  LatticePFieldC;
typedef TxqcdLatticeTensorField<vComplex>  LatticeTField;

NAMESPACE_END(Grid);
