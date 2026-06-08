#pragma once
// Doubled fermion type for DTXQCD: pair of DTXQCDFermionNf (Nf=2 doublets)
// for the upper and lower components of the Cstar doubled Dirac spinor.
//
// Arithmetic is defined componentwise.  innerProduct and norm2 sum over
// the upper and lower parts, matching the natural inner product on the
// 2 x Nf x 4 x Nc x V doubled vector space.

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>

NAMESPACE_BEGIN(Grid);

struct DTXQCDFermionDoubled {
  DTXQCDFermionNf upper;
  DTXQCDFermionNf lower;

  DTXQCDFermionDoubled(GridBase *grid) : upper(grid), lower(grid) {}

  GridBase *Grid() const { return upper.Grid(); }

  DTXQCDFermionDoubled &operator=(const Zero &) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      upper.f[a] = Zero();
      lower.f[a] = Zero();
    }
    return *this;
  }
  DTXQCDFermionDoubled &operator=(const DTXQCDFermionDoubled &rhs) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      upper.f[a] = rhs.upper.f[a];
      lower.f[a] = rhs.lower.f[a];
    }
    return *this;
  }
};

inline ComplexD innerProduct(const DTXQCDFermionDoubled &a,
                             const DTXQCDFermionDoubled &b) {
  return innerProduct(a.upper, b.upper) + innerProduct(a.lower, b.lower);
}

inline RealD norm2(const DTXQCDFermionDoubled &a) {
  return norm2(a.upper) + norm2(a.lower);
}

NAMESPACE_END(Grid);
