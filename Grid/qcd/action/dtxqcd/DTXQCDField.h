#pragma once
// Composite Field type for DTXQCD HMC (v2 roster).
//
// Bundles the gauge field U together with the six v2 auxiliary fields
// (sigma, pi, d, n -- each 6x6 color-flavor traceless Hermitian; s, p
// singlet scalars) so the HMC Integrator (templated on one
// FieldImplementation / one Field) can drive them with one shared
// momentum P of the same composite type.
//
// v1 -> v2 change: drop tensor field t; add singlet scalars s, p.
// Aux field storage shapes also change (color-flavor matrices replace
// Pauli triplets) -- see DTXQCDAuxFieldTypes.h for layout.

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

struct DTXQCDField {
  LatticeGaugeField   U;      // gauge links
  LatticeDtxqcdSigma  sigma;  // sigma^{ij}_{ab} CF traceless Hermitian
  LatticeDtxqcdPi     pi;     // pi^{ij}_{ab}    CF traceless Hermitian (gamma5)
  LatticeDtxqcdD      d;      // d^{ij}_{ab}     CF traceless Hermitian (off-diagonal, gamma5)
  LatticeDtxqcdN      n;      // n^{ij}_{ab}     CF traceless Hermitian (off-diagonal)
  LatticeDtxqcdS      s;      // s(x)            real singlet scalar
  LatticeDtxqcdP      p;      // p(x)            real singlet scalar (gamma5)

  DTXQCDField(GridBase *grid)
      : U(grid), sigma(grid), pi(grid), d(grid), n(grid), s(grid), p(grid) {}

  GridBase *Grid() const { return U.Grid(); }

  DTXQCDField &operator=(const DTXQCDField &rhs) {
    U = rhs.U;
    sigma = rhs.sigma;
    pi = rhs.pi;
    d = rhs.d;
    n = rhs.n;
    s = rhs.s;
    p = rhs.p;
    return *this;
  }

  DTXQCDField &operator=(const Zero &z) {
    U = Zero();
    sigma = Zero();
    pi = Zero();
    d = Zero();
    n = Zero();
    s = Zero();
    p = Zero();
    return *this;
  }

  DTXQCDField &operator+=(const DTXQCDField &rhs) {
    U += rhs.U;
    sigma += rhs.sigma;
    pi += rhs.pi;
    d += rhs.d;
    n += rhs.n;
    s += rhs.s;
    p += rhs.p;
    return *this;
  }

  DTXQCDField &operator-=(const DTXQCDField &rhs) {
    U -= rhs.U;
    sigma -= rhs.sigma;
    pi -= rhs.pi;
    d -= rhs.d;
    n -= rhs.n;
    s -= rhs.s;
    p -= rhs.p;
    return *this;
  }

  DTXQCDField &operator*=(double ep) {
    U = U * ep;
    sigma = sigma * ep;
    pi = pi * ep;
    d = d * ep;
    n = n * ep;
    s = s * ep;
    p = p * ep;
    return *this;
  }
};

inline DTXQCDField operator+(const DTXQCDField &a, const DTXQCDField &b) {
  DTXQCDField r(a.Grid()); r = a; r += b; return r;
}
inline DTXQCDField operator-(const DTXQCDField &a, const DTXQCDField &b) {
  DTXQCDField r(a.Grid()); r = a; r -= b; return r;
}
inline DTXQCDField operator*(const DTXQCDField &a, double ep) {
  DTXQCDField r(a.Grid()); r = a; r *= ep; return r;
}
inline DTXQCDField operator*(double ep, const DTXQCDField &a) { return a * ep; }

inline RealD norm2(const DTXQCDField &F) {
  return norm2(F.U) + norm2(F.sigma) + norm2(F.pi)
       + norm2(F.d) + norm2(F.n)
       + norm2(F.s) + norm2(F.p);
}

inline RealD maxLocalNorm2(const DTXQCDField &F) {
  RealD m = maxLocalNorm2(F.U);
  m = std::max(m, maxLocalNorm2(F.sigma));
  m = std::max(m, maxLocalNorm2(F.pi));
  m = std::max(m, maxLocalNorm2(F.d));
  m = std::max(m, maxLocalNorm2(F.n));
  m = std::max(m, maxLocalNorm2(F.s));
  m = std::max(m, maxLocalNorm2(F.p));
  return m;
}

NAMESPACE_END(Grid);
