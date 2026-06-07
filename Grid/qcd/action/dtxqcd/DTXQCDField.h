#pragma once
// Composite Field type for DTXQCD HMC.
//
// Bundles the gauge field U together with the five DTXQCD auxiliary fields
// (sigma^A, pi^A, t^A_{mu,nu}, d^{ij}, n^{ij}) so the HMC Integrator (which
// is templated on one FieldImplementation / one Field) can drive them with
// one shared momentum P of the same composite type.
//
// Interface contract (mirrors TXQCDField; see that file for the integrator
// surface area expected by Grid/qcd/hmc/integrators/Integrator.h and HMC.h).

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

struct DTXQCDField {
  LatticeGaugeField   U;      // gauge links
  LatticeDtxqcdSigma  sigma;  // sigma^A traceless flavor scalar
  LatticeDtxqcdPi     pi;     // pi^A    traceless flavor pseudoscalar
  LatticeDtxqcdT      t;      // t^A_{mu,nu} traceless flavor antisym tensor
  LatticeDtxqcdD      d;      // d^{ij} Hermitian color (off-diagonal)
  LatticeDtxqcdN      n;      // n^{ij} Hermitian color (off-diagonal)

  DTXQCDField(GridBase *grid)
      : U(grid), sigma(grid), pi(grid), t(grid), d(grid), n(grid) {}

  GridBase *Grid() const { return U.Grid(); }

  DTXQCDField &operator=(const DTXQCDField &rhs) {
    U = rhs.U;
    sigma = rhs.sigma;
    pi = rhs.pi;
    t = rhs.t;
    d = rhs.d;
    n = rhs.n;
    return *this;
  }

  DTXQCDField &operator=(const Zero &z) {
    U = Zero();
    sigma = Zero();
    pi = Zero();
    t = Zero();
    d = Zero();
    n = Zero();
    return *this;
  }

  DTXQCDField &operator+=(const DTXQCDField &rhs) {
    U += rhs.U;
    sigma += rhs.sigma;
    pi += rhs.pi;
    t += rhs.t;
    d += rhs.d;
    n += rhs.n;
    return *this;
  }

  DTXQCDField &operator-=(const DTXQCDField &rhs) {
    U -= rhs.U;
    sigma -= rhs.sigma;
    pi -= rhs.pi;
    t -= rhs.t;
    d -= rhs.d;
    n -= rhs.n;
    return *this;
  }

  DTXQCDField &operator*=(double ep) {
    U = U * ep;
    sigma = sigma * ep;
    pi = pi * ep;
    t = t * ep;
    d = d * ep;
    n = n * ep;
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
       + norm2(F.t) + norm2(F.d) + norm2(F.n);
}

inline RealD maxLocalNorm2(const DTXQCDField &F) {
  RealD m = maxLocalNorm2(F.U);
  m = std::max(m, maxLocalNorm2(F.sigma));
  m = std::max(m, maxLocalNorm2(F.pi));
  m = std::max(m, maxLocalNorm2(F.t));
  m = std::max(m, maxLocalNorm2(F.d));
  m = std::max(m, maxLocalNorm2(F.n));
  return m;
}

NAMESPACE_END(Grid);
