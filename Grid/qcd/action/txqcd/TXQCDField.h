#pragma once
// Composite Field type for TXQCD HMC.
//
// Bundles the gauge field U together with the auxiliary fields
// (sigma, pi, s, p, t) so that the HMC Integrator -- which is templated on a
// single FieldImplementation and a single Field -- can drive them with one
// shared momentum P (of the same composite type).
//
// The Integrator (Grid/qcd/hmc/integrators/Integrator.h) requires the Field
// to support:
//   - Field F(GridBase*);           // ctor
//   - F.Grid() -> GridBase*;        // grid accessor
//   - F = Zero();                   // zero-assign
//   - F = G; (copy)
//   - F += G; F -= G; F + G; F * scalar;
//   - norm2(F) -> RealD;
//   - maxLocalNorm2(F) -> RealD;
//
// The HMC driver (Grid/qcd/hmc/HMC.h) additionally uses:
//   - Field F(Ucur.Grid());         // ctor from another field's grid
//   - F = G;                        // copy-assign
//
// All operations delegate to the underlying Lattice components.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

struct TXQCDField {
  LatticeGaugeField U;      // gauge links
  LatticeSigmaField sigma;  // Nf x Nf Hermitian flavor
  LatticePiField    pi;     // Nf x Nf Hermitian flavor, gamma5-odd
  LatticeSFieldC    s;      // Nc x Nc Hermitian color
  LatticePFieldC    p;      // Nc x Nc Hermitian color, gamma5-odd
  LatticeTField     t;      // Nc x Nc Hermitian color, antisym (mu,nu)

  TXQCDField(GridBase *grid)
      : U(grid), sigma(grid), pi(grid), s(grid), p(grid), t(grid) {}

  // Required by HMC.h line 251: Field Ucopy(Ucur.Grid()).
  GridBase *Grid() const { return U.Grid(); }

  // Copy-assign: HMC.h line 268 Ucopy = Ucur.
  TXQCDField &operator=(const TXQCDField &rhs) {
    U = rhs.U;
    sigma = rhs.sigma;
    pi = rhs.pi;
    s = rhs.s;
    p = rhs.p;
    t = rhs.t;
    return *this;
  }

  // Zero-assign: Integrator.h line 131 level_force = Zero().
  TXQCDField &operator=(const Zero &z) {
    U = Zero();
    sigma = Zero();
    pi = Zero();
    s = Zero();
    p = Zero();
    t = Zero();
    return *this;
  }

  TXQCDField &operator+=(const TXQCDField &rhs) {
    U += rhs.U;
    sigma += rhs.sigma;
    pi += rhs.pi;
    s += rhs.s;
    p += rhs.p;
    t += rhs.t;
    return *this;
  }

  TXQCDField &operator-=(const TXQCDField &rhs) {
    U -= rhs.U;
    sigma -= rhs.sigma;
    pi -= rhs.pi;
    s -= rhs.s;
    p -= rhs.p;
    t -= rhs.t;
    return *this;
  }

  TXQCDField &operator*=(double ep) {
    U = U * ep;
    sigma = sigma * ep;
    pi = pi * ep;
    s = s * ep;
    p = p * ep;
    t = t * ep;
    return *this;
  }
};

// Binary operators built from the compound assignments.
inline TXQCDField operator+(const TXQCDField &a, const TXQCDField &b) {
  TXQCDField r(a.Grid());
  r = a;
  r += b;
  return r;
}

inline TXQCDField operator-(const TXQCDField &a, const TXQCDField &b) {
  TXQCDField r(a.Grid());
  r = a;
  r -= b;
  return r;
}

inline TXQCDField operator*(const TXQCDField &a, double ep) {
  TXQCDField r(a.Grid());
  r = a;
  r *= ep;
  return r;
}

inline TXQCDField operator*(double ep, const TXQCDField &a) { return a * ep; }

// norm2 and maxLocalNorm2: sum over components. maxLocalNorm2 takes the max
// across components so the "Force max" diagnostic printed by the Integrator
// is meaningful for the dominant component.
inline RealD norm2(const TXQCDField &F) {
  return norm2(F.U) + norm2(F.sigma) + norm2(F.pi) + norm2(F.s) + norm2(F.p) +
         norm2(F.t);
}

inline RealD maxLocalNorm2(const TXQCDField &F) {
  RealD m = maxLocalNorm2(F.U);
  m = std::max(m, maxLocalNorm2(F.sigma));
  m = std::max(m, maxLocalNorm2(F.pi));
  m = std::max(m, maxLocalNorm2(F.s));
  m = std::max(m, maxLocalNorm2(F.p));
  m = std::max(m, maxLocalNorm2(F.t));
  return m;
}

NAMESPACE_END(Grid);
