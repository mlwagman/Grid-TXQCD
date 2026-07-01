#pragma once
// Discrete spatial parity transformation on DTXQCD fields (gauge + aux +
// doubled pseudofermion), used by the Z_2 heatbath in the HMC driver and by
// the invariance unit test.
//
// Definition
// ----------
// P: (x, y, z, t) -> (-x, -y, -z, t)  with all spatial coordinates reflected
// modulo the periodic lattice.  The physical (Z_2) transformations are:
//
//   Aux even (unchanged sign, coordinate reflected):
//     sigma^P(x, t) =  sigma(-x, t)
//     s    ^P(x, t) =  s    (-x, t)
//     d    ^P(x, t) =  d    (-x, t)
//
//   Aux odd  (sign flipped after coordinate reflection):
//     pi   ^P(x, t) = -pi   (-x, t)
//     p    ^P(x, t) = -p    (-x, t)
//     n    ^P(x, t) = -n    (-x, t)
//
//   Gauge (temporal link unchanged direction, spatial link reversed):
//     U_0^P(x, t) = U_0(-x, t)                              [temporal reflect]
//     U_k^P(x, t) = U_k( -x - k̂, t )^dagger    for k=1..3   [spatial reflect
//                                                          + shift + dagger]
//     The spatial rule comes from: the parity image of the ordered edge
//     (x → x+k̂) is (-x → -x-k̂), which is stored as U_k^dagger at the
//     start-of-image site (-x-k̂).  Grid stores U_k(y) at site y for the
//     y → y+k̂ edge, so the parity-transformed link at site z is read from
//     site -z-k̂ and daggered.
//
//   Doubled pseudofermion chi (each of upper.f[0], upper.f[1], lower.f[0],
//   lower.f[1] transforms identically):
//     chi^P(x, t) = gamma_T * chi(-x, t)     [Grid uses GammaT for the
//                                             temporal Dirac matrix.]
//
// Verification
// ------------
// See tests/dtxqcd/Test_dtxqcd_parity_invariance.cc which checks that the
// gauge Wilson action, the aux Gaussian action, and the pseudofermion
// bilinear Phi^dag M^dag M Phi are all bit-invariant on a random 4^4 config.
//
// Implementation cost
// -------------------
// This first cut uses peekSite / pokeSite for the coordinate reflection.
// That is O(V) unvectorized host work; fine for a unit test on 4^4 and for
// the Z_2 heatbath (one call per accepted trajectory), NOT suitable for
// per-step use inside the integrator.

#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>

NAMESPACE_BEGIN(Grid);
namespace DtxqcdParity {

// Reflect a lattice field under (x, y, z, t) -> (-x, -y, -z, t).
// Value-copy: out(x, t) = in(-x, t) mod L.  Works for any Grid lattice
// scalar_object; caller supplies the returned lattice sized to `in.Grid()`.
template <class Obj>
inline void SpatialReflect(const Lattice<Obj> &in, Lattice<Obj> &out) {
  typedef typename Obj::scalar_object sobj;
  GridBase *grid = in.Grid();
  assert(grid == out.Grid());
  const int Nd_ = grid->_ndimension;
  assert(Nd_ == 4);
  Coordinate gd = grid->GlobalDimensions();
  Coordinate coor(Nd_), rcoor(Nd_);
  sobj tmp;
  for (int t = 0; t < gd[3]; ++t)
    for (int z = 0; z < gd[2]; ++z)
      for (int y = 0; y < gd[1]; ++y)
        for (int x = 0; x < gd[0]; ++x) {
          coor[0] = x;  coor[1] = y;  coor[2] = z;  coor[3] = t;
          // (L - c) mod L is the periodic reflection; L-0 -> 0, L-1 -> 1, etc.
          rcoor[0] = (gd[0] - x) % gd[0];
          rcoor[1] = (gd[1] - y) % gd[1];
          rcoor[2] = (gd[2] - z) % gd[2];
          rcoor[3] = t;
          peekSite(tmp, in, rcoor);
          pokeSite(tmp, out, coor);
        }
}

// Apply the parity transformation to the gauge field U_mu(x).
//
//   U_0^P(x, t) = U_0(-x, t)              (temporal, no dagger)
//   U_k^P(x, t) = U_k(-x - k̂, t)^dagger   (spatial, reflect + shift + dagger)
//
// The +k̂ shift matches Grid's Cshift(dim=k, shift=+1) convention: a lattice
// L' with L'(x) = L(x + k̂) is obtained via Cshift(L, k, +1).
inline void ApplyGauge(LatticeGaugeField &U) {
  GridBase *grid = U.Grid();
  const int Nd_ = grid->_ndimension;
  LatticeGaugeField U_refl(grid);
  SpatialReflect(U, U_refl);              // now U_refl(x) has value U(-x)

  LatticeGaugeField Uout(grid);
  Uout = U_refl;                          // start with the reflected copy

  for (int mu = 0; mu < Nd_; ++mu) {
    if (mu == 3) continue;                // temporal: already correct via reflect
    LatticeColourMatrix Umu(grid);
    Umu = PeekIndex<LorentzIndex>(U_refl, mu);
    // Cshift(+1 in mu): Umu_shift(x) = Umu(x + mû) = [in reflected coords]
    // U(-(x + mû)) = U(-x - mû).  Then dagger site-wise.
    LatticeColourMatrix Umu_shift(grid);
    Umu_shift = Cshift(Umu, mu, +1);
    LatticeColourMatrix Umu_dag(grid);
    Umu_dag = adj(Umu_shift);
    PokeIndex<LorentzIndex>(Uout, Umu_dag, mu);
  }
  U = Uout;
}

// Apply parity to the DTXQCDField aux slots.  Gauge is handled separately by
// ApplyGauge above; this only touches the aux + gauge together for
// convenience (calls ApplyGauge internally).  Sign flips per user-specified
// parity assignments:
//    even under P: sigma, s, d
//    odd  under P: pi,    p, n
inline void ApplyField(DTXQCDField &F) {
  ApplyGauge(F.U);

  {  // sigma  (even)
    LatticeDtxqcdSigma tmp(F.sigma.Grid());
    SpatialReflect(F.sigma, tmp);
    F.sigma = tmp;
  }
  {  // pi (odd)
    LatticeDtxqcdPi tmp(F.pi.Grid());
    SpatialReflect(F.pi, tmp);
    F.pi = ComplexD(-1.0, 0.0) * tmp;
  }
  {  // d (EVEN).  d couples the doubled fermion via the Majorana-like
     // pseudoscalar bilinear q^T C γ_5 q, which is P-even because both the
     // (P-odd) γ_5 factor and the (P-odd) q^T C q transposition cancel to
     // give an overall P-even scalar in the physical Dirac basis.  The
     // sign flip that shows up in the DTXQCDDeltaOp off-diagonal block
     // instead comes from ψ_lower ≡ q^c picking up an extra minus under P
     // (γ_T C = -C γ_T so q^c → -γ_T q^c(-x)); we absorb that in the
     // ApplyFermionDoubled transformation on the lower block below.
    LatticeDtxqcdD tmp(F.d.Grid());
    SpatialReflect(F.d, tmp);
    F.d = tmp;
  }
  {  // n (ODD).
    LatticeDtxqcdN tmp(F.n.Grid());
    SpatialReflect(F.n, tmp);
    F.n = ComplexD(-1.0, 0.0) * tmp;
  }
  {  // s (even)
    LatticeDtxqcdS tmp(F.s.Grid());
    SpatialReflect(F.s, tmp);
    F.s = tmp;
  }
  {  // p (odd)
    LatticeDtxqcdP tmp(F.p.Grid());
    SpatialReflect(F.p, tmp);
    F.p = ComplexD(-1.0, 0.0) * tmp;
  }
}

// Apply parity to a single LatticeFermion component of a doubled
// pseudofermion: chi^P(x, t) = gamma_T * chi(-x, t).  gamma_T is the
// Grid temporal Dirac matrix (Gamma::Algebra::GammaT).
inline void ApplyFermion(LatticeFermion &chi) {
  LatticeFermion tmp(chi.Grid());
  SpatialReflect(chi, tmp);
  Gamma gT(Gamma::Algebra::GammaT);
  chi = gT * tmp;
}

// Apply parity to a full DTXQCDFermionDoubled.  The upper block transforms
// like a normal Dirac fermion:  chi_upper^P(x, t) = gamma_T chi_upper(-x, t).
// The lower block equals q^c = C q^bar^T (the charge-conjugated field of the
// doubled construction) and picks up an EXTRA minus sign under P:
//    chi_lower^P(x, t) = -gamma_T chi_lower(-x, t)
// derived from  C gamma_T = -gamma_T C  in the standard Euclidean C convention.
// Without this minus the off-diagonal M_{UR} = c (d gamma5 + n) block breaks
// invariance by a sign in the (d, n) sector; with it the aux P-assignments
// take their physical values: {sigma, s, d} EVEN, {pi, p, n} ODD.
inline void ApplyFermionDoubled(DTXQCDFermionDoubled &chi) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    ApplyFermion(chi.upper.f[a]);
    ApplyFermion(chi.lower.f[a]);
    chi.lower.f[a] = ComplexD(-1.0, 0.0) * chi.lower.f[a];
  }
}

}  // namespace DtxqcdParity
NAMESPACE_END(Grid);
