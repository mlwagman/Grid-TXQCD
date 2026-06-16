#pragma once
// DTXQCD Wilson hopping (Meooe) on the doubled fermion.
//
// Upper block: standard Wilson hopping with gauge field U.
//   D_hop_upper[psi](x) = -(1/2) Σ_μ [(r - γ_μ) U_μ(x) psi(x+μ)
//                                    + (r + γ_μ) U_μ^†(x-μ) psi(x-μ)]
//
// Lower block — Cstar M_22 = C^T D_qcd^T C applied to the hopping piece.
// Derivation:  D_hop^T transposes (r ± γ_μ) -> (r ± γ_μ^T) and reverses the
// shift index; C^T (r ± γ_μ^T) C = (r ∓ γ_μ) (using C γ_μ^T C = γ_μ and
// C^T = -C from dtxqcd.tex).  Reorganizing the forward/backward shifts:
//   D_hop_lower[psi](x) = -(1/2) Σ_μ [(r - γ_μ) U_μ^*(x) psi(x+μ)
//                                    + (r + γ_μ) (U_μ^*(x-μ))^† psi(x-μ)]
// — identical to stock Wilson on the C-conjugated gauge field U^*.
//
// Implementation: hold two WilsonFermion engines (one per block), each fed
// the appropriate gauge field (U for upper, conjugate(U) for lower).  The
// Wilson hopping then proceeds independently per flavor of the Nf=2 doublet.

#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/qcd/action/fermion/WilsonImpl.h>

NAMESPACE_BEGIN(Grid);

// Build the C-conjugated gauge field U^* for the lower-block Wilson
// hopping.  conjugate() is a component-wise complex conjugation of every
// link's color entries — leaves the SU(N) structure intact since
// SU(N)^* = SU(N) under entry-wise conjugation.
inline LatticeGaugeField DtxqcdConjugateGauge(const LatticeGaugeField &U) {
  LatticeGaugeField Uc(U.Grid());
  Uc = conjugate(U);
  return Uc;
}

// Doubled Meooe wrapper: two WilsonFermion engines, one per block.
//
// Owns the conjugated gauge field; the upper-block gauge field lifetime is
// the caller's responsibility (same as TXQCDWilsonOp).
class DTXQCDMeooeDoubled {
 public:
  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;

  // APBC time (chroma physics convention).  Both block engines must agree.
  static typename Impl::ImplParams DefaultImplParams() {
    typename Impl::ImplParams p;
    p.boundary_phases.resize(Nd, 1.0);
    p.boundary_phases[Nd - 1] = -1.0;
    return p;
  }

  DTXQCDMeooeDoubled(LatticeGaugeField &U_upper,
                     GridCartesian &grid,
                     GridRedBlackCartesian &rbgrid,
                     RealD mass,
                     typename Impl::ImplParams impl_p = DefaultImplParams())
      : U_conj_(DtxqcdConjugateGauge(U_upper)),
        Dw_upper_(U_upper, grid, rbgrid, mass, impl_p),
        Dw_lower_(U_conj_, grid, rbgrid, mass, impl_p) {}

  // Apply the doubled Meooe hopping per block, per flavor.
  // Checkerboard semantics follow Grid's WilsonFermion::Meooe (in/out must
  // live on opposite checkerboards).
  void Meooe(const DTXQCDFermionNf &in_upper,
             const DTXQCDFermionNf &in_lower,
             DTXQCDFermionNf &out_upper,
             DTXQCDFermionNf &out_lower) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      Dw_upper_.Meooe(in_upper.f[a], out_upper.f[a]);
      Dw_lower_.Meooe(in_lower.f[a], out_lower.f[a]);
    }
  }

  void MeooeDag(const DTXQCDFermionNf &in_upper,
                const DTXQCDFermionNf &in_lower,
                DTXQCDFermionNf &out_upper,
                DTXQCDFermionNf &out_lower) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      Dw_upper_.MeooeDag(in_upper.f[a], out_upper.f[a]);
      Dw_lower_.MeooeDag(in_lower.f[a], out_lower.f[a]);
    }
  }

  // Refresh the conjugated gauge field after an external update to U.
  // The upper-block Dw_upper_ already sees U directly via its stored
  // reference; the lower-block Dw_lower_ must be re-imported with the
  // new U_conj.
  void ImportGauge(LatticeGaugeField &U_upper) {
    U_conj_ = DtxqcdConjugateGauge(U_upper);
    Dw_upper_.ImportGauge(U_upper);
    Dw_lower_.ImportGauge(U_conj_);
  }

  WilsonOp &UpperWilson() { return Dw_upper_; }
  WilsonOp &LowerWilson() { return Dw_lower_; }
  const LatticeGaugeField &ConjugatedGauge() const { return U_conj_; }

 private:
  LatticeGaugeField U_conj_;
  WilsonOp Dw_upper_;
  WilsonOp Dw_lower_;
};

NAMESPACE_END(Grid);
