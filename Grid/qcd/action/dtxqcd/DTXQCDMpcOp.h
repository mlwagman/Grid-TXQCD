#pragma once
// Schur-complement (Mpc) operator on the odd sub-lattice for the DTXQCD
// doubled Wilson-Clover fermion.
//
//   Mpc psi_o = Mooee psi_o - Meooe (MooeeInv (Meooe psi_o))
//             = (M_oo - M_oe M_ee^{-1} M_eo) psi_o
//
// Used by the rational-EO pseudofermion action: integrating M_ee out via
// the LogDet action leaves Mpc on the odd block.  The RHMC pseudofermion
// represents det(Mpc^dag Mpc)^{1/4} = |det(Mpc)|^{1/2} = |Pf(D)|/|det(M_ee)|^{1/2}.
//
// Wraps DTXQCDWilsonCloverFermionEO -- all of M_oo, M_ee^{-1}, M_eo / M_oe
// machinery (including doubled Mooee, per-site 48x48 LU inverse, and Wilson
// hopping with the C-conjugated lower gauge field) already lives there.

#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDMpcOp {
 public:
  typedef DTXQCDFermionDoubled Field;

  explicit DTXQCDMpcOp(DTXQCDWilsonCloverFermionEO &Dw) : Dw_(Dw) {}

  // Mpc psi_o on the odd sub-lattice (in / out both Odd-checkerboarded).
  void M(const Field &in, Field &out) {
    GridBase *grid = in.Grid();
    Field tmp_e(grid), tmp_e2(grid), tmp_o(grid);

    Dw_.Meooe(in, tmp_e);          // M_eo psi_o  (odd -> even)
    Dw_.MooeeInv(tmp_e, tmp_e2);   // M_ee^{-1} (...)
    Dw_.Meooe(tmp_e2, tmp_o);      // M_oe (...)
    Dw_.Mooee(in, out);            // M_oo psi_o
    SubInPlace(out, tmp_o);        // out -= tmp_o
  }

  // Mpc^dag psi_o.  Mpc is generally non-Hermitian; the daggered form swaps
  // each of Mooee/MooeeInv/Meooe for its Dag counterpart and reverses the
  // composition order (which falls out naturally from the same identity).
  void Mdag(const Field &in, Field &out) {
    GridBase *grid = in.Grid();
    Field tmp_e(grid), tmp_e2(grid), tmp_o(grid);

    Dw_.MeooeDag(in, tmp_e);
    Dw_.MooeeInvDag(tmp_e, tmp_e2);
    Dw_.MeooeDag(tmp_e2, tmp_o);
    Dw_.MooeeDag(in, out);
    SubInPlace(out, tmp_o);
  }

  DTXQCDWilsonCloverFermionEO &Wilson() { return Dw_; }

 private:
  static void SubInPlace(Field &x, const Field &y) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = x.upper.f[a] - y.upper.f[a];
      x.lower.f[a] = x.lower.f[a] - y.lower.f[a];
    }
  }

  DTXQCDWilsonCloverFermionEO &Dw_;
};

NAMESPACE_END(Grid);
