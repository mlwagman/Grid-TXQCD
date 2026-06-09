#pragma once
// Full-volume M (and M^dag) operator on the DTXQCD doubled Wilson-Clover
// fermion.  Sibling of DTXQCDMpcOp.h: same interface (M, Mdag, Field
// typedef) but acting on a full-volume DTXQCDFermionDoubled instead of
// the odd-checkerboard Schur complement.
//
// Used by DTXQCDWilsonCloverRationalFullAction (non-EO 1/4-root RHMC
// pseudofermion) and by anything else that needs to call multi-shift
// CG against the full operator.  Today's PSD diagnostic
// (Test_dtxqcd_psd_check) showed that the full M is well-conditioned
// across the entire aux range that breaks the EO Schur Mpc, so this
// is the operator that any HMC variant aiming to bypass the EO Schur
// cliff should target.
//
// All of M, Mdag already live on DTXQCDWilsonCloverFermionEO (see
// member functions M and Mdag at the top of that header).  This file
// only adds the thin op-interface wrapper that multi-shift CG expects.

#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDMOp {
 public:
  typedef DTXQCDFermionDoubled Field;

  explicit DTXQCDMOp(DTXQCDWilsonCloverFermionEO &Dw) : Dw_(Dw) {}

  void M(const Field &in, Field &out)    { Dw_.M(in, out); }
  void Mdag(const Field &in, Field &out) { Dw_.Mdag(in, out); }

  DTXQCDWilsonCloverFermionEO &Wilson() { return Dw_; }

 private:
  DTXQCDWilsonCloverFermionEO &Dw_;
};

NAMESPACE_END(Grid);
