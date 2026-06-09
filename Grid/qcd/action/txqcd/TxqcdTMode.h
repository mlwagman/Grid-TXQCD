#pragma once
// Compile-time helpers for the TXQCD_T_FLAVOR toggle.
//
// Two TXQCD variants are supported, selected at compile time by the
// TXQCD_T_FLAVOR macro defined in AuxFieldTypes.h:
//
//   Mode A (TXQCD_T_FLAVOR = 0, default): t_{mu,nu} is an Nc x Nc color
//     tensor.  In BuildSiteMatrix, t couples to the same Dirac structure as
//     the clover term (F_{mu,nu} . i sigma_{mu,nu}).  Pre-factors:
//       sigma, pi : 1
//       s, p      : 1/sqrt(2)
//       t         : 1
//
//   Mode B (TXQCD_T_FLAVOR = 1): t_{mu,nu} is an Nf x Nf flavor tensor,
//     stored in the upper-left Nf x Nf block of the existing Nc x Nc
//     storage; inactive color slots stay zero by invariant.  Pre-factors
//     swap:
//       sigma, pi : 1/sqrt(2)
//       s, p      : 1
//       t         : 1
//
// In both modes t storage type is unchanged (iScalar<iMatrix<iMatrix<v,Nc>,Nd>>)
// so the field algebra (norm2, +=, RNG, GPU pack/unpack, etc.) is mode-agnostic.
//
// Mode-dependent constants live in AuxFieldTypes.h:
//   TxqcdTIsFlavor : true if mode B
//   TxqcdTDim      : Nf in mode B, Nc in mode A  (block dimension for t loops)

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

// Returns true if t's Dirac coupling (via i sigma_{mu,nu}) shares its
// (color x Lorentz) tensor channel with the clover term.  This is the case
// only in color-t mode.  In flavor-t mode the t-derived gauge force through
// the shared clover channel is identically zero and must be skipped.
static constexpr bool TxqcdTSharesCloverDirac() {
  return !TxqcdTIsFlavor;
}

NAMESPACE_END(Grid);
