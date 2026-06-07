#pragma once
// Include hub for DTXQCD (Diquark-Tensor XQCD) auxiliary-field HMC machinery.
//
// Sibling of Txqcd.h.  TXQCD couples aux fields in the meson channels so the
// pion shows up at the saddle point; DTXQCD couples them in the diquark
// channels (color-Hermitian d, n; flavor-traceless sigma^A, pi^A, t^A) so the
// scalar SNR penalty lands on baryon channels — which already have severe SNR
// degradation in QCD — instead of on the pion.  Math: dtxqcd.tex.
//
// Coexists with TXQCD: the two trees can be linked into the same binary and
// the action wrappers / smearing / gauge adapters are shared (pulled from
// txqcd/).

#include <Grid/qcd/action/dtxqcd/DTXQCDAuxFieldTypes.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h>
// DTXQCDCheckpointer.h depends on NerscIO and is included separately by
// drivers (analogous to TXQCDCheckpointer.h relative to Txqcd.h).
