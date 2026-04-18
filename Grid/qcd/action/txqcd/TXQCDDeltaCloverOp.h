#pragma once
// Clover term application for the TXQCD Wilson-Clover operator.
// Applies -(csw/2) * sum_{mu<nu} FS_{mu,nu} * (i*sigma_{mu,nu}) * in.
// Flavor-diagonal. Used by TXQCDWilsonCloverFermionEO.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

inline void ApplyClover(RealD csw,
                        const std::vector<LatticeColourMatrix> &FS,
                        const TXQCDFermionNf &in, TXQCDFermionNf &out) {
  int cb = in.f[0].Checkerboard();
  if (csw == 0.0) {
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = Zero();
      out.f[a].Checkerboard() = cb;
    }
    return;
  }
  const RealD neg_csw_half = -0.5 * csw;
  int k = 0;
  for (int a = 0; a < TxqcdNf; ++a) {
    out.f[a] = Zero();
    out.f[a].Checkerboard() = cb;
  }
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu) {
      Gamma smn(SigmaMuNuAlgebra(mu, nu));
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion smn_v(in.Grid());
        smn_v = smn * in.f[a];
        out.f[a] = out.f[a] + neg_csw_half * (FS[k] * smn_v);
      }
      ++k;
    }
  }
}

NAMESPACE_END(Grid);
