#pragma once
// Clover term application for the TXQCD Wilson-Clover operator.
// Applies -(csw/2) * sum_{mu<nu} FS_{mu,nu} * (i*sigma_{mu,nu}) * in.
// Flavor-diagonal. Used by TXQCDWilsonCloverFermionEO.

#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>

NAMESPACE_BEGIN(Grid);

// Apply -(csw/2) * sum_{μ<ν} F_{μν} σ_{μν} v.  Same fused-accelerator_for
// approach as ApplyDeltaColor: pre-rotate v through the 6 Gamma σ_{μν}
// (which is GPU-accelerated lattice arithmetic) and then collapse the
// FS color-mat × v_spin sums into a single per-site kernel.  Avoids 12
// in-loop LatticeFermion allocations and 12 sequential Grid kernels per
// call, both major contributors to MultiShift CG cost.
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
  GridBase *grid = in.Grid();
  out.f[0].Checkerboard() = cb;
  out.f[1].Checkerboard() = cb;

  // Pre-rotate input through the 6 σ_{μν} (μ<ν) per flavor.
  std::vector<LatticeFermion> smn_in;
  smn_in.reserve(6 * TxqcdNf);
  for (int a = 0; a < TxqcdNf; ++a) {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        smn_in.emplace_back(grid);
        smn_in.back() = smn * in.f[a];
      }
    }
  }

  // FS layout: 6 LatticeColourMatrix indexed by (μ<ν) pair.  Same order as
  // SMU::FmnIndex (XY=0, XZ=1, XT=2, YZ=3, YT=4, ZT=5).
  autoView(F0v, FS[0], AcceleratorRead);
  autoView(F1v, FS[1], AcceleratorRead);
  autoView(F2v, FS[2], AcceleratorRead);
  autoView(F3v, FS[3], AcceleratorRead);
  autoView(F4v, FS[4], AcceleratorRead);
  autoView(F5v, FS[5], AcceleratorRead);
  autoView(s00v, smn_in[0], AcceleratorRead);
  autoView(s01v, smn_in[1], AcceleratorRead);
  autoView(s02v, smn_in[2], AcceleratorRead);
  autoView(s03v, smn_in[3], AcceleratorRead);
  autoView(s04v, smn_in[4], AcceleratorRead);
  autoView(s05v, smn_in[5], AcceleratorRead);
  autoView(s10v, smn_in[6], AcceleratorRead);
  autoView(s11v, smn_in[7], AcceleratorRead);
  autoView(s12v, smn_in[8], AcceleratorRead);
  autoView(s13v, smn_in[9], AcceleratorRead);
  autoView(s14v, smn_in[10], AcceleratorRead);
  autoView(s15v, smn_in[11], AcceleratorRead);
  autoView(out0v, out.f[0], AcceleratorWrite);
  autoView(out1v, out.f[1], AcceleratorWrite);

  const RealD neg_csw_half = -0.5 * csw;
  const int Nsimd = LatticeFermion::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto F0 = F0v(ss); auto F1 = F1v(ss); auto F2 = F2v(ss);
    auto F3 = F3v(ss); auto F4 = F4v(ss); auto F5 = F5v(ss);
    typedef typename std::remove_cv<typename std::remove_reference<decltype(s00v(ss))>::type>::type FermSitePerLane;
    FermSitePerLane out0_acc, out1_acc;
    for (int a = 0; a < TxqcdNf; ++a) {
      auto smn0 = (a == 0) ? s00v(ss) : s10v(ss);
      auto smn1 = (a == 0) ? s01v(ss) : s11v(ss);
      auto smn2 = (a == 0) ? s02v(ss) : s12v(ss);
      auto smn3 = (a == 0) ? s03v(ss) : s13v(ss);
      auto smn4 = (a == 0) ? s04v(ss) : s14v(ss);
      auto smn5 = (a == 0) ? s05v(ss) : s15v(ss);
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          decltype(F0()()(i, 0) * smn0()(alpha)(0)) val;
          zeroit(val);
          for (int j = 0; j < Nc; ++j) {
            val = val + F0()()(i, j) * smn0()(alpha)(j)
                      + F1()()(i, j) * smn1()(alpha)(j)
                      + F2()()(i, j) * smn2()(alpha)(j)
                      + F3()()(i, j) * smn3()(alpha)(j)
                      + F4()()(i, j) * smn4()(alpha)(j)
                      + F5()()(i, j) * smn5()(alpha)(j);
          }
          val = neg_csw_half * val;
          if (a == 0) out0_acc()(alpha)(i) = val;
          else        out1_acc()(alpha)(i) = val;
        }
      }
    }
    coalescedWrite(out0v[ss], out0_acc);
    coalescedWrite(out1v[ss], out1_acc);
  });
}

NAMESPACE_END(Grid);
