#pragma once
// Fused per-shift kernels for TXQCDMultiShiftCGSchur, with view caching.
//
// Replaces the per-iter inner loops over nshift × Nf separate Lattice
// expressions (~80 launches per CG iter) with a single accelerator_for over
// (oSites × nshift × Nf).  Views, host→device pointer arrays, and small
// coefficient buffers are cached across the entire CG via FusedShiftCtx —
// per iter we only update coefficients and launch one kernel.
//
// Three operations covered:
//   psUpdate    : ps[s].f[a]  = α_s · r.f[a] + β_s · ps[s].f[a]
//   psiUpdate   : psi[s].f[a] += coef_s · ps[s].f[a]
//   psiInit     : psi[s].f[a]  = scale_s · src.f[a]
//
// Toggle via env TXQCD_CG_FUSED=1 in TXQCDMultiShiftCGSchur.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>

NAMESPACE_BEGIN(Grid);

namespace TxqcdFusedShift {

template <class vobj>
struct FusedShiftCtx {
  // Cached views for fields that are ONLY touched by our fused kernels
  // (ps, psi).  r and src are NOT cached — Grid Lattice expressions
  // elsewhere in the CG also open views on them, and the MemoryManager
  // disallows overlapping locks on the same field.
  std::vector<LatticeView<vobj>> ps_views;     // [nshift * Nf]
  std::vector<LatticeView<vobj>> psi_views;    // [nshift * Nf]

  // Device-resident pointer arrays — built once per CG.
  deviceVector<vobj *> ps_ptrs_d;
  deviceVector<vobj *> psi_ptrs_d;
  // r and src ptrs scratch (rebuilt per call, kept here to avoid alloc churn).
  deviceVector<vobj *> r_ptrs_d;
  deviceVector<vobj *> src_ptrs_d;

  // Reusable per-iter coefficient buffers (resize once).
  deviceVector<RealD> alpha_d, beta_d, coef_d, scale_d;
  deviceVector<int>   conv_d;

  int nshift = 0;
  int Nf = 0;
  uint64_t oSites = 0;

  ~FusedShiftCtx() {
    for (auto &v : ps_views)  v.ViewClose();
    for (auto &v : psi_views) v.ViewClose();
  }
};

// Build the context: open views for ps[s].f[a] and psi[s].f[a] (these are
// ONLY accessed via our fused kernels in TXQCDMultiShiftCGSchur, so caching
// is safe).  r and src views are opened per-call.
template <class vobj>
inline void BuildContext(FusedShiftCtx<vobj> &ctx,
                         std::vector<TXQCDFermionNf> &ps,
                         std::vector<TXQCDFermionNf> &psi,
                         GridBase *grid) {
  constexpr int Nf = TxqcdNf;
  int nshift = static_cast<int>(ps.size());
  GRID_ASSERT(static_cast<int>(psi.size()) == nshift);
  ctx.nshift = nshift;
  ctx.Nf = Nf;
  ctx.oSites = grid->oSites();

  ctx.ps_views.reserve(nshift * Nf);
  ctx.psi_views.reserve(nshift * Nf);
  for (int s = 0; s < nshift; ++s) {
    for (int a = 0; a < Nf; ++a) {
      ctx.ps_views.emplace_back (ps[s].f[a].View (AcceleratorWrite));
      ctx.psi_views.emplace_back(psi[s].f[a].View(AcceleratorWrite));
    }
  }

  std::vector<vobj *> ps_ptrs_h(nshift * Nf);
  std::vector<vobj *> psi_ptrs_h(nshift * Nf);
  for (int i = 0; i < nshift * Nf; ++i) {
    ps_ptrs_h[i]  = ctx.ps_views[i].getHostPointer();
    psi_ptrs_h[i] = ctx.psi_views[i].getHostPointer();
  }

  ctx.ps_ptrs_d.resize(nshift * Nf);
  ctx.psi_ptrs_d.resize(nshift * Nf);
  ctx.r_ptrs_d.resize(Nf);
  ctx.src_ptrs_d.resize(Nf);
  ctx.alpha_d.resize(nshift);
  ctx.beta_d.resize(nshift);
  ctx.coef_d.resize(nshift);
  ctx.scale_d.resize(nshift);
  ctx.conv_d.resize(nshift);

  acceleratorCopyToDevice(ps_ptrs_h.data(),  &ctx.ps_ptrs_d[0],
                          nshift * Nf * sizeof(vobj *));
  acceleratorCopyToDevice(psi_ptrs_h.data(), &ctx.psi_ptrs_d[0],
                          nshift * Nf * sizeof(vobj *));
}

// ps[s].f[a] = alpha_per_s[s] * r.f[a] + beta_per_s[s] * ps[s].f[a]
// (skipping shifts with converged[s] != 0).  Uses cached context.
template <class vobj>
inline void psUpdate(FusedShiftCtx<vobj> &ctx,
                     std::vector<TXQCDFermionNf> &ps,
                     const TXQCDFermionNf &r,
                     const std::vector<int> &converged,
                     const std::vector<RealD> &alpha_per_s,
                     const std::vector<RealD> &beta_per_s) {
  int nshift = ctx.nshift;
  int Nf = ctx.Nf;
  if (nshift == 0) return;

  int cb = r.f[0].Checkerboard();
  for (int s = 0; s < nshift; ++s)
    for (int a = 0; a < Nf; ++a) ps[s].f[a].Checkerboard() = cb;

  // Open r views per call (r is read-modify-written by Grid Lattice
  // expressions elsewhere in the CG, so it's not safe to cache).
  std::vector<LatticeView<vobj>> r_views;
  r_views.reserve(Nf);
  for (int a = 0; a < Nf; ++a)
    r_views.emplace_back(r.f[a].View(AcceleratorRead));
  std::vector<vobj *> r_ptrs_h(Nf);
  for (int a = 0; a < Nf; ++a) r_ptrs_h[a] = r_views[a].getHostPointer();
  acceleratorCopyToDevice(r_ptrs_h.data(), &ctx.r_ptrs_d[0],
                          Nf * sizeof(vobj *));

  acceleratorCopyToDevice(const_cast<RealD *>(alpha_per_s.data()),
                          &ctx.alpha_d[0], nshift * sizeof(RealD));
  acceleratorCopyToDevice(const_cast<RealD *>(beta_per_s.data()),
                          &ctx.beta_d[0],  nshift * sizeof(RealD));
  acceleratorCopyToDevice(const_cast<int *>(converged.data()),
                          &ctx.conv_d[0],  nshift * sizeof(int));

  vobj **ps_p   = &ctx.ps_ptrs_d[0];
  vobj **r_p    = &ctx.r_ptrs_d[0];
  RealD *alpha_p = &ctx.alpha_d[0];
  RealD *beta_p  = &ctx.beta_d[0];
  int   *conv_p  = &ctx.conv_d[0];
  uint64_t oSites = ctx.oSites;
  constexpr int Nsimd = vobj::Nsimd();

  accelerator_for(idx, oSites * static_cast<uint64_t>(nshift) *
                          static_cast<uint64_t>(Nf), Nsimd, {
    int Nf_loc = Nf;
    int nshift_loc = nshift;
    uint64_t site = idx / (uint64_t(nshift_loc) * Nf_loc);
    int s = (idx / Nf_loc) % nshift_loc;
    int a = idx % Nf_loc;
    if (conv_p[s]) return;
    auto r_lane  = coalescedRead(r_p[a][site]);
    auto ps_lane = coalescedRead(ps_p[s * Nf_loc + a][site]);
    auto out     = alpha_p[s] * r_lane + beta_p[s] * ps_lane;
    coalescedWrite(ps_p[s * Nf_loc + a][site], out);
  });

  for (auto &v : r_views) v.ViewClose();
}

// psi[s].f[a] += coef_per_s[s] * ps[s].f[a]
template <class vobj>
inline void psiUpdate(FusedShiftCtx<vobj> &ctx,
                      std::vector<TXQCDFermionNf> &psi,
                      const std::vector<TXQCDFermionNf> &ps,
                      const std::vector<int> &converged,
                      const std::vector<RealD> &coef_per_s) {
  int nshift = ctx.nshift;
  int Nf = ctx.Nf;
  if (nshift == 0) return;

  int cb = ps[0].f[0].Checkerboard();
  for (int s = 0; s < nshift; ++s)
    for (int a = 0; a < Nf; ++a) psi[s].f[a].Checkerboard() = cb;

  acceleratorCopyToDevice(const_cast<RealD *>(coef_per_s.data()),
                          &ctx.coef_d[0], nshift * sizeof(RealD));
  acceleratorCopyToDevice(const_cast<int *>(converged.data()),
                          &ctx.conv_d[0], nshift * sizeof(int));

  vobj **psi_p = &ctx.psi_ptrs_d[0];
  vobj **ps_p  = &ctx.ps_ptrs_d[0];
  RealD *coef_p = &ctx.coef_d[0];
  int   *conv_p = &ctx.conv_d[0];
  uint64_t oSites = ctx.oSites;
  constexpr int Nsimd = vobj::Nsimd();

  accelerator_for(idx, oSites * static_cast<uint64_t>(nshift) *
                          static_cast<uint64_t>(Nf), Nsimd, {
    int Nf_loc = Nf;
    int nshift_loc = nshift;
    uint64_t site = idx / (uint64_t(nshift_loc) * Nf_loc);
    int s = (idx / Nf_loc) % nshift_loc;
    int a = idx % Nf_loc;
    if (conv_p[s]) return;
    auto psi_lane = coalescedRead(psi_p[s * Nf_loc + a][site]);
    auto ps_lane  = coalescedRead(ps_p[s * Nf_loc + a][site]);
    auto out      = psi_lane + coef_p[s] * ps_lane;
    coalescedWrite(psi_p[s * Nf_loc + a][site], out);
  });
}

// psi[s].f[a] = scale_per_s[s] * src.f[a]    (initial setup before loop)
template <class vobj>
inline void psiInit(FusedShiftCtx<vobj> &ctx,
                    std::vector<TXQCDFermionNf> &psi,
                    const TXQCDFermionNf &src,
                    const std::vector<RealD> &scale_per_s) {
  int nshift = ctx.nshift;
  int Nf = ctx.Nf;
  if (nshift == 0) return;

  int cb = src.f[0].Checkerboard();
  for (int s = 0; s < nshift; ++s)
    for (int a = 0; a < Nf; ++a) psi[s].f[a].Checkerboard() = cb;

  std::vector<LatticeView<vobj>> src_views;
  src_views.reserve(Nf);
  for (int a = 0; a < Nf; ++a)
    src_views.emplace_back(src.f[a].View(AcceleratorRead));
  std::vector<vobj *> src_ptrs_h(Nf);
  for (int a = 0; a < Nf; ++a) src_ptrs_h[a] = src_views[a].getHostPointer();
  acceleratorCopyToDevice(src_ptrs_h.data(), &ctx.src_ptrs_d[0],
                          Nf * sizeof(vobj *));

  acceleratorCopyToDevice(const_cast<RealD *>(scale_per_s.data()),
                          &ctx.scale_d[0], nshift * sizeof(RealD));

  vobj **psi_p  = &ctx.psi_ptrs_d[0];
  vobj **src_p  = &ctx.src_ptrs_d[0];
  RealD *scale_p = &ctx.scale_d[0];
  uint64_t oSites = ctx.oSites;
  constexpr int Nsimd = vobj::Nsimd();

  accelerator_for(idx, oSites * static_cast<uint64_t>(nshift) *
                          static_cast<uint64_t>(Nf), Nsimd, {
    int Nf_loc = Nf;
    int nshift_loc = nshift;
    uint64_t site = idx / (uint64_t(nshift_loc) * Nf_loc);
    int s = (idx / Nf_loc) % nshift_loc;
    int a = idx % Nf_loc;
    auto src_lane = coalescedRead(src_p[a][site]);
    auto out      = scale_p[s] * src_lane;
    coalescedWrite(psi_p[s * Nf_loc + a][site], out);
  });

  for (auto &v : src_views) v.ViewClose();
}

}  // namespace TxqcdFusedShift

NAMESPACE_END(Grid);
