#pragma once
// Shared aux-field wall-wall correlator computation.
//
// Used both by:
//   - meas_aux_txqcd.cc (standalone per-cfg measurement)
//   - gen_txqcd_cfgs_2plus1.cc (per-traj HMC diagnostic)
//
// All slice helpers do a Grid sliceSum (collective on the t-direction),
// so they MUST be called on every rank — the result is replicated.
// CorrelatorFromSlice and SliceMean are post-gather CPU loops; safe on any rank.

#include "params.h"

namespace TXQCDProduction {

// --- Slice helpers (collective: must be called on every rank) ---------------

// Σ_{x} (Tr_flavor of a flavor-Hermitian aux at site x), per t.
inline std::vector<ComplexD>
SliceSumFlavorTrace(const LatticeSigmaField &F) {
  GridBase *g = F.Grid();
  LatticeComplex tr(g);
  tr = trace(F);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Σ_{x} (Tr_color of a color-Hermitian aux at site x), per t.
inline std::vector<ComplexD>
SliceSumColorTrace(const LatticeSFieldC &F) {
  GridBase *g = F.Grid();
  LatticeComplex tr(g);
  tr = trace(F);
  std::vector<TComplex> sl;
  sliceSum(tr, sl, Nd - 1);
  std::vector<ComplexD> out(sl.size());
  for (size_t t = 0; t < sl.size(); ++t) out[t] = TensorRemove(sl[t]);
  return out;
}

// Σ_{x} π_ab(x) per (a,b,t), shape [Nf²][T] (row-major a*Nf+b).
inline std::vector<std::vector<ComplexD>>
SliceSumPiAll(const LatticePiField &piF) {
  GridBase *g = piF.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  const int Nf2 = TxqcdNf * TxqcdNf;
  std::vector<std::vector<ComplexD>> out(Nf2, std::vector<ComplexD>(T));
  for (int a = 0; a < TxqcdNf; ++a) {
    for (int b = 0; b < TxqcdNf; ++b) {
      LatticeComplex piab(g);
      piab = PeekIndex<2>(piF, a, b);
      std::vector<TComplex> sl;
      sliceSum(piab, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[a * TxqcdNf + b][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

// Σ_{x} Tr_color t_{μν}(x), 6 antisymmetric pairs (01,02,03,12,13,23) × T.
inline std::vector<std::vector<ComplexD>>
SliceSumTAll(const LatticeTField &tF) {
  GridBase *g = tF.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  const int Npairs = Nd * (Nd - 1) / 2;
  std::vector<std::vector<ComplexD>> out(Npairs, std::vector<ComplexD>(T));
  int idx = 0;
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu, ++idx) {
      auto t_munu = PeekIndex<1>(tF, mu, nu);
      LatticeComplex tr(g);
      tr = trace(reinterpret_cast<const LatticeColourMatrix &>(t_munu));
      std::vector<TComplex> sl;
      sliceSum(tr, sl, Nd - 1);
      for (int t = 0; t < T; ++t)
        out[idx][t] = TensorRemove(sl[t]);
    }
  }
  return out;
}

// --- Post-gather helpers (CPU loops; safe on any rank) ----------------------

inline std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &sA,
                    const std::vector<ComplexD> &sB,
                    RealD V4) {
  int T = (int)sA.size();
  std::vector<ComplexD> C(T, 0.0);
  for (int dt = 0; dt < T; ++dt)
    for (int t0 = 0; t0 < T; ++t0)
      C[dt] += sA[(t0 + dt) % T] * conjugate(sB[t0]);
  for (auto &c : C) c /= V4;
  return C;
}

inline std::vector<ComplexD>
CorrelatorFromSlice(const std::vector<ComplexD> &s, RealD V4) {
  return CorrelatorFromSlice(s, s, V4);
}

inline ComplexD SliceMean(const std::vector<ComplexD> &s) {
  ComplexD acc = 0.0;
  for (auto v : s) acc += v;
  return acc / RealD(s.size());
}

inline void ScaleVec(std::vector<ComplexD> &v, RealD c) {
  for (auto &x : v) x *= c;
}

// --- Bundled output --------------------------------------------------------

struct AuxWallCorrelators {
  int T;
  int Nf2;     // = TxqcdNf * TxqcdNf
  int NtPairs; // = Nd*(Nd-1)/2

  // λ⁴-scaled raw correlators (length T).  Analysis subtracts ensemble-mean
  // disconnected piece using wall_*_slice values.
  std::vector<ComplexD> C_sigma;
  std::vector<ComplexD> C_pi_iso;          // isovector: Σ_ab C_ab − Tr·Tr
  std::vector<ComplexD> C_pi_ab_flat;      // Nf² × T row-major
  std::vector<ComplexD> C_s;
  std::vector<ComplexD> C_p;
  std::vector<ComplexD> C_t_flat;          // NtPairs × T row-major

  // Per-cfg wall slices Σ_x f(x,t), length T (flat for multi-channel).
  std::vector<ComplexD> wall_sigma;
  std::vector<ComplexD> wall_pi_tr;
  std::vector<ComplexD> wall_pi_ab_flat;   // Nf² × T row-major
  std::vector<ComplexD> wall_s;
  std::vector<ComplexD> wall_p;
  std::vector<ComplexD> wall_t_flat;       // NtPairs × T row-major
};

// Compute all aux wall-wall correlators for one cfg.  Must be called on
// every rank (the slice helpers do collectives).
inline AuxWallCorrelators
ComputeAuxWallCorrelators(const TXQCDField &U) {
  GridBase *g = U.sigma.Grid();
  int T = g->GlobalDimensions()[Nd - 1];
  RealD V4 = 1.0;
  for (int mu = 0; mu < Nd; ++mu) V4 *= g->GlobalDimensions()[mu];
  const RealD lam4 = lambda * lambda * lambda * lambda;
  const int Nf2 = TxqcdNf * TxqcdNf;

  AuxWallCorrelators r;
  r.T = T;
  r.Nf2 = Nf2;

  // sigma channel (Tr_flavor σ)
  std::vector<ComplexD> sig_tr = SliceSumFlavorTrace(U.sigma);
  r.C_sigma = CorrelatorFromSlice(sig_tr, V4);
  ScaleVec(r.C_sigma, lam4);
  r.wall_sigma = std::move(sig_tr);

  // pi isovector + per-(a,b)
  auto pi_all = SliceSumPiAll(U.pi);
  std::vector<ComplexD> pi_tr = SliceSumFlavorTrace(U.pi);
  std::vector<ComplexD> C_disc = CorrelatorFromSlice(pi_tr, V4);
  std::vector<ComplexD> C_total(T, 0.0);
  r.C_pi_ab_flat.assign(Nf2 * T, ComplexD(0.0));
  for (int ab = 0; ab < Nf2; ++ab) {
    auto C_ab = CorrelatorFromSlice(pi_all[ab], V4);
    for (int t = 0; t < T; ++t) C_total[t] += C_ab[t];
    ScaleVec(C_ab, lam4);
    for (int t = 0; t < T; ++t) r.C_pi_ab_flat[ab * T + t] = C_ab[t];
  }
  r.C_pi_iso.assign(T, 0.0);
  for (int t = 0; t < T; ++t) r.C_pi_iso[t] = lam4 * (C_total[t] - C_disc[t]);
  r.wall_pi_tr = std::move(pi_tr);
  r.wall_pi_ab_flat.assign(Nf2 * T, ComplexD(0.0));
  for (int ab = 0; ab < Nf2; ++ab)
    for (int t = 0; t < T; ++t)
      r.wall_pi_ab_flat[ab * T + t] = pi_all[ab][t];

  // s, p (color-Hermitian)
  std::vector<ComplexD> s_tr = SliceSumColorTrace(U.s);
  r.C_s = CorrelatorFromSlice(s_tr, V4);
  ScaleVec(r.C_s, lam4);
  r.wall_s = std::move(s_tr);

  std::vector<ComplexD> p_tr = SliceSumColorTrace(U.p);
  r.C_p = CorrelatorFromSlice(p_tr, V4);
  ScaleVec(r.C_p, lam4);
  r.wall_p = std::move(p_tr);

  // antisymmetric tensor t_{μν} per-pair
  auto t_all = SliceSumTAll(U.t);
  const int NtPairs = (int)t_all.size();
  r.NtPairs = NtPairs;
  r.C_t_flat.assign(NtPairs * T, ComplexD(0.0));
  for (int p = 0; p < NtPairs; ++p) {
    auto C_p_munu = CorrelatorFromSlice(t_all[p], V4);
    ScaleVec(C_p_munu, lam4);
    for (int t = 0; t < T; ++t) r.C_t_flat[p * T + t] = C_p_munu[t];
  }
  r.wall_t_flat.assign(NtPairs * T, ComplexD(0.0));
  for (int p = 0; p < NtPairs; ++p)
    for (int t = 0; t < T; ++t)
      r.wall_t_flat[p * T + t] = t_all[p][t];

  return r;
}

} // namespace TXQCDProduction
