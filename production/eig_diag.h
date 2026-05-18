#pragma once
// Per-trajectory eigenvalue diagnostic for the HMC drivers.
//
//   RunEigDiagQcd  (Dw,    grid, pRNG, params, evals_M2, evals_g5M)
//   RunEigDiagTxqcd(Mop,   grid, pRNG, params, evals_M2, evals_g5M)
//
// Both compute the smallest few signed eigenvalues of γ5·M (Hermitian, since
// M† = γ5 M γ5).  Sign of the lowest eigenvalue flags the sign of det(M).
//
// Algorithm: Chebyshev-filter Lanczos on (γ5·M)² = M†M (standard polynomial
// filter for finding small-|λ| modes of an indefinite Hermitian op), with
// a Rayleigh-quotient sweep <u, γ5·M·u>/<u,u> on each converged Ritz vector
// to recover the signed eigenvalue.  Only the lowest 1–2 Ritz pairs are
// reliable at the cheap NEV=2 / NM=20 / order=21 defaults below; for finer
// resolution use eigspec_diag offline.
//
// Lifted from eigspec_diag.cc (which is now a thin standalone wrapper for
// offline sweeps) — keep the two in sync.

#include "params.h"
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/Grid_Eigen_Dense.h>
#include <Grid/Eigen/Eigenvalues>

namespace TXQCDProduction {

typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> EigDiagWCF;

struct EigDiagParams {
  // Defaults sized to actually CONVERGE the lowest few γ5·M modes on a
  // thermalized 16³×48 cfg.  The original NM=20/ord=21 did NOT converge
  // (g5M[0]² off from M†M[0] by ~1000× on λ=7 thermalized).  Anchors:
  // offline eigspec NM=80/ord=51/hi=120 converged lowest 1–2 modes on
  // production cfgs.  We go above that.  Cost ≈ NM·ord·(one M†M apply)
  // ≈ 120·61·10 ms ≈ 73 s/traj at λ=7 Nf=2 → ~6% of a ~1170 s MDS=10 traj.
  // cheby_hi MUST exceed λ_max(M†M) (~76 at λ=4, ~62 at λ≥6) or the filter
  // fails to suppress the top of the spectrum and the low modes never
  // converge — this (not just NM) was why NM=20/ord=21/hi=80 failed.
  int   Nev      = 6;
  int   Nm       = 120;
  int   cheby_ord = 61;
  RealD cheby_lo = 0.05;
  RealD cheby_hi = 120.0;  // > λ_max(M†M) across all production λ (≤76)
  // Sign-problem order-parameter threshold.  A det(M) sign change ⟺ an
  // eigenvalue of γ5·M crosses zero ⟺ min|γ5·M| → 0.  We log two
  // basis-independent scalars per traj: min|γ5·M| over the converged modes,
  // and the count with |γ5·M| < zero_eps.  zero_eps default 0.01 ≈ ¼ of the
  // typical smallest |λ|~0.04 — comfortably above the ~4e-4 convergence
  // floor, so a genuine near-zero event registers but normal ± shuffle does
  // not.  Override with EIG_ZERO_EPS.
  RealD zero_eps = 0.01;
};

// Read EIG_NEV / EIG_NM / EIG_CHEBY_ORD / EIG_CHEBY_LO / EIG_CHEBY_HI env vars
// into a params struct.  Use this to share the same env interface across
// drivers.  Returns false if `EIG_DIAG` is unset/0 (disabled).
inline bool eig_diag_enabled() {
  const char *e = std::getenv("EIG_DIAG");
  return e && *e && std::atoi(e) != 0;
}
inline EigDiagParams eig_diag_params_from_env() {
  EigDiagParams p;
  if (const char *e = std::getenv("EIG_NEV");       e && *e) p.Nev       = std::atoi(e);
  if (const char *e = std::getenv("EIG_NM");        e && *e) p.Nm        = std::atoi(e);
  if (const char *e = std::getenv("EIG_CHEBY_ORD"); e && *e) p.cheby_ord = std::atoi(e);
  if (const char *e = std::getenv("EIG_CHEBY_LO");  e && *e) p.cheby_lo  = std::atof(e);
  if (const char *e = std::getenv("EIG_CHEBY_HI");  e && *e) p.cheby_hi  = std::atof(e);
  if (const char *e = std::getenv("EIG_ZERO_EPS");  e && *e) p.zero_eps  = std::atof(e);
  return p;
}

// Sign-problem order parameters from the per-traj signed γ5·M Rayleigh
// quotients (the evals_g5M vector RunEigDiag* fills).  Returns:
//   min_abs   = min_k |γ5·M_k|   — must stay bounded away from 0
//   n_near    = #{ k : |γ5·M_k| < zero_eps }  — near-zero mode count
// Both are basis-independent: a det(M) sign change requires an eigenvalue
// to reach zero, so min_abs→0 / n_near>0 is THE sharp order parameter,
// immune to the ± near-degenerate-pair relabeling that makes per-mode
// sign "flips" look alarming but carry no determinant-sign information.
inline void eig_order_params(const std::vector<RealD> &evals_g5M,
                              RealD zero_eps,
                              RealD &min_abs, int &n_near) {
  min_abs = -1.0;
  n_near  = 0;
  for (RealD v : evals_g5M) {
    RealD a = std::fabs(v);
    if (min_abs < 0.0 || a < min_abs) min_abs = a;
    if (a < zero_eps) ++n_near;
  }
  if (min_abs < 0.0) min_abs = 0.0;  // empty (eig disabled) → 0
}

// ---- Field-generic helpers --------------------------------------------------
template <class Field>
inline void eig_axpy_ip(Field &y, const ComplexD &a, const Field &x);
template <class Field>
inline void eig_scale_ip(Field &y, const RealD &a);
template <class Field>
inline void eig_copy_field(Field &dst, const Field &src);
template <class Field>
inline void eig_random_field(GridParallelRNG &rng, Field &f);
template <class Field>
inline void eig_axpby_field(Field &out, const RealD &a, const RealD &b,
                             const Field &x, const Field &y);

// LatticeFermion specializations
template <> inline void eig_axpy_ip(LatticeFermion &y, const ComplexD &a,
                                     const LatticeFermion &x) { y = y + a*x; }
template <> inline void eig_scale_ip(LatticeFermion &y, const RealD &a) { y = a*y; }
template <> inline void eig_copy_field(LatticeFermion &dst, const LatticeFermion &src) {
  dst = src;
}
template <> inline void eig_random_field(GridParallelRNG &rng, LatticeFermion &f) {
  gaussian(rng, f);
}
template <> inline void eig_axpby_field(LatticeFermion &out, const RealD &a,
                                         const RealD &b, const LatticeFermion &x,
                                         const LatticeFermion &y) {
  out = a*x + b*y;
}

// TXQCDFermionNf specializations
template <> inline void eig_axpy_ip(TXQCDFermionNf &y, const ComplexD &a,
                                     const TXQCDFermionNf &x) {
  for (int aa = 0; aa < TxqcdNf; ++aa) y.f[aa] = y.f[aa] + a*x.f[aa];
}
template <> inline void eig_scale_ip(TXQCDFermionNf &y, const RealD &a) {
  for (int aa = 0; aa < TxqcdNf; ++aa) y.f[aa] = a*y.f[aa];
}
template <> inline void eig_copy_field(TXQCDFermionNf &dst, const TXQCDFermionNf &src) {
  for (int aa = 0; aa < TxqcdNf; ++aa) dst.f[aa] = src.f[aa];
}
template <> inline void eig_random_field(GridParallelRNG &rng, TXQCDFermionNf &f) {
  for (int aa = 0; aa < TxqcdNf; ++aa) gaussian(rng, f.f[aa]);
}
template <> inline void eig_axpby_field(TXQCDFermionNf &out, const RealD &a,
                                         const RealD &b, const TXQCDFermionNf &x,
                                         const TXQCDFermionNf &y) {
  for (int aa = 0; aa < TxqcdNf; ++aa) out.f[aa] = a*x.f[aa] + b*y.f[aa];
}

// ---- Operator wrappers ------------------------------------------------------
class EigQcdMdagM : public LinearFunction<LatticeFermion> {
public:
  EigDiagWCF &Dw_;
  EigQcdMdagM(EigDiagWCF &Dw) : Dw_(Dw) {}
  void operator()(const LatticeFermion &in, LatticeFermion &out) override {
    LatticeFermion tmp(in.Grid());
    Dw_.M(in, tmp);
    Dw_.Mdag(tmp, out);
  }
};

class EigQcdG5M : public LinearFunction<LatticeFermion> {
public:
  EigDiagWCF &Dw_;
  EigQcdG5M(EigDiagWCF &Dw) : Dw_(Dw) {}
  void operator()(const LatticeFermion &in, LatticeFermion &out) override {
    LatticeFermion tmp(in.Grid());
    Dw_.M(in, tmp);
    Gamma g5(Gamma::Algebra::Gamma5);
    out = g5 * tmp;
  }
};

class EigTxqcdMdagM : public LinearFunction<TXQCDFermionNf> {
public:
  TXQCDWilsonCloverOp &Mop_;
  EigTxqcdMdagM(TXQCDWilsonCloverOp &Mop) : Mop_(Mop) {}
  void operator()(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mop_.M(in, tmp);
    Mop_.Mdag(tmp, out);
  }
};

class EigTxqcdG5M : public LinearFunction<TXQCDFermionNf> {
public:
  TXQCDWilsonCloverOp &Mop_;
  EigTxqcdG5M(TXQCDWilsonCloverOp &Mop) : Mop_(Mop) {}
  void operator()(const TXQCDFermionNf &in, TXQCDFermionNf &out) override {
    TXQCDFermionNf tmp(in.Grid());
    Mop_.M(in, tmp);
    Gamma g5(Gamma::Algebra::Gamma5);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = g5 * tmp.f[a];
  }
};

// ---- Chebyshev filter -------------------------------------------------------
template <class Field>
void EigApplyChebyT(LinearFunction<Field> &Op, const Field &in, Field &out,
                    RealD lo, RealD hi, int order) {
  GridBase *grid = in.Grid();
  Field T0(grid), T1(grid), T2(grid), y(grid);
  RealD xscale = 2.0 / (hi - lo);
  RealD mscale = -(hi + lo) / (hi - lo);
  eig_copy_field(T0, in);
  Op(T0, y);
  eig_axpby_field(T1, xscale, mscale, y, T0);
  if (order <= 1) { eig_copy_field(out, T0); return; }
  if (order == 2) { eig_copy_field(out, T1); return; }
  Field *Tnm = &T0, *Tn = &T1, *Tnp = &T2;
  for (int n = 2; n < order; ++n) {
    Op(*Tn, y);
    eig_axpby_field(y, xscale, mscale, y, *Tn);
    eig_axpby_field(*Tnp, 2.0, -1.0, y, *Tnm);
    Field *sw = Tnm; Tnm = Tn; Tn = Tnp; Tnp = sw;
  }
  eig_copy_field(out, *Tn);
}

template <class Field>
class EigChebFilter : public LinearFunction<Field> {
public:
  LinearFunction<Field> &Op_;
  RealD lo_, hi_;
  int order_;
  EigChebFilter(LinearFunction<Field> &Op, RealD lo, RealD hi, int order)
      : Op_(Op), lo_(lo), hi_(hi), order_(order) {}
  void operator()(const Field &in, Field &out) override {
    EigApplyChebyT<Field>(Op_, in, out, lo_, hi_, order_);
  }
};

// ---- Lanczos with full re-orthogonalization ---------------------------------
template <class Field>
static void EigLanczos(LinearFunction<Field> &FilterOp,
                        LinearFunction<Field> &MdagM,
                        LinearFunction<Field> &G5M,
                        GridBase *grid, GridParallelRNG &rng, int Nm,
                        std::vector<RealD> &mdagm_evals,
                        std::vector<RealD> &g5m_evals,
                        int Nev) {
  std::vector<Field> V; V.reserve(Nm);
  for (int i = 0; i < Nm; ++i) V.emplace_back(grid);
  std::vector<RealD> alpha(Nm, 0.0), beta(Nm, 0.0);

  Field w(grid), tmp(grid);
  eig_random_field(rng, V[0]);
  RealD n0 = std::sqrt(norm2(V[0]));
  eig_scale_ip(V[0], 1.0 / n0);

  int k_done = 0;
  for (int k = 0; k < Nm; ++k) {
    FilterOp(V[k], w);
    if (k > 0) eig_axpy_ip(w, ComplexD(-beta[k-1], 0), V[k-1]);
    alpha[k] = real(innerProduct(V[k], w));
    eig_axpy_ip(w, ComplexD(-alpha[k], 0), V[k]);
    for (int pass = 0; pass < 2; ++pass) {
      for (int j = 0; j <= k; ++j) {
        ComplexD c = innerProduct(V[j], w);
        eig_axpy_ip(w, -c, V[j]);
      }
    }
    beta[k] = std::sqrt(norm2(w));
    k_done = k + 1;
    if (beta[k] < 1e-12) break;
    if (k + 1 < Nm) {
      eig_copy_field(V[k+1], w);
      eig_scale_ip(V[k+1], 1.0 / beta[k]);
    }
  }

  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k_done, k_done);
  for (int i = 0; i < k_done; ++i) {
    T(i, i) = alpha[i];
    if (i + 1 < k_done) { T(i+1, i) = beta[i]; T(i, i+1) = beta[i]; }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  auto evecs = es.eigenvectors();

  int top = std::min(Nev, k_done);
  mdagm_evals.clear(); g5m_evals.clear();
  std::vector<std::pair<RealD, RealD>> pairs; pairs.reserve(top);
  for (int rank = 0; rank < top; ++rank) {
    int idx = k_done - 1 - rank;  // largest FilterOp eval first
    Field u(grid); u = Zero();
    for (int j = 0; j < k_done; ++j)
      eig_axpy_ip(u, ComplexD(evecs(j, idx), 0), V[j]);
    RealD nu = norm2(u);
    if (nu < 1e-30) continue;
    MdagM(u, tmp);
    RealD ray_m2 = real(innerProduct(u, tmp)) / nu;
    G5M(u, tmp);
    RealD ray_g5 = real(innerProduct(u, tmp)) / nu;
    pairs.emplace_back(ray_m2, ray_g5);
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<RealD,RealD> &a,
               const std::pair<RealD,RealD> &b) { return a.first < b.first; });
  for (const auto &p : pairs) {
    mdagm_evals.push_back(p.first);
    g5m_evals.push_back(p.second);
  }
}

// ---- Public entrypoints -----------------------------------------------------
inline void RunEigDiagQcd(EigDiagWCF &Dw, GridBase *grid, GridParallelRNG &pRNG,
                           const EigDiagParams &p,
                           std::vector<RealD> &evals_M2,
                           std::vector<RealD> &evals_g5M) {
  EigQcdMdagM MdagM(Dw);
  EigQcdG5M   G5M(Dw);
  EigChebFilter<LatticeFermion> Filter(MdagM, p.cheby_lo, p.cheby_hi, p.cheby_ord);
  EigLanczos<LatticeFermion>(Filter, MdagM, G5M, grid, pRNG, p.Nm,
                              evals_M2, evals_g5M, p.Nev);
}

inline void RunEigDiagTxqcd(TXQCDWilsonCloverOp &Mop, GridBase *grid,
                             GridParallelRNG &pRNG, const EigDiagParams &p,
                             std::vector<RealD> &evals_M2,
                             std::vector<RealD> &evals_g5M) {
  EigTxqcdMdagM MdagM(Mop);
  EigTxqcdG5M   G5M(Mop);
  EigChebFilter<TXQCDFermionNf> Filter(MdagM, p.cheby_lo, p.cheby_hi, p.cheby_ord);
  EigLanczos<TXQCDFermionNf>(Filter, MdagM, G5M, grid, pRNG, p.Nm,
                              evals_M2, evals_g5M, p.Nev);
}

}  // namespace TXQCDProduction
