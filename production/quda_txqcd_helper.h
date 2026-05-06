#pragma once
// Drop-in QUDA-accelerated propagator solver for TXQCD measurement codes.
// Solves M_TXQCD · x = src on the full volume.
//
// TXQCD's diagonal Δ insertion (flavor / spin / color / tensor) means we
// can't call QUDA's invertQuda directly — QUDA doesn't know about Δ.
// Instead, we use **defect correction** with QUDA's plain-clover invert as
// the preconditioner:
//
//   x_0 = 0
//   for k = 0, 1, ...:
//     r_k = src - M_TXQCD x_k          // M_TXQCD includes Δ
//     if |r_k|² < tol² · |src|²: stop
//     δ_k = M_QCD^{-1} r_k             // QUDA invert, per flavor (no Δ)
//     x_{k+1} = x_k + δ_k
//
// Convergence is geometric with rate ‖M_QCD^{-1} · Δ‖.  For production TXQCD
// (λ=4..12, σ ≈ Σ/λ² ≈ 0.05–0.25), Δ contributes O(σ) off-diagonal entries
// while M_QCD diagonal is O(1) — typical contraction rate is 0.05–0.15 per
// outer iteration → ~5–10 outer iters for 1e-8 tolerance.
//
// Each outer iter is dominated by Nf QUDA invertQuda calls; on 16³×48 with
// Wilson-clover at light mass each takes O(10) seconds, so a propagator
// solve costs ~100 sec — typically several × faster than full-volume Grid
// CG on the TXQCD operator (which routinely runs to 5000+ iters at light
// mass).
//
// Drop-in API mirroring quda_helper.h::QudaPropSolver:
//   QudaTxqcdPropSolver solver(Mop, mass, csw, Usm, tol, max_iter);
//   solver.solve(src, x);
//
// Activated by env var QUDA_SOLVER=1 (same gate as the QCD QudaPropSolver).
// Otherwise falls back to the full-volume Grid TxqcdCG (M†M·x = M†·src).

#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/AuxFieldTypes.h>
#include <array>
#include <memory>
#include <vector>

#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#endif

namespace Grid {

class QudaTxqcdPropSolver {
public:
  QudaTxqcdPropSolver(TXQCDWilsonCloverOp &Mop,
                       const std::array<RealD, TxqcdNf> &mass,
                       RealD csw,
                       const LatticeGaugeField &Usm,
                       RealD tol, int max_iter,
                       bool anti_periodic_t = true)
      : Mop_(Mop), mass_(mass), csw_(csw),
        Usm_(Usm), tol_(tol), max_iter_(max_iter)
  {
#ifdef GRID_HAVE_QUDA
    use_quda_ = (std::getenv("QUDA_SOLVER") != nullptr);
    if (use_quda_) {
      Quda::initialize();
      // Build one QudaCloverInverter per UNIQUE flavor mass (typically 1
      // for degenerate Nf=2 light, may be 2 for split-mass setups).
      mass_to_idx_.fill(-1);
      std::vector<RealD> unique_masses;
      for (int a = 0; a < TxqcdNf; ++a) {
        int idx = -1;
        for (size_t k = 0; k < unique_masses.size(); ++k)
          if (std::abs(unique_masses[k] - mass[a]) < 1e-14) { idx = (int)k; break; }
        if (idx < 0) {
          idx = (int)unique_masses.size();
          unique_masses.push_back(mass[a]);
        }
        mass_to_idx_[a] = idx;
      }
      for (RealD m : unique_masses) {
        QudaCloverParams qp;
        qp.mass = m;
        qp.csw  = csw;
        qp.anti_periodic_t = anti_periodic_t;
        // Inner solve tighter than outer so inner tolerance never limits
        // the outer defect-correction convergence.
        qp.tol = std::min(tol * 0.1, 1e-10);
        qp.max_iter = max_iter;
        qp.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
        quda_inv_.emplace_back(std::make_unique<QudaCloverInverter>(Usm_.Grid(), qp));
        quda_inv_.back()->SetGauge(Usm_);
      }
      std::cout << GridLogMessage
                << "[QudaTxqcdPropSolver] QUDA defect-corrector active ("
                << unique_masses.size() << " unique mass(es), Nf=" << TxqcdNf
                << ")" << std::endl;
    }
#else
    use_quda_ = false;
#endif
  }

  // Solve M_TXQCD · x = src.  Caller does NOT need to apply Mdag first.
  void solve(const TXQCDFermionNf &src, TXQCDFermionNf &x) {
#ifdef GRID_HAVE_QUDA
    if (use_quda_) { solve_defect_correction(src, x); return; }
#endif
    solve_grid_cg(src, x);
  }

private:
  TXQCDWilsonCloverOp &Mop_;
  std::array<RealD, TxqcdNf> mass_;
  RealD csw_;
  const LatticeGaugeField &Usm_;
  RealD tol_;
  int max_iter_;
  bool use_quda_;

  std::array<int, TxqcdNf> mass_to_idx_;
#ifdef GRID_HAVE_QUDA
  std::vector<std::unique_ptr<QudaCloverInverter>> quda_inv_;
#endif

#ifdef GRID_HAVE_QUDA
  void solve_defect_correction(const TXQCDFermionNf &src, TXQCDFermionNf &x) {
    GridBase *g = src.Grid();
    TXQCDFermionNf r(g), Mx(g), delta(g);
    x = Zero();
    r = src;

    RealD src2 = std::max(norm2(src), 1e-30);
    RealD r2   = norm2(r);
    RealD tol2 = tol_ * tol_ * src2;

    // QudaCloverInverter::operator() requires a LinearOperator argument
    // (it's part of OperatorFunction's interface) but ignores it internally —
    // the inversion is done against QUDA's loaded gauge/clover.  Build a
    // dummy LinOp so we can satisfy the type signature.
    typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
    LatticeGaugeField &Usm_nc = const_cast<LatticeGaugeField&>(Usm_);
    GridCartesian *grid_cart = dynamic_cast<GridCartesian*>(Usm_.Grid());
    GridRedBlackCartesian rb(grid_cart);
    WilsonImplParams impl_p;
    impl_p.boundary_phases.resize(Nd, 1.0);
    impl_p.boundary_phases[Nd - 1] = -1.0;
    WCF dummy_dw(Usm_nc, *grid_cart, rb, mass_[0], csw_, csw_,
                 WilsonAnisotropyCoefficients(), impl_p);
    MdagMLinearOperator<WCF, LatticeFermion> dummy_HermOp(dummy_dw);

    constexpr int N_OUTER = 30;
    int it;
    for (it = 0; it < N_OUTER; ++it) {
      // Per-flavor QUDA QCD invert: δ_a = (M_QCD)^{-1}_a · r_a
      for (int a = 0; a < TxqcdNf; ++a) {
        delta.f[a] = Zero();
        int qidx = mass_to_idx_[a];
        (*quda_inv_[qidx])(dummy_HermOp, r.f[a], delta.f[a]);
      }
      // x += δ
      for (int a = 0; a < TxqcdNf; ++a) x.f[a] = x.f[a] + delta.f[a];

      // r = src − M_TXQCD · x  (full TXQCD operator with Δ)
      Mop_.M(x, Mx);
      for (int a = 0; a < TxqcdNf; ++a) r.f[a] = src.f[a] - Mx.f[a];
      r2 = norm2(r);

      std::cout << GridLogMessage
                << "[QudaTxqcdPropSolver] outer it=" << it
                << " |r|²/|src|²=" << r2 / src2 << std::endl;
      if (r2 < tol2) {
        std::cout << GridLogMessage
                  << "[QudaTxqcdPropSolver] CONVERGED after " << (it + 1)
                  << " outer iters" << std::endl;
        return;
      }
    }
    std::cout << GridLogMessage
              << "[QudaTxqcdPropSolver] WARNING: did not converge in "
              << N_OUTER << " outer iters, |r|²/|src|²=" << r2 / src2
              << std::endl;
  }
#endif  // GRID_HAVE_QUDA

  // Fallback: original full-volume Grid TxqcdCG that solves M†M·x = M†·src.
  void solve_grid_cg(const TXQCDFermionNf &src, TXQCDFermionNf &x) {
    GridBase *g = src.Grid();
    TXQCDFermionNf b(g), r(g), p(g), Mp(g), MdMp(g);
    Mop_.Mdag(src, b);
    x = Zero();
    r = b;
    p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(b), 1e-30);
    RealD tol2 = tol_ * tol_ * bsq;
    for (int it = 0; it < max_iter_; ++it) {
      Mop_.M(p, Mp);
      Mop_.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      axpy(x, alpha, p);
      axpy(r, -alpha, MdMp);
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) break;
      RealD beta_cg = rsq_new / rsq;
      for (int a = 0; a < TxqcdNf; ++a) p.f[a] = r.f[a] + beta_cg * p.f[a];
      rsq = rsq_new;
    }
  }
};

}  // namespace Grid
