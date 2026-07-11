#pragma once
// dtxqcd_meas_helper.h — measurement-side helper for the DTXQCD doubled 48×48
// Wilson-Clover operator.
//
// The point of this header is to build, for a measurement, the SAME fermion
// operator the sea quark saw during HMC, and to solve against it on the fast
// QUDA path:
//
//   * Stout-smeared gauge.  The checkpoint stores the THIN links; the HMC fed
//     the fermion action the stout-smeared links (DTXQCDSmearedConfiguration —
//     gauge through the stout chain, aux passed through unchanged).  So we
//     re-smear the thin gauge with the SAME stout params (stout_nsmear_inv,
//     stout_rho_inv from params.h) and build the operator on those links.
//     Mirrors meas_conn_txqcd.cc's U_inv recipe exactly.
//   * Production csw (1.2493…), NOT csw=0.  csw=0 only matches the old
//     unsmeared scout ensembles (STOUT_NSMEAR=0).
//   * APBC time — the DTXQCD operator's DefaultImplParams already sets
//     boundary_phases[Nd-1] = -1 (chroma convention); DTXQCDMpcOpQUDA defaults
//     anti_periodic_t=true.  So both operators agree with the sea action.
//   * Aux fields σ,π,d,n,s,p referenced as-is from the DTXQCDField.
//
// Solve path:
//
//   DTXQCD_MEAS_GRIDMG=1 (default OFF): full-volume Grid-MG solve —
//     HDCG/TwoLevelADEF2 + coarse deflation through the packed adapter on
//     the device M†M (dtxqcd_gridmg_solver.h).  The λ<=4 frontier path:
//     EO-Schur is sick there (M_ee cliff) and MG beats full CG (λ=4: MG
//     ~250 outers vs CG 3188 iters; λ=3: 2804 outers/724 s vs 70612
//     iters/1711 s).  Per-config setup (basis+coarsen+IRL) amortizes over
//     the solve count; cache with DTXQCD_MEAS_MG_SETUP_PREFIX.  Knobs:
//     DTXQCD_MEAS_MG_{NBASIS,BLOCK,GEOM,ORD,NEV,...} — see the header.
//   DTXQCD_MEAS_EO=1 (default under GRID_HAVE_QUDA, 2026-07-02): EO/Schur-
//     preconditioned solve — odd-block Mpc CG on the device (the same EO
//     Style C CG the HMC uses), even block back-substituted through the
//     site-local M_ee^{-1} caches.  ~10× per solve vs full-volume at λ=10
//     (fewer iterations × half the volume); validated correlator-identical.
//     The right default at λ>=4 (at λ=4: EO CG 68.9 s vs full CG 77 s).
//   DTXQCD_MEAS_EO=0 + DTXQCD_MEAS_QUDA=1: full-volume single-shift CG on
//     M†M through DTXQCDMpcOpQUDA — the exact full-volume operator the
//     non-EO HMC drives (QUDA-offloaded Wilson hop + device aux).
//   DTXQCD_MEAS_QUDA=0 (or non-QUDA build): pure-Grid CPU reference
//     (DTXQCDMOp on DTXQCDWilsonCloverFermionEO).  Bit-parity check target.
//
// The CG is the templated generalization of meas_baryon_dtxqcd.cc's proven
// hand-rolled DtxqcdCG (M†M x = M† source), so DTXQCDFermionDoubled sources map
// straight in.

#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC.h>
#include "dtxqcd_gridmg_solver.h"
#endif
#include <memory>

namespace TXQCDProduction {

// Re-apply the HMC's fermion-side stout smearing to the thin (checkpoint)
// links.  nsmear<=0 → no smearing (the unsmeared csw=0 scout path).
inline Grid::LatticeGaugeField
DtxqcdInversionSmear(Grid::GridCartesian &grid,
                     const Grid::LatticeGaugeField &U_thin) {
  using namespace Grid;
  if (stout_nsmear_inv <= 0) return U_thin;
  Smear_Stout<PeriodicGimplR> stout(stout_rho_inv);
  SmearedConfiguration<PeriodicGimplR> smear(&grid, stout_nsmear_inv, stout);
  LatticeGaugeField Uc = U_thin;  // set_Field takes a non-const ref
  smear.set_Field(Uc);
  return smear.get_SmearedU();
}

// GAUGE_TWIST=1: apply a random LOCAL gauge transformation Ω(x) to the loaded
// (U, aux) fields before measuring:
//   U_μ(x) → Ω(x) U_μ(x) Ω†(x+μ̂)
//   σ, π   → Ω σ Ω†   (fundamental ⊗ antifundamental colour indices)
//   d, n   → Ω d Ωᵀ   (BOTH indices fundamental — the diquark couples q q)
//   s, p   → unchanged (colour singlets)
// Every gauge-invariant output (plaq, source norms, and — iff every colour
// index in a correlator is correctly joined — the correlators themselves)
// must be IDENTICAL (to CG tol) between GAUGE_TWIST=0 and =1 runs on the same
// config.  A correlator that MOVES under the twist is gauge-variant, hence
// ensemble-ZERO by Elitzur ⇒ a mis-joined contraction.  Call AFTER the
// checkpoint load and BEFORE building any solver / derived field.
inline void DtxqcdMaybeGaugeTwist(Grid::DTXQCDField &U,
                                  Grid::GridParallelRNG &pRNG) {
  using namespace Grid;
  if (TXQCDProduction::detail::env_int("GAUGE_TWIST", 0) == 0) return;
  GridBase *g_ = U.U.Grid();
  LatticeColourMatrix om(g_);
  SU<Nc>::RandomGaugeTransform<PeriodicGimplR>(pRNG, U.U, om);
  LatticeColourMatrix omT(g_); omT = transpose(om);
  LatticeColourMatrix omA(g_); omA = adj(om);
  for (int a = 0; a < DtxqcdNf; ++a)
    for (int b = 0; b < DtxqcdNf; ++b) {
      { LatticeColourMatrix blk(PeekIndex<1>(U.sigma, a, b));
        blk = om * blk * omA; PokeIndex<1>(U.sigma, blk, a, b); }
      { LatticeColourMatrix blk(PeekIndex<1>(U.pi, a, b));
        blk = om * blk * omA; PokeIndex<1>(U.pi, blk, a, b); }
      { LatticeColourMatrix blk(PeekIndex<1>(U.d, a, b));
        blk = om * blk * omT; PokeIndex<1>(U.d, blk, a, b); }
      { LatticeColourMatrix blk(PeekIndex<1>(U.n, a, b));
        blk = om * blk * omT; PokeIndex<1>(U.n, blk, a, b); }
    }
  std::cout << GridLogMessage << "[GAUGE_TWIST] random Ω(x) applied: "
            << "U→ΩUΩ†, σ/π→ΩXΩ†, d/n→ΩXΩᵀ, s/p untouched" << std::endl;
}

class DtxqcdMeasSolver {
 public:
  // U is the freshly-loaded DTXQCDField (thin gauge + aux).  csw_val/mass are
  // the production values (params.h csw, mass_light_dtxqcd).  tol/maxit are the
  // CG stopping criteria.
  DtxqcdMeasSolver(Grid::GridCartesian &grid,
                   Grid::GridRedBlackCartesian &rbgrid,
                   Grid::DTXQCDField &U,
                   Grid::RealD mass, Grid::RealD csw_val,
                   Grid::RealD tol, int maxit)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), csw_(csw_val),
        tol_(tol), maxit_(maxit),
        U_inv_(DtxqcdInversionSmear(grid, U.U)),
        cpu_op_(U_inv_, grid, rbgrid, mass, csw_val,
                U.sigma, U.pi, U.d, U.n, U.s, U.p),
        Mop_cpu_(cpu_op_) {
    use_quda_ = ResolveUseQuda();
    use_eo_   = ResolveUseEO();
    use_gridmg_ = ResolveUseGridMG();
#ifdef GRID_HAVE_QUDA
    if (use_quda_) {
      Grid::Quda::initialize();  // idempotent
      if (use_gridmg_) {
        // Grid-MG on the full-volume M†M (the λ<=4 frontier path).  Rides
        // the same device op as the full-volume CG branch; the packed
        // adapter + MG hierarchy build once per config in the solver ctor.
        quda_op_.reset(new Grid::DTXQCDMpcOpQUDA(
            U_inv_, grid, rbgrid, mass, csw_val,
            U.sigma, U.pi, U.d, U.n, U.s, U.p));
        quda_op_->SetAuxStatic(true);
        packed_hop_.reset(new Grid::DtxqcdPackedFullHermOp(
            Mop_cpu_, *quda_op_, &grid, Grid::Even));
        Grid::GridParallelRNG mgRNG(&grid);
        mgRNG.SeedFixedIntegers({987654, 271828, 314159, 665772});
        const int nb = MgNbasis();
        if (nb == 64) {
          mg64_.reset(new Grid::DtxqcdGridMGSolver<64>(
              *packed_hop_, &grid, mgRNG, tol_, maxit_));
        } else {
          mg32_.reset(new Grid::DtxqcdGridMGSolver<32>(
              *packed_hop_, &grid, mgRNG, tol_, maxit_));
        }
      } else if (use_eo_) {
        eo_quda_op_.reset(new Grid::DTXQCDMpcOpQUDA_EO(
            cpu_op_, U_inv_, grid, rbgrid, mass, csw_val,
            U.sigma, U.pi, U.d, U.n, U.s, U.p));
      } else {
        quda_op_.reset(new Grid::DTXQCDMpcOpQUDA(
            U_inv_, grid, rbgrid, mass, csw_val,
            U.sigma, U.pi, U.d, U.n, U.s, U.p));
        // Aux fields are fixed for the whole measurement program: pack the
        // device aux cache once instead of twice per M†M CG iteration
        // (43c4d0ad SetAuxStatic; ~5x on the full-volume matvec at 16^3x48).
        quda_op_->SetAuxStatic(true);
      }
    }
#endif
    std::cout << Grid::GridLogMessage
              << "[DtxqcdMeasSolver] mass=" << mass_ << " csw=" << csw_
              << " stout(nsmear=" << stout_nsmear_inv
              << ", rho=" << stout_rho_inv << ")"
              << " tol=" << tol_ << " maxit=" << maxit_
              << " backend="
              << (use_quda_ ? (use_gridmg_ ? "QUDA-GridMG"
                                           : (use_eo_ ? "QUDA-EO" : "QUDA"))
                            : "Grid-CPU")
              << std::endl;
  }

  // Solve M x = source on the full volume (x = M^{-1} source), via CG on M†M:
  //   x = (M†M)^{-1} (M† source).
  void Solve(const Grid::DTXQCDFermionDoubled &source,
             Grid::DTXQCDFermionDoubled &x) {
#ifdef GRID_HAVE_QUDA
    if (use_gridmg_ && use_quda_) { SolveGridMG(source, x); return; }
    if (use_eo_ && use_quda_) { SolveEO(source, x); return; }
    if (use_quda_ && quda_op_) { SolveQuda(source, x); return; }
#endif
    SolveWith(Mop_cpu_, source, x);
  }

#ifdef GRID_HAVE_QUDA
  // Grid-MG solve: x = (M†M)^{-1} M† source through the packed adapter's
  // deflated ADEF2 (dtxqcd_gridmg_solver.h).  Same M† source prep as
  // SolveQuda; pack -> MG solve -> unpack.
  void SolveGridMG(const Grid::DTXQCDFermionDoubled &source,
                   Grid::DTXQCDFermionDoubled &x) {
    using namespace Grid;
    DTXQCDFermionDoubled b(source.Grid());
    quda_op_->Mdag(source, b);  // one-time M† source (host apply)
    LatticeDtxqcdPacked bp(source.Grid()), xp(source.Grid());
    DtxqcdPackDoubled(b, bp);
    if (mg64_) mg64_->solve(bp, xp);
    else       mg32_->solve(bp, xp);
    DtxqcdUnpackDoubled(xp, x);
  }
#endif

#ifdef GRID_HAVE_QUDA
  // EO-preconditioned solve (DTXQCD_MEAS_EO=1, 2026-07-02, rides the EO
  // Style C port): Schur-decompose M x = b,
  //   b'_o = b_o - M_oe M_ee^{-1} b_e
  //   Mpc x_o = b'_o   solved as (Mpc†Mpc) x_o = Mpc† b'_o  on the odd
  //                    half-volume via the device-resident EO Style C CG
  //   x_e  = M_ee^{-1} (b_e - M_eo x_o)
  // Half the volume per iteration vs the full-M path; site-local pieces ride
  // the existing per-site inverse caches.  Reliable update cross-checks
  // against the Grid Mpc, so the device solution is self-validated.
  void SolveEO(const Grid::DTXQCDFermionDoubled &source,
               Grid::DTXQCDFermionDoubled &x) {
    using namespace Grid;
    auto pick = [&](int cb, DTXQCDFermionDoubled &half,
                    const DTXQCDFermionDoubled &full) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        pickCheckerboard(cb, half.upper.f[a], full.upper.f[a]);
        pickCheckerboard(cb, half.lower.f[a], full.lower.f[a]);
      }
    };
    auto setcb = [&](DTXQCDFermionDoubled &full,
                     const DTXQCDFermionDoubled &half) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        setCheckerboard(full.upper.f[a], half.upper.f[a]);
        setCheckerboard(full.lower.f[a], half.lower.f[a]);
      }
    };
    auto stamp = [&](DTXQCDFermionDoubled &f, int cb) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        f.upper.f[a].Checkerboard() = cb;
        f.lower.f[a].Checkerboard() = cb;
      }
    };
    auto sub = [&](DTXQCDFermionDoubled &y,
                   const DTXQCDFermionDoubled &a_, int cb) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        y.upper.f[a] = y.upper.f[a] - a_.upper.f[a];
        y.lower.f[a] = y.lower.f[a] - a_.lower.f[a];
      }
      stamp(y, cb);
    };

    DTXQCDFermionDoubled b_e(&rbgrid_), b_o(&rbgrid_);
    pick(Even, b_e, source);
    pick(Odd,  b_o, source);

    // b'_o = b_o - Meooe(MooeeInv(b_e))
    DTXQCDFermionDoubled t_e(&rbgrid_), t_o(&rbgrid_);
    cpu_op_.MooeeInv(b_e, t_e);
    stamp(t_e, Even);
    cpu_op_.Meooe(t_e, t_o);
    stamp(t_o, Odd);
    sub(b_o, t_o, Odd);  // b_o -= t_o

    // rhs = Mpc† b'_o ; device single-zero-shift CG on Mpc†Mpc.
    DTXQCDMpcOp mpc(cpu_op_);
    DTXQCDFermionDoubled rhs(&rbgrid_);
    mpc.Mdag(b_o, rhs);
    stamp(rhs, Odd);
    std::vector<Grid::DTXQCDFermionDoubled> psi;
    psi.emplace_back(&rbgrid_);
    stamp(psi[0], Odd);
    std::vector<RealD> poles = {0.0};
    std::vector<RealD> tols = {tol_};
    DTXQCDMultiShiftCGQUDA_StyleC(mpc, *eo_quda_op_, poles, tols, rhs, psi,
                                  maxit_);
    stamp(psi[0], Odd);

    // x_e = MooeeInv(b_e - Meooe(x_o))
    DTXQCDFermionDoubled t_e2(&rbgrid_), x_e(&rbgrid_);
    cpu_op_.Meooe(psi[0], t_e2);
    stamp(t_e2, Even);
    sub(b_e, t_e2, Even);  // b_e -= Meooe x_o
    cpu_op_.MooeeInv(b_e, x_e);
    stamp(x_e, Even);

    setcb(x, x_e);
    setcb(x, psi[0]);
  }
#endif

#ifdef GRID_HAVE_QUDA
  // Device-resident single-shift CG: x = (M†M)^{-1} M† source = M^{-1} source,
  // via the SAME Style-C multishift CG the non-EO HMC uses, with a single zero
  // shift (base system = M†M, since Style-C requires poles[s] >= poles[0]).
  // All CG state lives on-device (native CSF); the aux Δ runs as an on-device
  // kernel per mat-vec — NO per-iteration host roundtrip (that was the
  // host-mode M/Mdag bottleneck — ~25 min/solve at near-critical mass).  The
  // one-time M† source apply on entry is host-mode (once per solve).  Style-C's
  // reliable update cross-checks the true DP residual against the Grid HermOp
  // (Mop_cpu_) every ReliableUpdateFreq iters, so the QUDA solution is
  // self-validated against the Grid path.
  void SolveQuda(const Grid::DTXQCDFermionDoubled &source,
                 Grid::DTXQCDFermionDoubled &x) {
    using namespace Grid;
    DTXQCDFermionDoubled b(source.Grid());
    quda_op_->Mdag(source, b);  // one-time M† source (host apply)
    std::vector<DTXQCDFermionDoubled> psi;
    psi.emplace_back(source.Grid());
    std::vector<RealD> poles = {0.0};
    std::vector<RealD> tols = {tol_};
    DTXQCDMultiShiftCGQUDA_StyleC(Mop_cpu_, *quda_op_, poles, tols, b, psi,
                                  maxit_);
    x = psi[0];
  }
#endif

  const Grid::LatticeGaugeField &SmearedGauge() const { return U_inv_; }
  bool UsingQuda() const { return use_quda_; }

 private:
  static bool ResolveUseQuda() {
#ifndef GRID_HAVE_QUDA
    return false;
#else
    const char *e = std::getenv("DTXQCD_MEAS_QUDA");
    if (!e || !*e) return true;  // default ON under QUDA
    return std::atoi(e) != 0;
#endif
  }

  // DTXQCD_MEAS_EO: EO-preconditioned (Schur) measurement solve on the
  // device path.  Default ON under QUDA (validated correlator-identical vs
  // the full-volume solve on the λ=10 thermal-aux config, ~10× per solve);
  // =0 restores the full-volume Style C solve.
  static bool ResolveUseEO() {
#ifndef GRID_HAVE_QUDA
    return false;
#else
    const char *e = std::getenv("DTXQCD_MEAS_EO");
    if (!e || !*e) return true;  // default ON under QUDA
    return std::atoi(e) != 0;
#endif
  }

  // DTXQCD_MEAS_GRIDMG: full-volume Grid-MG measurement solve (default OFF;
  // the λ<=4 frontier path).  Takes precedence over DTXQCD_MEAS_EO.
  static bool ResolveUseGridMG() {
#ifndef GRID_HAVE_QUDA
    return false;
#else
    const char *e = std::getenv("DTXQCD_MEAS_GRIDMG");
    return e && *e && std::atoi(e) != 0;
#endif
  }

  // Compile-time nbasis dispatch: 32 (λ>=4-ish spectra) or 64 (λ=3).
  static int MgNbasis() {
    const char *e = std::getenv("DTXQCD_MEAS_MG_NBASIS");
    int nb = (e && *e) ? std::atoi(e) : 32;
    GRID_ASSERT(nb == 32 || nb == 64);
    return nb;
  }

  // b = M† source ; then CG on (M†M) x = b.
  template <class Op>
  void SolveWith(Op &op, const Grid::DTXQCDFermionDoubled &source,
                 Grid::DTXQCDFermionDoubled &x) {
    Grid::DTXQCDFermionDoubled b(source.Grid());
    op.Mdag(source, b);
    CG(op, b, x);
  }

  // Templated generalization of meas_baryon_dtxqcd.cc's DtxqcdCG: solves
  // (M†M) x = b using op.M / op.Mdag.  Same relative-residual stop.
  template <class Op>
  void CG(Op &op, const Grid::DTXQCDFermionDoubled &b,
          Grid::DTXQCDFermionDoubled &x) {
    using namespace Grid;
    GridBase *g = b.Grid();
    DTXQCDFermionDoubled r(g), p(g), Mp(g), MdMp(g);
    x = Zero();
    r = b;
    p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(b), 1e-30);
    RealD tol2 = tol_ * tol_ * bsq;
    int it;
    for (it = 0; it < maxit_; ++it) {
      op.M(p, Mp);
      op.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      for (int a = 0; a < DtxqcdNf; ++a) {
        x.upper.f[a] = x.upper.f[a] + alpha * p.upper.f[a];
        x.lower.f[a] = x.lower.f[a] + alpha * p.lower.f[a];
        r.upper.f[a] = r.upper.f[a] - alpha * MdMp.upper.f[a];
        r.lower.f[a] = r.lower.f[a] - alpha * MdMp.lower.f[a];
      }
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) { rsq = rsq_new; break; }
      RealD beta = rsq_new / rsq;
      for (int a = 0; a < DtxqcdNf; ++a) {
        p.upper.f[a] = r.upper.f[a] + beta * p.upper.f[a];
        p.lower.f[a] = r.lower.f[a] + beta * p.lower.f[a];
      }
      rsq = rsq_new;
    }
    std::cout << Grid::GridLogMessage << "[DtxqcdMeasSolver::CG] iter=" << it
              << " rsq/bsq=" << rsq / bsq << std::endl;
  }

  Grid::GridCartesian &grid_;
  Grid::GridRedBlackCartesian &rbgrid_;
  Grid::RealD mass_, csw_, tol_;
  int maxit_;
  bool use_quda_ = false;
  bool use_eo_ = false;
  bool use_gridmg_ = false;
  Grid::LatticeGaugeField U_inv_;
  Grid::DTXQCDWilsonCloverFermionEO cpu_op_;
  Grid::DTXQCDMOp Mop_cpu_;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<Grid::DTXQCDMpcOpQUDA> quda_op_;
  std::unique_ptr<Grid::DTXQCDMpcOpQUDA_EO> eo_quda_op_;
  std::unique_ptr<Grid::DtxqcdPackedFullHermOp> packed_hop_;
  std::unique_ptr<Grid::DtxqcdGridMGSolver<32>> mg32_;
  std::unique_ptr<Grid::DtxqcdGridMGSolver<64>> mg64_;
#endif
};

}  // namespace TXQCDProduction
