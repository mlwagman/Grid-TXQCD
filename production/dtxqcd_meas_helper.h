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
// Solve path — full-volume single-shift CG on M†M (per CLAUDE.md: measurement
// inversions use full-volume MdagM CG, not EO):
//
//   DTXQCD_MEAS_QUDA=1 (default under GRID_HAVE_QUDA): route M/Mdag through
//     DTXQCDMpcOpQUDA — the exact full-volume operator the non-EO HMC drives
//     (QUDA-offloaded Wilson hop + device aux).
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
#include <Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC.h>
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
#ifdef GRID_HAVE_QUDA
    if (use_quda_) {
      Grid::Quda::initialize();  // idempotent
      quda_op_.reset(new Grid::DTXQCDMpcOpQUDA(
          U_inv_, grid, rbgrid, mass, csw_val,
          U.sigma, U.pi, U.d, U.n, U.s, U.p));
    }
#endif
    std::cout << Grid::GridLogMessage
              << "[DtxqcdMeasSolver] mass=" << mass_ << " csw=" << csw_
              << " stout(nsmear=" << stout_nsmear_inv
              << ", rho=" << stout_rho_inv << ")"
              << " tol=" << tol_ << " maxit=" << maxit_
              << " backend=" << (use_quda_ ? "QUDA" : "Grid-CPU")
              << std::endl;
  }

  // Solve M x = source on the full volume (x = M^{-1} source), via CG on M†M:
  //   x = (M†M)^{-1} (M† source).
  void Solve(const Grid::DTXQCDFermionDoubled &source,
             Grid::DTXQCDFermionDoubled &x) {
#ifdef GRID_HAVE_QUDA
    if (use_quda_ && quda_op_) { SolveQuda(source, x); return; }
#endif
    SolveWith(Mop_cpu_, source, x);
  }

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
  Grid::LatticeGaugeField U_inv_;
  Grid::DTXQCDWilsonCloverFermionEO cpu_op_;
  Grid::DTXQCDMOp Mop_cpu_;
#ifdef GRID_HAVE_QUDA
  std::unique_ptr<Grid::DTXQCDMpcOpQUDA> quda_op_;
#endif
};

}  // namespace TXQCDProduction
