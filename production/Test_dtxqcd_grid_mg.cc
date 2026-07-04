// Test_dtxqcd_grid_mg — Grid algebraic multigrid (HDCG / TwoLevelADEF2) on
// the DTXQCD EO Schur operator Mpc†Mpc, via the packed-field adapter
// (DTXQCDPackedField.h).  This is the route-(b) coarsening experiment: can
// geometric aggregation capture the doubled+aux operator's low modes at
// production lambda?  Stack cloned from tests/debug/Test_general_coarse_
// hdcg_phys.cc with the Mobius Schur op swapped for DtxqcdPackedMpcHermOp.
//
// Gates (all in one run):
//   A. pack/unpack round trip bit-exact + norm preservation
//   B. packed HermOp == struct-path Mpc†Mpc (Grid path ~exact; device 1e-10)
//   C. baseline CG on the packed HermOp (iters / time — the 380-400-iter
//      EO measurement solve reproduced through the adapter)
//   D. subspace setup + algebraic CoarsenOperator + TwoLevelADEF2 solve;
//      solution checked against the baseline CG solution; reports fine-op
//      call counts and wallclock per phase.
//
// Gauge/aux source: synthetic (GAUGE_KIND=tepid|hot|weak + FillAuxFields)
// by default, or a real checkpoint via CFG_DIR=<dir> TRAJ=<n> (thin links
// re-smeared with the production stout via DtxqcdInversionSmear, i.e. the
// exact operator the EO measurement solve inverts).
//
// Run (4^4 mechanics):   ./Test_dtxqcd_grid_mg --grid 4.4.4.4 --mpi 1.1.1.1
//                        env BLOCK=2.2.2.2 NBASIS=24
// Run (16^3x48 physics): env CFG_DIR=... TRAJ=... BLOCK=4.4.4.4 NBASIS=32
//                        ./Test_dtxqcd_grid_mg --grid 16.16.16.48 ...
//
// Env knobs: LAMBDA(4) MASS(-0.245) CSW(1.24930970916466) GAUGE_KIND(tepid)
//   AUX_SIGMA(0) CFG_DIR TRAJ NBASIS(32) BLOCK(4.4.4.4) GEOM(nnn|nearest)
//   SUBSPACE(cheby|powerlaw|inverse|multishift) — NB "cheby" carries a
//   hardwired ord-1000 refinement pass (Aggregates.h), "powerlaw" costs
//   exactly nbasis*CHEBY_ORD — CHEBY_ORD(300) CHEBY_LO(lmax*1e-3)
//   BENCH_HERMOP(20) DTXQCD_GRIDMG_SKIP_ZERO(0)
//   FULLVOL(0) — coarsen/solve the full-volume M†M instead of the odd-CB
//   Schur Mpc†Mpc (the better-conditioned substrate at small lambda: no
//   M_ee amplification; lam=4 full CG ~2100 iters vs EO 4005)
//   REFINE(0) REFINE_TOL(1e-3) REFINE_MAXIT(400) — inverse-iteration basis
//   polish after the filter (cost ~ nbasis*min(maxit, iters-to-tol))
//   SETUP_PREFIX(unset) — cache basis+coarse op via LIME per
//   (nbasis, geom, block); existing files -> load and skip setup entirely
//   (knob scans of smoother/coarse params then cost only the solve)
//   CHEBY_HI(lmax*1.1) MSHIFT_LO(lmax*1e-3) MSHIFT_TOL(1e-5) MSHIFT_MAXIT(2000)
//   SMOOTH_IT(8) MIRS_SHIFT(lmax*0.02) COARSE_TOL(1e-10) COARSE_TOL_SLOPPY(0.04)
//   CG_TOL(1e-8) CG_MAX(100000) MG_TOL(=CG_TOL) MG_MAXIT(700)
//   RUN_BASELINE(1) RUN_MG(1) DTXQCD_GRIDMG_QUDA_FINE(1)

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/AdefGeneric.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/util/QudaInit.h>
#endif

#include "params.h"
#include "dtxqcd_meas_helper.h"

#include <cstdlib>
#include <fstream>
#include <string>

using namespace Grid;

namespace {

// ---- setup save/load (from tests/debug/Test_general_coarse_hdcg_phys.cc) --
// SETUP_PREFIX=<path> caches the null-space basis + coarsened operator per
// (nbasis, geom, block) so smoother/coarse-solver knob scans skip the
// expensive setup.  Files: <prefix>.basis.<tag> / <prefix>.ldop.<tag>.
bool file_exists(const std::string &f) { return std::ifstream(f).good(); }

template <class aggregation>
void SaveBasis(aggregation &Agg, const std::string &file) {
#ifdef HAVE_LIME
  emptyUserRecord record;
  ScidacWriter WR(Agg.FineGrid->IsBoss());
  WR.open(file);
  for (int b = 0; b < (int)Agg.subspace.size(); b++)
    WR.writeScidacFieldRecord(Agg.subspace[b], record, 0, 0);
  WR.close();
#endif
}
template <class aggregation>
void LoadBasis(aggregation &Agg, const std::string &file) {
#ifdef HAVE_LIME
  emptyUserRecord record;
  ScidacReader RD;
  RD.open(file);
  for (int b = 0; b < (int)Agg.subspace.size(); b++)
    RD.readScidacFieldRecord(Agg.subspace[b], record, 0);
  RD.close();
#endif
}
template <class Coarsened>
void SaveOperator(Coarsened &Operator, const std::string &file) {
#ifdef HAVE_LIME
  emptyUserRecord record;
  ScidacWriter WR(Operator.Grid()->IsBoss());
  GRID_ASSERT(Operator._A.size() == Operator.geom.npoint);
  WR.open(file);
  for (int p = 0; p < (int)Operator._A.size(); p++) {
    auto tmp = Operator.Cell.Extract(Operator._A[p]);
    WR.writeScidacFieldRecord(tmp, record, 0, 0);
  }
  WR.close();
#endif
}
template <class Coarsened>
void LoadOperator(Coarsened &Operator, const std::string &file) {
#ifdef HAVE_LIME
  emptyUserRecord record;
  ScidacReader RD;
  RD.open(file);
  GRID_ASSERT(Operator._A.size() == Operator.geom.npoint);
  for (int p = 0; p < (int)Operator.geom.npoint; p++) {
    conformable(Operator._A[p].Grid(), Operator.CoarseGrid());
    RD.readScidacFieldRecord(Operator._A[p], record, 0);
  }
  RD.close();
  Operator.ExchangeCoarseLinks();
#endif
}

double env_real(const char *n, double d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atof(e) : d;
}
int env_int(const char *n, int d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atoi(e) : d;
}
std::string env_str(const char *n, const char *d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::string(e) : std::string(d);
}

Coordinate parse_dotted(const std::string &s, const Coordinate &dflt) {
  std::vector<int> dv;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, '.')) dv.push_back(std::stoi(t));
  if (dv.size() != (size_t)Nd) return dflt;
  return Coordinate(dv);
}

void pick_cb(int cb, DTXQCDFermionDoubled &half,
             const DTXQCDFermionDoubled &full) {
  for (int f = 0; f < DtxqcdNf; ++f) {
    pickCheckerboard(cb, half.upper.f[f], full.upper.f[f]);
    pickCheckerboard(cb, half.lower.f[f], full.lower.f[f]);
  }
}

RealD packed_rel_diff(const LatticeDtxqcdPacked &a,
                      const LatticeDtxqcdPacked &b) {
  LatticeDtxqcdPacked d(a.Grid());
  d.Checkerboard() = a.Checkerboard();
  LatticeDtxqcdPacked bb = b;
  bb.Checkerboard() = a.Checkerboard();  // flag-only alignment for the ET
  d = a - bb;
  return std::sqrt(norm2(d) / std::max(norm2(bb), 1e-300));
}

// Mirs smoother: fixed-iteration-count CG on a shifted HermOp (from the
// HDCG reference test; non-convergence is by design).
template <class Field>
class CGSmoother : public LinearFunction<Field> {
 public:
  using LinearFunction<Field>::operator();
  LinearOperatorBase<Field> &_SmootherOperator;
  int iters;
  CGSmoother(int _iters, LinearOperatorBase<Field> &SmootherOperator)
      : _SmootherOperator(SmootherOperator), iters(_iters) {}
  void operator()(const Field &in, Field &out) {
    ConjugateGradient<Field> CG(0.0, iters, false);
    CG(_SmootherOperator, in, out);
  }
};

struct MGParams {
  Coordinate block;
  std::string geom;
  std::string subspace;
  int cheby_ord;
  double cheby_lo, cheby_hi;     // <=0 -> derive from lmax
  double mshift_lo, mshift_tol;  // multishift subspace knobs
  int mshift_maxit;
  int smooth_it;
  double mirs_shift;             // <=0 -> lmax*0.02
  double coarse_tol, coarse_tol_sloppy;
  double mg_tol;
  int mg_maxit;
};

template <int nbasis, class HermOpT>
int run_mg(GridCartesian &Grid_, GridBase *FineGrid, int cb,
           HermOpT &FineHermOp, GridParallelRNG &pRNG,
           const LatticeDtxqcdPacked &src, const LatticeDtxqcdPacked *ref_sol,
           RealD lmax, const MGParams &P) {
  typedef LatticeDtxqcdPacked Field;
  typedef GeneralCoarsenedMatrix<vDtxqcdPackedVector, vTComplex, nbasis>
      LittleDiracOperator;
  typedef typename LittleDiracOperator::CoarseVector CoarseVector;
  typedef Aggregation<vDtxqcdPackedVector, vTComplex, nbasis> Subspace;

  int fails = 0;
  Coordinate latt = Grid_.GlobalDimensions();
  Coordinate clatt(Nd);
  for (int d = 0; d < Nd; ++d) {
    GRID_ASSERT(latt[d] % P.block[d] == 0);
    clatt[d] = latt[d] / P.block[d];
  }
  GridCartesian Coarse4d(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                         GridDefaultMpi());
  std::cout << GridLogMessage << "[gridmg] nbasis=" << nbasis << " block="
            << P.block << " coarse=" << clatt << " geom=" << P.geom
            << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Geometry + coarse operator shell (constructed before the subspace so
  // a cached setup can be loaded into it)
  //////////////////////////////////////////////////////////////////////
  std::unique_ptr<NonLocalStencilGeometry> geom;
  if (P.geom == "nearest")
    geom.reset(new NearestStencilGeometry4D(&Coarse4d));
  else
    geom.reset(new NextToNextToNextToNearestStencilGeometry4D(&Coarse4d));
  LittleDiracOperator LittleDiracOp(*geom, FineGrid, &Coarse4d);

  Subspace Aggregates(&Coarse4d, FineGrid, cb);
  double cheby_hi = (P.cheby_hi > 0) ? P.cheby_hi : 1.1 * lmax;
  double cheby_lo = (P.cheby_lo > 0) ? P.cheby_lo : 1.0e-3 * lmax;

  // Cached-setup path: SETUP_PREFIX + existing files -> skip subspace+coarsen.
  std::string setup_prefix = env_str("SETUP_PREFIX", "");
  std::string setup_tag;
  bool setup_loaded = false;
#ifdef HAVE_LIME
  if (!setup_prefix.empty()) {
    std::ostringstream tag;
    tag << nbasis << "." << P.geom << ".b" << P.block[0] << P.block[1]
        << P.block[2] << P.block[3];
    setup_tag = tag.str();
    std::string bf = setup_prefix + ".basis." + setup_tag;
    std::string of = setup_prefix + ".ldop." + setup_tag;
    if (file_exists(bf) && file_exists(of)) {
      double tl = usecond();
      LoadBasis(Aggregates, bf);
      Aggregates.Orthogonalise();  // idempotent on a saved-orthonormal basis
      LoadOperator(LittleDiracOp, of);
      setup_loaded = true;
      std::cout << GridLogMessage << "[gridmg] setup LOADED from "
                << setup_prefix << ".{basis,ldop}." << setup_tag << " in "
                << (usecond() - tl) * 1e-6 << " s" << std::endl;
    }
  }
#endif

  //////////////////////////////////////////////////////////////////////
  // Subspace (null vectors)
  //////////////////////////////////////////////////////////////////////
  size_t nc0 = FineHermOp.ncalls;
  double t0 = usecond();
  if (setup_loaded) {
    // nothing to do
  } else if (P.subspace == "inverse") {
    Aggregates.CreateSubspace(pRNG, FineHermOp, nbasis);
  } else if (P.subspace == "multishift") {
    double mlo = (P.mshift_lo > 0) ? P.mshift_lo : 1.0e-3 * lmax;
    Aggregates.CreateSubspaceMultishift(pRNG, FineHermOp, mlo, P.mshift_tol,
                                        P.mshift_maxit);
  } else if (P.subspace == "powerlaw") {
    // Single power-law Chebyshev pass: cost = nbasis * CHEBY_ORD, no
    // hardwired ord-1000 refinement (which dominates the "cheby" variant:
    // Aggregates.h:398 — CHEBY_ORD barely moves that one's cost).
    std::cout << GridLogMessage << "[gridmg] PowerLaw subspace: ord="
              << P.cheby_ord << " hi=" << cheby_hi << std::endl;
    Aggregates.CreateSubspaceChebyshevPowerLaw(pRNG, FineHermOp, nbasis,
                                               cheby_hi, P.cheby_ord);
  } else {
    std::cout << GridLogMessage << "[gridmg] Chebyshev subspace: ord="
              << P.cheby_ord << " [" << cheby_lo << "," << cheby_hi << "]"
              << std::endl;
    Aggregates.CreateSubspaceChebyshev(pRNG, FineHermOp, nbasis, cheby_hi,
                                       cheby_lo, P.cheby_ord);
  }
  // Optional inverse-iteration polish (REFINE=1): per-vector maxit-capped
  // sloppy CG on M†M — amplifies the low-mode content the filter can't reach
  // at large kappa (the HDCG-paper refinement).  Included in the saved setup.
  if (!setup_loaded && env_int("REFINE", 0) != 0) {
    double rtol = env_real("REFINE_TOL", 1e-3);
    int rmaxit = env_int("REFINE_MAXIT", 400);
    std::cout << GridLogMessage << "[gridmg] RefineSubspace: tol=" << rtol
              << " maxit=" << rmaxit << std::endl;
    Aggregates.RefineSubspace(FineHermOp, /*Lo (unused upstream)=*/0.0, rtol,
                              rmaxit);
  }
  if (!setup_loaded) Aggregates.Orthogonalise();
  double t_sub = (usecond() - t0) * 1e-6;
  size_t nc_sub = FineHermOp.ncalls - nc0;
  std::cout << GridLogMessage << "[gridmg] subspace done: " << nc_sub
            << " fine HermOps, " << t_sub << " s" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Coarsen (algebraic, Op-probe only)
  //////////////////////////////////////////////////////////////////////
  nc0 = FineHermOp.ncalls;
  t0 = usecond();
  if (!setup_loaded) {
    LittleDiracOp.CoarsenOperator(FineHermOp, Aggregates);
#ifdef HAVE_LIME
    if (!setup_prefix.empty()) {
      SaveBasis(Aggregates, setup_prefix + ".basis." + setup_tag);
      SaveOperator(LittleDiracOp, setup_prefix + ".ldop." + setup_tag);
      std::cout << GridLogMessage << "[gridmg] setup SAVED to "
                << setup_prefix << ".{basis,ldop}." << setup_tag << std::endl;
    }
#endif
  }
  double t_coarsen = (usecond() - t0) * 1e-6;
  std::cout << GridLogMessage << "[gridmg] CoarsenOperator done: "
            << (FineHermOp.ncalls - nc0) << " fine HermOps, " << t_coarsen
            << " s (npoint=" << geom->npoint << ")" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Two-level ADEF2
  //////////////////////////////////////////////////////////////////////
  typedef HermitianLinearOperator<LittleDiracOperator, CoarseVector>
      HermMatrix;
  HermMatrix CoarseOp(LittleDiracOp);

  ZeroGuesser<CoarseVector> CoarseZeroGuesser;
  ConjugateGradient<CoarseVector> CGcoarse(P.coarse_tol, 30000, false);
  ConjugateGradient<CoarseVector> CGcoarseSloppy(P.coarse_tol_sloppy, 30000,
                                                 false);
  HPDSolver<CoarseVector> HPDSolve(CoarseOp, CGcoarse, CoarseZeroGuesser);
  HPDSolver<CoarseVector> HPDSolveSloppy(CoarseOp, CGcoarseSloppy,
                                         CoarseZeroGuesser);

  double mirs = (P.mirs_shift > 0) ? P.mirs_shift : 0.02 * lmax;
  ShiftedHermOpLinearOperator<Field> ShiftedFineHermOp(FineHermOp, mirs);
  CGSmoother<Field> CGsmooth(P.smooth_it, ShiftedFineHermOp);
  std::cout << GridLogMessage << "[gridmg] smoother: CG(" << P.smooth_it
            << ") shift=" << mirs << std::endl;

  TwoLevelADEF2<Field, CoarseVector, Subspace> HDCG(
      P.mg_tol, P.mg_maxit, FineHermOp, CGsmooth, HPDSolveSloppy, HPDSolve,
      Aggregates);

  Field result(FineGrid);
  result = Zero();
  result.Checkerboard() = src.Checkerboard();
  nc0 = FineHermOp.ncalls;
  t0 = usecond();
  HDCG(src, result);
  double t_mg = (usecond() - t0) * 1e-6;
  size_t nc_mg = FineHermOp.ncalls - nc0;
  std::cout << GridLogMessage << "[gridmg] ADEF2 solve: " << nc_mg
            << " fine HermOps, " << t_mg << " s" << std::endl;

  // True residual through the fine op.
  Field tmp(FineGrid);
  tmp.Checkerboard() = src.Checkerboard();
  FineHermOp.HermOp(result, tmp);
  tmp = tmp - src;
  RealD true_res = std::sqrt(norm2(tmp) / norm2(src));
  bool res_ok = true_res < 10.0 * P.mg_tol;
  if (!res_ok) ++fails;
  std::cout << GridLogMessage << (res_ok ? "[ok]   " : "[FAIL] ")
            << "[gridmg] ADEF2 true residual = " << true_res << " (tol "
            << P.mg_tol << ")" << std::endl;

  if (ref_sol) {
    RealD rd = packed_rel_diff(result, *ref_sol);
    bool ok = rd < 100.0 * P.mg_tol;
    if (!ok) ++fails;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ")
              << "[gridmg] ADEF2 vs baseline CG solution rel = " << rd
              << std::endl;
  }
  std::cout << GridLogMessage << "[gridmg] totals: subspace " << t_sub
            << " s + coarsen " << t_coarsen << " s (setup), solve " << t_mg
            << " s" << std::endl;
  return fails;
}

// Gates A-D + benchmark + MG dispatch, templated over the (HermOp, Grid-op)
// pair so both the odd-CB Schur and full-volume instantiations run the same
// suite (FULLVOL=1 selects the latter in main).
template <class HermOpT, class MopT>
int run_suite(GridCartesian &Grid_, GridBase *FineGrid, int cb,
              HermOpT &FineHermOp, MopT &mop, GridParallelRNG &pRNG) {
  int fails = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = val < tol;
    if (!ok) ++fails;
    std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ") << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
  };

  //////////////////////////////////////////////////////////////////////
  // Gate A: pack/unpack round trip
  //////////////////////////////////////////////////////////////////////
  DTXQCDFermionDoubled psi_full(&Grid_), psi_o(FineGrid), psi_rt(FineGrid);
  for (int f = 0; f < DtxqcdNf; ++f) {
    gaussian(pRNG, psi_full.upper.f[f]);
    gaussian(pRNG, psi_full.lower.f[f]);
  }
  if (FineGrid->_isCheckerBoarded)
    pick_cb(cb, psi_o, psi_full);
  else
    psi_o = psi_full;

  LatticeDtxqcdPacked P(FineGrid);
  DtxqcdPackDoubled(psi_o, P);
  DtxqcdUnpackDoubled(P, psi_rt);
  RealD rt = 0, den = 0;
  for (int f = 0; f < DtxqcdNf; ++f) {
    LatticeFermion d(FineGrid);
    d.Checkerboard() = cb;
    d = psi_rt.upper.f[f] - psi_o.upper.f[f];
    rt += norm2(d);
    d = psi_rt.lower.f[f] - psi_o.lower.f[f];
    rt += norm2(d);
    den += norm2(psi_o.upper.f[f]) + norm2(psi_o.lower.f[f]);
  }
  check("gateA pack/unpack roundtrip rel", std::sqrt(rt / den), 1e-28);
  check("gateA norm preservation rel", std::abs(norm2(P) - den) / den, 1e-13);

  //////////////////////////////////////////////////////////////////////
  // Gate B: packed HermOp vs struct-path A†A
  //////////////////////////////////////////////////////////////////////
  LatticeDtxqcdPacked hop_out(FineGrid);
  hop_out.Checkerboard() = P.Checkerboard();
  FineHermOp.HermOp(P, hop_out);

  DTXQCDFermionDoubled t1(FineGrid), t2(FineGrid);
  mop.M(psi_o, t1);
  mop.Mdag(t1, t2);
  LatticeDtxqcdPacked ref_out(FineGrid);
  DtxqcdPackDoubled(t2, ref_out);
#ifdef GRID_HAVE_QUDA
  const RealD gateb_tol = 1e-9;  // device chain vs Grid reference
#else
  const RealD gateb_tol = 1e-13;  // identical arithmetic
#endif
  check("gateB packed HermOp vs struct", packed_rel_diff(hop_out, ref_out),
        gateb_tol);

  //////////////////////////////////////////////////////////////////////
  // Adapter-overhead micro-benchmark (per-phase ms; see header comment).
  //////////////////////////////////////////////////////////////////////
  const int nbench = env_int("BENCH_HERMOP", 20);
  if (nbench > 0) {
    FineHermOp.ResetCounters();
    for (int i = 0; i < nbench; ++i) FineHermOp.HermOp(P, hop_out);
    FineHermOp.ReportCounters("bench");
    FineHermOp.ResetCounters();
  }

  //////////////////////////////////////////////////////////////////////
  // Spectrum scale for Chebyshev / smoother windows
  //////////////////////////////////////////////////////////////////////
  LatticeDtxqcdPacked src(FineGrid);
  gaussian(pRNG, src);  // content-random; cb interpretation is the adapter's
  PowerMethod<LatticeDtxqcdPacked> PM;
  RealD lmax = PM(FineHermOp, src);
  std::cout << GridLogMessage << "[gridmg] PowerMethod lambda_max ~ " << lmax
            << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Gate C: baseline CG through the adapter
  //////////////////////////////////////////////////////////////////////
  const RealD cg_tol = env_real("CG_TOL", 1e-8);
  const int cg_max = env_int("CG_MAX", 100000);
  LatticeDtxqcdPacked sol_cg(FineGrid);
  sol_cg = Zero();
  sol_cg.Checkerboard() = src.Checkerboard();
  bool have_baseline = env_int("RUN_BASELINE", 1) != 0;
  if (have_baseline) {
    size_t nc0 = FineHermOp.ncalls;
    double t0 = usecond();
    ConjugateGradient<LatticeDtxqcdPacked> CG(cg_tol, cg_max, false);
    CG(FineHermOp, src, sol_cg);
    double t_cg = (usecond() - t0) * 1e-6;
    std::cout << GridLogMessage << "[gridmg] baseline CG: "
              << (FineHermOp.ncalls - nc0) << " fine HermOps, " << t_cg
              << " s (" << t_cg / std::max<size_t>(FineHermOp.ncalls - nc0, 1)
              << " s/HermOp)" << std::endl;
    LatticeDtxqcdPacked tmp(FineGrid);
    tmp.Checkerboard() = src.Checkerboard();
    FineHermOp.HermOp(sol_cg, tmp);
    tmp = tmp - src;
    check("gateC baseline CG true residual",
          std::sqrt(norm2(tmp) / norm2(src)), 10.0 * cg_tol);
  }

  //////////////////////////////////////////////////////////////////////
  // Gate D: the MG experiment
  //////////////////////////////////////////////////////////////////////
  if (env_int("RUN_MG", 1) != 0) {
    MGParams Pmg;
    Pmg.block = parse_dotted(env_str("BLOCK", "4.4.4.4"),
                             Coordinate(std::vector<int>{4, 4, 4, 4}));
    Pmg.geom = env_str("GEOM", "nnn");
    Pmg.subspace = env_str("SUBSPACE", "cheby");
    Pmg.cheby_ord = env_int("CHEBY_ORD", 300);
    Pmg.cheby_lo = env_real("CHEBY_LO", -1.0);
    Pmg.cheby_hi = env_real("CHEBY_HI", -1.0);
    Pmg.mshift_lo = env_real("MSHIFT_LO", -1.0);
    Pmg.mshift_tol = env_real("MSHIFT_TOL", 1e-5);
    Pmg.mshift_maxit = env_int("MSHIFT_MAXIT", 2000);
    Pmg.smooth_it = env_int("SMOOTH_IT", 8);
    Pmg.mirs_shift = env_real("MIRS_SHIFT", -1.0);
    Pmg.coarse_tol = env_real("COARSE_TOL", 1e-10);
    Pmg.coarse_tol_sloppy = env_real("COARSE_TOL_SLOPPY", 0.04);
    Pmg.mg_tol = env_real("MG_TOL", cg_tol);
    Pmg.mg_maxit = env_int("MG_MAXIT", 700);

    const LatticeDtxqcdPacked *ref = have_baseline ? &sol_cg : nullptr;
    const int nb = env_int("NBASIS", 32);
    switch (nb) {
      case 24:
        fails += run_mg<24>(Grid_, FineGrid, cb, FineHermOp, pRNG, src, ref,
                            lmax, Pmg);
        break;
      case 32:
        fails += run_mg<32>(Grid_, FineGrid, cb, FineHermOp, pRNG, src, ref,
                            lmax, Pmg);
        break;
      case 48:
        fails += run_mg<48>(Grid_, FineGrid, cb, FineHermOp, pRNG, src, ref,
                            lmax, Pmg);
        break;
      default:
        std::cout << GridLogMessage
                  << "[gridmg] NBASIS must be one of 24/32/48 (compiled "
                     "instantiations); got "
                  << nb << std::endl;
        ++fails;
    }
  }
  std::cout << GridLogMessage << "[gridmg] total fine HermOps this run: "
            << FineHermOp.ncalls << " (" << FineHermOp.t_herm_us * 1e-6
            << " s)" << std::endl;
  FineHermOp.ReportCounters("run-aggregate");
  return fails;
}

}  // namespace

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--grid" && i + 1 < argc)
      latt = parse_dotted(argv[i + 1], latt);
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridSerialRNG sRNG;
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({11, 12, 13, 14, 15});

  const RealD lambda = env_real("LAMBDA", 4.0);
  const RealD mass = env_real("MASS", -0.245);
  const RealD csw_v = env_real("CSW", 1.24930970916466);

  //////////////////////////////////////////////////////////////////////
  // Gauge + aux: checkpoint (production operator) or synthetic
  //////////////////////////////////////////////////////////////////////
  DTXQCDField U(&Grid_);
  std::string cfg_dir = env_str("CFG_DIR", "");
  if (!cfg_dir.empty()) {
    const int traj = env_int("TRAJ", 0);
    DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG, cfg_dir + "/ckpoint_lat",
                                   cfg_dir + "/ckpoint_rng", traj);
    std::cout << GridLogMessage << "[gridmg] cfg " << cfg_dir << " traj "
              << traj << " plaq "
              << WilsonLoops<PeriodicGimplR>::avgPlaquette(U.U) << std::endl;
  } else {
    std::string gk = env_str("GAUGE_KIND", "tepid");
    if (gk == "hot")
      SU<Nc>::HotConfiguration(pRNG, U.U);
    else if (gk == "weak")
      DTXQCDCompositeImpl::GenerateWeakFieldGauge(pRNG, U, 0.1);
    else
      PeriodicGimplR::TepidConfiguration(pRNG, U.U);
    DTXQCDCompositeImpl::FillAuxFields(pRNG, U, lambda,
                                       env_real("AUX_SIGMA", 0.0));
    std::cout << GridLogMessage << "[gridmg] synthetic gauge=" << gk
              << " lambda=" << lambda << std::endl;
  }

  // Same smearing the measurement/HMC operator sees (no-op for synthetic
  // runs unless STOUT_NSMEAR is set).
  LatticeGaugeField U_inv =
      TXQCDProduction::DtxqcdInversionSmear(Grid_, U.U);

  DTXQCDWilsonCloverFermionEO Dw(U_inv, Grid_, RBGrid, mass, csw_v, U.sigma,
                                 U.pi, U.d, U.n, U.s, U.p);

  // FULLVOL=1: run the suite on the full-volume operator M†M instead of the
  // odd-CB Schur Mpc†Mpc.  The M6 probe (2026-07-03) showed the full
  // operator is far better conditioned at small lambda (no M_ee
  // amplification): lam=4 traj 120 full CG ~2100 iters vs EO CG 4005 (EO
  // lambda_max 9846).
  const bool fullvol = env_int("FULLVOL", 0) != 0;
  int fails = 0;
  if (fullvol) {
    DTXQCDMOp mop(Dw);
#ifdef GRID_HAVE_QUDA
    Grid::Quda::initialize();
    DTXQCDMpcOpQUDA Mq(U_inv, Grid_, RBGrid, mass, csw_v, U.sigma, U.pi,
                       U.d, U.n, U.s, U.p);
    DtxqcdPackedFullHermOp FineHermOp(mop, Mq, &Grid_, Even);
#else
    DtxqcdPackedFullHermOp FineHermOp(mop, &Grid_, Even);
#endif
    std::cout << GridLogMessage << "[gridmg] FULLVOL mode: fine grid = full "
              << "volume, operator = M†M" << std::endl;
    fails = run_suite(Grid_, (GridBase *)&Grid_, Even, FineHermOp, mop, pRNG);
  } else {
    DTXQCDMpcOp mpc(Dw);
#ifdef GRID_HAVE_QUDA
    Grid::Quda::initialize();
    DTXQCDMpcOpQUDA_EO Mq(Dw, U_inv, Grid_, RBGrid, mass, csw_v, U.sigma,
                          U.pi, U.d, U.n, U.s, U.p);
    DtxqcdPackedMpcHermOp FineHermOp(mpc, Mq, &RBGrid, Odd);
#else
    DtxqcdPackedMpcHermOp FineHermOp(mpc, &RBGrid, Odd);
#endif
    fails = run_suite(Grid_, (GridBase *)&RBGrid, Odd, FineHermOp, mpc, pRNG);
  }

  std::cout << GridLogMessage << "==== gridmg "
            << (fails ? "FAILED" : "PASSED") << " (" << fails
            << " failures) ====" << std::endl;
  Grid_finalize();
  return fails ? 1 : 0;
}
