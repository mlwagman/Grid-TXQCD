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
//   COARSE_NEV(0) COARSE_IRL_LO(clmax*1e-3) COARSE_IRL_ORD(201) — deflate
//   the coarse solves with IRL evecs of the coarse operator
//   SMOOTH_SP(0) — run the smoother's matvecs in single precision via the
//   adapter's SP chain (FULLVOL substrate only; outer stays DP)
//   THREE_LEVEL(0) — nested-ADEF2 coarse solvers (NBASIS=64 only);
//   BLOCK2(2.2.2.2) L2_ORD(300) L2_SMOOTH_IT(8) L2_MIRS_SHIFT GEOM2(nnn)
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
#include <Grid/algorithms/iterative/AdefMrhs.h>
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
  int coarse_nev;                // >0: deflate coarse solves with IRL evecs
  double coarse_irl_lo;          // <=0 -> 1e-3 * coarse lmax
  int coarse_irl_ord;
  double coarse_irl_tol;         // deflation-grade default 1e-3
  bool smooth_sp;                // SP smoother matvecs (FULLVOL only)
};

// SP-matvec shim for the smoother: LinearOperatorBase whose HermOp routes
// through the adapter's single-precision chain.  Outer ADEF2 stays DP.
template <class AdapterT>
class SpHermOpShim : public LinearOperatorBase<LatticeDtxqcdPacked> {
 public:
  typedef LatticeDtxqcdPacked Field;
  explicit SpHermOpShim(AdapterT &a) : a_(a) {}
  void HermOp(const Field &in, Field &out) { a_.HermOpSp(in, out); }
  void Op(const Field &in, Field &out) { a_.HermOpSp(in, out); }
  void AdjOp(const Field &in, Field &out) { a_.HermOpSp(in, out); }
  void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) {
    a_.HermOpSp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void OpDiag(const Field &in, Field &out) { GRID_ASSERT(0); }
  void OpDir(const Field &in, Field &out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const Field &in, std::vector<Field> &out) { GRID_ASSERT(0); }

 private:
  AdapterT &a_;
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
  else if (P.geom == "nn")
    // 25-point middle ground: the 9-point truncation of the squared-op
    // Galerkin operator is INDEFINITE at lambda=3 (negative coarse evals,
    // v6 job 1298092), while the 81-point nnn stencil's depth-4 padded
    // links are OOM at nbasis=64.
    geom.reset(new NextToNearestStencilGeometry4D(&Coarse4d));
  else
    geom.reset(new NextToNextToNextToNearestStencilGeometry4D(&Coarse4d));
  LittleDiracOperator LittleDiracOp(*geom, FineGrid, &Coarse4d);

  Subspace Aggregates(&Coarse4d, FineGrid, cb);
  double cheby_hi = (P.cheby_hi > 0) ? P.cheby_hi : 1.1 * lmax;
  double cheby_lo = (P.cheby_lo > 0) ? P.cheby_lo : 1.0e-3 * lmax;

  // Cached-setup path: SETUP_PREFIX + existing files.  Basis and coarse
  // operator load INDEPENDENTLY — a coarsen-phase crash (e.g. the nb64 OOM,
  // 2026-07-04, which cost a 51-min basis) leaves a usable saved basis that
  // a rerun picks up, recomputing only the coarse links.
  std::string setup_prefix = env_str("SETUP_PREFIX", "");
  std::string setup_tag;
  bool basis_loaded = false, ldop_loaded = false;
#ifdef HAVE_LIME
  if (!setup_prefix.empty()) {
    std::ostringstream tag;
    tag << nbasis << "." << P.geom << ".b" << P.block[0] << P.block[1]
        << P.block[2] << P.block[3];
    setup_tag = tag.str();
    std::string bf = setup_prefix + ".basis." + setup_tag;
    std::string of = setup_prefix + ".ldop." + setup_tag;
    if (file_exists(bf)) {
      double tl = usecond();
      LoadBasis(Aggregates, bf);
      Aggregates.Orthogonalise();  // idempotent on a saved-orthonormal basis
      basis_loaded = true;
      std::cout << GridLogMessage << "[gridmg] basis LOADED from " << bf
                << " in " << (usecond() - tl) * 1e-6 << " s" << std::endl;
      if (file_exists(of)) {
        double tl2 = usecond();
        LoadOperator(LittleDiracOp, of);
        ldop_loaded = true;
        std::cout << GridLogMessage << "[gridmg] ldop LOADED from " << of
                  << " in " << (usecond() - tl2) * 1e-6 << " s" << std::endl;
      }
    }
  }
#endif
  const bool setup_loaded = basis_loaded;  // legacy name for the phases below

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
  if (!setup_loaded) {
    Aggregates.Orthogonalise();
#ifdef HAVE_LIME
    // Save the basis BEFORE coarsening: a coarsen OOM/crash must not lose
    // the (much more expensive) subspace work.
    if (!setup_prefix.empty()) {
      SaveBasis(Aggregates, setup_prefix + ".basis." + setup_tag);
      std::cout << GridLogMessage << "[gridmg] basis SAVED to "
                << setup_prefix << ".basis." << setup_tag << std::endl;
    }
#endif
  }
  double t_sub = (usecond() - t0) * 1e-6;
  size_t nc_sub = FineHermOp.ncalls - nc0;
  std::cout << GridLogMessage << "[gridmg] subspace done: " << nc_sub
            << " fine HermOps, " << t_sub << " s" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Coarsen (algebraic, Op-probe only)
  //////////////////////////////////////////////////////////////////////
  nc0 = FineHermOp.ncalls;
  t0 = usecond();
  if (!ldop_loaded) {
    LittleDiracOp.CoarsenOperator(FineHermOp, Aggregates);
#ifdef HAVE_LIME
    if (!setup_prefix.empty()) {
      SaveOperator(LittleDiracOp, setup_prefix + ".ldop." + setup_tag);
      std::cout << GridLogMessage << "[gridmg] ldop SAVED to "
                << setup_prefix << ".ldop." << setup_tag << std::endl;
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

  // Coarse deflation (COARSE_NEV>0): IRL on the coarse operator once per
  // setup, DeflatedGuesser feeds both coarse solvers.  Kills the dominant
  // non-matvec cost (sloppy coarse CG was ~135 its/outer on the lam4 EO
  // spectrum) at a one-time coarse-Lanczos cost.
  ZeroGuesser<CoarseVector> CoarseZeroGuesser;
  std::vector<RealD> c_eval;
  std::vector<CoarseVector> c_evec;
  std::unique_ptr<DeflatedGuesser<CoarseVector>> CoarseDeflGuesser;
  LinearFunction<CoarseVector> *CoarseGuess = &CoarseZeroGuesser;
  if (P.coarse_nev > 0) {
    double t_irl0 = usecond();
    GridParallelRNG CRNG(&Coarse4d);
    CRNG.SeedFixedIntegers({21, 22, 23, 24});
    CoarseVector c_src(&Coarse4d);
    gaussian(CRNG, c_src);
    PowerMethod<CoarseVector> cPM;
    RealD clmax = cPM(CoarseOp, c_src);
    double irl_lo =
        (P.coarse_irl_lo > 0) ? P.coarse_irl_lo : 1.0e-3 * clmax;
    const int Nstop = P.coarse_nev;
    // Nk > Nstop: convergence-window buffer.  With Nk == Nstop, deep requests
    // (nev=128, s3A 2026-07-08) end 10 restarts with individually-converged
    // modes but a 0/N contiguous count -> Grid IRL abort()s; a 16-vector
    // buffer between the wanted window and the restart truncation fixes the
    // counting.  Restart cap env COARSE_IRL_MAXIT (was hardcoded 10).
    const int Nk = P.coarse_nev + 16;
    // 2x Krylov headroom (the phys-test ratio); deflation-guesser evecs only
    // need ~1e-3 residuals (IRL's internal target is eresid^2) — demanding
    // measurement-grade 1e-5 ground restarts to 0/N converged (o1 failure).
    // Chebyshev hi MUST clear the true coarse lmax: T_ord grows exponentially
    // outside the window, and a PowerMethod ESTIMATE undershooting by >10%
    // NaN'd the whole Lanczos (p2/p3 failure) — 1.5x margin, env-overridable
    // via COARSE_IRL_HI.  Note deflation presumes the coarse op HAS low
    // modes; at toy volumes (4^3x8: lowest coarse eval O(1)) there is
    // nothing below lo and IRL grinds — leave COARSE_NEV=0 there.
    const int Nm = Nk + std::max(64, Nk);
    double irl_hi = env_real("COARSE_IRL_HI", 1.5 * clmax);
    const int irl_restarts = env_int("COARSE_IRL_MAXIT", 10);
    Chebyshev<CoarseVector> IRLCheby(irl_lo, irl_hi, P.coarse_irl_ord);
    FunctionHermOp<CoarseVector> IRLOpCheby(IRLCheby, CoarseOp);
    PlainHermOp<CoarseVector> IRLOp(CoarseOp);
    ImplicitlyRestartedLanczos<CoarseVector> IRL(IRLOpCheby, IRLOp, Nstop, Nk,
                                                 Nm, P.coarse_irl_tol,
                                                 irl_restarts);
    c_eval.resize(Nm);
    c_evec.resize(Nm, &Coarse4d);
    int Nconv = 0;
    IRL.calc(c_eval, c_evec, c_src, Nconv);
    // IRL returns exactly Nstop entries but its (inclusive, j<=Nk) final
    // test can report Nconv = Nstop+1 — resizing UP appends a zero
    // eval/evec pair and the DeflatedGuesser divides by it (NaN guess).
    const int nuse = std::min(Nconv, (int)c_eval.size());
    c_eval.resize(nuse);
    c_evec.resize(nuse, &Coarse4d);
    CoarseDeflGuesser.reset(
        new DeflatedGuesser<CoarseVector>(c_evec, c_eval));
    CoarseGuess = CoarseDeflGuesser.get();
    std::cout << GridLogMessage << "[gridmg] coarse IRL: " << Nconv
              << " evecs (lmax_c=" << clmax << ", lo=" << irl_lo << ") in "
              << (usecond() - t_irl0) * 1e-6 << " s" << std::endl;
  }
  ConjugateGradient<CoarseVector> CGcoarse(P.coarse_tol, 30000, false);
  ConjugateGradient<CoarseVector> CGcoarseSloppy(P.coarse_tol_sloppy, 30000,
                                                 false);
  HPDSolver<CoarseVector> HPDSolve(CoarseOp, CGcoarse, *CoarseGuess);
  HPDSolver<CoarseVector> HPDSolveSloppy(CoarseOp, CGcoarseSloppy,
                                         *CoarseGuess);

  double mirs = (P.mirs_shift > 0) ? P.mirs_shift : 0.02 * lmax;
  ShiftedHermOpLinearOperator<Field> ShiftedFineHermOp(FineHermOp, mirs);
  CGSmoother<Field> CGsmooth(P.smooth_it, ShiftedFineHermOp);
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  SpHermOpShim<HermOpT> SpShim(FineHermOp);
  ShiftedHermOpLinearOperator<Field> ShiftedSpHermOp(SpShim, mirs);
  CGSmoother<Field> CGsmoothSp(P.smooth_it, ShiftedSpHermOp);
  LinearFunction<Field> &Smoother =
      P.smooth_sp ? static_cast<LinearFunction<Field> &>(CGsmoothSp)
                  : static_cast<LinearFunction<Field> &>(CGsmooth);
#else
  LinearFunction<Field> &Smoother = CGsmooth;
#endif
  std::cout << GridLogMessage << "[gridmg] smoother: CG(" << P.smooth_it
            << ") shift=" << mirs << (P.smooth_sp ? " [SP matvec]" : "")
            << std::endl;

  // Shared solve phase, parameterized on the level-2 solvers so the
  // three-level variant below swaps in nested ADEF2s.
  auto solve_phase = [&](LinearFunction<CoarseVector> &CSloppy,
                         LinearFunction<CoarseVector> &CPrecise) -> int {
    int lfails = 0;
    TwoLevelADEF2<Field, CoarseVector, Subspace> HDCG(
        P.mg_tol, P.mg_maxit, FineHermOp, Smoother, CSloppy, CPrecise,
        Aggregates);

    Field result(FineGrid);
    result = Zero();
    result.Checkerboard() = src.Checkerboard();
    size_t nc1 = FineHermOp.ncalls;
    double ts = usecond();
    HDCG(src, result);
    double t_mg = (usecond() - ts) * 1e-6;
    size_t nc_mg = FineHermOp.ncalls - nc1;
    std::cout << GridLogMessage << "[gridmg] ADEF2 solve: " << nc_mg
              << " fine HermOps, " << t_mg << " s" << std::endl;

    // True residual through the fine op.
    Field tmp(FineGrid);
    tmp.Checkerboard() = src.Checkerboard();
    FineHermOp.HermOp(result, tmp);
    tmp = tmp - src;
    RealD true_res = std::sqrt(norm2(tmp) / norm2(src));
    bool res_ok = true_res < 10.0 * P.mg_tol;
    if (!res_ok) ++lfails;
    std::cout << GridLogMessage << (res_ok ? "[ok]   " : "[FAIL] ")
              << "[gridmg] ADEF2 true residual = " << true_res << " (tol "
              << P.mg_tol << ")" << std::endl;

    if (ref_sol) {
      RealD rd = packed_rel_diff(result, *ref_sol);
      bool ok = rd < 100.0 * P.mg_tol;
      if (!ok) ++lfails;
      std::cout << GridLogMessage << (ok ? "[ok]   " : "[FAIL] ")
                << "[gridmg] ADEF2 vs baseline CG solution rel = " << rd
                << std::endl;
    }
    std::cout << GridLogMessage << "[gridmg] totals: subspace " << t_sub
              << " s + coarsen " << t_coarsen << " s (setup), solve " << t_mg
              << " s" << std::endl;
    return lfails;
  };

  //////////////////////////////////////////////////////////////////////
  // MRHS_NRHS=N (mrhs-M1, compiled for nbasis 32/64): batched ADEF2 over
  // N simultaneous sources.  The mrhs coarse grid makes nrhs the LEADING
  // dimension carrying all the SIMD lanes (rhs-as-vector-lanes — the
  // phys48_blockcg idiom), so block projects and coarse solves become
  // full-width BLAS.  Fine matvecs + smoother remain per-rhs loops in
  // stock TwoLevelADEF2mrhs (fine batching = the M2 program).  Reuses the
  // aggregation basis (ImportBasis) and the coarse IRL eigenpairs
  // (ImportEigenBasis) — COARSE_NEV>0 required.  Replaces the single-RHS
  // solve when active (baselines are banked from prior runs).
  //////////////////////////////////////////////////////////////////////
  const int mrhs_nrhs = env_int("MRHS_NRHS", 0);
  if constexpr (nbasis == 32 || nbasis == 64) {
    if (mrhs_nrhs > 0) {
      GRID_ASSERT(P.coarse_nev > 0 &&
                  CoarseDeflGuesser);  // deflation eigenpairs required
      GRID_ASSERT(mrhs_nrhs % vComplex::Nsimd() == 0 &&
                  "MRHS_NRHS must be a multiple of vComplex::Nsimd()");
      Coordinate mpi5 = GridDefaultMpi();
      Coordinate rhMpi({1, mpi5[0], mpi5[1], mpi5[2], mpi5[3]});
      Coordinate rhLatt({mrhs_nrhs, clatt[0], clatt[1], clatt[2], clatt[3]});
      Coordinate rhSimd({(int)vComplex::Nsimd(), 1, 1, 1, 1});
      GridCartesian CoarseMrhs(rhLatt, rhSimd, rhMpi);
      std::cout << GridLogMessage << "[gridmg] MRHS: nrhs=" << mrhs_nrhs
                << " coarse-mrhs grid " << rhLatt << std::endl;

      typedef MultiGeneralCoarsenedMatrix<vDtxqcdPackedVector, vTComplex,
                                          nbasis>
          MrhsOp_t;
      MrhsOp_t mrhsOp(*geom, &CoarseMrhs);
      size_t ncm0 = FineHermOp.ncalls;
      double tcm0 = usecond();
      mrhsOp.CoarsenOperator(FineHermOp, Aggregates, &Coarse4d);
      std::cout << GridLogMessage << "[gridmg] MRHS CoarsenOperator: "
                << (FineHermOp.ncalls - ncm0) << " fine HermOps, "
                << (usecond() - tcm0) * 1e-6 << " s" << std::endl;

      typedef HermitianLinearOperator<MrhsOp_t, CoarseVector> MrhsHermMatrix;
      MrhsHermMatrix MrhsCoarseOp(mrhsOp);
      ConjugateGradient<CoarseVector> mrhsCGsloppy(P.coarse_tol_sloppy,
                                                   30000, false);
      ConjugateGradient<CoarseVector> mrhsCG(P.coarse_tol, 30000, false);
      DoNothingGuesser<CoarseVector> DoNothing;
      HPDSolver<CoarseVector> MrhsSolveSloppy(MrhsCoarseOp, mrhsCGsloppy,
                                              DoNothing);
      HPDSolver<CoarseVector> MrhsSolve(MrhsCoarseOp, mrhsCG, DoNothing);

      MultiRHSDeflation<CoarseVector> MrhsDeflator;
      MrhsDeflator.Allocate((int)c_eval.size(), &Coarse4d);
      MrhsDeflator.ImportEigenBasis(c_evec, c_eval);

      MultiRHSBlockProject<Field> MrhsProjector;
      MrhsProjector.Allocate(nbasis, FineGrid, &Coarse4d);
      MrhsProjector.ImportBasis(Aggregates.subspace);

      TwoLevelADEF2mrhs<Field, CoarseVector> HDCGmrhs(
          P.mg_tol, P.mg_maxit, FineHermOp, Smoother, MrhsSolveSloppy,
          MrhsSolve, MrhsProjector, MrhsDeflator, &CoarseMrhs);

      std::vector<Field> src_m(mrhs_nrhs, FineGrid);
      std::vector<Field> res_m(mrhs_nrhs, FineGrid);
      for (int r = 0; r < mrhs_nrhs; ++r) {
        gaussian(pRNG, src_m[r]);
        res_m[r] = Zero();
        res_m[r].Checkerboard() = src_m[r].Checkerboard();
      }
      size_t ncs0 = FineHermOp.ncalls;
      double ts0 = usecond();
      HDCGmrhs(src_m, res_m);
      double t_mrhs = (usecond() - ts0) * 1e-6;
      std::cout << GridLogMessage << "[gridmg] MRHS solve: " << mrhs_nrhs
                << " rhs in " << t_mrhs << " s ("
                << t_mrhs / mrhs_nrhs << " s/rhs amortized, "
                << (FineHermOp.ncalls - ncs0) << " fine HermOps total)"
                << std::endl;
      int bad = 0;
      for (int r = 0; r < mrhs_nrhs; ++r) {
        Field tmp_r(FineGrid);
        tmp_r.Checkerboard() = src_m[r].Checkerboard();
        FineHermOp.HermOp(res_m[r], tmp_r);
        tmp_r = tmp_r - src_m[r];
        RealD rr = std::sqrt(norm2(tmp_r) / norm2(src_m[r]));
        if (!(rr < 10.0 * P.mg_tol)) ++bad;
        if (r == 0 || !(rr < 10.0 * P.mg_tol))
          std::cout << GridLogMessage << (rr < 10.0 * P.mg_tol ? "[ok]   "
                                                               : "[FAIL] ")
                    << "[gridmg] MRHS rhs " << r << " true residual = " << rr
                    << std::endl;
      }
      if (bad) fails += bad;
      std::cout << GridLogMessage << "[gridmg] MRHS totals: coarsen "
                << (usecond() - tcm0) * 1e-6 << " s incl-solve, solve "
                << t_mrhs << " s, " << t_mrhs / mrhs_nrhs << " s/rhs"
                << std::endl;
      return fails;
    }
  } else if (mrhs_nrhs > 0) {
    std::cout << GridLogMessage
              << "[gridmg] MRHS is compiled for NBASIS=32/64 only" << std::endl;
  }

  //////////////////////////////////////////////////////////////////////
  // THREE_LEVEL=1 (compiled for nbasis==64 only): the outer ADEF2's coarse
  // solvers become nested TwoLevelADEF2s — the level-2 operator (the
  // 64-color coarse op) is itself aggregated (BLOCK2, nbasis2=32) and
  // corrected on a third level.  The coarse vector is a proper
  // Lattice<iVector<vTComplex,64>>, so the same algebraic machinery
  // composes with no adapter.
  //////////////////////////////////////////////////////////////////////
  const bool three_level = env_int("THREE_LEVEL", 0) != 0;
  if constexpr (nbasis == 64) {
    if (three_level) {
      typedef typename LittleDiracOperator::siteVector CoarseSiteVector;
      constexpr int nbasis2 = 32;
      // Level-2 CComplex must be iScalar-wrapped (the HDCR 3-level idiom,
      // tests/solver/Test_dwf_hdcr_48_rb.cc:307,320) or tensor arithmetic
      // over iVector<vTComplex,64> sites does not close.
      typedef Aggregation<CoarseSiteVector, iScalar<vTComplex>, nbasis2>
          Subspace2;
      typedef GeneralCoarsenedMatrix<CoarseSiteVector, iScalar<vTComplex>,
                                     nbasis2>
          LittleDiracOperator2;
      typedef typename LittleDiracOperator2::CoarseVector CCVector;
      typedef HermitianLinearOperator<LittleDiracOperator2, CCVector>
          HermMatrix2;

      Coordinate block2 = parse_dotted(env_str("BLOCK2", "2.2.2.2"),
                                       Coordinate(std::vector<int>{2, 2, 2, 2}));
      Coordinate cclatt(Nd);
      for (int d = 0; d < Nd; ++d) {
        GRID_ASSERT(clatt[d] % block2[d] == 0);
        cclatt[d] = clatt[d] / block2[d];
      }
      GridCartesian Coarse2(cclatt, GridDefaultSimd(Nd, vComplex::Nsimd()),
                            GridDefaultMpi());
      std::cout << GridLogMessage << "[gridmg] THREE_LEVEL: coarse2="
                << cclatt << " nbasis2=" << nbasis2 << std::endl;

      GridParallelRNG CRNG(&Coarse4d);
      CRNG.SeedFixedIntegers({31, 32, 33, 34});
      CoarseVector c_probe(&Coarse4d);
      gaussian(CRNG, c_probe);
      PowerMethod<CoarseVector> cPM;
      RealD clmax2 = cPM(CoarseOp, c_probe);

      double t2_0 = usecond();
      Subspace2 Aggregates2(&Coarse2, &Coarse4d, 0);
      Aggregates2.CreateSubspaceChebyshevPowerLaw(
          CRNG, CoarseOp, nbasis2, 1.5 * clmax2, env_int("L2_ORD", 300));
      Aggregates2.Orthogonalise();
      std::unique_ptr<NonLocalStencilGeometry> geom2;
      std::string g2 = env_str("GEOM2", "nnn");
      if (g2 == "nearest")
        geom2.reset(new NearestStencilGeometry4D(&Coarse2));
      else if (g2 == "nn")
        geom2.reset(new NextToNearestStencilGeometry4D(&Coarse2));
      else
        geom2.reset(new NextToNextToNextToNearestStencilGeometry4D(&Coarse2));
      LittleDiracOperator2 LittleDiracOp2(*geom2, &Coarse4d, &Coarse2);
      LittleDiracOp2.CoarsenOperator(CoarseOp, Aggregates2);
      std::cout << GridLogMessage << "[gridmg] level-2 coarsen done in "
                << (usecond() - t2_0) * 1e-6 << " s" << std::endl;

      HermMatrix2 CoarseOp2(LittleDiracOp2);
      ZeroGuesser<CCVector> zg2;
      ConjugateGradient<CCVector> CGcc(P.coarse_tol, 30000, false);
      ConjugateGradient<CCVector> CGccSloppy(P.coarse_tol_sloppy, 30000,
                                             false);
      HPDSolver<CCVector> L3Solve(CoarseOp2, CGcc, zg2);
      HPDSolver<CCVector> L3SolveSloppy(CoarseOp2, CGccSloppy, zg2);

      double mirs2 = env_real("L2_MIRS_SHIFT", 0.02 * clmax2);
      ShiftedHermOpLinearOperator<CoarseVector> ShiftedCoarseOp(CoarseOp,
                                                                mirs2);
      CGSmoother<CoarseVector> CGsmooth2(env_int("L2_SMOOTH_IT", 8),
                                         ShiftedCoarseOp);

      TwoLevelADEF2<CoarseVector, CCVector, Subspace2> InnerSloppy(
          P.coarse_tol_sloppy, 3000, CoarseOp, CGsmooth2, L3SolveSloppy,
          L3Solve, Aggregates2);
      TwoLevelADEF2<CoarseVector, CCVector, Subspace2> InnerPrecise(
          P.coarse_tol, 3000, CoarseOp, CGsmooth2, L3SolveSloppy, L3Solve,
          Aggregates2);

      return fails + solve_phase(InnerSloppy, InnerPrecise);
    }
  } else if (three_level) {
    std::cout << GridLogMessage << "[gridmg] THREE_LEVEL is compiled for "
                                   "NBASIS=64 only — running two-level"
              << std::endl;
  }
  return fails + solve_phase(HPDSolveSloppy, HPDSolve);
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
    Pmg.coarse_nev = env_int("COARSE_NEV", 0);
    Pmg.coarse_irl_lo = env_real("COARSE_IRL_LO", -1.0);
    Pmg.coarse_irl_ord = env_int("COARSE_IRL_ORD", 201);
    Pmg.coarse_irl_tol = env_real("COARSE_IRL_TOL", 1e-3);
    Pmg.smooth_sp = env_int("SMOOTH_SP", 0) != 0;

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
      case 64:
        fails += run_mg<64>(Grid_, FineGrid, cb, FineHermOp, pRNG, src, ref,
                            lmax, Pmg);
        break;
      default:
        std::cout << GridLogMessage
                  << "[gridmg] NBASIS must be one of 24/32/48/64 (compiled "
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
    // Aux fields are fixed for the whole run: hoist the per-apply aux
    // re-pack (it was ~85 of the 119 ms full-volume HermOp).
    Mq.SetAuxStatic(true);
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
  // House pattern for QUDA-linked drivers (see the meas programs): skip
  // Grid/QUDA teardown entirely — the finalize path can abort (rc 134,
  // benign but ugly) or DEADLOCK (a smoke gate hung a batch chain for 5 h
  // on 2026-07-04).  All results are printed and flushed by now.
  std::cout.flush();
  std::_Exit(fails ? 1 : 0);
}
