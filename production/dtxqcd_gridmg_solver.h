// dtxqcd_gridmg_solver.h — per-config Grid-MG (HDCG / TwoLevelADEF2 with
// coarse-eigenvector deflation) solver for DTXQCD measurement inversions on
// the FULL-VOLUME M†M, the right substrate at the lambda<=4 frontier where
// the M_ee cliff poisons the EO Schur complement.
//
// Factored from the validated Test_dtxqcd_grid_mg stack (commits 93043526,
// abb0ff20, f03dd41f).  Measured at 16^3x48 on the chromafork configs:
//   lambda=4 (nb32, ord300):        MG 250 outers vs full CG 3188 iters
//   lambda=3 (nb64, nn, nev96):     MG 2804 outers / 724 s vs CG 70612
//                                   iters / 1711 s   (single RHS)
// The per-config setup (basis -> coarsen -> coarse IRL) is cached to
// <setup_prefix>.{basis,ldop,...}.<tag> LIME sidecars and amortizes over
// the measurement solve count (32-192 baryon wall solves per config).
// CAUTION: cached setups must be consumed at the MPI decomposition they
// were written with (the LIME parallel read is decomposition-sensitive —
// job 1299205 checksum failure at 1.1.2.2 on a 1.1.1.2-written basis).
//
// Env knobs (all DTXQCD_MEAS_MG_*, defaults = the lambda=3/4 winners):
//   DTXQCD_MEAS_MG_NBASIS   32|64      (32: lambda>=4; 64: lambda=3)
//   DTXQCD_MEAS_MG_BLOCK    4.4.4.4
//   DTXQCD_MEAS_MG_GEOM     nn|nearest|nnn   (nn: the nb64-safe 25-pt)
//   DTXQCD_MEAS_MG_ORD      1500       (powerlaw filter order)
//   DTXQCD_MEAS_MG_REFINE_MAXIT 400    (0 disables the inverse-iteration
//                                       basis polish)
//   DTXQCD_MEAS_MG_NEV      96         (coarse IRL deflation depth; the
//                                       lambda=3 band supports ~112)
//   DTXQCD_MEAS_MG_SMOOTH_IT 8
//   DTXQCD_MEAS_MG_TOL_SLOPPY 4e-2  DTXQCD_MEAS_MG_TOL_COARSE 1e-10
//   DTXQCD_MEAS_MG_SETUP_PREFIX <path>  (empty: no caching)
//   COARSE_IRL_MAXIT / COARSE_IRL_HI respected as in the test.

#pragma once

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
#include <Grid/algorithms/iterative/AdefGeneric.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace Grid {
namespace DtxqcdGridMG {

inline bool file_exists(const std::string &f) {
  return std::ifstream(f).good();
}
inline double env_real(const char *n, double d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atof(e) : d;
}
inline int env_int(const char *n, int d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::atoi(e) : d;
}
inline std::string env_str(const char *n, const char *d) {
  const char *e = std::getenv(n);
  return (e && *e) ? std::string(e) : std::string(d);
}
inline Coordinate parse_dotted(const std::string &s, const Coordinate &dflt) {
  std::vector<int> dv;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, '.')) dv.push_back(std::stoi(t));
  if (dv.size() != (size_t)Nd) return dflt;
  return Coordinate(dv);
}

// Setup save/load (phys-test idiom; see Test_dtxqcd_grid_mg.cc provenance).
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

// Mirs smoother: fixed-iteration CG on a shifted HermOp.
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

// SP-matvec shim for the smoother (outer ADEF2 stays DP).
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
  void OpDir(const Field &in, Field &out, int dir, int disp) {
    GRID_ASSERT(0);
  }
  void OpDirAll(const Field &in, std::vector<Field> &out) { GRID_ASSERT(0); }

 private:
  AdapterT &a_;
};

}  // namespace DtxqcdGridMG

// The solver.  nbasis is compile-time (Grid's Aggregation template);
// instantiate 32 and 64 and pick at runtime via DTXQCD_MEAS_MG_NBASIS.
template <int nbasis>
class DtxqcdGridMGSolver {
 public:
  typedef LatticeDtxqcdPacked Field;
  typedef GeneralCoarsenedMatrix<vDtxqcdPackedVector, vTComplex, nbasis>
      LittleDiracOperator;
  typedef typename LittleDiracOperator::CoarseVector CoarseVector;
  typedef Aggregation<vDtxqcdPackedVector, vTComplex, nbasis> Subspace;

  // hop: the packed full-volume HermOp adapter (M†M through the device op).
  // FineGrid: the full 4D grid.  Setup runs in the ctor (gen or cache-load).
  DtxqcdGridMGSolver(DtxqcdPackedFullHermOp &hop, GridCartesian *FineGrid,
                     GridParallelRNG &pRNG, RealD mg_tol, int mg_maxit)
      : hop_(hop), fine_(FineGrid), mg_tol_(mg_tol), mg_maxit_(mg_maxit) {
    using namespace DtxqcdGridMG;

    Coordinate latt = FineGrid->GlobalDimensions();
    Coordinate block =
        parse_dotted(env_str("DTXQCD_MEAS_MG_BLOCK", "4.4.4.4"),
                     Coordinate({4, 4, 4, 4}));
    Coordinate clatt(Nd);
    for (int d = 0; d < Nd; d++) {
      GRID_ASSERT(latt[d] % block[d] == 0);
      clatt[d] = latt[d] / block[d];
    }
    coarse_.reset(new GridCartesian(clatt,
                                    GridDefaultSimd(Nd, vComplex::Nsimd()),
                                    FineGrid->_processors));

    std::string geom_s = env_str("DTXQCD_MEAS_MG_GEOM", "nn");
    if (geom_s == "nearest")
      geom_.reset(new NearestStencilGeometry4D(coarse_.get()));
    else if (geom_s == "nnn")
      geom_.reset(
          new NextToNextToNextToNearestStencilGeometry4D(coarse_.get()));
    else
      geom_.reset(new NextToNearestStencilGeometry4D(coarse_.get()));

    ldop_.reset(new LittleDiracOperator(*geom_, FineGrid, coarse_.get()));
    aggregates_.reset(new Subspace(coarse_.get(), FineGrid, Even));

    // Spectrum scale.
    Field src(FineGrid);
    gaussian(pRNG, src);
    PowerMethod<Field> PM;
    lmax_ = PM(hop_, src);
    std::cout << GridLogMessage << "[measMG] PowerMethod lambda_max ~ "
              << lmax_ << std::endl;

    // ---- basis: load or generate (powerlaw + optional refine) ----
    std::string prefix = env_str("DTXQCD_MEAS_MG_SETUP_PREFIX", "");
    std::string tag;
    bool basis_loaded = false, ldop_loaded = false;
    std::string bf, of;
#ifdef HAVE_LIME
    if (!prefix.empty()) {
      std::ostringstream t;
      t << nbasis << "." << geom_s << ".b" << block[0] << block[1] << block[2]
        << block[3];
      tag = t.str();
      bf = prefix + ".basis." + tag;
      of = prefix + ".ldop." + tag;
      if (file_exists(bf)) {
        LoadBasis(*aggregates_, bf);
        aggregates_->Orthogonalise();
        basis_loaded = true;
        std::cout << GridLogMessage << "[measMG] basis LOADED " << bf
                  << std::endl;
        if (file_exists(of)) {
          LoadOperator(*ldop_, of);
          ldop_loaded = true;
          std::cout << GridLogMessage << "[measMG] ldop LOADED " << of
                    << std::endl;
        }
      }
    }
#endif
    if (!basis_loaded) {
      int ord = env_int("DTXQCD_MEAS_MG_ORD", 1500);
      double hi = env_real("DTXQCD_MEAS_MG_CHEBY_HI", 1.1 * lmax_);
      double t0 = usecond();
      aggregates_->CreateSubspaceChebyshevPowerLaw(pRNG, hop_, nbasis, hi,
                                                   ord);
      int rmaxit = env_int("DTXQCD_MEAS_MG_REFINE_MAXIT", 400);
      if (rmaxit > 0)
        aggregates_->RefineSubspace(hop_, 0.0,
                                    env_real("DTXQCD_MEAS_MG_REFINE_TOL",
                                             1e-3),
                                    rmaxit);
      aggregates_->Orthogonalise();
      std::cout << GridLogMessage << "[measMG] subspace built in "
                << (usecond() - t0) * 1e-6 << " s" << std::endl;
#ifdef HAVE_LIME
      if (!prefix.empty()) SaveBasis(*aggregates_, bf);
#endif
    }
    if (!ldop_loaded) {
      double t0 = usecond();
      ldop_->CoarsenOperator(hop_, *aggregates_);
      std::cout << GridLogMessage << "[measMG] CoarsenOperator in "
                << (usecond() - t0) * 1e-6 << " s" << std::endl;
#ifdef HAVE_LIME
      if (!prefix.empty()) SaveOperator(*ldop_, of);
#endif
    }

    coarse_herm_.reset(
        new HermitianLinearOperator<LittleDiracOperator, CoarseVector>(
            *ldop_));

    // ---- coarse deflation (IRL with the Nk buffer, deflation-grade tol) --
    // NEV default 0 (OFF): deflation only pays where a dense low coarse band
    // exists (lambda=3: nev=96 -> 2.8x).  At lambda>=4 the coarse band is
    // nearly empty, so an IRL asking for N low modes that do NOT exist grinds
    // 0/N every restart and ABORTS inside calc() (job 1299370, nev=16 @lam4).
    // Set DTXQCD_MEAS_MG_NEV=96 explicitly at lambda=3; leave 0 elsewhere.
    int nev = env_int("DTXQCD_MEAS_MG_NEV", 0);
    if (nev > 0) {
      GridParallelRNG CRNG(coarse_.get());
      CRNG.SeedFixedIntegers({271828, 314159, 161803, 141421});
      CoarseVector c_src(coarse_.get());
      gaussian(CRNG, c_src);
      PowerMethod<CoarseVector> CPM;
      double clmax = CPM(*coarse_herm_, c_src);
      double irl_lo = env_real("DTXQCD_MEAS_MG_IRL_LO", 1.0e-3 * clmax);
      double irl_hi = env_real("COARSE_IRL_HI", 1.5 * clmax);
      const int Nstop = nev;
      const int Nk = nev + 16;  // convergence-window buffer (s3D lesson)
      const int Nm = Nk + std::max(64, Nk);
      const int restarts = env_int("COARSE_IRL_MAXIT", 10);
      Chebyshev<CoarseVector> cheb(irl_lo, irl_hi,
                                   env_int("DTXQCD_MEAS_MG_IRL_ORD", 201));
      FunctionHermOp<CoarseVector> OpCheby(cheb, *coarse_herm_);
      PlainHermOp<CoarseVector> Op(*coarse_herm_);
      ImplicitlyRestartedLanczos<CoarseVector> IRL(
          OpCheby, Op, Nstop, Nk, Nm,
          env_real("DTXQCD_MEAS_MG_IRL_TOL", 1e-3), restarts);
      c_eval_.resize(Nm);
      c_evec_.resize(Nm, coarse_.get());
      int Nconv = 0;
      IRL.calc(c_eval_, c_evec_, c_src, Nconv);
      const int nuse = std::min(Nconv, (int)c_eval_.size());
      c_eval_.resize(nuse);
      c_evec_.resize(nuse, coarse_.get());
      guesser_.reset(new DeflatedGuesser<CoarseVector>(c_evec_, c_eval_));
      std::cout << GridLogMessage << "[measMG] coarse IRL: " << nuse
                << " evecs (clmax=" << clmax << ")" << std::endl;
    } else {
      nothing_.reset(new DoNothingGuesser<CoarseVector>());
    }

    // ---- coarse solvers + smoother + driver pieces ----
    cg_coarse_.reset(new ConjugateGradient<CoarseVector>(
        env_real("DTXQCD_MEAS_MG_TOL_COARSE", 1e-10), 30000, false));
    cg_coarse_sloppy_.reset(new ConjugateGradient<CoarseVector>(
        env_real("DTXQCD_MEAS_MG_TOL_SLOPPY", 4e-2), 30000, false));
    LinearFunction<CoarseVector> *dummy = nullptr;
    (void)dummy;
    if (guesser_) {
      hpd_.reset(new HPDSolver<CoarseVector>(*coarse_herm_, *cg_coarse_,
                                             *guesser_));
      hpd_sloppy_.reset(new HPDSolver<CoarseVector>(
          *coarse_herm_, *cg_coarse_sloppy_, *guesser_));
    } else {
      hpd_.reset(new HPDSolver<CoarseVector>(*coarse_herm_, *cg_coarse_,
                                             *nothing_));
      hpd_sloppy_.reset(new HPDSolver<CoarseVector>(
          *coarse_herm_, *cg_coarse_sloppy_, *nothing_));
    }

    double mirs = env_real("DTXQCD_MEAS_MG_MIRS", 0.02 * lmax_);
    shifted_.reset(new ShiftedHermOpLinearOperator<Field>(hop_, mirs));
    int sit = env_int("DTXQCD_MEAS_MG_SMOOTH_IT", 8);
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
    if (env_int("DTXQCD_MEAS_MG_SMOOTH_SP", 1)) {
      sp_shim_.reset(new DtxqcdGridMG::SpHermOpShim<DtxqcdPackedFullHermOp>(
          hop_));
      shifted_sp_.reset(
          new ShiftedHermOpLinearOperator<Field>(*sp_shim_, mirs));
      smoother_.reset(
          new DtxqcdGridMG::CGSmoother<Field>(sit, *shifted_sp_));
    } else
#endif
    {
      smoother_.reset(new DtxqcdGridMG::CGSmoother<Field>(sit, *shifted_));
    }

    hdcg_.reset(new TwoLevelADEF2<Field, CoarseVector, Subspace>(
        mg_tol_, mg_maxit_, hop_, *smoother_, *hpd_sloppy_, *hpd_,
        *aggregates_));
    std::cout << GridLogMessage << "[measMG] ready: nbasis=" << nbasis
              << " block=" << block << " geom=" << geom_s << " nev=" << nev
              << " smooth_it=" << sit << std::endl;
  }

  // Solve (M†M) x = b on the packed full grid.  b is the packed M†source.
  void solve(const Field &b, Field &x) {
    x = Zero();
    x.Checkerboard() = b.Checkerboard();
    (*hdcg_)(b, x);
  }

 private:
  DtxqcdPackedFullHermOp &hop_;
  GridCartesian *fine_;
  RealD mg_tol_;
  int mg_maxit_;
  RealD lmax_;
  std::unique_ptr<GridCartesian> coarse_;
  std::unique_ptr<NonLocalStencilGeometry> geom_;
  std::unique_ptr<LittleDiracOperator> ldop_;
  std::unique_ptr<Subspace> aggregates_;
  std::vector<RealD> c_eval_;
  std::vector<CoarseVector> c_evec_;
  std::unique_ptr<HermitianLinearOperator<LittleDiracOperator, CoarseVector>>
      coarse_herm_;
  std::unique_ptr<DeflatedGuesser<CoarseVector>> guesser_;
  std::unique_ptr<DoNothingGuesser<CoarseVector>> nothing_;
  std::unique_ptr<ConjugateGradient<CoarseVector>> cg_coarse_;
  std::unique_ptr<ConjugateGradient<CoarseVector>> cg_coarse_sloppy_;
  std::unique_ptr<HPDSolver<CoarseVector>> hpd_;
  std::unique_ptr<HPDSolver<CoarseVector>> hpd_sloppy_;
  std::unique_ptr<ShiftedHermOpLinearOperator<Field>> shifted_;
#if defined(GRID_HAVE_QUDA) && defined(GRID_CUDA)
  std::unique_ptr<DtxqcdGridMG::SpHermOpShim<DtxqcdPackedFullHermOp>>
      sp_shim_;
  std::unique_ptr<ShiftedHermOpLinearOperator<Field>> shifted_sp_;
#endif
  std::unique_ptr<LinearFunction<Field>> smoother_;
  std::unique_ptr<TwoLevelADEF2<Field, CoarseVector, Subspace>> hdcg_;
};

}  // namespace Grid
