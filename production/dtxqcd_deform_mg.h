// dtxqcd_deform_mg.h: the two-level Petrov-Galerkin multigrid for the
// DEFORMED EO Schur operator Mpc~, extracted from the mtilde prototype
// (Test_dtxqcd_deform_mg_mtilde.cc) for production use.  See
// DTXQCD_DEFORM_MG_NOTE.tex for the full design + diagnostics.
//
//   - solves Mpc~ x = b DIRECTLY (unsquared kappa) with flexible PGCR
//   - right basis: inverse iteration through CGNR at NV_TOL (default 1e-3)
//   - left basis: CGNR on the dagger system (equivalent to the quasi-g5
//     recip trick, no profile toggling needed)
//   - coarse op: Grid's native two-subspace (Petrov-Galerkin)
//     CoarsenOperator, 81-point geometry, calibrated link-transpose
//   - coarse solve: EXACT dense LU (cusolverDnZgetrf) on a SECOND GPU
//     (cudaSetDevice bracketing; ~20 s once per config+profile)
//   - V-cycle: MR smoothing (8 its) around the exact coarse correction
//
// VALIDITY DOMAIN (measured): smooth profiles (tent_rounded2 / cos) with
// <~6 supercritical slices at nbasis=16 — cos eta<=0.6 converges (5,187
// fine Ops at eta=0.6 = 1.7x the best plain-CG point at equal grading
// differential); cos eta>=0.7 stalls at every tested nbasis (16/32/48,
// tol-1e-3 vectors); sharp tents ALWAYS fail (wall-mode localization).
// Requires >= 2 visible GPUs and a single rank.
//
// Usage:
//   DtxqcdDeformMG<16> mg(Qdef, Grid, RBGrid);
//   mg.Setup(pRNG);                              // ~6 min, once per cfg+profile
//   Qdef.SetOddSolver([&](const F &b, F &x, RealD tol, int mx, RealD &r) {
//     return mg.Solve(b, x, tol, mx, r); });
//   ... Qdef.SolveTilEO(...) now routes its odd block through the MG.
#pragma once

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidualNonHermitian.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDPackedField.h>
#if defined(GRID_CUDA) && defined(GRID_HAVE_QUDA)
#include <cusolverDn.h>
#endif

#include <functional>
#include <vector>

namespace Grid {

#if defined(GRID_CUDA) && defined(GRID_HAVE_QUDA)

template <int NB = 16>
class DtxqcdDeformMG {
 public:
  typedef DTXQCDFermionDoubled F;
  typedef LatticeDtxqcdPacked Field;
  typedef GeneralCoarsenedMatrix<vDtxqcdPackedVector, vTComplex, NB>
      LittleDiracOperator;
  typedef typename LittleDiracOperator::CoarseVector CoarseVector;
  typedef Aggregation<vDtxqcdPackedVector, vTComplex, NB> Subspace;

  // Non-Hermitian fine shim: Op = Mpc~, AdjOp = Mpc~^dag on packed fields.
  // Packed cb METADATA stays Even throughout (the grid_mg RB convention —
  // CoarsenOperator's phase fields are default-Even and its ET products
  // assert on mismatch); physical Odd parity is stamped struct-side.
  struct Shim : public LinearOperatorBase<Field> {
    DtxqcdDeformQuda &q;
    GridBase *rb;
    mutable size_t nops = 0;
    Shim(DtxqcdDeformQuda &q_, GridBase *rb_) : q(q_), rb(rb_) {}
    void Apply_(const Field &in, Field &out, bool dag) {
      F fi(rb), fo(rb);
      DtxqcdUnpackDoubled(in, fi);
      for (int a = 0; a < DtxqcdNf; ++a) {
        fi.upper.f[a].Checkerboard() = Odd;
        fi.lower.f[a].Checkerboard() = Odd;
      }
      if (dag)
        q.MpcTilDag(fi, fo);
      else
        q.MpcTil(fi, fo);
      DtxqcdPackDoubled(fo, out);
      out.Checkerboard() = Even;
      ++nops;
    }
    void Op(const Field &in, Field &out) { Apply_(in, out, false); }
    void AdjOp(const Field &in, Field &out) { Apply_(in, out, true); }
    void HermOp(const Field &in, Field &out) {
      Field t(in.Grid());
      Op(in, t);
      AdjOp(t, out);
    }
    void HermOpAndNorm(const Field &in, Field &out, RealD &n1, RealD &n2) {
      HermOp(in, out);
      ComplexD dot = innerProduct(in, out);
      n1 = real(dot);
      n2 = norm2(out);
    }
    void OpDiag(const Field &, Field &) { GRID_ASSERT(0); }
    void OpDir(const Field &, Field &, int, int) { GRID_ASSERT(0); }
    void OpDirAll(const Field &, std::vector<Field> &) { GRID_ASSERT(0); }
  };

  DtxqcdDeformMG(DtxqcdDeformQuda &q, GridCartesian &grid,
                 GridRedBlackCartesian &rb)
      : q_(q), grid_(grid), rb_(rb), shim_(q, &rb) {
    block_ = ParseBlock_(EnvStr_("DTXQCD_MG_BLOCK", "4.4.4.2"));
    Coordinate latt = grid_.GlobalDimensions();
    clatt_.resize(Nd);
    for (int d = 0; d < Nd; ++d) {
      GRID_ASSERT(latt[d] % block_[d] == 0);
      clatt_[d] = latt[d] / block_[d];
    }
    coarse_.reset(new GridCartesian(clatt_, GridDefaultSimd(Nd,
                                    vComplex::Nsimd()), GridDefaultMpi()));
    geom_.reset(new NextToNextToNextToNearestStencilGeometry4D(coarse_.get()));
    Ac_.reset(new LittleDiracOperator(*geom_, &rb_, coarse_.get()));
    aggR_.reset(new Subspace(coarse_.get(), &rb_, Odd));
    aggL_.reset(new Subspace(coarse_.get(), &rb_, Odd));
  }

  // Null vectors (right: Mpc~^{-1} r; left: Mpc~^{-dag} r — both by CGNR
  // on the corresponding normal equation) -> PG coarsen -> calibrated
  // dense assembly -> LU on GPU 1.
  void Setup(GridParallelRNG &pRNG) {
    const RealD nvtol = EnvReal_("DTXQCD_MG_NVTOL", 1e-3);
    const int nvmax = EnvInt_("DTXQCD_MG_NVMAX", 50000);
    // Null vectors through the PRODUCTION solve path (SolveTilEO Schur
    // sandwich at loose tol; the odd-solver hook must NOT be installed
    // yet), mirroring the validated standalone recipe exactly — a fresh
    // internal-CGNR reimplementation of the setup produced a solver that
    // STALLED in production (relres 0.6 vs the standalone's clean
    // convergence; A/B gate 2026-07-16).  Right: M~^{-1}-enriched; left:
    // M~^{-dag} via the dagger CGNR on the same shim.
    double t0 = usecond();
    // DTXQCD_MG_NV_RBQUIRK=1 (default): reproduce the standalone's
    // VALIDATED null-vector pipeline verbatim — RB-allocated struct fields
    // fed to SolveTilEO.  The RB allocation makes SolveTilEO's internal
    // pickCheckerboard iterate the wrong grid, i.e. the solve's deep
    // inverse iterate is partially MASKED back toward white noise
    // (|M~v|/|v| = 4.38 = the white-noise value) while keeping the band's
    // spatial envelope.  Clean full-grid recipes measured on the depth
    // ladder (2026-07-16, jobs 1301886/91/92): converged 1e-3 -> metric
    // 2.86, V-cycle AMPLIFIES 3.9; 23 its -> 3.20/1.57; 6 its ->
    // 3.63/0.60 with a slow-converging selftest — none reach the masked
    // basis's 5.2k-Op class.  Quirk-derived but deterministic across
    // builds/nodes (5,187 Ops thrice bit-identical; 5,607 after an
    // unrelated recompile) and gated every Setup by S2 + the contraction
    // probe (+ selftest in production runs).  =0 selects the clean
    // full-grid pure-odd recipe (the ladder's subject) for future study.
    const int rbquirk = EnvInt_("DTXQCD_MG_NV_RBQUIRK", 1);
    // DTXQCD_MG_REUSE_NV=1: keep the aggregates from the previous Setup
    // (shared null space).  At SUBCRITICAL strengths the masked basis is
    // white-noise-Rayleigh and barely operator-specific, so it transfers
    // across s/eta nodes; the coarse op + LU are always rebuilt (they DO
    // change).  Probe-gated: if the one-V-cycle contraction exceeds
    // DTXQCD_MG_REUSE_MAXCONTR (default 0.3) the basis is regenerated
    // in full — reuse can degrade quality but never correctness.
    const bool reuse_nv =
        EnvInt_("DTXQCD_MG_REUSE_NV", 0) != 0 && ready_;
    if (reuse_nv)
      std::cout << GridLogMessage
                << "[deform-mg] REUSING null-vector basis (probe-gated)"
                << std::endl;
    for (int i = reuse_nv ? NB : 0; i < NB; ++i) {
      F rf(&grid_), xf(&grid_);
      GaussFull_(pRNG, rf);
      RealD rel = 0;
      if (rbquirk) {
        F rq(&rb_), xq(&rb_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          pickCheckerboard(Odd, rq.upper.f[a], rf.upper.f[a]);
          pickCheckerboard(Odd, rq.lower.f[a], rf.lower.f[a]);
          xq.upper.f[a] = Zero();
          xq.lower.f[a] = Zero();
          xq.upper.f[a].Checkerboard() = Odd;
          xq.lower.f[a].Checkerboard() = Odd;
        }
        const int itr = q_.SolveTilEO(rq, xq, nvtol, nvmax, rel);
        DtxqcdPackDoubled(xq, aggR_->subspace[i]);
        aggR_->subspace[i].Checkerboard() = Even;
        NormalizeP_(aggR_->subspace[i]);
        {
          Field Mv(&rb_);
          shim_.Op(aggR_->subspace[i], Mv);
          std::cout << GridLogMessage << "[deform-mg] right nv[" << i
                    << "]: invit iters=" << itr
                    << "  |M~v|/|v|=" << std::sqrt(norm2(Mv)) << std::endl;
        }
        // left: G5 x recip solve, same RB pipeline
        GaussFull_(pRNG, rf);
        for (int a = 0; a < DtxqcdNf; ++a) {
          pickCheckerboard(Odd, rq.upper.f[a], rf.upper.f[a]);
          pickCheckerboard(Odd, rq.lower.f[a], rf.lower.f[a]);
          xq.upper.f[a] = Zero();
          xq.lower.f[a] = Zero();
          xq.upper.f[a].Checkerboard() = Odd;
          xq.lower.f[a].Checkerboard() = Odd;
        }
        q_.SetEtaSign(-1);
        const int itl = q_.SolveTilEO(rq, xq, nvtol, nvmax, rel);
        q_.SetEtaSign(+1);
        {
          Gamma g5(Gamma::Algebra::Gamma5);
          for (int a = 0; a < DtxqcdNf; ++a) {
            xq.upper.f[a] = g5 * xq.upper.f[a];
            xq.lower.f[a] = g5 * xq.lower.f[a];
          }
        }
        DtxqcdPackDoubled(xq, aggL_->subspace[i]);
        aggL_->subspace[i].Checkerboard() = Even;
        NormalizeP_(aggL_->subspace[i]);
        {
          Field Mv(&rb_);
          shim_.AdjOp(aggL_->subspace[i], Mv);
          std::cout << GridLogMessage << "[deform-mg] left  nv[" << i
                    << "]: invit iters=" << itl
                    << "  |M~dag u|/|u|=" << std::sqrt(norm2(Mv)) << std::endl;
        }
        continue;
      }
      // ZERO the even checkerboard of the source: with b_e = 0 the Schur
      // prep is the identity and the odd solve sees a PURE-ODD gaussian
      // RHS — the recipe the standalone validated.  A full-grid gaussian's
      // prepped RHS b'_o = b_o - M~oe Mee^{-1} b_e yields right vectors
      // with |M~v|/|v| ~ 2.6 (vs 4.38) whose coarse space STALLS the
      // two-level solve (NV_RHS_FULL discriminator, job 1301868: the
      // production gate's relres-0.56 stall reproduced on the standalone
      // by this flip alone; left/dagger recipe exonerated).
      {
        LatticeFermion o(&rb_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          xf.upper.f[a] = Zero();
          xf.lower.f[a] = Zero();
          pickCheckerboard(Odd, o, rf.upper.f[a]);
          rf.upper.f[a] = Zero();
          setCheckerboard(rf.upper.f[a], o);
          pickCheckerboard(Odd, o, rf.lower.f[a]);
          rf.lower.f[a] = Zero();
          setCheckerboard(rf.lower.f[a], o);
        }
      }
      const int itr = q_.SolveTilEO(rf, xf, nvtol, nvmax, rel);
      PackOdd_(xf, aggR_->subspace[i]);
      NormalizeP_(aggR_->subspace[i]);
      {
        Field Mv(&rb_);
        shim_.Op(aggR_->subspace[i], Mv);
        std::cout << GridLogMessage << "[deform-mg] right nv[" << i
                  << "]: invit iters=" << itr
                  << "  |M~v|/|v|=" << std::sqrt(norm2(Mv)) << std::endl;
      }
      // left vector: G5 x M~(-eta)^{-1} r — the standalone's recipe via the
      // quasi-g5 identity (left null space of M~(eta) = G5 x right of
      // M~(-eta)).  Same pure-odd RHS discipline as the right vectors.
      GaussFull_(pRNG, rf);
      {
        LatticeFermion o(&rb_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          xf.upper.f[a] = Zero();
          xf.lower.f[a] = Zero();
          pickCheckerboard(Odd, o, rf.upper.f[a]);
          rf.upper.f[a] = Zero();
          setCheckerboard(rf.upper.f[a], o);
          pickCheckerboard(Odd, o, rf.lower.f[a]);
          rf.lower.f[a] = Zero();
          setCheckerboard(rf.lower.f[a], o);
        }
      }
      q_.SetEtaSign(-1);
      const int itl = q_.SolveTilEO(rf, xf, nvtol, nvmax, rel);
      q_.SetEtaSign(+1);
      {
        Gamma g5(Gamma::Algebra::Gamma5);
        for (int a = 0; a < DtxqcdNf; ++a) {
          xf.upper.f[a] = g5 * xf.upper.f[a];
          xf.lower.f[a] = g5 * xf.lower.f[a];
        }
      }
      PackOdd_(xf, aggL_->subspace[i]);
      NormalizeP_(aggL_->subspace[i]);
      {
        Field Mv(&rb_);
        shim_.AdjOp(aggL_->subspace[i], Mv);
        std::cout << GridLogMessage << "[deform-mg] left  nv[" << i
                  << "]: invit iters=" << itl
                  << "  |M~dag u|/|u|=" << std::sqrt(norm2(Mv)) << std::endl;
      }
    }
    std::cout << GridLogMessage << "[deform-mg] " << 2 * NB
              << " null vectors in " << (usecond() - t0) * 1e-6 << " s"
              << std::endl;
    t0 = usecond();
    Ac_->CoarsenOperator(shim_, *aggL_, *aggR_);
    // S2 gate (ported from the standalone): Ac c must equal
    // ProjL(M~ PromR(c)) against the OPERATOR — the dense-vs-Ac
    // calibration alone cannot catch a wrong AGGREGATION (both sides
    // would agree).  The production selftest stalled at relres 0.57 on a
    // gaussian RHS (2026-07-16) with this gate absent.
    {
      GridParallelRNG CR(coarse_.get());
      CR.SeedFixedIntegers({61, 62, 63, 64});
      CoarseVector c(coarse_.get()), y1(coarse_.get()), y2(coarse_.get());
      gaussian(CR, c);
      Ac_->M(c, y1);
      Field f(&rb_), Mf(&rb_);
      f.Checkerboard() = Even;
      aggR_->PromoteFromSubspace(c, f);
      shim_.Op(f, Mf);
      aggL_->ProjectToSubspace(y2, Mf);
      CoarseVector d(coarse_.get());
      d = y1 - y2;
      const RealD s2 = std::sqrt(norm2(d) / norm2(y2));
      std::cout << GridLogMessage << "[deform-mg] S2 coarse-op consistency "
                << s2 << (s2 < 1e-8 ? " OK" : "  ** FAIL **") << std::endl;
    }
    BuildDirect_();
    std::cout << GridLogMessage << "[deform-mg] coarsen+LU in "
              << (usecond() - t0) * 1e-6 << " s (N=" << Ncc_() << ")"
              << std::endl;
    ready_ = true;
    // MG-lite knobs for the SUBCRITICAL reweighting nodes (ep < 0.54 for
    // cos): DTXQCD_MG_LUDEV picks the dense-LU GPU (default 1 = the
    // certified two-GPU layout; 0 = single-GPU when the coarse fits, e.g.
    // BLOCK=4.4.4.4 -> N=12,288 -> 2.4 GB).  DTXQCD_MG_COARSE=gmres
    // replaces the dense LU with restarted GMRES on A_c (CTOL/CMAXIT/
    // CRESTART) -- the historical "GMRES does not converge on the coarse
    // system" verdict was measured on the SUPERCRITICAL inherited band;
    // the subcritical coarse system may be iterable, freeing the second
    // GPU entirely (2x fleet throughput).  Gate every configuration with
    // the contraction probe + selftest before production use.
    // One-V-cycle contraction probe: |r - M~ V(r)| / |r| on a random odd
    // vector.  Separates BASIS quality (probe >~ 1: coarse correction
    // useless -> PGCR stalls) from outer-solver issues (probe ~ 0.6 like
    // the winning standalone, yet the solve stalls anyway).
    {
      GridParallelRNG PR(&grid_);
      PR.SeedFixedIntegers({91, 92, 93, 94});
      Field r(&rb_), z(&rb_), t(&rb_);
      {
        F full(&grid_);
        GaussFull_(PR, full);
        PackOdd_(full, r);
      }
      VCycle vc(*this);
      vc(r, z);
      shim_.Op(z, t);
      t = t - r;
      last_contr_ = std::sqrt(norm2(t) / norm2(r));
      std::cout << GridLogMessage << "[deform-mg] V-cycle contraction |r-Mz|/|r|="
                << last_contr_ << std::endl;
    }
    if (reuse_nv &&
        last_contr_ > EnvReal_("DTXQCD_MG_REUSE_MAXCONTR", 0.3)) {
      std::cout << GridLogMessage << "[deform-mg] reuse probe FAILED (contr="
                << last_contr_ << ") -- regenerating basis in full"
                << std::endl;
      ready_ = false;
      Setup(pRNG);  // depth <= 1: ready_=false disables reuse on the retry
      return;
    }
    // DTXQCD_MG_SELFTEST=1: solve a gaussian odd RHS through the freshly
    // built hierarchy and report.  Discriminates hierarchy-broken from
    // RHS-class-specific behavior: the standalone validation used gaussian
    // RHSs (= the reweighting estimator's Hutchinson noise, MG's actual
    // production use case); Schur-prepped WALL sources measured a
    // relres-0.64 stall on the same hierarchy (2026-07-16 gate).
    if (EnvInt_("DTXQCD_MG_SELFTEST", 0)) {
      GridParallelRNG SR(&grid_);
      SR.SeedFixedIntegers({81, 82, 83, 84});
      F rb_src(&rb_), rx(&rb_);
      {
        F full(&grid_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          gaussian(SR, full.upper.f[a]);
          gaussian(SR, full.lower.f[a]);
          pickCheckerboard(Odd, rb_src.upper.f[a], full.upper.f[a]);
          pickCheckerboard(Odd, rb_src.lower.f[a], full.lower.f[a]);
        }
      }
      RealD rel = 0;
      const int ops = Solve(rb_src, rx, 1e-7, 0, rel);
      std::cout << GridLogMessage << "[deform-mg] SELFTEST gaussian RHS: "
                << ops << " fine Ops, relres " << rel << std::endl;
    }
  }
  bool Ready() const { return ready_; }

  // Odd-block solve, plugs into DtxqcdDeformQuda::SetOddSolver.
  int Solve(const F &b_o, F &x_o, RealD tol, int maxit, RealD &rel_out) {
    GRID_ASSERT(ready_);
    Field pb(&rb_), px(&rb_);
    DtxqcdPackDoubled(b_o, pb);
    pb.Checkerboard() = Even;
    px = Zero();
    px.Checkerboard() = Even;
    VCycle vc(*this);
    shim_.nops = 0;
    PrecGeneralisedConjugateResidualNonHermitian<Field> PGCR(
        tol, EnvInt_("DTXQCD_MG_MAXIT", 400), shim_, vc, 16, 16);
    PGCR(pb, px);
    Field t(&rb_);
    shim_.Op(px, t);
    t = t - pb;
    rel_out = std::sqrt(norm2(t) / norm2(pb));
    DtxqcdUnpackDoubled(px, x_o);
    for (int a = 0; a < DtxqcdNf; ++a) {
      x_o.upper.f[a].Checkerboard() = Odd;
      x_o.lower.f[a].Checkerboard() = Odd;
    }
    std::cout << GridLogMessage << "[deform-mg] solve: " << shim_.nops
              << " fine Ops, relres " << rel_out << std::endl;
    return (int)shim_.nops;
  }

 private:
  struct CoarseDirect {
    int N = 0, dev = 1;
    cusolverDnHandle_t h = nullptr;
    cuDoubleComplex *dA = nullptr, *dWork = nullptr, *dB = nullptr;
    int *dPiv = nullptr, *dInfo = nullptr;
    bool ready = false;
    void Factor(std::vector<std::complex<double>> &Dcol, int n) {
      N = n;
      int dev0 = 0;
      cudaGetDevice(&dev0);
      cudaSetDevice(dev);
      if (ready) {
        cudaFree(dA);
        cudaFree(dWork);
        cudaFree(dPiv);
        cudaFree(dInfo);
        cudaFree(dB);
        cusolverDnDestroy(h);
        ready = false;
      }
      cudaMalloc(&dA, sizeof(cuDoubleComplex) * (size_t)N * N);
      cudaMemcpy(dA, Dcol.data(), sizeof(cuDoubleComplex) * (size_t)N * N,
                 cudaMemcpyHostToDevice);
      cusolverDnCreate(&h);
      int lwork = 0;
      cusolverDnZgetrf_bufferSize(h, N, N, dA, N, &lwork);
      cudaMalloc(&dWork, sizeof(cuDoubleComplex) * (size_t)lwork);
      cudaMalloc(&dPiv, sizeof(int) * N);
      cudaMalloc(&dInfo, sizeof(int));
      cudaMalloc(&dB, sizeof(cuDoubleComplex) * N);
      cusolverDnZgetrf(h, N, N, dA, N, dWork, dPiv, dInfo);
      int info = -1;
      cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost);
      GRID_ASSERT(info == 0 && "deform-mg coarse LU failed");
      {
        std::vector<cuDoubleComplex> diag(N);
        cudaMemcpy2D(diag.data(), sizeof(cuDoubleComplex), dA,
                     (size_t)(N + 1) * sizeof(cuDoubleComplex),
                     sizeof(cuDoubleComplex), N, cudaMemcpyDeviceToHost);
        double lo = 1e300, hi = 0;
        for (int i = 0; i < N; ++i) {
          const double m = std::hypot(diag[i].x, diag[i].y);
          lo = std::min(lo, m);
          hi = std::max(hi, m);
        }
        std::cout << GridLogMessage << "[deform-mg] A_c LU diag |U_ii|: min "
                  << lo << " max " << hi << " ratio " << hi / std::max(lo, 1e-300)
                  << std::endl;
      }
      ready = true;
      cudaSetDevice(dev0);
    }
    void Solve(std::vector<ComplexD> &b) {
      int dev0 = 0;
      cudaGetDevice(&dev0);
      cudaSetDevice(dev);
      cudaMemcpy(dB, b.data(), sizeof(cuDoubleComplex) * N,
                 cudaMemcpyHostToDevice);
      cusolverDnZgetrs(h, CUBLAS_OP_N, N, 1, dA, N, dPiv, dB, N, dInfo);
      cudaMemcpy(b.data(), dB, sizeof(cuDoubleComplex) * N,
                 cudaMemcpyDeviceToHost);
      cudaSetDevice(dev0);
    }
  };

  struct VCycle : public LinearFunction<Field> {
    DtxqcdDeformMG &m;
    explicit VCycle(DtxqcdDeformMG &m_) : m(m_) {}
    void MR_(const Field &r, Field &z) {
      Field rs(r.Grid()), Ar(r.Grid());
      z = Zero();
      z.Checkerboard() = r.Checkerboard();
      rs = r;
      const int si = 8;
      for (int k = 0; k < si; ++k) {
        m.shim_.Op(rs, Ar);
        ComplexD num = innerProduct(Ar, rs);
        RealD den = norm2(Ar);
        ComplexD alpha = num * (1.0 / std::max(den, 1e-300));
        z = z + alpha * rs;
        rs = rs - alpha * Ar;
      }
    }
    void operator()(const Field &r, Field &z) {
      Field t(r.Grid()), resid(r.Grid());
      t.Checkerboard() = r.Checkerboard();
      resid.Checkerboard() = r.Checkerboard();
      MR_(r, z);
      m.shim_.Op(z, t);
      resid = r - t;
      CoarseVector cr(m.coarse_.get()), cx(m.coarse_.get());
      m.aggL_->ProjectToSubspace(cr, resid);
      cx = Zero();
      if (m.coarse_gmres_) {
        NonHermitianLinearOperator<LittleDiracOperator, CoarseVector> cop(
            *m.Ac_);
        GeneralisedMinimalResidual<CoarseVector> gmres(
            m.EnvReal_("DTXQCD_MG_CTOL", 1e-2),
            m.EnvInt_("DTXQCD_MG_CMAXIT", 2000),
            m.EnvInt_("DTXQCD_MG_CRESTART", 64), false);
        gmres(cop, cr, cx);
      } else {
        std::vector<ComplexD> cb;
        m.Packc_(cr, cb);
        m.direct_.Solve(cb);
        m.Pokec_(cb, cx);
      }
      Field corr(r.Grid());
      corr.Checkerboard() = r.Checkerboard();
      m.aggR_->PromoteFromSubspace(cx, corr);
      z = z + corr;
      m.shim_.Op(z, t);
      resid = r - t;
      Field z2(r.Grid());
      z2.Checkerboard() = r.Checkerboard();
      MR_(resid, z2);
      z = z + z2;
    }
  };
  friend struct VCycle;

  int Ncc_() const { return (int)(coarse_->gSites()) * NB; }
  int64_t Cflat_(const Coordinate &c) const {
    int64_t idx = 0;
    for (int mu = Nd - 1; mu >= 0; --mu) idx = idx * clatt_[mu] + c[mu];
    return idx;
  }
  void Packc_(const CoarseVector &v, std::vector<ComplexD> &out) const {
    out.resize(Ncc_());
    Coordinate c(Nd);
    for (c[3] = 0; c[3] < clatt_[3]; ++c[3])
      for (c[2] = 0; c[2] < clatt_[2]; ++c[2])
        for (c[1] = 0; c[1] < clatt_[1]; ++c[1])
          for (c[0] = 0; c[0] < clatt_[0]; ++c[0]) {
            auto s = peekSite(v, c);
            const int64_t b = Cflat_(c) * NB;
            for (int i = 0; i < NB; ++i) out[b + i] = TensorRemove(s(i));
          }
  }
  void Pokec_(const std::vector<ComplexD> &in, CoarseVector &v) const {
    Coordinate c(Nd);
    for (c[3] = 0; c[3] < clatt_[3]; ++c[3])
      for (c[2] = 0; c[2] < clatt_[2]; ++c[2])
        for (c[1] = 0; c[1] < clatt_[1]; ++c[1])
          for (c[0] = 0; c[0] < clatt_[0]; ++c[0]) {
            decltype(peekSite(v, c)) s;
            const int64_t b = Cflat_(c) * NB;
            for (int i = 0; i < NB; ++i) s(i) = in[b + i];
            pokeSite(s, v, c);
          }
  }
  // Dense assembly, convention-calibrated once (measured winner: shift +1,
  // TRANSPOSED link indices, link at x — variant 2).
  void BuildDirect_() {
    // GMRES coarse mode: no dense assembly, no LU (S2 already gates the
    // coarse operator against the fine one) -- single-GPU, zero coarse
    // memory beyond A_c itself.
    coarse_gmres_ = EnvStr_("DTXQCD_MG_COARSE", "lu") == std::string("gmres");
    if (coarse_gmres_) {
      std::cout << GridLogMessage
                << "[deform-mg] coarse solve: restarted GMRES (no LU)"
                << std::endl;
      return;
    }
    const int npoint = geom_->npoint;
    const int N = Ncc_();
    std::vector<std::complex<double>> Dcol;
    CoarseVector tv(coarse_.get()), ty(coarse_.get());
    GridParallelRNG CR(coarse_.get());
    CR.SeedFixedIntegers({71, 72, 73, 74});
    gaussian(CR, tv);
    Ac_->M(tv, ty);
    std::vector<ComplexD> vin, yref;
    Packc_(tv, vin);
    Packc_(ty, yref);
    auto assemble = [&](int variant) {
      const int sign = (variant & 1) ? -1 : +1;
      const int trans = (variant >> 1) & 1;
      const int atnbr = (variant >> 2) & 1;
      Dcol.assign((size_t)N * N, std::complex<double>(0, 0));
      Coordinate x(Nd);
      for (x[3] = 0; x[3] < clatt_[3]; ++x[3])
        for (x[2] = 0; x[2] < clatt_[2]; ++x[2])
          for (x[1] = 0; x[1] < clatt_[1]; ++x[1])
            for (x[0] = 0; x[0] < clatt_[0]; ++x[0]) {
              const int64_t rb = Cflat_(x) * NB;
              for (int p = 0; p < npoint; ++p) {
                Coordinate y = x;
                for (int mu = 0; mu < Nd; ++mu)
                  y[mu] = (x[mu] + sign * geom_->shifts[p][mu] +
                           clatt_[mu]) % clatt_[mu];
                const int64_t cb = Cflat_(y) * NB;
                auto link = peekSite(Ac_->_A[p], atnbr ? y : x);
                for (int i = 0; i < NB; ++i)
                  for (int j = 0; j < NB; ++j) {
                    const ComplexD z =
                        TensorRemove(trans ? link(j, i) : link(i, j));
                    Dcol[(size_t)(rb + i) + (size_t)N * (cb + j)] +=
                        std::complex<double>(DtxqcdToStd(z).real(),
                                             DtxqcdToStd(z).imag());
                  }
              }
            }
    };
    auto rel = [&]() -> RealD {
      RealD dn = 0, yn = 0;
      for (int r = 0; r < N; ++r) {
        std::complex<double> acc(0, 0);
        for (int cidx = 0; cidx < N; ++cidx)
          acc += Dcol[(size_t)r + (size_t)N * cidx] *
                 std::complex<double>(DtxqcdToStd(vin[cidx]).real(),
                                      DtxqcdToStd(vin[cidx]).imag());
        const std::complex<double> ry(DtxqcdToStd(yref[r]).real(),
                                      DtxqcdToStd(yref[r]).imag());
        dn += std::norm(acc - ry);
        yn += std::norm(ry);
      }
      return std::sqrt(dn / yn);
    };
    RealD best = 1e30;
    if (variant_ < 0) {
      for (int v = 0; v < 8; ++v) {
        assemble(v);
        const RealD r = rel();
        if (r < best) {
          best = r;
          variant_ = v;
        }
        if (r < 1e-10) break;
      }
    } else {
      assemble(variant_);
      best = rel();
    }
    GRID_ASSERT(best < 1e-8 && "deform-mg dense assembly calibration failed");
    direct_.dev = EnvInt_("DTXQCD_MG_LUDEV", 1);
    direct_.Factor(Dcol, N);
  }
  // Internal CGNR (dag=false: solve M~ z = r; dag=true: M~^dag z = r).
  void Cgnr_(const Field &r, Field &z, RealD tol, int maxit, bool dag) {
    Field b(&rb_), res(&rb_), p(&rb_), t(&rb_), Ap(&rb_);
    if (dag)
      shim_.Op(r, b);  // normal eq for M~^dag: (M~ M~^dag) z = M~ r? no:
    else
      shim_.AdjOp(r, b);
    // dag=false: solve (M~^dag M~) z = M~^dag r  => z ~ M~^{-1} r
    // dag=true : solve (M~ M~^dag) z = M~ r      => z ~ M~^{-dag} r
    z = Zero();
    z.Checkerboard() = Even;
    res = b;
    p = b;
    RealD r2 = norm2(res);
    const RealD b2 = std::max(norm2(b), 1e-300);
    const RealD tol2 = tol * tol * b2;
    for (int k = 0; k < maxit && r2 > tol2; ++k) {
      if (dag) {
        shim_.AdjOp(p, t);
        shim_.Op(t, Ap);
      } else {
        shim_.Op(p, t);
        shim_.AdjOp(t, Ap);
      }
      RealD pAp = real(innerProduct(p, Ap));
      RealD alpha = r2 / std::max(pAp, 1e-300);
      z = z + alpha * p;
      res = res - alpha * Ap;
      RealD rn = norm2(res);
      RealD beta = rn / r2;
      p = res + beta * p;
      r2 = rn;
    }
  }
  void Gauss_(GridParallelRNG &pRNG, Field &x) {
    Field full(&grid_);
    gaussian(pRNG, full);
    pickCheckerboard(Odd, x, full);
    x.Checkerboard() = Even;  // packed metadata convention
  }
  void GaussFull_(GridParallelRNG &pRNG, F &x) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, x.upper.f[a]);
      gaussian(pRNG, x.lower.f[a]);
    }
  }
  void PackOdd_(const F &full, Field &out) {
    F o(&rb_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      pickCheckerboard(Odd, o.upper.f[a], full.upper.f[a]);
      pickCheckerboard(Odd, o.lower.f[a], full.lower.f[a]);
    }
    DtxqcdPackDoubled(o, out);
    out.Checkerboard() = Even;  // packed metadata convention
  }
  void NormalizeP_(Field &x) {
    RealD n = std::sqrt(norm2(x));
    x = (1.0 / std::max(n, (RealD)1e-300)) * x;
    x.Checkerboard() = Even;
  }
  void Normalize_(Field &x) {
    RealD n = std::sqrt(norm2(x));
    x = (1.0 / std::max(n, (RealD)1e-300)) * x;
  }
  static Coordinate ParseBlock_(const std::string &s) {
    Coordinate out(std::vector<int>{4, 4, 4, 2});
    size_t p = 0;
    for (int d = 0; d < Nd && p < s.size(); ++d) {
      size_t q = s.find('.', p);
      out[d] = std::atoi(s.substr(p, q - p).c_str());
      if (q == std::string::npos) break;
      p = q + 1;
    }
    return out;
  }
  static std::string EnvStr_(const char *n, const char *d) {
    const char *e = std::getenv(n);
    return (e && *e) ? std::string(e) : std::string(d);
  }
  static double EnvReal_(const char *n, double d) {
    const char *e = std::getenv(n);
    return (e && *e) ? std::atof(e) : d;
  }
  static int EnvInt_(const char *n, int d) {
    const char *e = std::getenv(n);
    return (e && *e) ? std::atoi(e) : d;
  }

  DtxqcdDeformQuda &q_;
  GridCartesian &grid_;
  GridRedBlackCartesian &rb_;
  Shim shim_;
  Coordinate block_, clatt_;
  std::unique_ptr<GridCartesian> coarse_;
  std::unique_ptr<NextToNextToNextToNearestStencilGeometry4D> geom_;
  std::unique_ptr<LittleDiracOperator> Ac_;
  std::unique_ptr<Subspace> aggR_, aggL_;
  CoarseDirect direct_;
  int variant_ = -1;
  bool ready_ = false;
  bool coarse_gmres_ = false;
  RealD last_contr_ = 0;
};

#endif  // GRID_CUDA && GRID_HAVE_QUDA

}  // namespace Grid
