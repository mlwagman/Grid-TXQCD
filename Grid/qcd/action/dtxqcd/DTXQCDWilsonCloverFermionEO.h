#pragma once
// DTXQCDWilsonCloverFermionEO: full doubled Wilson-Clover Dirac operator
// for DTXQCD, exposing the standard Grid fermion-operator interface so it
// plugs into CG / RHMC infrastructure that already works on a doubled
// fermion type.
//
// Operator structure (Cstar doubled, with corrected M_22 = C^T D^T C):
//
//   M | psi_u   = | (mass + D_W[U]_hop + Delta_diag(u) + clover_upper) psi_u
//                 + (2 d gamma_5 + 2 n) psi_l                                |
//     | psi_l   = | (mass + D_W[U^*]_hop + Delta_diag_lower(l) + clover_lower) psi_l
//                 + (2 d gamma_5 + 2 n) psi_u                                |
//
// where
//   clover_upper psi = -(csw/2) sum_{mu<nu} F_{mu,nu}    (sigma_{mu,nu} psi)
//   clover_lower psi = +(csw/2) sum_{mu<nu} F^T_{mu,nu}  (sigma_{mu,nu} psi)
//
// gamma_5-Hermiticity: gamma_5 M gamma_5 = M^dag (validated in
// Test_dtxqcd_gamma5_herm_full); Mdag uses this identity.
//
// EO production caches.  ImportFields() (called from the constructor and
// from ImportGauge) builds:
//
//   1. Per-CB aux + F_{mu,nu} copies (LatticeDtxqcd{Sigma,Pi,T,D,N} on
//      rbgrid_): forward Mooee on a CB fermion uses these so the
//      DtxqcdApplyMooeeDoubled lattice op gets conformable inputs (its
//      aux refs and the fermion live on the SAME grid).
//
//   2. Per-CB cached per-site 48x48 Mooee^{-1} stored as
//      Lattice<iScalar<iScalar<iMatrix<vComplex, 48>>>> on rbgrid_:
//      MooeeInv on a CB fermion is a single SIMD-vectorized gemv per
//      oSite, replacing the v1 per-call Eigen LU rebuild that dominated
//      multi-shift CG runtime.
//
// Caches must be rebuilt whenever the gauge field or aux fields change
// externally.  ImportGauge(U) rebuilds everything (gauge + aux + caches);
// callers mutating only aux should call ImportFields() afterwards.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDBatchedInverse48.h>
#include <Grid/qcd/utils/WilsonLoops.h>
#include <cstring>

NAMESPACE_BEGIN(Grid);

// Cached 48x48 per-site matrix type (Mooee^{-1}).  SIMD-vectorized over
// outer sites; one such matrix per CB site stored as an iMatrix tensor.
template <class vtype>
using DtxqcdSiteInvMat = iScalar<iScalar<iMatrix<vtype, kDtxqcdSiteDim48>>>;

class DTXQCDWilsonCloverFermionEO {
 public:
  typedef WilsonImplR Impl;
  typedef DTXQCDFermionDoubled Field;
  typedef LatticeGaugeField GaugeField;
  typedef typename LatticeFermion::vector_object::vector_type FermVtype;
  typedef DtxqcdSiteInvMat<FermVtype> SiteInvMat;
  typedef Lattice<SiteInvMat> InvField;

  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDWilsonCloverFermionEO(GaugeField &U,
                              GridCartesian &grid,
                              GridRedBlackCartesian &rbgrid,
                              RealD mass,
                              RealD csw,
                              const LatticeDtxqcdSigma &sigma,
                              const LatticeDtxqcdPi    &pi,
                              const LatticeDtxqcdD     &d,
                              const LatticeDtxqcdN     &n,
                              const LatticeDtxqcdS     &s,
                              const LatticeDtxqcdP     &p,
                              typename Impl::ImplParams impl_p =
                                  DTXQCDMeooeDoubled::DefaultImplParams())
      : mass_(mass),
        csw_(csw),
        sigma_(sigma),
        pi_(pi),
        d_(d),
        n_(n),
        s_(s),
        p_(p),
        grid_(grid),
        rbgrid_(rbgrid),
        Umu_(U),
        meooe_(U, grid, rbgrid, mass, impl_p),
        sigma_e_(&rbgrid), sigma_o_(&rbgrid),
        pi_e_(&rbgrid),    pi_o_(&rbgrid),
        d_e_(&rbgrid),     d_o_(&rbgrid),
        n_e_(&rbgrid),     n_o_(&rbgrid),
        s_e_(&rbgrid),     s_o_(&rbgrid),
        p_e_(&rbgrid),     p_o_(&rbgrid),
        inv_e_(&rbgrid),   inv_o_(&rbgrid),
        spin_(grid) {
    ImportFields();
  }

  // Refresh internal state after the caller updates U (e.g. HMC integrator).
  // Rebuilds the doubled-Wilson hopping engine, the field strength, the
  // per-CB aux copies, and the per-CB Mooee^{-1} caches.
  void ImportGauge(GaugeField &U) {
    meooe_.ImportGauge(U);
    ImportFields();
  }

  // Refresh per-CB aux copies, field strength (gauge unchanged here), and
  // the per-CB Mooee^{-1} caches.  Callers that mutate only aux fields
  // (not gauge) should call this after each mutation; ImportGauge calls
  // this internally.
  void ImportFields() {
    // Per-CB aux copies (pickCheckerboard from the full-grid refs).
    pickCheckerboard(Even, sigma_e_, sigma_);
    pickCheckerboard(Odd,  sigma_o_, sigma_);
    pickCheckerboard(Even, pi_e_,    pi_);
    pickCheckerboard(Odd,  pi_o_,    pi_);
    pickCheckerboard(Even, d_e_,     d_);
    pickCheckerboard(Odd,  d_o_,     d_);
    pickCheckerboard(Even, n_e_,     n_);
    pickCheckerboard(Odd,  n_o_,     n_);
    pickCheckerboard(Even, s_e_,     s_);
    pickCheckerboard(Odd,  s_o_,     s_);
    pickCheckerboard(Even, p_e_,     p_);
    pickCheckerboard(Odd,  p_o_,     p_);

    // F_{mu,nu} from U (full grid) + per-CB copies (csw != 0 only).
    BuildFieldStrength();

    // Mooee^{-1} caches per CB.
    BuildInverseCacheCB(Even, inv_e_, inv_lex_e_);
    BuildInverseCacheCB(Odd,  inv_o_, inv_lex_o_);
  }

  // -------- Full-volume apply --------

  // M psi: full operator on a full-volume doubled fermion.  Uses
  // WilsonFermion::M per block (mass + hopping) plus site-local Delta + cross
  // + clover.  Matches the assembly validated in Test_dtxqcd_gamma5_herm_full.
  void M(const Field &in, Field &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      meooe_.UpperWilson().M(in.upper.f[a], out.upper.f[a]);
      meooe_.LowerWilson().M(in.lower.f[a], out.lower.f[a]);
    }
    AddDiagAndCrossAndClover(in, out);
  }

  // Mdag psi via gamma_5 M gamma_5.
  void Mdag(const Field &in, Field &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5_in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5_in.upper.f[a] = g5 * in.upper.f[a];
      g5_in.lower.f[a] = g5 * in.lower.f[a];
    }
    M(g5_in, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
    }
  }

  // -------- EO-decomposed pieces --------

  // Mooee psi: site-local diagonal block of M on the input checkerboard.
  // CB input dispatches to DtxqcdApplyMooeeDoubled with the matching per-CB
  // aux / FS copies (cached at ImportFields time).  Full-grid input keeps
  // the full-grid path.
  void Mooee(const Field &in, Field &out) {
    if (in.upper.f[0].Grid() == &rbgrid_) {
      int cb = in.upper.f[0].Checkerboard();
      const auto &sig = (cb == Even) ? sigma_e_ : sigma_o_;
      const auto &pp  = (cb == Even) ? pi_e_    : pi_o_;
      const auto &dd  = (cb == Even) ? d_e_     : d_o_;
      const auto &nn  = (cb == Even) ? n_e_     : n_o_;
      const auto &ss  = (cb == Even) ? s_e_     : s_o_;
      const auto &qq  = (cb == Even) ? p_e_     : p_o_;
      const auto *fs  = (csw_ != 0.0)
                         ? ((cb == Even) ? &FS_e_ : &FS_o_)
                         : nullptr;
      DtxqcdApplyMooeeDoubled(mass_, sig, pp, dd, nn, ss, qq,
                              in.upper, in.lower, out.upper, out.lower,
                              csw_, fs);
    } else {
      DtxqcdApplyMooeeDoubled(mass_, sigma_, pi_, d_, n_, s_, p_,
                              in.upper, in.lower, out.upper, out.lower,
                              csw_, (csw_ != 0.0 ? &FS_ : nullptr));
    }
  }

  // MooeeDag psi via gamma_5 Mooee gamma_5 (per-block gamma_5-Hermiticity).
  void MooeeDag(const Field &in, Field &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5_in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5_in.upper.f[a] = g5 * in.upper.f[a];
      g5_in.lower.f[a] = g5 * in.lower.f[a];
    }
    Mooee(g5_in, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
    }
  }

  // Meooe psi: hopping piece (cross-checkerboard).  Forwards to
  // DTXQCDMeooeDoubled which wraps two WilsonFermion engines (one with U
  // for upper, one with conjugate(U) for lower).
  void Meooe(const Field &in, Field &out) {
    meooe_.Meooe(in.upper, in.lower, out.upper, out.lower);
  }
  void MeooeDag(const Field &in, Field &out) {
    meooe_.MeooeDag(in.upper, in.lower, out.upper, out.lower);
  }

  // MooeeInv psi via the cached SIMD per-site 48x48 inverse.  CB-only path.
  void MooeeInv(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    GRID_ASSERT(in.upper.f[0].Grid() == &rbgrid_);
    ApplyInverseSimd(in, out, cb);
  }
  // MooeeInvDag psi via gamma_5 MooeeInv gamma_5 (per-block gamma_5-Hermiticity
  // of Mooee carries over to its inverse).
  void MooeeInvDag(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    GRID_ASSERT(in.upper.f[0].Grid() == &rbgrid_);
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5in.upper.f[a] = g5 * in.upper.f[a];
      g5in.lower.f[a] = g5 * in.lower.f[a];
      g5in.upper.f[a].Checkerboard() = cb;
      g5in.lower.f[a].Checkerboard() = cb;
    }
    ApplyInverseSimd(g5in, tmp, cb);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }
  }

  // -------- Multi-RHS variants for batched-gemm hot loops --------
  //
  // MooeeInvN: applies Mooee^{-1} to NRHS fermions in one threaded sweep
  // over CB sites.  Each thread does a single Eigen 48 x NRHS gemm instead
  // of NRHS separate 48-component gemvs -- the per-site matrix is loaded
  // once per site and reused across all RHS columns.  For NRHS = (multi-shift
  // degree) this gives the matrix-load savings that the cuBLAS-batched
  // production path on GPU provides.  CPU benefit scales with NRHS up to
  // the point where the 48 x NRHS Eigen matrix overflows L2.
  //
  // Used by DTXQCDWilsonCloverRationalEOAction::deriv()'s post-CG pole loop
  // (Npole independent Mpc applies on the same gauge configuration).
  void MooeeInvN(const std::vector<const Field *> &ins,
                 const std::vector<Field *> &outs) {
    ApplyInverseLex(ins, outs, /*dag=*/false);
  }

  void MooeeInvDagN(const std::vector<const Field *> &ins,
                    const std::vector<Field *> &outs) {
    ApplyInverseLex(ins, outs, /*dag=*/true);
  }

  // -------- Accessors / introspection (mainly for tests) --------

  RealD Mass() const { return mass_; }
  RealD Csw()  const { return csw_; }
  const std::vector<LatticeColourMatrix> &FieldStrength() const { return FS_; }
  DTXQCDMeooeDoubled &MeooeEngine() { return meooe_; }

 private:
  // Build F_{mu,nu} for the 6 (mu<nu) pairs from U via WilsonLoops, plus
  // pickCheckerboard into per-CB copies.  No-op when csw == 0.
  void BuildFieldStrength() {
    FS_.clear();
    FS_e_.clear();
    FS_o_.clear();
    if (csw_ == 0.0) return;
    FS_.reserve(6);
    FS_e_.reserve(6);
    FS_o_.reserve(6);
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(&grid_);
        WilsonLoops<WilsonImplR>::FieldStrength(F, Umu_, mu, nu);
        LatticeColourMatrix F_e(&rbgrid_), F_o(&rbgrid_);
        pickCheckerboard(Even, F_e, F);
        pickCheckerboard(Odd,  F_o, F);
        FS_.push_back(std::move(F));
        FS_e_.push_back(std::move(F_e));
        FS_o_.push_back(std::move(F_o));
      }
    }
  }

  // Site-local Delta + cross + clover, added on top of (mass + hopping) which
  // WilsonFermion::M already produced into `out`.  Used by M() only.
  void AddDiagAndCrossAndClover(const Field &in, Field &out) {
    GridBase *grid = in.Grid();

    DTXQCDFermionNf delta_u(grid), delta_l(grid);
    DtxqcdApplyDeltaDiag     (sigma_, pi_, s_, p_, in.upper, delta_u);
    DtxqcdApplyDeltaDiagLower(sigma_, pi_, s_, p_, in.lower, delta_l);

    DTXQCDFermionNf cross_u(grid), cross_l(grid);
    DtxqcdApplyDnCross(d_, n_, in.lower, cross_u, /*apply_conj=*/false);  // M_UR
    DtxqcdApplyDnCross(d_, n_, in.upper, cross_l, /*apply_conj=*/true);   // M_LL = conj(M_UR) under DN_COMPLEX_SYMMETRIC

    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = out.upper.f[a] + delta_u.f[a] + cross_u.f[a];
      out.lower.f[a] = out.lower.f[a] + delta_l.f[a] + cross_l.f[a];
    }

    if (csw_ != 0.0) {
      DTXQCDFermionNf clov_u(grid), clov_l(grid);
      DtxqcdApplyCloverUpper(csw_, FS_, in.upper, clov_u);
      DtxqcdApplyCloverLower(csw_, FS_, in.lower, clov_l);
      for (int a = 0; a < DtxqcdNf; ++a) {
        out.upper.f[a] = out.upper.f[a] + clov_u.f[a];
        out.lower.f[a] = out.lower.f[a] + clov_l.f[a];
      }
    }
  }

  // Build the per-CB 48x48 Mooee^{-1} cache.  Iterates sites of the chosen
  // CB, assembles M48 from the full-grid aux + clover at that site (Eigen),
  // computes M48^{-1} via partialPivLu, and pokes the result into the
  // SIMD-vectorized lattice slot.  thread_for-parallel over sites; the
  // SIMD repack is hidden inside Grid's pokeSite.
  void BuildInverseCacheCB(int cb, InvField &inv,
                            std::vector<Eigen::MatrixXcd> &inv_lex) {
    typedef typename InvField::vector_object::scalar_object SmatSobj;
    inv = Zero();
    inv.Checkerboard() = cb;

    // LOCAL, per-rank build.  Unvectorize the per-CB aux (+FS) copies ONCE into
    // lex-ordered host arrays for THIS rank's sublattice, then build/invert M48
    // per local site.  Previously this looped grid_.GlobalDimensions() with
    // peekSite -> every rank built the entire global cache (3.6 GB buffers ->
    // OOM at mpi != 1.1.1.1) and the work was N_ranks-redundant.  Unvectorizing
    // also removes peekSite from the hot loop, so the build is GPU-safe to
    // thread_for (no per-lattice view-lock corruption).  Mirrors TXQCD.
    const auto &sig_L = (cb == Even) ? sigma_e_ : sigma_o_;
    const auto &pi_L  = (cb == Even) ? pi_e_    : pi_o_;
    const auto &d_L   = (cb == Even) ? d_e_     : d_o_;
    const auto &n_L   = (cb == Even) ? n_e_     : n_o_;
    const auto &s_L   = (cb == Even) ? s_e_     : s_o_;
    const auto &p_L   = (cb == Even) ? p_e_     : p_o_;

    typedef typename LatticeDtxqcdSigma::vector_object::scalar_object SigSob;
    typedef typename LatticeDtxqcdPi::vector_object::scalar_object    PiSob;
    typedef typename LatticeDtxqcdD::vector_object::scalar_object     DSob;
    typedef typename LatticeDtxqcdN::vector_object::scalar_object     NSob;
    typedef typename LatticeDtxqcdS::vector_object::scalar_object     SSob;
    typedef typename LatticeDtxqcdP::vector_object::scalar_object     PSob;
    typedef typename LatticeColourMatrix::vector_object::scalar_object CMSob;

    std::vector<SigSob> sig_a; std::vector<PiSob> pi_a;
    std::vector<DSob>   d_a;   std::vector<NSob>  n_a;
    std::vector<SSob>   s_a;   std::vector<PSob>  p_a;
    unvectorizeToLexOrdArray(sig_a, sig_L);
    unvectorizeToLexOrdArray(pi_a,  pi_L);
    unvectorizeToLexOrdArray(d_a,   d_L);
    unvectorizeToLexOrdArray(n_a,   n_L);
    unvectorizeToLexOrdArray(s_a,   s_L);
    unvectorizeToLexOrdArray(p_a,   p_L);
    const uint64_t Nsite = sig_a.size();

    std::vector<std::array<CMSob, 6>> fs_a;  // per-local-site F_{mu,nu} (csw!=0)
    if (csw_ != 0.0) {
      const auto &FS_L = (cb == Even) ? FS_e_ : FS_o_;
      std::array<std::vector<CMSob>, 6> fk;
      for (int k = 0; k < 6; ++k) unvectorizeToLexOrdArray(fk[k], FS_L[k]);
      fs_a.resize(Nsite);
      for (uint64_t x = 0; x < Nsite; ++x)
        for (int k = 0; k < 6; ++k) fs_a[x][k] = fk[k][x];
    }

    // Build the forward 48x48 M_ee at local lex site idx (host Eigen, no
    // peekSite -> safe + parallel under thread_for).
    auto build_M48 = [&](uint64_t idx, Eigen::MatrixXcd &M48) {
      DtxqcdSiteAux aux = DtxqcdSiteAux::FromSobjs(
          sig_a[idx], pi_a[idx], d_a[idx], n_a[idx], s_a[idx], p_a[idx]);
      Eigen::MatrixXcd M_upper, M_lower, M_off;
      if (csw_ != 0.0) {
        DtxqcdSiteClover clover = DtxqcdSiteClover::FromSobjs(fs_a[idx]);
        DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper, csw_, &clover);
        DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower, csw_, &clover);
      } else {
        DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper);
        DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower);
      }
      DtxqcdBuildOffDiagBlock24(aux, spin_, M_off);
      DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);
    };

    std::vector<SmatSobj> sobjs(Nsite);

    // Gate the per-site 48x48 inversion -- the dominant cost -- onto one batched
    // cuBLAS getrf+getri (DTXQCD_PRECOMPUTE_GPU, default ON on CUDA).  CPU
    // reference keeps the per-site Eigen inverse.  Both fill sobjs identically.
    static int use_gpu = []() {
#ifndef GRID_CUDA
      return 0;
#else
      const char *e = std::getenv("DTXQCD_PRECOMPUTE_GPU");
      if (!e || !*e) return 1;
      return std::atoi(e);
#endif
    }();

#ifdef GRID_CUDA
    if (use_gpu) {
      const int N = kDim48, N2 = N * N;
      std::vector<std::complex<double>> h_fwd((size_t)Nsite * N2);
      thread_for(idx, Nsite, {
        Eigen::MatrixXcd M48;
        build_M48(idx, M48);                   // Eigen storage is column-major
        std::memcpy(&h_fwd[(size_t)idx * N2], M48.data(),
                    (size_t)N2 * sizeof(std::complex<double>));
      });
      DtxqcdBlas::BatchedInverse48::Ensure(Nsite, /*need_inv=*/true);
      acceleratorCopyToDevice((void *)h_fwd.data(),
                              (void *)&DtxqcdBlas::BatchedInverse48::M_fwd[0],
                              (size_t)Nsite * N2 * sizeof(ComplexD));
      DtxqcdBlas::BatchedInverse48::Invert(Nsite);
      std::vector<std::complex<double>> h_inv((size_t)Nsite * N2);
      acceleratorCopyFromDevice((void *)&DtxqcdBlas::BatchedInverse48::M_inv[0],
                                (void *)h_inv.data(),
                                (size_t)Nsite * N2 * sizeof(ComplexD));
      thread_for(idx, Nsite, {
        const std::complex<double> *src = &h_inv[(size_t)idx * N2];
        SmatSobj sobj;
        sobj = Zero();
        for (int r = 0; r < N; ++r)
          for (int c = 0; c < N; ++c)
            sobj()()(r, c) = src[(size_t)c * N + r];  // column-major -> (r,c)
        sobjs[idx] = sobj;
      });
    } else
#endif
    {
      thread_for(idx, Nsite, {
        Eigen::MatrixXcd M48;
        build_M48(idx, M48);
        Eigen::MatrixXcd Minv = M48.inverse();
        SmatSobj sobj;
        sobj = Zero();
        for (int r = 0; r < kDim48; ++r)
          for (int c = 0; c < kDim48; ++c)
            sobj()()(r, c) = Minv(r, c);
        sobjs[idx] = sobj;
      });
    }

#ifdef GRID_CUDA
    // Phase 1b: persistent per-CB device inverse (batched column-major) for the
    // cuBLAS MooeeInvN path in ApplyInverseLex.  Same local lex order as sobjs.
    {
      const int Nb = kDim48, Nb2 = Nb * Nb;
      deviceVector<ComplexD> &inv_dev = (cb == Even) ? inv_dev_e_ : inv_dev_o_;
      uint64_t &inv_dev_n = (cb == Even) ? inv_dev_n_e_ : inv_dev_n_o_;
      if (inv_dev.size() < (size_t)Nsite * Nb2) inv_dev.resize((size_t)Nsite * Nb2);
      std::vector<ComplexD> h_id((size_t)Nsite * Nb2);
      thread_for(idx, Nsite, {
        ComplexD *dst = &h_id[(size_t)idx * Nb2];
        for (int r = 0; r < Nb; ++r)
          for (int c = 0; c < Nb; ++c)
            dst[(size_t)c * Nb + r] = sobjs[idx]()()(r, c);  // column-major
      });
      acceleratorCopyToDevice((void *)h_id.data(), (void *)&inv_dev[0],
                              (size_t)Nsite * Nb2 * sizeof(ComplexD));
      inv_dev_n = Nsite;
      // PR 1: bump gen so any cached pointer arrays in ApplyInverseLex rebuild.
      if (cb == Even) ++inv_dev_gen_e_; else ++inv_dev_gen_o_;
    }
#endif

    // Pack the local lex-ordered inverses into the SIMD cache (inverse of the
    // unvectorize above) and the Eigen multi-RHS cache.  inv_lex[s] aligns with
    // the fermion lex order in ApplyInverseLex (same grid + CB).
    vectorizeFromLexOrdArray(sobjs, inv);
    inv_lex.clear();
    inv_lex.resize(Nsite);
    thread_for(s, Nsite, {
      Eigen::MatrixXcd M(kDim48, kDim48);
      for (int r = 0; r < kDim48; ++r)
        for (int c = 0; c < kDim48; ++c)
          M(r, c) = ComplexD(sobjs[s]()()(r, c));
      inv_lex[s] = std::move(M);
    });
  }

  // SIMD per-oSite gemv applying the cached 48x48 inverse to a doubled
  // fermion.  Reads inv_v(s) as a single iMatrix lane and the four
  // LatticeFermion views (upper × DtxqcdNf, lower × DtxqcdNf) as the input
  // vector lanes.  Direct port of TXQCD's ApplyMooeeInvSimd pattern with
  // an outer block (upper, lower) dimension added.
 public:
  // Public because nvcc (--expt-extended-lambda) forbids an extended
  // __host__ __device__ lambda (the accelerator_for below) inside a private
  // or protected member function.
  void ApplyInverseSimd(const Field &in, Field &out, int cb) {
    InvField &inv = (cb == Even) ? inv_e_ : inv_o_;
    GridBase *fg = in.upper.f[0].Grid();
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }

    autoView(inv_v, inv, AcceleratorRead);
    auto in_up_v  = MakeFermViewsRead<DtxqcdNf>(in.upper,  std::make_index_sequence<DtxqcdNf>{});
    auto in_lo_v  = MakeFermViewsRead<DtxqcdNf>(in.lower,  std::make_index_sequence<DtxqcdNf>{});
    auto out_up_v = MakeFermViewsWrite<DtxqcdNf>(out.upper, std::make_index_sequence<DtxqcdNf>{});
    auto out_lo_v = MakeFermViewsWrite<DtxqcdNf>(out.lower, std::make_index_sequence<DtxqcdNf>{});

    typedef decltype(coalescedRead(in_up_v[0][0])) FermSitePerLane;
    const int Nsimd = LatticeFermion::vector_object::Nsimd();

    accelerator_for(s, fg->oSites(), Nsimd, {
      auto Mlane = inv_v(s);
      FermSitePerLane in_up[DtxqcdNf], in_lo[DtxqcdNf];
      for (int a = 0; a < DtxqcdNf; ++a) {
        in_up[a] = in_up_v[a](s);
        in_lo[a] = in_lo_v[a](s);
      }
      FermSitePerLane out_up[DtxqcdNf], out_lo[DtxqcdNf];
      typedef typename std::remove_reference<decltype(Mlane()()(0, 0))>::type MEl;

      for (int r_blk = 0; r_blk < 2; ++r_blk) {
        for (int r_a = 0; r_a < DtxqcdNf; ++r_a) {
          for (int r_alpha = 0; r_alpha < Ns; ++r_alpha) {
            for (int r_i = 0; r_i < Nc; ++r_i) {
              int r = r_blk * kDim24
                    + r_a * Ns * Nc + r_alpha * Nc + r_i;
              MEl sum;
              zeroit(sum);
              for (int c_blk = 0; c_blk < 2; ++c_blk) {
                for (int c_a = 0; c_a < DtxqcdNf; ++c_a) {
                  for (int c_alpha = 0; c_alpha < Ns; ++c_alpha) {
                    for (int c_i = 0; c_i < Nc; ++c_i) {
                      int c = c_blk * kDim24
                            + c_a * Ns * Nc + c_alpha * Nc + c_i;
                      auto Mrc = Mlane()()(r, c);
                      auto vc = (c_blk == 0)
                                ? in_up[c_a]()(c_alpha)(c_i)
                                : in_lo[c_a]()(c_alpha)(c_i);
                      sum = sum + Mrc * vc;
                    }
                  }
                }
              }
              if (r_blk == 0) out_up[r_a]()(r_alpha)(r_i) = sum;
              else            out_lo[r_a]()(r_alpha)(r_i) = sum;
            }
          }
        }
      }

      for (int a = 0; a < DtxqcdNf; ++a) {
        coalescedWrite(out_up_v[a][s], out_up[a]);
        coalescedWrite(out_lo_v[a][s], out_lo[a]);
      }
    });

    for (int a = 0; a < DtxqcdNf; ++a) {
      in_up_v[a].ViewClose();
      in_lo_v[a].ViewClose();
      out_up_v[a].ViewClose();
      out_lo_v[a].ViewClose();
    }
  }

 private:
  // Multi-RHS apply via per-site Eigen 48 x NRHS gemm.  Mirrors TXQCD's
  // ApplyMooeeInvScalar pattern with an outer NRHS dimension and the
  // doubled (upper, lower) block layout.  Threaded across CB sites; each
  // thread holds a stack-resident 48 x NRHS input + output Eigen matrix.
  //
  //   dag = false:  out_k = inv_lex[site] *      input_k
  //   dag = true :  out_k = inv_lex[site]^dag *  input_k
  //                       = (inv^dag is the inverse of M^dag = gamma_5 M gamma_5,
  //                          so we wrap in gamma_5 outside and call dag=false here).
  //
  // The dag path is unused on the inside (we provide MooeeInvDagN above as a
  // gamma_5 wrapper), but kept symmetric in case a caller wants the raw
  // adjoint apply.
  void ApplyInverseLex(const std::vector<const Field *> &ins,
                        const std::vector<Field *> &outs,
                        bool dag) {
    const int NRHS = static_cast<int>(ins.size());
    GRID_ASSERT(NRHS > 0);
    GRID_ASSERT(static_cast<int>(outs.size()) == NRHS);

    const int cb = ins[0]->upper.f[0].Checkerboard();
    GRID_ASSERT(ins[0]->upper.f[0].Grid() == &rbgrid_);

    if (dag) {
      // Wrap in gamma_5 on each RHS and recurse with dag=false.
      Gamma g5(Gamma::Algebra::Gamma5);
      std::vector<Field> g5_in;       g5_in.reserve(NRHS);
      std::vector<Field> tmp;         tmp.reserve(NRHS);
      std::vector<const Field *> g5_in_p(NRHS);
      std::vector<Field *>       tmp_p(NRHS);
      for (int k = 0; k < NRHS; ++k) {
        g5_in.emplace_back(&rbgrid_);
        tmp.emplace_back(&rbgrid_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          g5_in[k].upper.f[a] = g5 * ins[k]->upper.f[a];
          g5_in[k].lower.f[a] = g5 * ins[k]->lower.f[a];
          g5_in[k].upper.f[a].Checkerboard() = cb;
          g5_in[k].lower.f[a].Checkerboard() = cb;
        }
        g5_in_p[k] = &g5_in[k];
        tmp_p[k]   = &tmp[k];
      }
      ApplyInverseLex(g5_in_p, tmp_p, /*dag=*/false);
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          outs[k]->upper.f[a] = g5 * tmp[k].upper.f[a];
          outs[k]->lower.f[a] = g5 * tmp[k].lower.f[a];
          outs[k]->upper.f[a].Checkerboard() = cb;
          outs[k]->lower.f[a].Checkerboard() = cb;
        }
      }
      return;
    }

    // Forward (no-dag) batched apply.
    const auto &inv_lex = (cb == Even) ? inv_lex_e_ : inv_lex_o_;
    typedef typename LatticeFermion::vector_object::scalar_object SiteFerm;

    // Unvectorize each RHS into per-(rhs, flavor, block) lex arrays.  The
    // lex order Grid uses here matches the order inv_lex was populated in
    // (via the same unvectorize on the SIMD InvField in BuildInverseCacheCB),
    // so per-site indices line up directly.
    std::vector<std::vector<std::vector<SiteFerm>>> in_up_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> in_lo_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> out_up_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> out_lo_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    for (int k = 0; k < NRHS; ++k) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        unvectorizeToLexOrdArray(in_up_lex[k][a], ins[k]->upper.f[a]);
        unvectorizeToLexOrdArray(in_lo_lex[k][a], ins[k]->lower.f[a]);
        out_up_lex[k][a].resize(in_up_lex[k][a].size());
        out_lo_lex[k][a].resize(in_lo_lex[k][a].size());
      }
    }
    const uint64_t Nsite = in_up_lex[0][0].size();
    GRID_ASSERT(Nsite == inv_lex.size());

    // ----- PR 1: DTXQCD_MOOEEINV_CUBLAS=1 path -----------------------------
    // gemmBatched(48, NRHS, 48) over Nsite sites, sourcing A from the
    // already-on-device inv_dev_e_/o_ populated by BuildInverseCacheCB.
    // Falls through to the existing thread_for/Eigen path when the env var
    // is unset (default OFF for safety).
    static int use_cublas = []() {
#ifndef GRID_CUDA
      return 0;
#else
      const char *e = std::getenv("DTXQCD_MOOEEINV_CUBLAS");
      return (e && *e && std::atoi(e)) ? 1 : 0;
#endif
    }();

#ifdef GRID_CUDA
    if (use_cublas) {
      const int N = kDim48;
      const int Nmat = N * N;
      const int N_x_NRHS = N * NRHS;
      const size_t flat_n = (size_t)Nsite * N_x_NRHS;

      deviceVector<ComplexD>  &inv_dev   = (cb == Even) ? inv_dev_e_   : inv_dev_o_;
      uint64_t                &inv_dev_n = (cb == Even) ? inv_dev_n_e_ : inv_dev_n_o_;
      uint64_t                inv_gen    = (cb == Even) ? inv_dev_gen_e_ : inv_dev_gen_o_;
      deviceVector<ComplexD>  &fin_dev   = (cb == Even) ? ferm_in_dev_e_  : ferm_in_dev_o_;
      deviceVector<ComplexD>  &fout_dev  = (cb == Even) ? ferm_out_dev_e_ : ferm_out_dev_o_;
      deviceVector<ComplexD*> &Amk       = (cb == Even) ? Amk_e_ : Amk_o_;
      deviceVector<ComplexD*> &Bkn       = (cb == Even) ? Bkn_e_ : Bkn_o_;
      deviceVector<ComplexD*> &Cmn       = (cb == Even) ? Cmn_e_ : Cmn_o_;
      uint64_t &ptrs_gen   = (cb == Even) ? cublas_ptrs_gen_e_   : cublas_ptrs_gen_o_;
      int      &ptrs_NRHS  = (cb == Even) ? cublas_ptrs_NRHS_e_  : cublas_ptrs_NRHS_o_;
      uint64_t &ptrs_Nsite = (cb == Even) ? cublas_ptrs_Nsite_e_ : cublas_ptrs_Nsite_o_;

      GRID_ASSERT(inv_dev_n == Nsite &&
                  "DTXQCD_MOOEEINV_CUBLAS requires DTXQCD_PRECOMPUTE_GPU=1 / "
                  "ImportFields to have populated inv_dev for this CB");

      // (1) Pack flat host buffer [Nsite × 48 × NRHS] column-major.
      //     Per site s, RHS k: rows 0..23 = upper(a,alpha,i), rows 24..47 = lower.
      std::vector<ComplexD> h_B(flat_n);
      thread_for(s, Nsite, {
        for (int k = 0; k < NRHS; ++k) {
          size_t off = ((size_t)s * NRHS + (size_t)k) * (size_t)N;  // column k of site s
          for (int a = 0; a < DtxqcdNf; ++a) {
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int i = 0; i < Nc; ++i) {
                int r24 = a * Ns * Nc + alpha * Nc + i;
                h_B[off + r24]          = ComplexD(in_up_lex[k][a][s]()(alpha)(i));
                h_B[off + kDim24 + r24] = ComplexD(in_lo_lex[k][a][s]()(alpha)(i));
              }
            }
          }
        }
      });

      // (2) Ensure device scratch + pointer arrays match shape.
      if (fin_dev.size() < flat_n) {
        fin_dev.resize(flat_n);
        fout_dev.resize(flat_n);
        ptrs_gen = 0;  // pointers stale
      }
      acceleratorCopyToDevice((void*)h_B.data(), (void*)&fin_dev[0],
                              flat_n * sizeof(ComplexD));

      bool need_rebuild = (ptrs_gen != inv_gen)
                       || (ptrs_NRHS != NRHS)
                       || (ptrs_Nsite != Nsite)
                       || (Amk.size() != Nsite);
      if (need_rebuild) {
        Amk.resize(Nsite);
        Bkn.resize(Nsite);
        Cmn.resize(Nsite);
        ComplexD *A_ptr = &inv_dev[0];
        ComplexD *B_ptr = &fin_dev[0];
        ComplexD *C_ptr = &fout_dev[0];
        // Host shadow + bulk D2D-by-H copy.  Rare (per resize / generation
        // bump), so cheaper than wrapping the build in a public helper just
        // to satisfy the "no extended __device__ lambda in private function"
        // CUDA constraint.
        std::vector<ComplexD*> Amk_host(Nsite), Bkn_host(Nsite), Cmn_host(Nsite);
        thread_for(s, Nsite, {
          Amk_host[s] = &A_ptr[(size_t)s * Nmat];
          Bkn_host[s] = &B_ptr[(size_t)s * N_x_NRHS];
          Cmn_host[s] = &C_ptr[(size_t)s * N_x_NRHS];
        });
        acceleratorCopyToDevice((void*)Amk_host.data(), (void*)&Amk[0],
                                Nsite * sizeof(ComplexD*));
        acceleratorCopyToDevice((void*)Bkn_host.data(), (void*)&Bkn[0],
                                Nsite * sizeof(ComplexD*));
        acceleratorCopyToDevice((void*)Cmn_host.data(), (void*)&Cmn[0],
                                Nsite * sizeof(ComplexD*));
        ptrs_gen   = inv_gen;
        ptrs_NRHS  = NRHS;
        ptrs_Nsite = Nsite;
      }

      // (3) Batched gemv/gemm: A[48,48] · B[48,NRHS] = C[48,NRHS] per site.
      //     Column-major matches the inv_dev layout and our flat fermion layout.
      GridBLAS blas;
      blas.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                       N, NRHS, N,
                       ComplexD(1.0, 0.0),
                       Amk, Bkn,
                       ComplexD(0.0, 0.0),
                       Cmn);

      // (4) D2H + unpack into out_{up,lo}_lex (which then go through the
      //     existing vectorizeFromLexOrdArray below).
      std::vector<ComplexD> h_C(flat_n);
      acceleratorCopyFromDevice((void*)&fout_dev[0], (void*)h_C.data(),
                                flat_n * sizeof(ComplexD));
      thread_for(s, Nsite, {
        for (int k = 0; k < NRHS; ++k) {
          size_t off = ((size_t)s * NRHS + (size_t)k) * (size_t)N;
          for (int a = 0; a < DtxqcdNf; ++a) {
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int i = 0; i < Nc; ++i) {
                int r24 = a * Ns * Nc + alpha * Nc + i;
                out_up_lex[k][a][s]()(alpha)(i) = h_C[off + r24];
                out_lo_lex[k][a][s]()(alpha)(i) = h_C[off + kDim24 + r24];
              }
            }
          }
        }
      });
    } else
#endif
    thread_for(s, Nsite, {
      Eigen::MatrixXcd B(kDim48, NRHS);
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int i = 0; i < Nc; ++i) {
              int r24 = a * Ns * Nc + alpha * Nc + i;
              B(r24,           k) = ComplexD(in_up_lex[k][a][s]()(alpha)(i));
              B(kDim24 + r24,  k) = ComplexD(in_lo_lex[k][a][s]()(alpha)(i));
            }
          }
        }
      }
      Eigen::MatrixXcd C = inv_lex[s] * B;
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int i = 0; i < Nc; ++i) {
              int r24 = a * Ns * Nc + alpha * Nc + i;
              out_up_lex[k][a][s]()(alpha)(i) = C(r24,          k);
              out_lo_lex[k][a][s]()(alpha)(i) = C(kDim24 + r24, k);
            }
          }
        }
      }
    });

    for (int k = 0; k < NRHS; ++k) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        vectorizeFromLexOrdArray(out_up_lex[k][a], outs[k]->upper.f[a]);
        vectorizeFromLexOrdArray(out_lo_lex[k][a], outs[k]->lower.f[a]);
        outs[k]->upper.f[a].Checkerboard() = cb;
        outs[k]->lower.f[a].Checkerboard() = cb;
      }
    }
  }

  // std::array-of-LatticeView helpers (LatticeView has no default ctor; the
  // index_sequence trick aggregate-initialises the array).  Mirror of
  // TXQCD's MakeFermViewsRead/Write.
  template <std::size_t N, class FieldT, std::size_t... Is>
  static auto MakeFermViewsRead(const FieldT &fld, std::index_sequence<Is...>)
      -> std::array<decltype(fld.f[0].View(AcceleratorRead)), N> {
    return {{ fld.f[Is].View(AcceleratorRead)... }};
  }
  template <std::size_t N, class FieldT, std::size_t... Is>
  static auto MakeFermViewsWrite(FieldT &fld, std::index_sequence<Is...>)
      -> std::array<decltype(fld.f[0].View(AcceleratorWrite)), N> {
    return {{ fld.f[Is].View(AcceleratorWrite)... }};
  }

  RealD mass_;
  RealD csw_;
  const LatticeDtxqcdSigma &sigma_;
  const LatticeDtxqcdPi    &pi_;
  const LatticeDtxqcdD     &d_;
  const LatticeDtxqcdN     &n_;
  const LatticeDtxqcdS     &s_;
  const LatticeDtxqcdP     &p_;
  GridCartesian            &grid_;
  GridRedBlackCartesian    &rbgrid_;
  GaugeField               &Umu_;

  DTXQCDMeooeDoubled       meooe_;
  std::vector<LatticeColourMatrix> FS_;
  std::vector<LatticeColourMatrix> FS_e_, FS_o_;

  LatticeDtxqcdSigma sigma_e_, sigma_o_;
  LatticeDtxqcdPi    pi_e_,    pi_o_;
  LatticeDtxqcdD     d_e_,     d_o_;
  LatticeDtxqcdN     n_e_,     n_o_;
  LatticeDtxqcdS     s_e_,     s_o_;
  LatticeDtxqcdP     p_e_,     p_o_;

  InvField inv_e_, inv_o_;

  // Multi-RHS scratch: per-CB lex-ordered Eigen 48x48 inverses.  Built by
  // unvectorizing the SIMD InvField so the lex index matches what
  // unvectorizeToLexOrdArray produces on input fermions in ApplyInverseLex.
  std::vector<Eigen::MatrixXcd> inv_lex_e_, inv_lex_o_;

#ifdef GRID_CUDA
  // Phase 1b: persistent per-CB device inverse in batched [Nsite*48*48]
  // column-major layout for the cuBLAS MooeeInvN path (ApplyInverseLex).
  // inline static (grow-only, shared) to avoid per-MakeEOp device churn -- safe
  // because only one FermionEO is alive at a time (MakeEOp is sequential).
  // Same local lex order as inv_lex_e_/o_.
  inline static deviceVector<ComplexD> inv_dev_e_, inv_dev_o_;
  inline static uint64_t inv_dev_n_e_{0}, inv_dev_n_o_{0};
  // PR 1 (DTXQCD_MOOEEINV_CUBLAS=1): cuBLAS gemmBatched scratch for
  // ApplyInverseLex.  Flat column-major fermion buffers (lSites * 48 * NRHS)
  // and per-site pointer arrays Amk -> inv_dev[s*48*48], Bkn -> fin_flat[s*48*NRHS],
  // Cmn -> fout_flat[s*48*NRHS].  Rebuilt when scratch resizes or inv_dev
  // is repopulated (gen counter bump in BuildInverseCacheCB).
  inline static deviceVector<ComplexD>  ferm_in_dev_e_,  ferm_in_dev_o_;
  inline static deviceVector<ComplexD>  ferm_out_dev_e_, ferm_out_dev_o_;
  inline static deviceVector<ComplexD*> Amk_e_, Amk_o_;
  inline static deviceVector<ComplexD*> Bkn_e_, Bkn_o_;
  inline static deviceVector<ComplexD*> Cmn_e_, Cmn_o_;
  inline static uint64_t inv_dev_gen_e_{0}, inv_dev_gen_o_{0};      // bumped each rebuild
  inline static uint64_t cublas_ptrs_gen_e_{0}, cublas_ptrs_gen_o_{0};
  inline static int      cublas_ptrs_NRHS_e_{0}, cublas_ptrs_NRHS_o_{0};
  inline static uint64_t cublas_ptrs_Nsite_e_{0}, cublas_ptrs_Nsite_o_{0};
#endif

  DtxqcdSpinMatrices spin_;
};

NAMESPACE_END(Grid);
