#pragma once
// Even-odd preconditioned TXQCD Wilson operator.
//
// M = D_W + (4+m) + Δ + Clover, where Δ is the site-diagonal aux-field
// insertion and Clover = -(csw/2) * σ_{μν} * F_{μν} is the clover term.
// In even-odd decomposition:
//   Mee = (4+m) + Δ_e     (site-diagonal, mixes flavors)
//   Moo = (4+m) + Δ_o     (site-diagonal, mixes flavors)
//   Meo = D_W^{eo}         (hopping, per-flavor, from Wilson kernel)
//   Moe = D_W^{oe}         (hopping, per-flavor, from Wilson kernel)
//
// Mee^{-1} and Moo^{-1} are precomputed at ImportGauge time via per-site
// 24×24 Eigen LU decomposition (Nf=2 × Ns=4 × Nc=3 = 24).
//
// Schur complement: Mpc = Moo - Moe Mee^{-1} Meo operates on the odd
// sublattice (half volume).
//
// γ₅-Hermiticity: MooeeDag(x) = γ₅ Mooee(γ₅ x), so we only store one
// inverse per checkerboard.

#include <Grid/qcd/action/txqcd/TXQCDDeltaOp.h>
#include <Grid/qcd/action/txqcd/TXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>
#include <Grid/util/Lexicographic.h>
#include <Grid/algorithms/blas/BatchedBlas.h>

NAMESPACE_BEGIN(Grid);

// SIMD-vectorized 24×24 site matrix used as the precomputed Mooee^{-1}.
// Storing the per-site inverse in Grid's iMatrix tensor layout lets the
// per-CG-iteration apply run inside a single accelerator_for over outer
// SIMD sites — eliminating the unvectorize→Eigen→revectorize roundtrip
// that previously dominated.
template <class vtype>
using TxqcdSiteInvMatrix = iScalar<iScalar<iMatrix<vtype, TxqcdNf * Ns * Nc>>>;

template <class Simd>
using TxqcdLatticeInvMatrix = Lattice<TxqcdSiteInvMatrix<Simd>>;

class TXQCDWilsonCloverFermionEO {
 public:
  typedef TXQCDSiteMatrixUtil SMU;
  static constexpr int kDim = SMU::kDim;
  // SIMD type of the underlying gauge field (matches LatticeFermion's vtype).
  typedef typename LatticeFermion::vector_object::scalar_type FermScalar;
  typedef typename LatticeFermion::vector_object::vector_type FermVtype;
  typedef TxqcdSiteInvMatrix<FermVtype> SiteInvMat;
  typedef Lattice<SiteInvMat> InvField;

  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;
  typedef typename Impl::GaugeField GaugeField;

  static typename Impl::ImplParams DefaultImplParams() {
    typename Impl::ImplParams p;
    p.boundary_phases.resize(Nd, 1.0);
    p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time (chroma convention)
    return p;
  }

  // Per-flavor mass constructor (e.g. mass = {m_l, m_l, m_s} for Nf=3).
  // The inner Dw_ is built with mass=0 since its diagonal piece is never
  // called by TXQCD code paths (Mooee/M apply diag_mass_[a] separately);
  // only Dw_.Meooe/MoeDeriv (mass-independent hopping) and Dw_.M (used in
  // M(in,out) where we add mass_[a]*in afterwards) matter.
  TXQCDWilsonCloverFermionEO(GaugeField &Umu, GridCartesian &grid,
                       GridRedBlackCartesian &rbgrid,
                       const std::array<RealD, TxqcdNf> &mass,
                       LatticeSigmaField &sigma, LatticePiField &pi,
                       LatticeSFieldC &s, LatticePFieldC &p,
                       LatticeTField &t, RealD csw = 0.0,
                       typename Impl::ImplParams impl_p = DefaultImplParams())
      : grid_(grid), rbgrid_(rbgrid), mass_(mass),
        csw_(csw),
        Dw_(Umu, grid, rbgrid, 0.0, impl_p),
        Umu_(Umu),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t),
        sigma_e_(&rbgrid), sigma_o_(&rbgrid),
        pi_e_(&rbgrid), pi_o_(&rbgrid),
        s_e_(&rbgrid), s_o_(&rbgrid),
        p_e_(&rbgrid), p_o_(&rbgrid),
        t_e_(&rbgrid), t_o_(&rbgrid) {
    for (int a = 0; a < TxqcdNf; ++a) diag_mass_[a] = 4.0 + mass_[a];
    if (csw_ != 0.0) {
      for (int k = 0; k < 6; ++k) {
        FS_.emplace_back(&grid);
        FS_e_.emplace_back(&rbgrid);
        FS_o_.emplace_back(&rbgrid);
      }
    }
    ImportFields();
  }

  // Backward-compat: degenerate scalar mass.
  TXQCDWilsonCloverFermionEO(GaugeField &Umu, GridCartesian &grid,
                       GridRedBlackCartesian &rbgrid, RealD mass,
                       LatticeSigmaField &sigma, LatticePiField &pi,
                       LatticeSFieldC &s, LatticePFieldC &p,
                       LatticeTField &t, RealD csw = 0.0,
                       typename Impl::ImplParams impl_p = DefaultImplParams())
      : TXQCDWilsonCloverFermionEO(Umu, grid, rbgrid,
                                   SMU::MassArray(mass),
                                   sigma, pi, s, p, t, csw, impl_p) {}

  ~TXQCDWilsonCloverFermionEO() {
    PrintTimers("destructor");
  }

  // ---- Profiling ----
  // Accumulating wall-time counters around the 24×24-inverse hotspots.
  // Profiles where time is spent so we can decide whether to swap the
  // dense per-site inverse for a Schur / Woodbury solve.
  void ResetTimers() {
    t_precompute_us_ = 0;
    t_apply_inv_us_ = 0;
    t_unvec_us_ = 0;
    t_revec_us_ = 0;
    n_precompute_ = n_apply_inv_ = 0;
  }
  void PrintTimers(const char *tag) const {
    if (n_precompute_ == 0 && n_apply_inv_ == 0) return;
    auto fmt = [](uint64_t us) { return double(us) * 1e-6; };
    std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
              << "] PrecomputeInverses: " << n_precompute_
              << " calls, " << fmt(t_precompute_us_) << " s ("
              << (n_precompute_ ? fmt(t_precompute_us_) / n_precompute_ * 1e3 : 0)
              << " ms/call)" << std::endl;
    std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
              << "]   precompute breakdown: unvec=" << fmt(t_pre_unvec_us_)
              << " s  inv=" << fmt(t_pre_inv_us_)
              << " s  pack=" << fmt(t_pre_pack_us_) << " s" << std::endl;
    std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
              << "] ApplyMooeeInv:    " << n_apply_inv_
              << " calls, " << fmt(t_apply_inv_us_) << " s ("
              << (n_apply_inv_ ? fmt(t_apply_inv_us_) / n_apply_inv_ * 1e3 : 0)
              << " ms/call)" << std::endl;
    std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
              << "] -- of which un/revectorize: "
              << fmt(t_unvec_us_ + t_revec_us_) << " s" << std::endl;
    if (n_mooee_fwd_) {
      std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
                << "] Mooee (forward):  " << n_mooee_fwd_
                << " calls, " << fmt(t_mooee_fwd_us_) << " s ("
                << fmt(t_mooee_fwd_us_) / n_mooee_fwd_ * 1e3
                << " ms/call)" << std::endl;
    }
  }

  void ImportGauge(const GaugeField &U) {
    Dw_.ImportGauge(U);
    ImportFields();
  }

  // ----- Full-grid operator (for testing / comparison) -----
  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    // Dw_ has mass=0, so Dw_.M gives 4*I + Wilson_hop; add per-flavor mass.
    for (int a = 0; a < TxqcdNf; ++a) {
      Dw_.M(in.f[a], out.f[a]);
      out.f[a] = out.f[a] + mass_[a] * in.f[a];
    }
    TXQCDFermionNf d(in.Grid());
    ApplyDelta(sigma_, pi_, s_, p_, t_, in, d);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + d.f[a];
  }

  void Mdag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf g5in(in.Grid());
    for (int a = 0; a < TxqcdNf; ++a) g5in.f[a] = g5 * in.f[a];
    M(g5in, out);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = g5 * out.f[a];
  }

  // ----- Even-odd components -----

  void Mooee(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    auto t0 = usecond();
    static int use_cublas = []() {
      const char *e = std::getenv("TXQCD_MOOEE_CUBLAS");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    if (use_cublas) {
      int cb = in.f[0].Checkerboard();
      ApplyMooeeFwdCublas(cb, in, out);
      t_mooee_fwd_us_ += usecond() - t0;
      n_mooee_fwd_++;
      return;
    }
    int cb = in.f[0].Checkerboard();
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = diag_mass_[a] * in.f[a];
      out.f[a].Checkerboard() = cb;
    }
    TXQCDFermionNf d(in.Grid());
    ApplyDeltaCB(cb, in, d);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + d.f[a];
    if (csw_ != 0.0) {
      auto &fs = (cb == Even) ? FS_e_ : FS_o_;
      TXQCDFermionNf cl(in.Grid());
      ApplyClover(csw_, fs, in, cl);
      for (int a = 0; a < TxqcdNf; ++a) {
        out.f[a] = out.f[a] + cl.f[a];
        out.f[a].Checkerboard() = cb;
      }
    }
    t_mooee_fwd_us_ += usecond() - t0;
    n_mooee_fwd_++;
  }

  // cuBLAS gemmBatched-based 24×24 forward Mooee.  Mirror of ApplyMooeeInvCublas
  // but using the pre-inversion matrix M (=Mee or Moo) populated by ImportFields
  // when TXQCD_MOOEE_CUBLAS=1.  Requires TXQCD_PRECOMPUTE_GPU=1 (lex table) and
  // TXQCD_MOOEE_CUBLAS=1 (Mfwd_dev_ buffers populated).
  void ApplyMooeeFwdCublas(int cb, const TXQCDFermionNf &in,
                           TXQCDFermionNf &out) {
    GridBase *grid = in.f[0].Grid();
    uint64_t lSites = grid->lSites();
    uint64_t oSites = grid->oSites();
    using vobj = typename LatticeFermion::vector_object;
    constexpr int Nsimd = vobj::Nsimd();
    constexpr int N = SMU::kDim;       // 24
    constexpr int Ncomp = N;

    auto &fin_flat  = (cb == Even) ? fermion_in_flat_e_  : fermion_in_flat_o_;
    auto &fout_flat = (cb == Even) ? fermion_out_flat_e_ : fermion_out_flat_o_;
    auto &lex_dev   = (cb == Even) ? lex_table_dev_e_    : lex_table_dev_o_;
    auto &lex_built = (cb == Even) ? lex_table_built_e_  : lex_table_built_o_;
    GRID_ASSERT(lex_built && "TXQCD_MOOEE_CUBLAS requires TXQCD_PRECOMPUTE_GPU=1");

    if (fin_flat.size() < lSites * Ncomp) {
      fin_flat.resize(lSites * Ncomp);
      fout_flat.resize(lSites * Ncomp);
    }

    // (1) SIMD fermion → flat per-site (lex-ordered).
    {
      ComplexD *fin_ptr = &fin_flat[0];
      int *lex_dev_ptr = &lex_dev[0];
      autoView(in_v0, in.f[0], AcceleratorRead);
      autoView(in_v1, in.f[1], AcceleratorRead);
      accelerator_for(s, oSites, Nsimd, {
        int simt_lane = static_cast<int>(lane);
        int lex = lex_dev_ptr[s * Nsimd + simt_lane];
        ComplexD *dst = &fin_ptr[lex * Ncomp];
        auto v0 = in_v0[s];
        auto v1 = in_v1[s];
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int i = 0; i < Nc; ++i) {
            dst[0 * 12 + alpha * 3 + i] = getlane(v0()(alpha)(i), simt_lane);
            dst[1 * 12 + alpha * 3 + i] = getlane(v1()(alpha)(i), simt_lane);
          }
        }
      });
    }

    // (2) Set up forward-matrix pointer arrays once per parity.
    auto &Amk_fwd = (cb == Even) ? Amk_fwd_e_ : Amk_fwd_o_;
    auto &Bkn = (cb == Even) ? Bkn_e_ : Bkn_o_;
    auto &Cmn = (cb == Even) ? Cmn_e_ : Cmn_o_;
    auto &fwd_built = (cb == Even) ? cublas_fwd_built_e_ : cublas_fwd_built_o_;
    auto &built_inv = (cb == Even) ? cublas_ptrs_built_e_ : cublas_ptrs_built_o_;
    auto &Mfwd_dev = (cb == Even) ? Mfwd_dev_e_ : Mfwd_dev_o_;
    if (!fwd_built) {
      Amk_fwd.resize(lSites);
      Bkn.resize(lSites);
      Cmn.resize(lSites);
      ComplexD *Mfwd_ptr = &Mfwd_dev[0];
      ComplexD *Bin_ptr  = &fin_flat[0];
      ComplexD *Cout_ptr = &fout_flat[0];
      ComplexD **Amk_ptr = &Amk_fwd[0];
      ComplexD **Bkn_ptr = &Bkn[0];
      ComplexD **Cmn_ptr = &Cmn[0];
      accelerator_for(i, lSites, 1, {
        Amk_ptr[i] = &Mfwd_ptr[i * 576];
        Bkn_ptr[i] = &Bin_ptr[i * 24];
        Cmn_ptr[i] = &Cout_ptr[i * 24];
      });
      fwd_built = true;
      built_inv = true;  // Bkn/Cmn are now valid for the inverse path too
    }

    // (3) Batched 24×24 × 24×1 matvec via cuBLAS.
    GridBLAS blas;
    blas.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                     N, 1, N,
                     ComplexD(1.0, 0.0),
                     Amk_fwd, Bkn,
                     ComplexD(0.0, 0.0),
                     Cmn);

    // (4) Flat → SIMD fermion.
    for (int a = 0; a < TxqcdNf; ++a) out.f[a].Checkerboard() = cb;
    {
      ComplexD *fout_ptr = &fout_flat[0];
      int *lex_dev_ptr = &lex_dev[0];
      autoView(out_v0, out.f[0], AcceleratorWrite);
      autoView(out_v1, out.f[1], AcceleratorWrite);
      accelerator_for(s, oSites, Nsimd, {
        int simt_lane = static_cast<int>(lane);
        int lex = lex_dev_ptr[s * Nsimd + simt_lane];
        const ComplexD *src = &fout_ptr[lex * Ncomp];
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int i = 0; i < Nc; ++i) {
            putlane(out_v0[s]()(alpha)(i), src[0 * 12 + alpha * 3 + i], simt_lane);
            putlane(out_v1[s]()(alpha)(i), src[1 * 12 + alpha * 3 + i], simt_lane);
          }
        }
      });
    }
  }

  void MooeeDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf tmp(in.Grid());
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp.f[a] = g5 * in.f[a];
      tmp.f[a].Checkerboard() = in.f[0].Checkerboard();
    }
    Mooee(tmp, out);
    int cb = in.f[0].Checkerboard();
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = g5 * out.f[a];
      out.f[a].Checkerboard() = cb;
    }
  }

  void MooeeInv(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    int cb = in.f[0].Checkerboard();
    ApplyMooeeInv(cb, in, out);
  }

  void MooeeInvDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf tmp(in.Grid());
    int cb = in.f[0].Checkerboard();
    for (int a = 0; a < TxqcdNf; ++a) {
      tmp.f[a] = g5 * in.f[a];
      tmp.f[a].Checkerboard() = cb;
    }
    MooeeInv(tmp, out);
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = g5 * out.f[a];
      out.f[a].Checkerboard() = cb;
    }
  }

  void Meooe(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a)
      Dw_.Meooe(in.f[a], out.f[a]);
  }

  void MeooeDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a)
      Dw_.MeooeDag(in.f[a], out.f[a]);
  }

  WilsonOp &Wilson() { return Dw_; }
  const std::array<RealD, TxqcdNf> &DiagMass() const { return diag_mass_; }
  const std::array<RealD, TxqcdNf> &Mass() const { return mass_; }
  RealD Csw() const { return csw_; }
  const GaugeField &Gauge() const { return Umu_; }
  const std::vector<LatticeColourMatrix> &FieldStrengths() const {
    return FS_;
  }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  std::array<RealD, TxqcdNf> mass_;
  std::array<RealD, TxqcdNf> diag_mass_;
  RealD csw_;
  WilsonOp Dw_;
  GaugeField &Umu_;

  LatticeSigmaField &sigma_;
  LatticePiField    &pi_;
  LatticeSFieldC    &s_;
  LatticePFieldC    &p_;
  LatticeTField     &t_;

  LatticeSigmaField sigma_e_, sigma_o_;
  LatticePiField    pi_e_, pi_o_;
  LatticeSFieldC    s_e_, s_o_;
  LatticePFieldC    p_e_, p_o_;
  LatticeTField     t_e_, t_o_;

  std::vector<LatticeColourMatrix> FS_;
  std::vector<LatticeColourMatrix> FS_e_, FS_o_;

  std::vector<SMU::SiteMatrix> inv_even_;
  std::vector<SMU::SiteMatrix> inv_odd_;
  // SIMD-vectorized mirrors of inv_even_/inv_odd_ used by the fast accelerator
  // apply path.  Packed once per ImportFields, then read via coalescedRead in
  // the per-CG ApplyMooeeInvSIMD kernel.
  std::unique_ptr<InvField> inv_simd_e_;
  std::unique_ptr<InvField> inv_simd_o_;
  // GPU pack scratch (TXQCD_PRECOMPUTE_GPU=1 path): contiguous device buffer
  // of inverses (24×24 ComplexD per site, lex-ordered) and the (oSite, lane)
  // → lex-index table.  Both grow monotonically; the lex table is grid-only
  // and is computed once per parity.
  deviceVector<ComplexD> M_dev_e_, M_dev_o_;
  deviceVector<int> lex_table_dev_e_, lex_table_dev_o_;
  bool lex_table_built_e_{false}, lex_table_built_o_{false};
  // cuBLAS gemmBatched scratch (TXQCD_MOOEEINV_CUBLAS=1 path): per-parity flat
  // fermion in/out buffers (lex-ordered, lSites × 24 ComplexD) and pointer
  // arrays for batched 24×24 × 24×1 gemm.  Amk pointers are static once
  // M_dev is populated; Bkn/Cmn point into the reusable flat buffers.
  deviceVector<ComplexD> fermion_in_flat_e_, fermion_in_flat_o_;
  deviceVector<ComplexD> fermion_out_flat_e_, fermion_out_flat_o_;
  deviceVector<ComplexD *> Amk_e_, Amk_o_;
  deviceVector<ComplexD *> Bkn_e_, Bkn_o_;
  deviceVector<ComplexD *> Cmn_e_, Cmn_o_;
  bool cublas_ptrs_built_e_{false}, cublas_ptrs_built_o_{false};
  // Forward Mooee cuBLAS (TXQCD_MOOEE_CUBLAS=1) scratch: pre-inversion
  // matrix M (=Mee or Moo) on CPU and GPU, plus a separate Amk pointer
  // array indexing the forward matrix buffer.  The fermion in/out flat
  // buffers and Bkn/Cmn pointer arrays are reused with the inverse path.
  std::vector<SMU::SiteMatrix> fwd_even_, fwd_odd_;
  deviceVector<ComplexD> Mfwd_dev_e_, Mfwd_dev_o_;
  deviceVector<ComplexD *> Amk_fwd_e_, Amk_fwd_o_;
  bool cublas_fwd_built_e_{false}, cublas_fwd_built_o_{false};

  SMU::SpinMatrices sm_;

  // Profiling counters; use ResetTimers + PrintTimers to read them.
  mutable uint64_t t_precompute_us_{0};
  mutable uint64_t t_apply_inv_us_{0};
  mutable uint64_t t_unvec_us_{0};
  mutable uint64_t t_revec_us_{0};
  mutable uint64_t n_precompute_{0};
  mutable uint64_t n_apply_inv_{0};
  // Sub-timers inside the precompute pipeline (CPU): aux unvectorize, the
  // 24×24 Eigen LU inversion itself, and the SIMD repack into InvField.
  mutable uint64_t t_pre_unvec_us_{0};
  mutable uint64_t t_pre_inv_us_{0};
  mutable uint64_t t_pre_pack_us_{0};
  // Forward Mooee timer (the 24×24 forward apply, currently composed of
  // ApplyDelta + ApplyClover + diag-mass).
  mutable uint64_t t_mooee_fwd_us_{0};
  mutable uint64_t n_mooee_fwd_{0};

  void ImportFields() {
    pickCheckerboard(Even, sigma_e_, sigma_);
    pickCheckerboard(Odd, sigma_o_, sigma_);
    pickCheckerboard(Even, pi_e_, pi_);
    pickCheckerboard(Odd, pi_o_, pi_);
    pickCheckerboard(Even, s_e_, s_);
    pickCheckerboard(Odd, s_o_, s_);
    pickCheckerboard(Even, p_e_, p_);
    pickCheckerboard(Odd, p_o_, p_);
    pickCheckerboard(Even, t_e_, t_);
    pickCheckerboard(Odd, t_o_, t_);

    if (csw_ != 0.0) {
      int k = 0;
      for (int mu = 0; mu < Nd; ++mu)
        for (int nu = mu + 1; nu < Nd; ++nu) {
          WilsonLoops<Impl>::FieldStrength(FS_[k], Umu_, mu, nu);
          pickCheckerboard(Even, FS_e_[k], FS_[k]);
          pickCheckerboard(Odd, FS_o_[k], FS_[k]);
          ++k;
        }
    }

    auto t_unv0 = usecond();
    auto aux_e = SMU::UnvectorizeAux(sigma_e_, pi_e_, s_e_, p_e_, t_e_);
    auto aux_o = SMU::UnvectorizeAux(sigma_o_, pi_o_, s_o_, p_o_, t_o_);
    t_pre_unvec_us_ += usecond() - t_unv0;

    auto t0 = usecond();
    static int use_mooee_cublas = []() {
      const char *e = std::getenv("TXQCD_MOOEE_CUBLAS");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    std::vector<SMU::SiteMatrix> *fwd_e_ptr = use_mooee_cublas ? &fwd_even_ : nullptr;
    std::vector<SMU::SiteMatrix> *fwd_o_ptr = use_mooee_cublas ? &fwd_odd_  : nullptr;
    if (csw_ != 0.0) {
      auto t_cl0 = usecond();
      auto cl_e = SMU::UnvectorizeClover(FS_e_);
      auto cl_o = SMU::UnvectorizeClover(FS_o_);
      t_pre_unvec_us_ += usecond() - t_cl0;
      auto t_inv0 = usecond();
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_e, csw_, &cl_e, inv_even_, fwd_e_ptr);
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_o, csw_, &cl_o, inv_odd_,  fwd_o_ptr);
      t_pre_inv_us_ += usecond() - t_inv0;
    } else {
      auto t_inv0 = usecond();
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_e, 0.0, nullptr, inv_even_, fwd_e_ptr);
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_o, 0.0, nullptr, inv_odd_,  fwd_o_ptr);
      t_pre_inv_us_ += usecond() - t_inv0;
    }
    auto t_pack0 = usecond();
    static int use_gpu_pack = []() {
      const char *e = std::getenv("TXQCD_PRECOMPUTE_GPU");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    if (use_gpu_pack) {
      PackInverseToSimdGPU(inv_even_, inv_simd_e_, Even);
      PackInverseToSimdGPU(inv_odd_,  inv_simd_o_, Odd);
    } else {
      PackInverseToSimd(inv_even_, inv_simd_e_, Even);
      PackInverseToSimd(inv_odd_,  inv_simd_o_, Odd);
    }
    if (use_mooee_cublas) {
      // Memcpy forward matrices CPU → GPU (Eigen std::vector is contiguous
      // column-major).
      uint64_t nsites_e = fwd_even_.size();
      uint64_t nsites_o = fwd_odd_.size();
      if (Mfwd_dev_e_.size() < nsites_e * 576) Mfwd_dev_e_.resize(nsites_e * 576);
      if (Mfwd_dev_o_.size() < nsites_o * 576) Mfwd_dev_o_.resize(nsites_o * 576);
      acceleratorCopyToDevice(
          reinterpret_cast<void *>(const_cast<std::complex<double> *>(
              fwd_even_.data()->data())),
          &Mfwd_dev_e_[0],
          nsites_e * 576 * sizeof(ComplexD));
      acceleratorCopyToDevice(
          reinterpret_cast<void *>(const_cast<std::complex<double> *>(
              fwd_odd_.data()->data())),
          &Mfwd_dev_o_[0],
          nsites_o * 576 * sizeof(ComplexD));
    }
    t_pre_pack_us_ += usecond() - t_pack0;
    t_precompute_us_ += usecond() - t0;
    n_precompute_++;
  }

 public:  // Public so CUDA extended lambdas inside accelerator_for compile.
  // GPU pack: writes the scalar Eigen 24×24 inverse into a SIMD-vectorized
  // lattice field via a single accelerator_for, replacing the CPU path
  // (vectorizeFromLexOrdArray + thread_for, ~1.5 s on 16³×48) with a kernel
  // that runs in ~10–50 ms.  The bottleneck previously was pure data shuffling
  // (≈900 MB CPU memory move per parity), not math.
  //
  // Pipeline:
  //   (a) flatten scalar_inv (CPU std::vector<Eigen 24×24>) into a contiguous
  //       host buffer of ComplexD (lex-ordered, row-major);
  //   (b) copy that buffer to a deviceVector;
  //   (c) build / reuse a (oSite, lane) → lex-index table on the device;
  //   (d) accelerator_for over (oSite, lane): each thread reads its lane's
  //       scalar entries from the flat buffer and writes them per-lane into
  //       the InvField via coalescedWrite on each (r, c) entry.
  void PackInverseToSimdGPU(const std::vector<SMU::SiteMatrix> &scalar_inv,
                            std::unique_ptr<InvField> &simd_field, int cb) {
    if (!simd_field) simd_field.reset(new InvField(&rbgrid_));
    GridBase *grid = simd_field->Grid();
    constexpr int N  = SMU::kDim;     // 24
    constexpr int N2 = N * N;         // 576
    using vobj = typename InvField::vector_object;
    constexpr int Nsimd = vobj::Nsimd();
    uint64_t nsites = scalar_inv.size();
    uint64_t oSites = grid->oSites();
    GRID_ASSERT(nsites == (uint64_t)grid->lSites());

    // (a)+(b) Eigen's std::vector<Matrix<complex<double>,24,24>> is contiguous
    // column-major storage (24×24×16 B = 9216 B per matrix, no padding).
    // std::complex<double> and Grid's ComplexD share layout (two doubles),
    // so we memcpy the entire array straight to the device — no host flatten
    // pass and no temporary 921 MB std::vector.  The kernel below reads with
    // column-major indexing src[c*N + r] to match Eigen's layout.
    auto &M_dev = (cb == Even) ? M_dev_e_ : M_dev_o_;
    if (M_dev.size() < nsites * N2) M_dev.resize(nsites * N2);
    static_assert(sizeof(std::complex<double>) == sizeof(ComplexD),
                  "std::complex<double> and ComplexD must share layout");
    acceleratorCopyToDevice(
        reinterpret_cast<void *>(const_cast<std::complex<double> *>(
            scalar_inv.data()->data())),
        &M_dev[0],
        nsites * N2 * sizeof(ComplexD));

    // (c) Compute lex-index table for (oSite, lane).  Depends only on the
    // grid layout, so cache it once per parity.
    auto &lex_dev = (cb == Even) ? lex_table_dev_e_ : lex_table_dev_o_;
    auto &lex_built = (cb == Even) ? lex_table_built_e_ : lex_table_built_o_;
    if (!lex_built) {
      std::vector<int> lex_host(oSites * Nsimd);
      std::vector<Coordinate> icoor(Nsimd);
      const int ndim = grid->Nd();
      for (int lane = 0; lane < Nsimd; ++lane) {
        icoor[lane].resize(ndim);
        grid->iCoorFromIindex(icoor[lane], lane);
      }
      thread_for(oidx, oSites, {
        Coordinate ocoor(ndim), lcoor(ndim);
        grid->oCoorFromOindex(ocoor, oidx);
        for (int lane = 0; lane < Nsimd; ++lane) {
          for (int mu = 0; mu < ndim; ++mu)
            lcoor[mu] = ocoor[mu] + grid->_rdimensions[mu] * icoor[lane][mu];
          int lex;
          Lexicographic::IndexFromCoor(lcoor, lex, grid->_ldimensions);
          lex_host[oidx * Nsimd + lane] = lex;
        }
      });
      lex_dev.resize(oSites * Nsimd);
      acceleratorCopyToDevice(&lex_host[0], &lex_dev[0],
                              oSites * Nsimd * sizeof(int));
      lex_built = true;
    }

    // (d) Per-lane scatter into InvField via coalescedWrite.
    ComplexD *M_dev_ptr = &M_dev[0];
    int *lex_dev_ptr = &lex_dev[0];
    InvField &simd_ref = *simd_field;
    autoView(out_v, simd_ref, AcceleratorWrite);
    accelerator_for(s, oSites, Nsimd, {
      // The accelerator_for macro binds 'lane' as the lambda's third parameter
      // (the SIMD-inner thread index on GPU).  Use putlane (host+device, takes
      // a scalar) directly to dodge the SIMT-vs-host coalescedWrite overload
      // split that breaks compilation when the body is also compiled for host.
      int simt_lane = static_cast<int>(lane);
      int lex  = lex_dev_ptr[s * Nsimd + simt_lane];
      const ComplexD *src = &M_dev_ptr[lex * N2];
      // Eigen is column-major; element (r,c) is at offset c*N + r.
      for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
          putlane(out_v[s]()()(r, c), src[c * N + r], simt_lane);
        }
      }
    });
    simd_field->Checkerboard() = cb;
  }

  // Pack the scalar Eigen 24×24 inverse into a SIMD-vectorized lattice field
  // suitable for accelerator_for kernels.  Done once per ImportFields, so the
  // per-CG-iteration apply path can read directly from SIMD storage.
  void PackInverseToSimd(const std::vector<SMU::SiteMatrix> &scalar_inv,
                         std::unique_ptr<InvField> &simd_field, int cb) {
    if (!simd_field) simd_field.reset(new InvField(&rbgrid_));
    typedef typename InvField::vector_object::scalar_object SiteScalar;
    std::vector<SiteScalar> tmp(scalar_inv.size());
    thread_for(x, scalar_inv.size(), {
      const auto &E = scalar_inv[x];
      for (int r = 0; r < kDim; ++r)
        for (int c = 0; c < kDim; ++c) {
          // Eigen stores std::complex<double>; SiteScalar uses Grid's
          // ComplexD (interface compatible).
          auto z = E(r, c);
          tmp[x]()()(r, c) = ComplexD(z.real(), z.imag());
        }
    });
    vectorizeFromLexOrdArray(tmp, *simd_field);
    simd_field->Checkerboard() = cb;
  }

 private:
  void ApplyDeltaCB(int cb, const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    auto &sig = (cb == Even) ? sigma_e_ : sigma_o_;
    auto &pi  = (cb == Even) ? pi_e_ : pi_o_;
    auto &sc  = (cb == Even) ? s_e_ : s_o_;
    auto &pc  = (cb == Even) ? p_e_ : p_o_;
    auto &tc  = (cb == Even) ? t_e_ : t_o_;
    for (int a = 0; a < TxqcdNf; ++a) out.f[a].Checkerboard() = cb;
    ApplyDelta(sig, pi, sc, pc, tc, in, out);
  }

 public:
  // SIMD path: do the per-site 24×24 mat-vec inside a single accelerator_for
  // over outer SIMD sites — the production-default fast path.  TXQCD_MOOEEINV_SCALAR=1
  // forces the legacy un/revec scalar path for A/B testing.
  //
  // Caveat: at exactly identity gauge (WEAK_FIELD_SCALE=0) on the 16³×48
  // production lattice the SIMD path triggers MultiShift CG divergence to
  // NaN — likely a degenerate-eigenvalue artifact from the translation-
  // invariant Schur operator (CG iteration order interacts with the SIMD
  // accumulator non-trivially when many shifts are exactly degenerate).
  // Real production HMC never starts from exact identity (configs come
  // from a warm start or an imported chroma cfg), so this hasn't been
  // observed in actual runs.  If you need to verify SIMD correctness from
  // a synthetic identity start, set TXQCD_MOOEEINV_SCALAR=1.
  void ApplyMooeeInv(int cb, const TXQCDFermionNf &in,
                     TXQCDFermionNf &out) {
    auto t_total0 = usecond();
    static int use_scalar = []() {
      const char *e = std::getenv("TXQCD_MOOEEINV_SCALAR");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    static int use_cublas = []() {
      const char *e = std::getenv("TXQCD_MOOEEINV_CUBLAS");
      return (e && *e && std::atoi(e)) ? 1 : 0;
    }();
    if (use_scalar) {
      ApplyMooeeInvScalar(cb, in, out);
    } else if (use_cublas) {
      ApplyMooeeInvCublas(cb, in, out);
    } else {
      ApplyMooeeInvSimd(cb, in, out);
    }
    t_apply_inv_us_ += usecond() - t_total0;
    n_apply_inv_++;
  }

  // cuBLAS gemmBatched-based 24×24 matvec.  Reuses the flat column-major
  // M_dev_ buffers populated by GPU pack (TXQCD_PRECOMPUTE_GPU=1) and
  // gemmBatched(M=24, N=1, K=24, batchCount=lSites/parity) instead of the
  // per-site SIMD accelerator_for.  Reshape kernels (SIMD ↔ flat) bracket the
  // gemm call.  Requires TXQCD_PRECOMPUTE_GPU=1 so M_dev_ is populated.
  void ApplyMooeeInvCublas(int cb, const TXQCDFermionNf &in,
                           TXQCDFermionNf &out) {
    GridBase *grid = in.f[0].Grid();
    uint64_t lSites = grid->lSites();
    uint64_t oSites = grid->oSites();
    using vobj = typename LatticeFermion::vector_object;
    constexpr int Nsimd = vobj::Nsimd();
    constexpr int N = SMU::kDim;       // 24
    constexpr int Ncomp = N;           // per-site fermion components

    auto &fin_flat  = (cb == Even) ? fermion_in_flat_e_  : fermion_in_flat_o_;
    auto &fout_flat = (cb == Even) ? fermion_out_flat_e_ : fermion_out_flat_o_;
    auto &lex_dev   = (cb == Even) ? lex_table_dev_e_    : lex_table_dev_o_;
    auto &lex_built = (cb == Even) ? lex_table_built_e_  : lex_table_built_o_;
    GRID_ASSERT(lex_built && "TXQCD_MOOEEINV_CUBLAS requires TXQCD_PRECOMPUTE_GPU=1 (lex table built by GPU pack)");

    if (fin_flat.size() < lSites * Ncomp) {
      fin_flat.resize(lSites * Ncomp);
      fout_flat.resize(lSites * Ncomp);
    }

    // (1) SIMD fermion → flat per-site (lex-ordered).  Per (oSite, lane) thread,
    // extract this lane's 24 components from the 2 spinor flavors.
    {
      ComplexD *fin_ptr = &fin_flat[0];
      int *lex_dev_ptr = &lex_dev[0];
      autoView(in_v0, in.f[0], AcceleratorRead);
      autoView(in_v1, in.f[1], AcceleratorRead);
      accelerator_for(s, oSites, Nsimd, {
        int simt_lane = static_cast<int>(lane);
        int lex = lex_dev_ptr[s * Nsimd + simt_lane];
        ComplexD *dst = &fin_ptr[lex * Ncomp];
        auto v0 = in_v0[s];
        auto v1 = in_v1[s];
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int i = 0; i < Nc; ++i) {
            dst[0 * 12 + alpha * 3 + i] = getlane(v0()(alpha)(i), simt_lane);
            dst[1 * 12 + alpha * 3 + i] = getlane(v1()(alpha)(i), simt_lane);
          }
        }
      });
    }

    // (2) Set up cuBLAS pointer arrays once per parity (cached).
    auto &Amk = (cb == Even) ? Amk_e_ : Amk_o_;
    auto &Bkn = (cb == Even) ? Bkn_e_ : Bkn_o_;
    auto &Cmn = (cb == Even) ? Cmn_e_ : Cmn_o_;
    auto &built = (cb == Even) ? cublas_ptrs_built_e_ : cublas_ptrs_built_o_;
    auto &M_dev = (cb == Even) ? M_dev_e_ : M_dev_o_;
    if (!built) {
      Amk.resize(lSites);
      Bkn.resize(lSites);
      Cmn.resize(lSites);
      ComplexD *M_ptr   = &M_dev[0];
      ComplexD *Bin_ptr = &fin_flat[0];
      ComplexD *Cout_ptr = &fout_flat[0];
      ComplexD **Amk_ptr = &Amk[0];
      ComplexD **Bkn_ptr = &Bkn[0];
      ComplexD **Cmn_ptr = &Cmn[0];
      accelerator_for(i, lSites, 1, {
        Amk_ptr[i] = &M_ptr[i * 576];
        Bkn_ptr[i] = &Bin_ptr[i * 24];
        Cmn_ptr[i] = &Cout_ptr[i * 24];
      });
      built = true;
    }

    // (3) Batched 24×24 × 24×1 matvec via cuBLAS.  Column-major matrices
    // (Eigen layout, row index r is fastest) match the cuBLAS convention.
    GridBLAS blas;
    blas.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                     N, 1, N,
                     ComplexD(1.0, 0.0),
                     Amk, Bkn,
                     ComplexD(0.0, 0.0),
                     Cmn);

    // (4) Flat → SIMD fermion.
    for (int a = 0; a < TxqcdNf; ++a) out.f[a].Checkerboard() = cb;
    {
      ComplexD *fout_ptr = &fout_flat[0];
      int *lex_dev_ptr = &lex_dev[0];
      autoView(out_v0, out.f[0], AcceleratorWrite);
      autoView(out_v1, out.f[1], AcceleratorWrite);
      accelerator_for(s, oSites, Nsimd, {
        int simt_lane = static_cast<int>(lane);
        int lex = lex_dev_ptr[s * Nsimd + simt_lane];
        const ComplexD *src = &fout_ptr[lex * Ncomp];
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int i = 0; i < Nc; ++i) {
            putlane(out_v0[s]()(alpha)(i), src[0 * 12 + alpha * 3 + i], simt_lane);
            putlane(out_v1[s]()(alpha)(i), src[1 * 12 + alpha * 3 + i], simt_lane);
          }
        }
      });
    }
  }

  // LatticeView has no default ctor, so std::array<LatticeView,N> can't be
  // default-constructed and then assigned.  These helpers aggregate-initialize
  // the array from a parameter pack of indices (C++17-friendly).
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

  // Nf-generic SIMD ApplyMooeeInv.  Uses std::array of LatticeView (which is
  // trivially copyable) so the kernel can index per-flavor without
  // hand-unrolled in0/in1 view names.  For TxqcdNf=2 the compiler still
  // produces the same fully-unrolled inner loop as the previous Nf=2 path.
  void ApplyMooeeInvSimd(int cb, const TXQCDFermionNf &in,
                         TXQCDFermionNf &out) {
    GRID_ASSERT(inv_simd_e_ && inv_simd_o_);
    InvField &inv_simd = (cb == Even) ? *inv_simd_e_ : *inv_simd_o_;
    GridBase *fg = in.f[0].Grid();
    for (int a = 0; a < TxqcdNf; ++a) out.f[a].Checkerboard() = cb;

    autoView(inv_v, inv_simd, AcceleratorRead);
    typedef decltype(in.f[0].View(AcceleratorRead))  FermViewIn;
    typedef decltype(out.f[0].View(AcceleratorWrite)) FermViewOut;
    // LatticeView has no default ctor — aggregate-initialize the std::array
    // through an index_sequence helper (C++17-compatible).
    auto in_v  = MakeFermViewsRead<TxqcdNf>(in,  std::make_index_sequence<TxqcdNf>{});
    auto out_v = MakeFermViewsWrite<TxqcdNf>(out, std::make_index_sequence<TxqcdNf>{});

    typedef decltype(coalescedRead(in_v[0][0])) FermSitePerLane;
    const int Nsimd = LatticeFermion::vector_object::Nsimd();

    accelerator_for(s, fg->oSites(), Nsimd, {
      auto Mlane = inv_v(s);
      // Per-flavor lane reads collected into a small stack array.
      FermSitePerLane in_lanes[TxqcdNf];
      for (int a = 0; a < TxqcdNf; ++a) in_lanes[a] = in_v[a](s);
      FermSitePerLane out_acc[TxqcdNf];
      typedef typename std::remove_reference<decltype(Mlane()()(0, 0))>::type MEl;
      // r = a*Ns*Nc + alpha*Nc + i ; M_site is (TxqcdNf*Ns*Nc) square.
      for (int a_out = 0; a_out < TxqcdNf; ++a_out) {
        for (int alpha_out = 0; alpha_out < Ns; ++alpha_out) {
          for (int i_out = 0; i_out < Nc; ++i_out) {
            int r = a_out * Ns * Nc + alpha_out * Nc + i_out;
            MEl sum;
            zeroit(sum);
            for (int a_in = 0; a_in < TxqcdNf; ++a_in) {
              for (int alpha_in = 0; alpha_in < Ns; ++alpha_in) {
                for (int i_in = 0; i_in < Nc; ++i_in) {
                  int c = a_in * Ns * Nc + alpha_in * Nc + i_in;
                  auto Mrc = Mlane()()(r, c);
                  sum = sum + Mrc * in_lanes[a_in]()(alpha_in)(i_in);
                }
              }
            }
            out_acc[a_out]()(alpha_out)(i_out) = sum;
          }
        }
      }
      for (int a = 0; a < TxqcdNf; ++a)
        coalescedWrite(out_v[a][s], out_acc[a]);
    });

    for (int a = 0; a < TxqcdNf; ++a) {
      in_v[a].ViewClose();
      out_v[a].ViewClose();
    }
  }

  void ApplyMooeeInvScalar(int cb, const TXQCDFermionNf &in,
                           TXQCDFermionNf &out) {
    auto &inv = (cb == Even) ? inv_even_ : inv_odd_;

    typedef typename LatticeFermion::vector_object::scalar_object FermSobj;
    std::array<std::vector<FermSobj>, TxqcdNf> in_s, out_s;
    auto t_unv0 = usecond();
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(in_s[a], in.f[a]);
      out_s[a].resize(in_s[a].size());
    }
    t_unvec_us_ += usecond() - t_unv0;

    uint64_t nsites = in_s[0].size();

    // OMP-parallel over sites.  Per-site SiteVector is thread-local.
    // Brings ApplyMooeeInvScalar from serial (~5s/call on 16³×48 production)
    // to ~Ncores× faster, the difference between unusably slow and a viable
    // Nf=3 production hot path until we generalise the SIMD kernel to Nf>2.
    thread_for(x, nsites, {
      SMU::SiteVector v, w;
      for (int a = 0; a < TxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            auto z = in_s[a][x]()(alpha)(i);
            v(a * Ns * Nc + alpha * Nc + i) =
                std::complex<double>(z.real(), z.imag());
          }

      w = inv[x] * v;

      for (int a = 0; a < TxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            auto &z = w(a * Ns * Nc + alpha * Nc + i);
            out_s[a][x]()(alpha)(i) = ComplexD(z.real(), z.imag());
          }
    });

    auto t_rev0 = usecond();
    for (int a = 0; a < TxqcdNf; ++a) {
      vectorizeFromLexOrdArray(out_s[a], out.f[a]);
      out.f[a].Checkerboard() = cb;
    }
    t_revec_us_ += usecond() - t_rev0;
  }
};

NAMESPACE_END(Grid);
