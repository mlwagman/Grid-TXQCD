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
              << "] ApplyMooeeInv:    " << n_apply_inv_
              << " calls, " << fmt(t_apply_inv_us_) << " s ("
              << (n_apply_inv_ ? fmt(t_apply_inv_us_) / n_apply_inv_ * 1e3 : 0)
              << " ms/call)" << std::endl;
    std::cout << GridLogMessage << "[TXQCD-EO timers/" << tag
              << "] -- of which un/revectorize: "
              << fmt(t_unvec_us_ + t_revec_us_) << " s" << std::endl;
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

  SMU::SpinMatrices sm_;

  // Profiling counters; use ResetTimers + PrintTimers to read them.
  mutable uint64_t t_precompute_us_{0};
  mutable uint64_t t_apply_inv_us_{0};
  mutable uint64_t t_unvec_us_{0};
  mutable uint64_t t_revec_us_{0};
  mutable uint64_t n_precompute_{0};
  mutable uint64_t n_apply_inv_{0};

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

    auto aux_e = SMU::UnvectorizeAux(sigma_e_, pi_e_, s_e_, p_e_, t_e_);
    auto aux_o = SMU::UnvectorizeAux(sigma_o_, pi_o_, s_o_, p_o_, t_o_);

    auto t0 = usecond();
    if (csw_ != 0.0) {
      auto cl_e = SMU::UnvectorizeClover(FS_e_);
      auto cl_o = SMU::UnvectorizeClover(FS_o_);
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_e, csw_, &cl_e, inv_even_);
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_o, csw_, &cl_o, inv_odd_);
    } else {
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_e, 0.0, nullptr, inv_even_);
      SMU::PrecomputeInverses(sm_, diag_mass_, aux_o, 0.0, nullptr, inv_odd_);
    }
    PackInverseToSimd(inv_even_, inv_simd_e_, Even);
    PackInverseToSimd(inv_odd_,  inv_simd_o_, Odd);
    t_precompute_us_ += usecond() - t0;
    n_precompute_++;
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
    // The SIMD path is hand-unrolled for Nf=2 only; force the scalar
    // (Nf-generic) path for any other Nf so the test suite can exercise
    // non-degenerate Nf=3 setups.
    if constexpr (TxqcdNf != 2) {
      ApplyMooeeInvScalar(cb, in, out);
    } else if (use_scalar) {
      ApplyMooeeInvScalar(cb, in, out);
    } else {
      ApplyMooeeInvSimd(cb, in, out);
    }
    t_apply_inv_us_ += usecond() - t_total0;
    n_apply_inv_++;
  }

  void ApplyMooeeInvSimd(int cb, const TXQCDFermionNf &in,
                         TXQCDFermionNf &out) {
    GRID_ASSERT(inv_simd_e_ && inv_simd_o_);
    InvField &inv_simd = (cb == Even) ? *inv_simd_e_ : *inv_simd_o_;
    GridBase *fg = in.f[0].Grid();
    out.f[0].Checkerboard() = cb;
    out.f[1].Checkerboard() = cb;

    autoView(inv_v, inv_simd, AcceleratorRead);
    autoView(in0_v, in.f[0], AcceleratorRead);
    autoView(in1_v, in.f[1], AcceleratorRead);
    autoView(out0_v, out.f[0], AcceleratorWrite);
    autoView(out1_v, out.f[1], AcceleratorWrite);

    typedef decltype(coalescedRead(in0_v[0])) FermSitePerLane;
    const int Nsimd = LatticeFermion::vector_object::Nsimd();

    accelerator_for(s, fg->oSites(), Nsimd, {
      // Per-lane reads: view(s) returns the scalar object for the current
      // SIMD lane (each GPU thread handles one lane).  view[s] is the
      // raw vobj and is only used for coalescedWrite.
      auto Mlane = inv_v(s);
      auto in0   = in0_v(s);
      auto in1   = in1_v(s);
      FermSitePerLane out0_acc;
      FermSitePerLane out1_acc;
      // Use the actual element type returned by these reads to declare the
      // accumulator (Coalesced types differ between CPU/GPU builds).
      typedef typename std::remove_reference<decltype(Mlane()()(0, 0))>::type MEl;
      // r = a*Ns*Nc + alpha*Nc + i  ; flavor a in {0,1}.
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
                  auto v_in = (a_in == 0)
                       ? in0()(alpha_in)(i_in)
                       : in1()(alpha_in)(i_in);
                  sum = sum + Mrc * v_in;
                }
              }
            }
            if (a_out == 0) out0_acc()(alpha_out)(i_out) = sum;
            else            out1_acc()(alpha_out)(i_out) = sum;
          }
        }
      }
      coalescedWrite(out0_v[s], out0_acc);
      coalescedWrite(out1_v[s], out1_acc);
    });
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
    SMU::SiteVector v, w;

    for (uint64_t x = 0; x < nsites; ++x) {
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
    }

    auto t_rev0 = usecond();
    for (int a = 0; a < TxqcdNf; ++a) {
      vectorizeFromLexOrdArray(out_s[a], out.f[a]);
      out.f[a].Checkerboard() = cb;
    }
    t_revec_us_ += usecond() - t_rev0;
  }
};

NAMESPACE_END(Grid);
