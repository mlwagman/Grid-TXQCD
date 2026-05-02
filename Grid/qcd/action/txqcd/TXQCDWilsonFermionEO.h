#pragma once
// Even-odd preconditioned TXQCD Wilson operator.
//
// M = D_W + (4+m) + Δ, where Δ is the site-diagonal aux-field insertion.
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
#include <Grid/qcd/action/fermion/WilsonFermion.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonFermionEO {
 public:
  static constexpr int kDim = TxqcdNf * Ns * Nc;  // 24

  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;
  typedef typename Impl::GaugeField GaugeField;

  static typename Impl::ImplParams DefaultImplParams() {
    typename Impl::ImplParams p;
    p.boundary_phases.resize(Nd, 1.0);
    p.boundary_phases[Nd - 1] = -1.0;  // antiperiodic time
    return p;
  }

  TXQCDWilsonFermionEO(GaugeField &Umu, GridCartesian &grid,
                       GridRedBlackCartesian &rbgrid, RealD mass,
                       LatticeSigmaField &sigma, LatticePiField &pi,
                       LatticeSFieldC &s, LatticePFieldC &p,
                       LatticeTField &t,
                       typename Impl::ImplParams impl_p = DefaultImplParams())
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), diag_mass_(4.0 + mass),
        Dw_(Umu, grid, rbgrid, mass, impl_p),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t),
        sigma_e_(&rbgrid), sigma_o_(&rbgrid),
        pi_e_(&rbgrid), pi_o_(&rbgrid),
        s_e_(&rbgrid), s_o_(&rbgrid),
        p_e_(&rbgrid), p_o_(&rbgrid),
        t_e_(&rbgrid), t_o_(&rbgrid) {
    PrecomputeSpinMatrices();
    ImportFields();
  }

  void ImportGauge(const GaugeField &U) {
    Dw_.ImportGauge(U);
    ImportFields();
  }

  // ----- Full-grid operator (for testing / comparison) -----
  void M(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a) Dw_.M(in.f[a], out.f[a]);
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

  // Mooee: (4+m)*in + Δ(x)*in on the input's checkerboard.
  void Mooee(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    int cb = in.f[0].Checkerboard();
    for (int a = 0; a < TxqcdNf; ++a) {
      out.f[a] = diag_mass_ * in.f[a];
      out.f[a].Checkerboard() = cb;
    }
    TXQCDFermionNf d(in.Grid());
    ApplyDeltaCB(cb, in, d);
    for (int a = 0; a < TxqcdNf; ++a) out.f[a] = out.f[a] + d.f[a];
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

  // MooeeInv: per-site multiply by precomputed (Mooee)^{-1}.
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

  // Meooe: Wilson hopping per flavor, from one CB to the other.
  void Meooe(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a)
      Dw_.Meooe(in.f[a], out.f[a]);
  }

  void MeooeDag(const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    for (int a = 0; a < TxqcdNf; ++a)
      Dw_.MeooeDag(in.f[a], out.f[a]);
  }

  // Expose for force computation.
  WilsonOp &Wilson() { return Dw_; }
  RealD DiagMass() const { return diag_mass_; }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD diag_mass_;
  WilsonOp Dw_;

  // Aux field references (borrowed from TXQCDField, updated in place by HMC).
  LatticeSigmaField &sigma_;
  LatticePiField    &pi_;
  LatticeSFieldC    &s_;
  LatticePFieldC    &p_;
  LatticeTField     &t_;

  // RB-projected aux fields, refreshed at ImportFields().
  LatticeSigmaField sigma_e_, sigma_o_;
  LatticePiField    pi_e_, pi_o_;
  LatticeSFieldC    s_e_, s_o_;
  LatticePFieldC    p_e_, p_o_;
  LatticeTField     t_e_, t_o_;

  // Precomputed 24×24 Mooee inverse per site, per checkerboard.
  // Indexed by the lex-order site index from unvectorizeToLexOrdArray.
  std::vector<Eigen::Matrix<std::complex<double>, kDim, kDim>> inv_even_;
  std::vector<Eigen::Matrix<std::complex<double>, kDim, kDim>> inv_odd_;

  // Precomputed spin matrices: γ₅ and i·σ_{μν} as kDim × kDim blocks.
  Eigen::Matrix<std::complex<double>, Ns, Ns> gamma5_mat_;
  std::array<std::array<Eigen::Matrix<std::complex<double>, Ns, Ns>, Nd>, Nd>
      isigma_mat_;  // isigma_mat_[mu][nu] for mu < nu

  void PrecomputeSpinMatrices() {
    gamma5_mat_ = Eigen::Matrix<std::complex<double>, Ns, Ns>::Zero();
    Gamma g5(Gamma::Algebra::Gamma5);
    for (int b = 0; b < Ns; ++b) {
      SpinVector e;
      e = Zero();
      e()(b) = ComplexD(1.0, 0.0);
      SpinVector r = g5 * e;
      for (int a = 0; a < Ns; ++a)
        gamma5_mat_(a, b) = std::complex<double>(
            TensorRemove(r()(a)).real(), TensorRemove(r()(a)).imag());
    }
    for (int mu = 0; mu < Nd; ++mu)
      for (int nu = mu + 1; nu < Nd; ++nu) {
        Gamma smn(SigmaMuNuAlgebra(mu, nu));
        isigma_mat_[mu][nu] =
            Eigen::Matrix<std::complex<double>, Ns, Ns>::Zero();
        for (int b = 0; b < Ns; ++b) {
          SpinVector e;
          e = Zero();
          e()(b) = ComplexD(1.0, 0.0);
          SpinVector r = smn * e;
          for (int a = 0; a < Ns; ++a) {
            std::complex<double> val(TensorRemove(r()(a)).real(),
                                     TensorRemove(r()(a)).imag());
            isigma_mat_[mu][nu](a, b) = std::complex<double>(0, 1) * val;
          }
        }
      }
  }

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

    PrecomputeInverse(Even, inv_even_);
    PrecomputeInverse(Odd, inv_odd_);
  }

  // Apply Δ using the RB-projected aux fields for the given checkerboard.
  void ApplyDeltaCB(int cb, const TXQCDFermionNf &in, TXQCDFermionNf &out) {
    auto &sig = (cb == Even) ? sigma_e_ : sigma_o_;
    auto &pi  = (cb == Even) ? pi_e_ : pi_o_;
    auto &sc  = (cb == Even) ? s_e_ : s_o_;
    auto &pc  = (cb == Even) ? p_e_ : p_o_;
    auto &tc  = (cb == Even) ? t_e_ : t_o_;
    for (int a = 0; a < TxqcdNf; ++a) out.f[a].Checkerboard() = cb;
    ApplyDelta(sig, pi, sc, pc, tc, in, out);
  }

  // Build the 24×24 site matrix M_site = (4+m)I + Δ_site for a single site,
  // given scalar site objects for each aux field.
  template <class SigSobj, class PiSobj, class SSobj, class PSobj, class TSobj>
  void BuildSiteMatrix(
      const SigSobj &sig_site, const PiSobj &pi_site,
      const SSobj &s_site, const PSobj &p_site, const TSobj &t_site,
      Eigen::Matrix<std::complex<double>, kDim, kDim> &M) const {
    M = Eigen::Matrix<std::complex<double>, kDim, kDim>::Zero();

    // Diagonal: (4+m)*I
    for (int r = 0; r < kDim; ++r) M(r, r) = diag_mass_;

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // row = a*Ns*Nc + alpha*Nc + i
    // col = b*Ns*Nc + beta*Nc + j
    for (int a = 0; a < TxqcdNf; ++a) {
      for (int b = 0; b < TxqcdNf; ++b) {
        // sigma: δ(i,j) * δ(α,β) * σ(a,b)
        std::complex<double> sig_ab(sig_site()()(a, b).real(),
                                    sig_site()()(a, b).imag());
        // pi: δ(i,j) * γ₅(α,β) * π(a,b)
        std::complex<double> pi_ab(pi_site()()(a, b).real(),
                                   pi_site()()(a, b).imag());
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int beta = 0; beta < Ns; ++beta) {
            std::complex<double> g5 = gamma5_mat_(alpha, beta);
            for (int i = 0; i < Nc; ++i) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = b * Ns * Nc + beta * Nc + i;
              if (alpha == beta) M(r, c) += sig_ab;
              M(r, c) += pi_ab * g5;
            }
          }
        }
      }
    }

    // Color sector: shared across flavors (δ(a,b) implicit).
    for (int a = 0; a < TxqcdNf; ++a) {
      for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
          std::complex<double> s_ij(s_site()()(i, j).real(),
                                    s_site()()(i, j).imag());
          std::complex<double> p_ij(p_site()()(i, j).real(),
                                    p_site()()(i, j).imag());
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              int r = a * Ns * Nc + alpha * Nc + i;
              int c = a * Ns * Nc + beta * Nc + j;
              // s term: δ(α,β) * s(i,j)/√2
              if (alpha == beta) M(r, c) += inv_sqrt2 * s_ij;
              // p term: γ₅(α,β) * p(i,j)/√2
              M(r, c) += inv_sqrt2 * p_ij * gamma5_mat_(alpha, beta);
              // tensor term: Σ_{μ<ν} t_{μν}(i,j) * (iσ_{μν})(α,β)
              for (int mu = 0; mu < Nd; ++mu)
                for (int nu = mu + 1; nu < Nd; ++nu) {
                  std::complex<double> t_ij(
                      t_site()(mu, nu)(i, j).real(),
                      t_site()(mu, nu)(i, j).imag());
                  M(r, c) += t_ij * isigma_mat_[mu][nu](alpha, beta);
                }
            }
          }
        }
      }
    }
  }

  void PrecomputeInverse(
      int cb,
      std::vector<Eigen::Matrix<std::complex<double>, kDim, kDim>> &inv) {
    auto &sig = (cb == Even) ? sigma_e_ : sigma_o_;
    auto &pi  = (cb == Even) ? pi_e_ : pi_o_;
    auto &sc  = (cb == Even) ? s_e_ : s_o_;
    auto &pc  = (cb == Even) ? p_e_ : p_o_;
    auto &tc  = (cb == Even) ? t_e_ : t_o_;

    typedef typename LatticeSigmaField::vector_object::scalar_object SigSobj;
    typedef typename LatticePiField::vector_object::scalar_object PiSobj;
    typedef typename LatticeSFieldC::vector_object::scalar_object SSobj;
    typedef typename LatticePFieldC::vector_object::scalar_object PSobj;
    typedef typename LatticeTField::vector_object::scalar_object TSobj;

    std::vector<SigSobj> sig_s;
    unvectorizeToLexOrdArray(sig_s, sig);
    std::vector<PiSobj> pi_s;
    unvectorizeToLexOrdArray(pi_s, pi);
    std::vector<SSobj> s_s;
    unvectorizeToLexOrdArray(s_s, sc);
    std::vector<PSobj> p_s;
    unvectorizeToLexOrdArray(p_s, pc);
    std::vector<TSobj> t_s;
    unvectorizeToLexOrdArray(t_s, tc);

    uint64_t nsites = sig_s.size();
    inv.resize(nsites);

    // Per-site BuildSiteMatrix + 24x24 Eigen.inverse() is independent across
    // sites — parallelize across CPU cores (matches the clover variant in
    // TXQCDSiteMatrix::PrecomputeInverses).
    thread_for(x, nsites, {
      Eigen::Matrix<std::complex<double>, kDim, kDim> M;
      BuildSiteMatrix(sig_s[x], pi_s[x], s_s[x], p_s[x], t_s[x], M);
      inv[x] = M.inverse();
    });
  }

  void ApplyMooeeInv(int cb, const TXQCDFermionNf &in,
                     TXQCDFermionNf &out) {
    auto &inv = (cb == Even) ? inv_even_ : inv_odd_;

    // Unvectorize input fermion fields to scalar site objects.
    typedef typename LatticeFermion::vector_object::scalar_object FermSobj;
    std::array<std::vector<FermSobj>, TxqcdNf> in_s, out_s;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(in_s[a], in.f[a]);
      out_s[a].resize(in_s[a].size());
    }

    uint64_t nsites = in_s[0].size();

    // Per-site 24x24 mat-vec is the dominant CG cost — parallelize.  v/w
    // declared inside so each thread has its own.
    thread_for(x, nsites, {
      Eigen::Matrix<std::complex<double>, kDim, 1> v, w;
      // Pack site vector: v[a*Ns*Nc + alpha*Nc + i]
      for (int a = 0; a < TxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            auto z = in_s[a][x]()(alpha)(i);
            v(a * Ns * Nc + alpha * Nc + i) =
                std::complex<double>(z.real(), z.imag());
          }

      w = inv[x] * v;

      // Unpack back to site objects.
      for (int a = 0; a < TxqcdNf; ++a)
        for (int alpha = 0; alpha < Ns; ++alpha)
          for (int i = 0; i < Nc; ++i) {
            auto &z = w(a * Ns * Nc + alpha * Nc + i);
            out_s[a][x]()(alpha)(i) = ComplexD(z.real(), z.imag());
          }
    });

    for (int a = 0; a < TxqcdNf; ++a) {
      vectorizeFromLexOrdArray(out_s[a], out.f[a]);
      out.f[a].Checkerboard() = cb;
    }
  }
};

NAMESPACE_END(Grid);
