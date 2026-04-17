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
#include <Grid/qcd/action/txqcd/TXQCDSiteMatrix.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonFermionEO {
 public:
  typedef TXQCDSiteMatrixUtil SMU;
  static constexpr int kDim = SMU::kDim;

  typedef WilsonImplR Impl;
  typedef WilsonFermion<Impl> WilsonOp;
  typedef typename Impl::GaugeField GaugeField;

  TXQCDWilsonFermionEO(GaugeField &Umu, GridCartesian &grid,
                       GridRedBlackCartesian &rbgrid, RealD mass,
                       LatticeSigmaField &sigma, LatticePiField &pi,
                       LatticeSFieldC &s, LatticePFieldC &p,
                       LatticeTField &t)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass), diag_mass_(4.0 + mass),
        Dw_(Umu, grid, rbgrid, mass),
        sigma_(sigma), pi_(pi), s_(s), p_(p), t_(t),
        sigma_e_(&rbgrid), sigma_o_(&rbgrid),
        pi_e_(&rbgrid), pi_o_(&rbgrid),
        s_e_(&rbgrid), s_o_(&rbgrid),
        p_e_(&rbgrid), p_o_(&rbgrid),
        t_e_(&rbgrid), t_o_(&rbgrid) {
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
  RealD DiagMass() const { return diag_mass_; }

 private:
  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD diag_mass_;
  WilsonOp Dw_;

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

  std::vector<SMU::SiteMatrix> inv_even_;
  std::vector<SMU::SiteMatrix> inv_odd_;

  SMU::SpinMatrices sm_;

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

    auto aux_e = SMU::UnvectorizeAux(sigma_e_, pi_e_, s_e_, p_e_, t_e_);
    SMU::PrecomputeInverses(sm_, diag_mass_, aux_e, inv_even_);
    auto aux_o = SMU::UnvectorizeAux(sigma_o_, pi_o_, s_o_, p_o_, t_o_);
    SMU::PrecomputeInverses(sm_, diag_mass_, aux_o, inv_odd_);
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

  void ApplyMooeeInv(int cb, const TXQCDFermionNf &in,
                     TXQCDFermionNf &out) {
    auto &inv = (cb == Even) ? inv_even_ : inv_odd_;

    typedef typename LatticeFermion::vector_object::scalar_object FermSobj;
    std::array<std::vector<FermSobj>, TxqcdNf> in_s, out_s;
    for (int a = 0; a < TxqcdNf; ++a) {
      unvectorizeToLexOrdArray(in_s[a], in.f[a]);
      out_s[a].resize(in_s[a].size());
    }

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

    for (int a = 0; a < TxqcdNf; ++a) {
      vectorizeFromLexOrdArray(out_s[a], out.f[a]);
      out.f[a].Checkerboard() = cb;
    }
  }
};

NAMESPACE_END(Grid);
