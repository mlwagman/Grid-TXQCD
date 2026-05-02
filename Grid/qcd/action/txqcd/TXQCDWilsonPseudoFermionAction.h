#pragma once
// Two-flavor pseudofermion action for the TXQCD Wilson Dirac operator
//
//   S = phi^dag (M^dag M)^{-1} phi,    M = D_W + Delta(aux),
//
// implemented as Action<TXQCDField> so it sits in the same MD level family as
// AuxiliaryFieldGaussianAction. M includes the local Delta insertion in
// flavor (sigma, pi) and color (s, p, t_{mu,nu}) sectors; D_W is a per-flavor
// stock WilsonFermion<WilsonImplR>.
//
// refresh: chi ~ N(0,1/2) on each flavor; phi = M^dag chi. Then
//   S = chi^dag M (M^dag M)^{-1} M^dag chi = chi^dag chi.
//
// deriv: with X = (M^dag M)^{-1} phi, Y = M X,
//   dS = -2 Re[Y^dag dM X].
// Each piece of dM is purely local and linear in the corresponding aux field,
// so the aux-field force is a per-site bilinear in (Y, X). Sign and Hermitian
// projection are folded into the closed forms below.
//   F_sigma(x)_{a,b} = -(<Y_b|X_a>(x) + <Y_a|X_b>*(x))   (Hermitian)
//   F_pi(x)_{a,b}    = -(<Y_b|g5 X_a> + <Y_a|g5 X_b>*)
// Color sector: per-site Nc x Nc Hermitian outer products summed over flavor
// and spin, with explicit (1/sqrt 2) and (i sigma_{mu,nu}) factors matching
// ApplyDeltaColor.
//
// Gauge force: standard recipe applied per flavor:
//   dSdU = sum_a [ MDeriv(Y_a, X_a, DaggerNo) + MDeriv(X_a, Y_a, DaggerYes) ].

#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>

NAMESPACE_BEGIN(Grid);

// -------------------- helpers: aux-force builders ----------------------

// Per-site flavor matrix g_{a,b}(x) = <Y_a(x)|X_b(x)>_(spin,color).
inline LatticeSigmaField FlavorBilinear(const TXQCDFermionNf &Y,
                                        const TXQCDFermionNf &X) {
  GridBase *grid = Y.Grid();
  LatticeSigmaField G(grid); G = Zero();
  G.Checkerboard() = Y.f[0].Checkerboard();

  if constexpr (TxqcdNf != 2) {
    // Generic-Nf path: assemble G via per-(a,b) localInnerProduct + per-site
    // poke.  Slower than the Nf=2 SIMD unroll but Nf-agnostic.
    for (int a = 0; a < TxqcdNf; ++a) {
      for (int b = 0; b < TxqcdNf; ++b) {
        // sum_{alpha,i} conj(Y_a)(alpha,i) * X_b(alpha,i) per site.
        auto inner = localInnerProduct(Y.f[a], X.f[b]);
        autoView(Gv, G, CpuWrite);
        autoView(iv, inner, CpuRead);
        thread_for(ss, grid->oSites(), {
          // tensor_reduced of SpinColourVector is iScalar<iScalar<iScalar<v>>>.
          Gv[ss]()()(a, b) = iv[ss]()()();
        });
      }
    }
    return G;
  }

  // Fuse the Nf=2 outer loops into one accelerator_for over outer SIMD sites.
  autoView(Gv,  G,       AcceleratorWrite);
  autoView(Y0v, Y.f[0],  AcceleratorRead);
  autoView(Y1v, Y.f[1],  AcceleratorRead);
  autoView(X0v, X.f[0],  AcceleratorRead);
  autoView(X1v, X.f[1],  AcceleratorRead);
  const int Nsimd = LatticeFermion::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto Y0 = Y0v(ss); auto Y1 = Y1v(ss);
    auto X0 = X0v(ss); auto X1 = X1v(ss);
    typedef typename std::remove_cv<typename std::remove_reference<decltype(Gv(ss))>::type>::type SigSitePerLane;
    SigSitePerLane g_acc;
    for (int a = 0; a < TxqcdNf; ++a) {
      auto Y_a = (a == 0) ? Y0 : Y1;
      for (int b = 0; b < TxqcdNf; ++b) {
        auto X_b = (b == 0) ? X0 : X1;
        decltype(conjugate(Y_a()(0)(0)) * X_b()(0)(0)) acc;
        zeroit(acc);
        for (int alpha = 0; alpha < Ns; ++alpha) {
          for (int i = 0; i < Nc; ++i) {
            acc = acc + conjugate(Y_a()(alpha)(i)) * X_b()(alpha)(i);
          }
        }
        g_acc()()(a, b) = acc;
      }
    }
    coalescedWrite(Gv[ss], g_acc);
  });
  return G;
}

// F = -(G^T + G^*) per site, with G the flavor bilinear above. Hermitian.
inline LatticeSigmaField HermitianFlavorForce(const LatticeSigmaField &G) {
  GridBase *grid = G.Grid();
  LatticeSigmaField F(grid); F = Zero();
  F.Checkerboard() = G.Checkerboard();
  autoView(Fv, F, AcceleratorWrite);
  autoView(Gv, G, AcceleratorRead);
  const int Nsimd = LatticeSigmaField::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto g = Gv(ss);
    typedef typename std::remove_cv<typename std::remove_reference<decltype(g)>::type>::type SigSitePerLane;
    SigSitePerLane f_acc;
    for (int a = 0; a < TxqcdNf; ++a) {
      for (int b = 0; b < TxqcdNf; ++b) {
        f_acc()()(a, b) = -(g()()(b, a) + conjugate(g()()(a, b)));
      }
    }
    coalescedWrite(Fv[ss], f_acc);
  });
  return F;
}

// Per-site Nc x Nc bilinear summed over flavor and spin, with optional spin
// matrix Op acting on X first. Returns G with G(x)_{i,j} =
//   sum_{a,alpha,beta} conj(Y_a(x)_{alpha,i}) (Op)_{alpha,beta} X_a(x)_{beta,j}.
//
// Implemented with autoView + thread_for since Grid does not export a
// site-local "color outer product" for fermions.
template <class Spin4Op>
inline LatticeSFieldC ColorBilinearSpinOp(const TXQCDFermionNf &Y,
                                          const TXQCDFermionNf &X,
                                          const Spin4Op &Op) {
  GridBase *grid = Y.Grid();
  LatticeSFieldC G(grid); G = Zero();
  G.Checkerboard() = Y.f[0].Checkerboard();
  // Spin matrix Op is small (4×4 ComplexD).  Capture it via a flat 16-entry
  // array so the lambda can access it through a __device__-friendly local.
  std::array<ComplexD, Ns * Ns> opflat{};
  for (int alpha = 0; alpha < Ns; ++alpha)
    for (int beta = 0; beta < Ns; ++beta)
      opflat[alpha * Ns + beta] = Op(alpha, beta);

  if constexpr (TxqcdNf != 2) {
    // Generic-Nf CPU thread_for: per-flavor, per-site, accumulate into G.
    autoView(Gv, G, CpuWrite);
    int cb = G.Checkerboard();
    for (int a = 0; a < TxqcdNf; ++a) {
      autoView(Yav, Y.f[a], CpuRead);
      autoView(Xav, X.f[a], CpuRead);
      thread_for(ss, grid->oSites(), {
        for (int i = 0; i < Nc; ++i) {
          for (int j = 0; j < Nc; ++j) {
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int beta = 0; beta < Ns; ++beta) {
                ComplexD op_ab = opflat[alpha * Ns + beta];
                Gv[ss]()()(i, j) = Gv[ss]()()(i, j)
                  + conjugate(Yav[ss]()(alpha)(i)) * op_ab
                  * Xav[ss]()(beta)(j);
              }
            }
          }
        }
      });
    }
    G.Checkerboard() = cb;
    return G;
  }

  autoView(Gv,  G,       AcceleratorWrite);
  autoView(Y0v, Y.f[0],  AcceleratorRead);
  autoView(Y1v, Y.f[1],  AcceleratorRead);
  autoView(X0v, X.f[0],  AcceleratorRead);
  autoView(X1v, X.f[1],  AcceleratorRead);
  const int Nsimd = LatticeFermion::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto Y0 = Y0v(ss); auto Y1 = Y1v(ss);
    auto X0 = X0v(ss); auto X1 = X1v(ss);
    typedef typename std::remove_cv<typename std::remove_reference<decltype(Gv(ss))>::type>::type SSitePerLane;
    SSitePerLane g_acc;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        decltype(conjugate(Y0()(0)(0)) * X0()(0)(0)) acc;
        zeroit(acc);
        for (int a = 0; a < TxqcdNf; ++a) {
          auto Y_a = (a == 0) ? Y0 : Y1;
          auto X_a = (a == 0) ? X0 : X1;
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int beta = 0; beta < Ns; ++beta) {
              ComplexD op_ab = opflat[alpha * Ns + beta];
              acc = acc + conjugate(Y_a()(alpha)(i)) * op_ab * X_a()(beta)(j);
            }
          }
        }
        g_acc()()(i, j) = acc;
      }
    }
    coalescedWrite(Gv[ss], g_acc);
  });
  return G;
}

// F = -(G^T + G^*) on color indices. Hermitian.
inline LatticeSFieldC HermitianColorForce(const LatticeSFieldC &G) {
  GridBase *grid = G.Grid();
  LatticeSFieldC F(grid); F = Zero();
  F.Checkerboard() = G.Checkerboard();
  autoView(Fv, F, AcceleratorWrite);
  autoView(Gv, G, AcceleratorRead);
  const int Nsimd = LatticeSFieldC::vector_object::Nsimd();
  accelerator_for(ss, grid->oSites(), Nsimd, {
    auto g = Gv(ss);
    typedef typename std::remove_cv<typename std::remove_reference<decltype(g)>::type>::type CSitePerLane;
    CSitePerLane f_acc;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        f_acc()()(i, j) = -(g()()(j, i) + conjugate(g()()(i, j)));
      }
    }
    coalescedWrite(Fv[ss], f_acc);
  });
  return F;
}

// Build the 6 spin-matrix entries of i*sigma_{mu,nu} as a 4x4 ComplexD array
// for use with ColorBilinearSpinOp. Returns Op[Ns][Ns].
inline std::array<std::array<ComplexD, Ns>, Ns>
ISigmaMatrix(int mu, int nu) {
  // Build by applying Gamma to identity columns. The factor i is included.
  std::array<std::array<ComplexD, Ns>, Ns> M{};
  Gamma g(SigmaMuNuAlgebra(mu, nu));
  for (int beta = 0; beta < Ns; ++beta) {
    SpinVector e; e = Zero(); e()(beta) = ComplexD(1.0, 0.0);
    SpinVector r = g * e;
    for (int alpha = 0; alpha < Ns; ++alpha) {
      ComplexD val(TensorRemove(r()(alpha)).real(),
                   TensorRemove(r()(alpha)).imag());
      M[alpha][beta] = ComplexD(0.0, 1.0) * val;
    }
  }
  return M;
}

// Same but for gamma5 (no factor of i).
inline std::array<std::array<ComplexD, Ns>, Ns> Gamma5Matrix() {
  std::array<std::array<ComplexD, Ns>, Ns> M{};
  Gamma g5(Gamma::Algebra::Gamma5);
  for (int beta = 0; beta < Ns; ++beta) {
    SpinVector e; e = Zero(); e()(beta) = ComplexD(1.0, 0.0);
    SpinVector r = g5 * e;
    for (int alpha = 0; alpha < Ns; ++alpha) {
      M[alpha][beta] = ComplexD(TensorRemove(r()(alpha)).real(),
                                TensorRemove(r()(alpha)).imag());
    }
  }
  return M;
}

inline std::array<std::array<ComplexD, Ns>, Ns> IdentitySpinMatrix() {
  std::array<std::array<ComplexD, Ns>, Ns> M{};
  for (int a = 0; a < Ns; ++a) M[a][a] = ComplexD(1.0, 0.0);
  return M;
}

// Wrap a ComplexD[Ns][Ns] table as a callable matching ColorBilinearSpinOp.
struct SpinTable {
  std::array<std::array<ComplexD, Ns>, Ns> M;
  ComplexD operator()(int a, int b) const { return M[a][b]; }
};

// -------------------- the action --------------------

class TXQCDWilsonPseudoFermionAction : public Action<TXQCDField> {
 public:
  TXQCDWilsonPseudoFermionAction(GridCartesian &grid,
                                 GridRedBlackCartesian &rbgrid, RealD mass,
                                 RealD cg_tol = 1e-12, int cg_maxiter = 10000)
      : grid_(grid), rbgrid_(rbgrid), mass_(mass),
        cg_tol_(cg_tol), cg_maxiter_(cg_maxiter), Phi(&grid) {}

  std::string action_name() override {
    return "TXQCDWilsonPseudoFermionAction";
  }
  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] mass=" << mass_
       << " cg_tol=" << cg_tol_ << " cg_maxiter=" << cg_maxiter_ << std::endl;
    return os.str();
  }

  void refresh(const TXQCDField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    TXQCDFermionNf chi(&grid_);
    const RealD scale = std::sqrt(0.5);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, chi.f[a]);
      chi.f[a] = scale * chi.f[a];
    }
    TXQCDWilsonOp Mop = MakeOp(U);
    Mop.Mdag(chi, Phi);
  }

  RealD S(const TXQCDField &U) override {
    TXQCDWilsonOp Mop = MakeOp(U);
    TXQCDFermionNf X(&grid_); X = Zero();
    SolveMdagM(Mop, Phi, X);
    TXQCDFermionNf Y(&grid_);
    Mop.M(X, Y);
    return norm2(Y);
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    TXQCDWilsonOp Mop = MakeOp(U);
    TXQCDFermionNf X(&grid_); X = Zero();
    SolveMdagM(Mop, Phi, X);
    TXQCDFermionNf Y(&grid_);
    Mop.M(X, Y);

    // ----- aux forces -----
    // sigma, pi via flavor bilinears.
    auto Gsig = FlavorBilinear(Y, X);
    dSdU.sigma = HermitianFlavorForce(Gsig);

    Gamma g5(Gamma::Algebra::Gamma5);
    TXQCDFermionNf g5X(&grid_);
    for (int a = 0; a < TxqcdNf; ++a) g5X.f[a] = g5 * X.f[a];
    auto Gpi = FlavorBilinear(Y, g5X);
    dSdU.pi = HermitianFlavorForce(Gpi);

    // s, p via color bilinears summed over spin/flavor with (1/sqrt 2) prefactor.
    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};
    auto rescale = [&](LatticeSFieldC &G) { G = inv_sqrt2 * G; };
    auto Gs = ColorBilinearSpinOp(Y, X, Id);
    rescale(Gs);
    dSdU.s = HermitianColorForce(Gs);
    auto Gp = ColorBilinearSpinOp(Y, X, G5);
    rescale(Gp);
    dSdU.p = HermitianColorForce(Gp);

    // t_{mu,nu}: per (mu<nu) piece is i*sigma_{mu,nu} on spin. Antisym in (mu,nu).
    dSdU.t = Zero();
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        SpinTable iSig{ISigmaMatrix(mu, nu)};
        auto Gt = ColorBilinearSpinOp(Y, X, iSig);
        auto Ft = HermitianColorForce(Gt);
        // Antisymmetrize: F_t[mu][nu] = +Ft, F_t[nu][mu] = -Ft.
        autoView(dst, dSdU.t, CpuWrite);
        autoView(src, Ft, CpuRead);
        thread_for(ss, grid_.oSites(), {
          for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
              dst[ss]()(mu, nu)(i, j) =  src[ss]()()(i, j);
              dst[ss]()(nu, mu)(i, j) = -src[ss]()()(i, j);
            }
          }
        });
      }
    }

    // ----- gauge force per flavor -----
    LatticeGaugeField gforce(&grid_); gforce = Zero();
    LatticeGaugeField tmp(&grid_);
    Mop.Wilson().ImportGauge(U.U);
    for (int a = 0; a < TxqcdNf; ++a) {
      Mop.Wilson().MDeriv(tmp, Y.f[a], X.f[a], DaggerNo);
      gforce = gforce + tmp;
      Mop.Wilson().MDeriv(tmp, X.f[a], Y.f[a], DaggerYes);
      gforce = gforce + tmp;
    }
    dSdU.U = gforce;
  }

  // Expose for tests.
  TXQCDFermionNf &PseudoFermion() { return Phi; }

 private:
  // Construct the operator for the current composite field. The aux fields
  // are referenced (not copied), so the operator is only valid while U lives.
  TXQCDWilsonOp MakeOp(const TXQCDField &U) {
    TXQCDField &Unc = const_cast<TXQCDField &>(U);
    return TXQCDWilsonOp(Unc.U, grid_, rbgrid_, mass_, Unc.sigma, Unc.pi,
                         Unc.s, Unc.p, Unc.t);
  }

  // Hand-rolled CG for MdagM x = b on TXQCDFermionNf. Stock Grid CG would
  // require a full LinearOperatorBase wrapper; this is simpler and adequate
  // for Phase-4 testing.
  void SolveMdagM(TXQCDWilsonOp &Mop, const TXQCDFermionNf &b,
                  TXQCDFermionNf &x) {
    TXQCDFermionNf r(&grid_), p(&grid_), Mp(&grid_), MdMp(&grid_);
    // r = b - MdagM x. Start x = 0 -> r = b.
    r = b;
    p = r;
    RealD rsq = norm2(r);
    RealD bsq = std::max(norm2(b), 1e-30);
    RealD tol2 = cg_tol_ * cg_tol_ * bsq;
    int it;
    for (it = 0; it < cg_maxiter_; ++it) {
      Mop.M(p, Mp);
      Mop.Mdag(Mp, MdMp);
      ComplexD pAp = innerProduct(p, MdMp);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      axpy(x,  alpha, p);
      axpy(r, -alpha, MdMp);
      RealD rsq_new = norm2(r);
      if (rsq_new < tol2) {
        rsq = rsq_new;
        break;
      }
      RealD beta = rsq_new / rsq;
      // p = r + beta * p
      for (int aa = 0; aa < TxqcdNf; ++aa) p.f[aa] = r.f[aa] + beta * p.f[aa];
      rsq = rsq_new;
    }
    std::cout << GridLogMessage << "[TXQCDPF CG] iter=" << it
              << " rsq=" << rsq << " tol2=" << tol2 << std::endl;
  }

  GridCartesian &grid_;
  GridRedBlackCartesian &rbgrid_;
  RealD mass_;
  RealD cg_tol_;
  int cg_maxiter_;
  TXQCDFermionNf Phi;
};

NAMESPACE_END(Grid);
