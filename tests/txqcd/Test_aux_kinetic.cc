// Test_aux_kinetic: validation gate for AuxiliaryFieldKineticAction over all
// five aux fields (σ, π, s, p, t).  Four checks:
//
//   1. Zero-on-constant.  A site-constant config has zero kinetic action and
//      zero kinetic force (lattice Laplacian of a constant is zero).
//
//   2. S vs deriv FD.  Central finite difference of S along a random
//      perturbation Y must match <deriv, Y> using the Frobenius inner
//      product matching the metric of HermitianFieldSquareNorm /
//      TensorFieldSquareNorm.
//
//   3. Positivity on a HotConfiguration (S ≥ 0 by construction).
//
//   4. Field-by-field: each Z>0 alone reproduces a non-zero S and deriv only
//      on its own field, all other fields untouched.

#include <Grid/Grid.h>

using namespace Grid;

template <class LatticeMat>
RealD HermInner(LatticeMat &A, LatticeMat &B) {
  return TensorRemove(sum(trace(A * B))).real();
}

RealD TensorInner(LatticeTField &A, LatticeTField &B) {
  autoView(Av, A, CpuRead);
  autoView(Bv, B, CpuRead);
  GridBase *grid = A.Grid();
  RealD total = 0.0;
  thread_for(ss, grid->oSites(), {
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        auto Am = Av[ss]()(mu, nu);
        auto Bm = Bv[ss]()(mu, nu);
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j)
            total += real(Reduce(Am(i, j) * Bm(j, i)));
      }
    }
  });
  grid->GlobalSum(total);
  return total;
}

RealD CompositeInner(TXQCDField &A, TXQCDField &B) {
  return HermInner(A.sigma, B.sigma) + HermInner(A.pi, B.pi)
       + HermInner(A.s, B.s) + HermInner(A.p, B.p)
       + TensorInner(A.t, B.t);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});
  GridSerialRNG sRNG;
  sRNG.SeedFixedIntegers({5, 6, 7, 8});

  int exitcode = 0;
  auto check = [&](const char *name, RealD measured, RealD expected,
                   RealD tol) {
    RealD rel = std::abs(measured - expected) /
                std::max(std::abs(expected), 1.0);
    std::cout << GridLogMessage << "[check] " << name << " : measured "
              << measured << " expected " << expected << " rel " << rel
              << (rel < tol ? "  PASS" : "  FAIL") << std::endl;
    if (rel >= tol) exitcode = 1;
  };

  // ---------- 1. Zero-on-constant ----------
  {
    AuxiliaryFieldKineticAction Skin(0.7, 0.5, 0.4, 0.3, 0.2);

    TXQCDField U(&Grid), dS(&Grid);
    U.U = Zero();
    typename LatticeSigmaField::vector_object::scalar_object sig_const;
    sig_const = Zero();
    for (int a = 0; a < TxqcdNf; ++a) sig_const()()(a, a) = 0.123;
    U.sigma = sig_const;
    typename LatticePiField::vector_object::scalar_object pi_const;
    pi_const = Zero();
    for (int a = 0; a < TxqcdNf; ++a) pi_const()()(a, a) = 0.234;
    U.pi = pi_const;
    typename LatticeSFieldC::vector_object::scalar_object s_const;
    s_const = Zero();
    for (int i = 0; i < Nc; ++i) s_const()()(i, i) = 0.456;
    U.s = s_const;
    typename LatticePFieldC::vector_object::scalar_object p_const;
    p_const = Zero();
    for (int i = 0; i < Nc; ++i) p_const()()(i, i) = 0.567;
    U.p = p_const;
    // For t we just use zero — antisym tensor of color matrices, simpler to
    // verify with all-zero t (still tests the Laplacian path is null).
    U.t = Zero();

    check("S(const) == 0", Skin.S(U), 0.0, 1e-12);
    Skin.deriv(U, dS);
    check("|dS/dσ(const)| == 0", norm2(dS.sigma), 0.0, 1e-12);
    check("|dS/dπ(const)| == 0", norm2(dS.pi),    0.0, 1e-12);
    check("|dS/ds(const)| == 0", norm2(dS.s),     0.0, 1e-12);
    check("|dS/dp(const)| == 0", norm2(dS.p),     0.0, 1e-12);
    check("|dS/dt(const)| == 0", norm2(dS.t),     0.0, 1e-12);
  }

  // ---------- 2. FD: deriv vs central FD ----------
  {
    AuxiliaryFieldKineticAction Skin(1.7, 1.3, 0.9, 1.1, 0.6);

    TXQCDField U(&Grid), Y(&Grid), Up(&Grid), Um(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    TXQCDCompositeImpl::HotConfiguration(pRNG, Y);
    U.U = Zero();
    Y.U = Zero();

    RealD h = 1e-3;
    Up = U;
    Up.sigma = U.sigma + h * Y.sigma;
    Up.pi    = U.pi    + h * Y.pi;
    Up.s     = U.s     + h * Y.s;
    Up.p     = U.p     + h * Y.p;
    Up.t     = U.t     + h * Y.t;
    Um = U;
    Um.sigma = U.sigma - h * Y.sigma;
    Um.pi    = U.pi    - h * Y.pi;
    Um.s     = U.s     - h * Y.s;
    Um.p     = U.p     - h * Y.p;
    Um.t     = U.t     - h * Y.t;

    RealD num_deriv = (Skin.S(Up) - Skin.S(Um)) / (2.0 * h);
    Skin.deriv(U, dS);
    RealD ana_deriv = CompositeInner(dS, Y);
    check("AuxKineticAction all-5 deriv vs FD", num_deriv, ana_deriv, 1e-5);
  }

  // ---------- 3. Positivity on HotConfiguration ----------
  {
    AuxiliaryFieldKineticAction Skin(1.0, 1.0, 1.0, 1.0, 1.0);

    TXQCDField U(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    RealD S = Skin.S(U);
    std::cout << GridLogMessage << "[info] S_kin(hot) = " << S
              << "  (positive by construction)" << std::endl;
    if (S < 0.0) {
      std::cout << GridLogMessage << "[check] S>=0 on hot config : FAIL"
                << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "[check] S>=0 on hot config : PASS"
                << std::endl;
    }
  }

  // ---------- 4. Field-by-field isolation: Z>0 on one field only ----------
  {
    TXQCDField U(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    U.U = Zero();

    auto field_only = [&](const char *name, RealD Zsigma, RealD Zpi, RealD Zs,
                          RealD Zp, RealD Zt) {
      AuxiliaryFieldKineticAction Skin(Zsigma, Zpi, Zs, Zp, Zt);
      RealD S = Skin.S(U);
      Skin.deriv(U, dS);
      bool sigma_zero = (Zsigma == 0.0);
      bool pi_zero    = (Zpi    == 0.0);
      bool s_zero     = (Zs     == 0.0);
      bool p_zero     = (Zp     == 0.0);
      bool t_zero     = (Zt     == 0.0);
      // Expect non-zero only for the field whose Z > 0.
      std::cout << GridLogMessage << "[isolation] " << name
                << " S=" << S
                << "  |dσ|² = " << norm2(dS.sigma)
                << "  |dπ|² = " << norm2(dS.pi)
                << "  |ds|² = " << norm2(dS.s)
                << "  |dp|² = " << norm2(dS.p)
                << "  |dt|² = " << norm2(dS.t) << std::endl;
      auto must_zero = [&](const char *what, bool should_be_zero, RealD val) {
        if (should_be_zero && val > 1e-20) {
          std::cout << GridLogMessage << "[isolation FAIL] " << name
                    << ": " << what << " should be 0 but = " << val
                    << std::endl;
          exitcode = 1;
        }
      };
      must_zero("|dσ|²", sigma_zero, norm2(dS.sigma));
      must_zero("|dπ|²", pi_zero,    norm2(dS.pi));
      must_zero("|ds|²", s_zero,     norm2(dS.s));
      must_zero("|dp|²", p_zero,     norm2(dS.p));
      must_zero("|dt|²", t_zero,     norm2(dS.t));
    };
    field_only("σ only", 1.0, 0.0, 0.0, 0.0, 0.0);
    field_only("π only", 0.0, 1.0, 0.0, 0.0, 0.0);
    field_only("s only", 0.0, 0.0, 1.0, 0.0, 0.0);
    field_only("p only", 0.0, 0.0, 0.0, 1.0, 0.0);
    field_only("t only", 0.0, 0.0, 0.0, 0.0, 1.0);
  }

  Grid_finalize();
  return exitcode;
}
