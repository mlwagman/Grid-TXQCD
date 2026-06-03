// Standalone FD test for AuxiliaryFieldKineticAction, run from production/.
// Builds via the production Makefile catch-all (no autotools needed).
//
// Checks:
//   1. Zero-on-constant aux config — S=0, all derivs=0.
//   2. Central finite-difference vs <deriv,Y> along random Y for all 5 fields.
//   3. Field-by-field isolation: Z>0 on one field at a time, verify all other
//      derivs are exactly zero.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>

using namespace Grid;

template <class LatticeMat>
RealD HermInner(LatticeMat &A, LatticeMat &B) {
  return TensorRemove(sum(trace(A * B))).real();
}

RealD TensorInner(LatticeTField &A, LatticeTField &B) {
  // Σ_{x, μ<ν} Re Tr(A_{μν}·B_{μν}).  Σ_x done via Grid's bit-deterministic
  // sum() on a complex scalar field — earlier `thread_for { total += ... }`
  // had a data race that produced non-deterministic, fractionally wrong
  // partial sums.
  GridBase *grid = A.Grid();
  Lattice<iScalar<iScalar<iScalar<vComplex>>>> site_inner(grid);
  site_inner = Zero();
  {
    autoView(Av, A, CpuRead);
    autoView(Bv, B, CpuRead);
    autoView(sv, site_inner, CpuWrite);
    thread_for(ss, grid->oSites(), {
      vComplex acc; acc = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = mu + 1; nu < Nd; ++nu) {
          auto Am = Av[ss]()(mu, nu);
          auto Bm = Bv[ss]()(mu, nu);
          for (int i = 0; i < Nc; ++i)
            for (int j = 0; j < Nc; ++j)
              acc = acc + Am(i, j) * Bm(j, i);
        }
      }
      sv[ss]()()() = acc;
    });
  }
  ComplexD total = TensorRemove(sum(site_inner));
  return real(total);
}

RealD CompositeInner(TXQCDField &A, TXQCDField &B) {
  // Inner product matching the FD-vs-deriv convention used by the HMC
  // integrator on the antisymmetric tensor field.  The deriv code stores t's
  // gradient with an extra factor 2 (see AuxGaussianAction::deriv and the
  // matching factor in AuxKineticAction::deriv) to compensate for the
  // integrator's halved metric on t (FieldSquareNorm uses TFSN/2 = norm2/4).
  // Combined: TensorInner over μ<ν of the stored 2× gradient against Y
  // equals the per-element FD of S — no extra doubling needed.  σ/π/s/p
  // contribute via plain HermInner (sum over (a,b) storage).
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

  int exitcode = 0;
  auto check = [&](const char *name, RealD measured, RealD expected,
                   RealD tol) {
    RealD denom = std::max(std::abs(expected), 1.0);
    RealD rel = std::abs(measured - expected) / denom;
    std::cout << GridLogMessage << "[check] " << name
              << " : measured=" << measured << " expected=" << expected
              << " rel=" << rel << (rel < tol ? "  PASS" : "  FAIL")
              << std::endl;
    if (rel >= tol) exitcode = 1;
  };

  // ---------- 1. Zero on constant ----------
  {
    AuxiliaryFieldKineticAction Skin(0.7, 0.5, 0.4, 0.3, 0.2);
    TXQCDField U(&Grid), dS(&Grid);
    U.U = Zero();
    typename LatticeSigmaField::vector_object::scalar_object sig_c;
    sig_c = Zero();
    for (int a = 0; a < TxqcdNf; ++a) sig_c()()(a, a) = 0.123;
    U.sigma = sig_c;
    typename LatticePiField::vector_object::scalar_object pi_c;
    pi_c = Zero();
    for (int a = 0; a < TxqcdNf; ++a) pi_c()()(a, a) = 0.234;
    U.pi = pi_c;
    typename LatticeSFieldC::vector_object::scalar_object s_c;
    s_c = Zero();
    for (int i = 0; i < Nc; ++i) s_c()()(i, i) = 0.456;
    U.s = s_c;
    typename LatticePFieldC::vector_object::scalar_object p_c;
    p_c = Zero();
    for (int i = 0; i < Nc; ++i) p_c()()(i, i) = 0.567;
    U.p = p_c;
    U.t = Zero();

    check("S(const) == 0", Skin.S(U), 0.0, 1e-12);
    Skin.deriv(U, dS);
    check("|dS/dσ(const)|² == 0", norm2(dS.sigma), 0.0, 1e-12);
    check("|dS/dπ(const)|² == 0", norm2(dS.pi),    0.0, 1e-12);
    check("|dS/ds(const)|² == 0", norm2(dS.s),     0.0, 1e-12);
    check("|dS/dp(const)|² == 0", norm2(dS.p),     0.0, 1e-12);
    check("|dS/dt(const)|² == 0", norm2(dS.t),     0.0, 1e-12);
  }

  // ---------- 2. Central FD vs <deriv, Y> ----------
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
    check("FD vs <deriv,Y> all-5", num_deriv, ana_deriv, 1e-5);
  }

  // ---------- 2b. Per-field FD (localize any discrepancy) ----------
  {
    // Re-seed so this block doesn't depend on previous random consumption.
    pRNG.SeedFixedIntegers({20, 21, 22, 23});
    TXQCDField U(&Grid), Y(&Grid), Up(&Grid), Um(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    TXQCDCompositeImpl::HotConfiguration(pRNG, Y);
    U.U = Zero();
    Y.U = Zero();
    RealD h = 1e-3;

    auto field_fd = [&](const char *name, RealD Zs, RealD Zpi, RealD Zss,
                        RealD Zp, RealD Zt) {
      AuxiliaryFieldKineticAction Skin(Zs, Zpi, Zss, Zp, Zt);
      Up = U; Up.sigma = U.sigma + h*Y.sigma;  Up.pi = U.pi + h*Y.pi;
      Up.s = U.s + h*Y.s;  Up.p = U.p + h*Y.p;  Up.t = U.t + h*Y.t;
      Um = U; Um.sigma = U.sigma - h*Y.sigma;  Um.pi = U.pi - h*Y.pi;
      Um.s = U.s - h*Y.s;  Um.p = U.p - h*Y.p;  Um.t = U.t - h*Y.t;
      RealD num = (Skin.S(Up) - Skin.S(Um)) / (2.0 * h);
      Skin.deriv(U, dS);
      RealD ana = CompositeInner(dS, Y);
      check(name, num, ana, 1e-5);
    };
    field_fd("FD σ-only",  1.7, 0.0, 0.0, 0.0, 0.0);
    field_fd("FD π-only",  0.0, 1.3, 0.0, 0.0, 0.0);
    field_fd("FD s-only",  0.0, 0.0, 0.9, 0.0, 0.0);
    field_fd("FD p-only",  0.0, 0.0, 0.0, 1.1, 0.0);
    field_fd("FD t-only",  0.0, 0.0, 0.0, 0.0, 0.6);
  }

  // ---------- 3. Field-by-field isolation ----------
  {
    TXQCDField U(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    U.U = Zero();

    auto field_only = [&](const char *name, RealD Zs, RealD Zpi, RealD Zss,
                          RealD Zp, RealD Zt) {
      AuxiliaryFieldKineticAction Skin(Zs, Zpi, Zss, Zp, Zt);
      RealD S = Skin.S(U);
      Skin.deriv(U, dS);
      std::cout << GridLogMessage << "[iso] " << name
                << "  S=" << S
                << "  |dσ|²=" << norm2(dS.sigma)
                << "  |dπ|²=" << norm2(dS.pi)
                << "  |ds|²=" << norm2(dS.s)
                << "  |dp|²=" << norm2(dS.p)
                << "  |dt|²=" << norm2(dS.t) << std::endl;
      auto need_zero = [&](const char *what, bool z, RealD v) {
        if (z && v > 1e-20) {
          std::cout << GridLogMessage << "[iso FAIL] " << name << ": "
                    << what << "=" << v << std::endl;
          exitcode = 1;
        }
      };
      need_zero("|dσ|²", Zs == 0.0, norm2(dS.sigma));
      need_zero("|dπ|²", Zpi == 0.0, norm2(dS.pi));
      need_zero("|ds|²", Zss == 0.0, norm2(dS.s));
      need_zero("|dp|²", Zp == 0.0, norm2(dS.p));
      need_zero("|dt|²", Zt == 0.0, norm2(dS.t));
    };
    field_only("σ only", 1.0, 0.0, 0.0, 0.0, 0.0);
    field_only("π only", 0.0, 1.0, 0.0, 0.0, 0.0);
    field_only("s only", 0.0, 0.0, 1.0, 0.0, 0.0);
    field_only("p only", 0.0, 0.0, 0.0, 1.0, 0.0);
    field_only("t only", 0.0, 0.0, 0.0, 0.0, 1.0);
  }

  if (exitcode == 0) {
    std::cout << GridLogMessage << "ALL CHECKS PASSED" << std::endl;
  } else {
    std::cout << GridLogMessage << "SOME CHECKS FAILED" << std::endl;
  }
  Grid_finalize();
  return exitcode;
}
