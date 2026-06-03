// FD test for AuxFierzShift + FierzShiftedAction.
//
// Tests:
//   1. Z=0 wrapped == unwrapped (bit-exact for AuxiliaryFieldGaussianAction).
//   2. Z>0 FD == ⟨wrapped.deriv, Y⟩ for AuxiliaryFieldGaussianAction
//      (generic chain-rule validation; underlying action analytical).
//   3. Z>0 sanity: uniform Y → Lap(Y) = 0 → wrap.deriv == raw.deriv.
//   4. Z>0 FD == ⟨wrapped.deriv, Y⟩ for TXQCDLogDetCloverEOAction
//      (validation against a real σ-in-Dirac-op fermion action).

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>

using namespace Grid;

template <class LatticeMat>
RealD HermInner(LatticeMat &A, LatticeMat &B) {
  return TensorRemove(sum(trace(A * B))).real();
}

// Same μ<ν deterministic TensorInner from the kinetic FD test (race-fixed).
RealD TensorInner(LatticeTField &A, LatticeTField &B) {
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
  return real(TensorRemove(sum(site_inner)));
}

RealD CompositeInner(TXQCDField &A, TXQCDField &B) {
  return HermInner(A.sigma, B.sigma) + HermInner(A.pi, B.pi)
       + HermInner(A.s, B.s) + HermInner(A.p, B.p)
       + TensorInner(A.t, B.t);   // matches AuxGaussian factor-2 t convention
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({30, 31, 32, 33});

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

  const RealD lambda = 7.0;
  const RealD Z      = 10.0;

  AuxiliaryFieldGaussianAction Aux(lambda);

  // ---------- 1. Z=0 wrapped == unwrapped ----------
  {
    AuxFierzShift shift_off(lambda, 0, 0, 0, 0, 0);
    FierzShiftedAction wrapped(Aux, shift_off);

    TXQCDField U(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    U.U = Zero();

    RealD S_raw  = Aux.S(U);
    RealD S_wrap = wrapped.S(U);
    check("Z=0: S(wrapped) == S(raw)", S_wrap, S_raw, 1e-13);

    TXQCDField dS_raw(&Grid), dS_wrap(&Grid);
    Aux.deriv(U, dS_raw);
    wrapped.deriv(U, dS_wrap);
    check("Z=0: norm2(dS_wrap.σ - dS_raw.σ)",
          norm2(dS_wrap.sigma - dS_raw.sigma), 0.0, 1e-13);
    check("Z=0: norm2(dS_wrap.t - dS_raw.t)",
          norm2(dS_wrap.t - dS_raw.t), 0.0, 1e-13);
  }

  // ---------- 2. Z>0 FD vs <deriv, Y> ----------
  {
    AuxFierzShift shift_on(lambda, Z, Z, Z, Z, Z);
    FierzShiftedAction wrapped(Aux, shift_on);

    TXQCDField U(&Grid), Y(&Grid), Up(&Grid), Um(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    TXQCDCompositeImpl::HotConfiguration(pRNG, Y);
    U.U = Zero();
    Y.U = Zero();

    const RealD h = 1e-3;
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

    RealD num = (wrapped.S(Up) - wrapped.S(Um)) / (2.0 * h);
    wrapped.deriv(U, dS);
    RealD ana = CompositeInner(dS, Y);

    check("Z>0: FD == <wrapped.deriv, Y>", num, ana, 1e-7);
  }

  // ---------- 3. Z>0 sanity: uniform Y → Lap Y = 0 → shift inert ----------
  {
    AuxFierzShift shift_on(lambda, Z, Z, Z, Z, Z);
    FierzShiftedAction wrapped(Aux, shift_on);

    TXQCDField U(&Grid), Y(&Grid), dS_wrap(&Grid), dS_raw(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    U.U = Zero();
    // Y constant per site, σ_aa diagonal only.
    Y.U = Zero();
    typename LatticeSigmaField::vector_object::scalar_object yconst;
    yconst = Zero();
    for (int a = 0; a < TxqcdNf; ++a) yconst()()(a, a) = 0.3;
    Y.sigma = yconst;
    Y.pi    = Zero();
    Y.s     = Zero();
    Y.p     = Zero();
    Y.t     = Zero();

    wrapped.deriv(U, dS_wrap);
    Aux.deriv(U, dS_raw);
    // ⟨wrapped.deriv, uniform Y⟩ should equal ⟨raw.deriv, uniform Y⟩,
    // because the shift on Y vanishes (Lap of uniform = 0) AND the shift on
    // U's σ contributes the same Lap term to both sides through symmetry.
    // Strict equality only holds for the constant-Y direction.
    RealD inner_wrap = HermInner(dS_wrap.sigma, Y.sigma);
    RealD inner_raw  = HermInner(dS_raw.sigma,  Y.sigma);
    check("Z>0 uniform-Y: <wrap.dσ, Yconst> == <raw.dσ, Yconst>",
          inner_wrap, inner_raw, 1e-9);
  }

  // ---------- 4. Z>0 FD on TXQCDLogDetCloverEOAction (real fermion action) ----
  {
    GridRedBlackCartesian RBGrid(&Grid);
    const RealD mass = -0.245;
    const RealD csw  = 1.24930970916466;
    TXQCDLogDetCloverEOAction LogDet(Grid, RBGrid, mass, csw);
    AuxFierzShift shift_on(lambda, Z, Z, Z, Z, Z);
    FierzShiftedAction wrapped(LogDet, shift_on);

    TXQCDField U(&Grid), Y(&Grid), Up(&Grid), Um(&Grid), dS(&Grid);
    TXQCDCompositeImpl::HotConfiguration(pRNG, U);
    TXQCDCompositeImpl::HotConfiguration(pRNG, Y);
    // Y has no gauge component — only probe σ direction.
    Y.U = Zero();

    const RealD h = 1e-6;  // tighter h: LogDet S is cubic+ in σ
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

    RealD num = (wrapped.S(Up) - wrapped.S(Um)) / (2.0 * h);
    wrapped.deriv(U, dS);
    RealD ana = CompositeInner(dS, Y);

    check("Z>0: FD == <wrapped.deriv, Y>  [LogDet]", num, ana, 1e-4);
  }

  if (exitcode == 0) {
    std::cout << GridLogMessage << "ALL CHECKS PASSED" << std::endl;
  } else {
    std::cout << GridLogMessage << "SOME CHECKS FAILED" << std::endl;
  }
  Grid_finalize();
  return exitcode;
}
