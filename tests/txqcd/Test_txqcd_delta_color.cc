// Hermiticity test for the full Delta-operator (sigma + pi + s + p + t).
//
// Each of the five pieces is individually Hermitian:
//   - sigma_{a,b}, pi_{a,b} are Hermitian in flavor (pi carries gamma5, which
//     is itself Hermitian and commutes structurally with color),
//   - s, p are Hermitian in color (p carries gamma5),
//   - t_{mu,nu} is Hermitian in color and antisymmetric in (mu,nu); combined
//     with sigma_{mu,nu} (also antisymmetric in (mu,nu) and Hermitian in
//     spin), the product in each pair is Hermitian.
// So the full Delta is Hermitian and the composite inner product should
// satisfy <w, Delta v> = conj(<v, Delta w>) to FP precision.
//
// Also verifies that the individual pieces sum correctly: ApplyDelta returns
// the same result as ApplyDeltaSigmaPi + ApplyDeltaColor.

#include <Grid/Grid.h>

using namespace Grid;

static ComplexD CompositeInner(const TXQCDFermionNf &x,
                               const TXQCDFermionNf &y) {
  ComplexD acc = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) acc += innerProduct(x.f[a], y.f[a]);
  return acc;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({11, 22, 33, 44});

  LatticeSigmaField sigma(&Grid);
  LatticePiField pi(&Grid);
  LatticeSFieldC s(&Grid);
  LatticePFieldC p(&Grid);
  LatticeTField t(&Grid);
  HermitianGaussian(pRNG, sigma);
  HermitianGaussian(pRNG, pi);
  HermitianGaussian(pRNG, s);
  HermitianGaussian(pRNG, p);
  GaussianAntisymTensor(pRNG, t);

  TXQCDFermionNf v(&Grid), w(&Grid);
  for (int a = 0; a < TxqcdNf; ++a) {
    gaussian(pRNG, v.f[a]);
    gaussian(pRNG, w.f[a]);
  }

  int exitcode = 0;
  auto hermcheck = [&](const char *name, const TXQCDFermionNf &Dv,
                       const TXQCDFermionNf &Dw) {
    ComplexD lhs = CompositeInner(w, Dv);
    ComplexD rhs = conjugate(CompositeInner(v, Dw));
    RealD err = std::abs(lhs - rhs) / std::max(std::abs(lhs), 1.0);
    std::cout << GridLogMessage << "[" << name << "] <w,Dv>=" << lhs
              << " <v,Dw>*=" << rhs << " rel=" << err
              << (err < 1e-12 ? "  PASS" : "  FAIL") << std::endl;
    if (err >= 1e-12) exitcode = 1;
  };

  // Sigma/pi sector alone (sanity against previous test).
  {
    TXQCDFermionNf Dv(&Grid), Dw(&Grid);
    ApplyDeltaSigmaPi(sigma, pi, v, Dv);
    ApplyDeltaSigmaPi(sigma, pi, w, Dw);
    hermcheck("sigma+pi", Dv, Dw);
  }

  // s alone: zero out p, t.
  {
    LatticePFieldC p0(&Grid); p0 = Zero();
    LatticeTField t0(&Grid); t0 = Zero();
    TXQCDFermionNf Dv(&Grid), Dw(&Grid);
    ApplyDeltaColor(s, p0, t0, v, Dv);
    ApplyDeltaColor(s, p0, t0, w, Dw);
    hermcheck("s only", Dv, Dw);
  }

  // p alone.
  {
    LatticeSFieldC s0(&Grid); s0 = Zero();
    LatticeTField t0(&Grid); t0 = Zero();
    TXQCDFermionNf Dv(&Grid), Dw(&Grid);
    ApplyDeltaColor(s0, p, t0, v, Dv);
    ApplyDeltaColor(s0, p, t0, w, Dw);
    hermcheck("p only", Dv, Dw);
  }

  // t alone.
  {
    LatticeSFieldC s0(&Grid); s0 = Zero();
    LatticePFieldC p0(&Grid); p0 = Zero();
    TXQCDFermionNf Dv(&Grid), Dw(&Grid);
    ApplyDeltaColor(s0, p0, t, v, Dv);
    ApplyDeltaColor(s0, p0, t, w, Dw);
    hermcheck("t only", Dv, Dw);
  }

  // Full Delta.
  {
    TXQCDFermionNf Dv(&Grid), Dw(&Grid);
    ApplyDelta(sigma, pi, s, p, t, v, Dv);
    ApplyDelta(sigma, pi, s, p, t, w, Dw);
    hermcheck("full Delta", Dv, Dw);
  }

  // Consistency: full == sigma/pi + color piece.
  {
    TXQCDFermionNf Dfull(&Grid), Dsp(&Grid), Dcol(&Grid);
    ApplyDelta(sigma, pi, s, p, t, v, Dfull);
    ApplyDeltaSigmaPi(sigma, pi, v, Dsp);
    ApplyDeltaColor(s, p, t, v, Dcol);
    RealD resid = 0.0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion diff(&Grid);
      diff = Dfull.f[a] - (Dsp.f[a] + Dcol.f[a]);
      resid += norm2(diff);
    }
    resid = std::sqrt(resid);
    std::cout << GridLogMessage << "[decomposition] |full - (sp+col)| = "
              << resid << (resid < 1e-12 ? "  PASS" : "  FAIL") << std::endl;
    if (resid >= 1e-12) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
