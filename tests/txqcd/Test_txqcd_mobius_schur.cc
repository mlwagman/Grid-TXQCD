// TXQCDMobiusSchurOp validation:
//   1. Mpc / MpcDag adjoint consistency: <w|Mpc v> == conj<v|MpcDag w>.
//   2. Full EO solve: solve M x = b via Schur reconstruction, verify
//      ||M x - b|| / ||b|| small (compared against the full TXQCDMobiusOp).
//
// EO solve (standard red-black):
//   src_o = b_o - Moe Mee^{-1} b_e
//   solve Mpc x_o = src_o      (CG on Mpc† Mpc)
//   x_e = Mee^{-1}(b_e - Meo x_o)

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusSchurOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt4(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG pRNG4(UGrid); pRNG4.SeedFixedIntegers({7, 8, 9, 10});
  GridParallelRNG pRNG5(FGrid); pRNG5.SeedFixedIntegers({11, 12, 13, 14});

  LatticeGaugeField Umu(UGrid);
  SU<Nc>::HotConfiguration(pRNG4, Umu);

  LatticeSigmaField sigma(UGrid); LatticePiField pi(UGrid);
  LatticeSFieldC s(UGrid);        LatticePFieldC p(UGrid);
  LatticeTField  t(UGrid);
  const RealD aux_scale = 0.05;
  HermitianGaussian(pRNG4, sigma); sigma = aux_scale * sigma;
  HermitianGaussian(pRNG4, pi);    pi    = aux_scale * pi;
  HermitianGaussian(pRNG4, s);     s     = aux_scale * s;
  HermitianGaussian(pRNG4, p);     p     = aux_scale * p;
  GaussianAntisymTensor(pRNG4, t); t     = aux_scale * t;

  RealD mass = 0.05, M5 = 1.8, b = 1.5, c = 0.5;

  TXQCDMobiusOp        Mop(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                            mass, M5, b, c, sigma, pi, s, p, t);
  TXQCDMobiusFermionEO Meo(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                            mass, M5, b, c, sigma, pi, s, p, t);
  TXQCDMobiusSchurOp   Schur(Meo);

  int exitcode = 0;

  // ---- Test 1: Mpc adjoint consistency ----
  {
    TXQCDFermionNf v(FrbGrid), w(FrbGrid), Mpv(FrbGrid), MdW(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      v.f[a].Checkerboard()   = Odd; w.f[a].Checkerboard()   = Odd;
      Mpv.f[a].Checkerboard() = Odd; MdW.f[a].Checkerboard() = Odd;
      LatticeFermion tmp(FGrid); gaussian(pRNG5, tmp);
      pickCheckerboard(Odd, v.f[a], tmp);
      gaussian(pRNG5, tmp);
      pickCheckerboard(Odd, w.f[a], tmp);
    }
    Schur.Mpc(v, Mpv);
    Schur.MpcDag(w, MdW);
    ComplexD lhs = innerProduct(w, Mpv);   // <w | Mpc v>
    ComplexD rhs = innerProduct(v, MdW);   // <v | MpcDag w>
    RealD rel = std::abs(lhs - std::conj(rhs)) / std::max(std::abs(lhs), 1e-30);
    bool pass = rel < 1e-9;
    std::cout << GridLogMessage << "Test 1 (Mpc adjoint): <w|Mpc v>=" << lhs
              << "  conj<v|MpcDag w>=" << std::conj(rhs)
              << "  rel=" << rel << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  // ---- Test 2: full EO solve reconstructs M^{-1} ----
  {
    // Random full-grid source b.
    TXQCDFermionNf bsrc(FGrid), xsol(FGrid), Mx(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG5, bsrc.f[a]);

    // Split b into checkerboards.
    TXQCDFermionNf b_e(FrbGrid), b_o(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      b_e.f[a].Checkerboard() = Even; b_o.f[a].Checkerboard() = Odd;
      pickCheckerboard(Even, b_e.f[a], bsrc.f[a]);
      pickCheckerboard(Odd,  b_o.f[a], bsrc.f[a]);
    }
    // src_o = b_o - Moe Mee^{-1} b_e
    TXQCDFermionNf t_e(FrbGrid), t_e2(FrbGrid), src_o(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      t_e.f[a].Checkerboard()  = Even; t_e2.f[a].Checkerboard() = Even;
      src_o.f[a].Checkerboard() = Odd;
    }
    Meo.MooeeInv(b_e, t_e2);    // Mee^{-1} b_e (even)
    Meo.Meooe(t_e2, src_o);     // Moe (even→odd)
    for (int a = 0; a < TxqcdNf; ++a) src_o.f[a] = b_o.f[a] - src_o.f[a];

    // Solve Mpc x_o = src_o via CG on Mpc† Mpc:  (Mpc†Mpc) x_o = Mpc† src_o
    TXQCDFermionNf rhs_o(FrbGrid), x_o(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      rhs_o.f[a].Checkerboard() = Odd;
      x_o.f[a].Checkerboard()   = Odd;
      x_o.f[a] = Zero();
    }
    Schur.MpcDag(src_o, rhs_o);

    // Hand-rolled CG on HermOp = Mpc† Mpc.
    TXQCDFermionNf r(FrbGrid), pp(FrbGrid), Ap(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      r.f[a].Checkerboard()  = Odd; pp.f[a].Checkerboard() = Odd;
      Ap.f[a].Checkerboard() = Odd;
    }
    r = rhs_o; pp = r;
    RealD rsq = norm2(r), bsq = std::max(norm2(rhs_o), 1e-30);
    RealD tol2 = 1e-18 * bsq;
    int it;
    for (it = 0; it < 20000; ++it) {
      if (rsq < tol2) break;
      Schur.HermOp(pp, Ap);
      ComplexD pAp = innerProduct(pp, Ap);
      ComplexD alpha = ComplexD(rsq, 0.0) / pAp;
      axpy(x_o,  alpha, pp);
      axpy(r,   -alpha, Ap);
      RealD rn = norm2(r);
      RealD beta = rn / rsq;
      for (int a = 0; a < TxqcdNf; ++a) pp.f[a] = r.f[a] + beta * pp.f[a];
      rsq = rn;
    }
    std::cout << GridLogMessage << "  [Schur CG] iter=" << it
              << " rsq=" << rsq << " tol2=" << tol2 << std::endl;

    // Reconstruct x_e = Mee^{-1}(b_e - Meo x_o)
    TXQCDFermionNf x_e(FrbGrid);
    for (int a = 0; a < TxqcdNf; ++a) x_e.f[a].Checkerboard() = Even;
    Meo.Meooe(x_o, t_e);                                  // Meo x_o (odd→even)
    for (int a = 0; a < TxqcdNf; ++a) t_e.f[a] = b_e.f[a] - t_e.f[a];
    Meo.MooeeInv(t_e, x_e);

    // Recombine x.
    for (int a = 0; a < TxqcdNf; ++a) {
      xsol.f[a].Checkerboard() = Even;
      setCheckerboard(xsol.f[a], x_e.f[a]);
      setCheckerboard(xsol.f[a], x_o.f[a]);
    }
    // Verify M x ≈ b.
    Mop.M(xsol, Mx);
    RealD resid = 0.0, nb = 0.0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(FGrid); d = Mx.f[a] - bsrc.f[a];
      resid += norm2(d);
      nb    += norm2(bsrc.f[a]);
    }
    RealD rel = std::sqrt(resid) / std::sqrt(std::max(nb, 1e-30));
    bool pass = rel < 1e-6;
    std::cout << GridLogMessage << "Test 2 (EO solve ||Mx-b||/||b||): "
              << rel << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
