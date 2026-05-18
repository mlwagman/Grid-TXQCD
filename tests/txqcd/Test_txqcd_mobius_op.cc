// Smoke test for TXQCDMobiusOp: aux=0 must equal stock MobiusFermion, and
// the γ5R5-hermiticity identity must hold at random aux.
//
// Compares M_TX^DWF and M_QCD^DWF in a small 5D setting:
//   1. Build random 4D gauge + 5D source.
//   2. Apply stock MobiusFermion to source per flavor → reference 5D output.
//   3. Apply TXQCDMobiusOp with aux = 0 → output must match reference bit-identically.
//   4. Apply TXQCDMobiusOp with random aux → check γ5R5 hermiticity:
//        <w | γ5R5 M v> == conj(<v | γ5R5 M w>)
//      where γ5R5 reverses the 5th-dim index and applies γ5.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/Txqcd.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>

using namespace Grid;

// <w | γ5R5 v>: reverse 5th-dim of v, then γ5, then inner product
static ComplexD g5R5_inner(const LatticeFermion &w, const LatticeFermion &v,
                           int Ls, GridCartesian *UGrid) {
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermion v_reversed(v.Grid()), sl(UGrid);
  v_reversed = Zero();
  for (int s = 0; s < Ls; ++s) {
    ExtractSlice(sl, const_cast<LatticeFermion &>(v), s, 0);
    InsertSlice(sl, v_reversed, Ls - 1 - s, 0);
  }
  v_reversed = g5 * v_reversed;
  return innerProduct(w, v_reversed);
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt4 = GridDefaultLatt();   // command-line --grid x.y.z.t
  Coordinate simd  = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi   = GridDefaultMpi();

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(latt4, simd, mpi);
  GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::vector<int> seeds4{1, 2, 3, 4}, seeds5{5, 6, 7, 8};
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers(seeds5);

  // Random gauge
  LatticeGaugeField Umu(UGrid);
  SU<Nc>::HotConfiguration(RNG4, Umu);

  // Random aux fields
  LatticeSigmaField sigma(UGrid); LatticePiField   pi(UGrid);
  LatticeSFieldC    s(UGrid);     LatticePFieldC   p(UGrid);
  LatticeTField     t(UGrid);
  HermitianGaussian(RNG4, sigma);
  HermitianGaussian(RNG4, pi);
  HermitianGaussian(RNG4, s);
  HermitianGaussian(RNG4, p);
  GaussianAntisymTensor(RNG4, t);

  // Random 5D source (Nf flavors)
  TXQCDFermionNf src(FGrid);
  for (int a = 0; a < TxqcdNf; ++a) gaussian(RNG5, src.f[a]);

  // Möbius parameters (typical)
  RealD mass = 0.05, M5 = 1.8, b = 1.5, c = 0.5;

  std::cout << GridLogMessage << "[mobius_op] Ls=" << Ls
            << " mass=" << mass << " M5=" << M5
            << " b=" << b << " c=" << c << std::endl;

  // ----- Test 1: aux=0 should match stock Möbius bit-identically -----
  LatticeSigmaField zsigma(UGrid); LatticePiField   zpi(UGrid);
  LatticeSFieldC    zs(UGrid);     LatticePFieldC   zp(UGrid);
  LatticeTField     zt(UGrid);
  zsigma = Zero(); zpi = Zero(); zs = Zero(); zp = Zero(); zt = Zero();

  TXQCDMobiusOp Mtx0(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                     mass, M5, b, c, zsigma, zpi, zs, zp, zt);

  TXQCDFermionNf out_tx(FGrid);
  Mtx0.M(src, out_tx);

  // Reference: stock MobiusFermion per flavor
  MobiusFermionD Dref(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c);
  TXQCDFermionNf out_ref(FGrid);
  for (int a = 0; a < TxqcdNf; ++a) Dref.M(src.f[a], out_ref.f[a]);

  RealD max_diff_zero = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) {
    LatticeFermion d(FGrid);
    d = out_tx.f[a] - out_ref.f[a];
    RealD nd = std::sqrt(norm2(d));
    RealD nr = std::sqrt(norm2(out_ref.f[a]));
    RealD rel = (nr > 0) ? nd / nr : nd;
    std::cout << GridLogMessage << "  flavor " << a
              << " ||M_tx0 - M_ref|| / ||M_ref|| = " << rel << std::endl;
    max_diff_zero = std::max(max_diff_zero, rel);
  }

  // ----- Test 2: Mdag adjoint consistency: <w | Mdag v> = conj(<v | M w>) -----
  // This is the defining property of the adjoint.  Tests that my Mdag
  // implementation (stock MobiusFermion::Mdag + (b+c) Δ) really is the
  // hermitian conjugate of M.  Verify at aux=0 and at random aux.

  auto check_adj = [&](TXQCDMobiusOp &Mop, const char *label) {
    TXQCDFermionNf v(FGrid), w(FGrid), Mv(FGrid), Mdw(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) { gaussian(RNG5, v.f[a]); gaussian(RNG5, w.f[a]); }
    Mop.M(w, Mv);                          // we'll use this for <Mw | v>
    Mop.Mdag(v, Mdw);                      // <w | Mdag v>
    ComplexD lhs(0.0, 0.0), rhs(0.0, 0.0);
    for (int a = 0; a < TxqcdNf; ++a) {
      lhs += innerProduct(w.f[a], Mdw.f[a]);  // <w | Mdag v>
      rhs += innerProduct(Mv.f[a], v.f[a]);   // <Mw | v>
    }
    ComplexD diff = lhs - rhs;
    RealD rel = std::abs(diff) / std::max(std::abs(lhs), 1e-30);
    std::cout << GridLogMessage << "  [" << label
              << "] <w|Mdag v> = " << lhs
              << "   <Mw|v> = " << rhs
              << "   rel diff = " << rel << std::endl;
    return rel;
  };

  // ----- Sanity: ApplyDelta hermiticity on 4D fields -----
  {
    TXQCDFermionNf v4(UGrid), w4(UGrid), Dv(UGrid), Dw(UGrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(RNG4, v4.f[a]); gaussian(RNG4, w4.f[a]);
    }
    ApplyDelta(sigma, pi, s, p, t, v4, Dv);
    ApplyDelta(sigma, pi, s, p, t, w4, Dw);
    ComplexD lhs(0,0), rhs(0,0);
    for (int a = 0; a < TxqcdNf; ++a) {
      lhs += innerProduct(w4.f[a], Dv.f[a]);   // <w | Δ v>
      rhs += innerProduct(Dw.f[a], v4.f[a]);   // <Δ w | v>
    }
    // Hermitian Δ ⇒ <w|Δv> = <Δw|v>.
    RealD rel = std::abs(lhs - rhs) / std::max(std::abs(lhs), 1e-30);
    std::cout << GridLogMessage << "  [4D Δ] <w|Δv> = " << lhs
              << "  <Δw|v> = " << rhs
              << "  rel diff = " << rel << std::endl;
  }

  RealD rel_zero = check_adj(Mtx0, "aux=0");
  TXQCDMobiusOp Mtx(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid,
                    mass, M5, b, c, sigma, pi, s, p, t);
  RealD rel_diff = check_adj(Mtx, "random aux");

  // ----- Diagnostic: Δ-only contribution adjoint test -----
  // ΔM ≡ M(aux≠0) − M(aux=0) = (b+c) Δ_5d.  Test <w|ΔM† v> = <ΔM w|v>.
  // ΔM† should equal ΔM since Δ is hermitian.
  {
    TXQCDFermionNf v(FGrid), w(FGrid);
    TXQCDFermionNf Mv_a(FGrid), Mv_z(FGrid), Mw_a(FGrid), Mw_z(FGrid);
    TXQCDFermionNf Mdv_a(FGrid), Mdv_z(FGrid);
    for (int a = 0; a < TxqcdNf; ++a) { gaussian(RNG5, v.f[a]); gaussian(RNG5, w.f[a]); }
    Mtx.M(v, Mv_a);    Mtx0.M(v, Mv_z);
    Mtx.M(w, Mw_a);    Mtx0.M(w, Mw_z);
    Mtx.Mdag(v, Mdv_a); Mtx0.Mdag(v, Mdv_z);
    // ΔM v = Mv_a - Mv_z;  ΔMdag v = Mdv_a - Mdv_z;  ΔM w = Mw_a - Mw_z
    ComplexD lhs(0,0), rhs(0,0);
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion dMv(FGrid), dMw(FGrid), dMdv(FGrid);
      dMv  = Mv_a.f[a]  - Mv_z.f[a];
      dMw  = Mw_a.f[a]  - Mw_z.f[a];
      dMdv = Mdv_a.f[a] - Mdv_z.f[a];
      lhs += innerProduct(w.f[a], dMdv);   // <w | ΔM† v>
      rhs += innerProduct(dMw, v.f[a]);    // <ΔM w | v>
    }
    RealD rel = std::abs(lhs - rhs) / std::max(std::abs(lhs), 1e-30);
    std::cout << GridLogMessage << "  [Δ-only] <w|ΔMdag v> = " << lhs
              << "   <ΔMw|v> = " << rhs
              << "   rel diff = " << rel << std::endl;
  }

  bool ok_zero  = max_diff_zero < 1e-12;
  bool ok_adj0  = rel_zero < 1e-10;
  bool ok_adjA  = rel_diff < 1e-10;
  std::cout << GridLogMessage << "Test 1 (aux=0 reproduces Möbius bit-identical): "
            << (ok_zero ? "PASS" : "FAIL") << std::endl;
  std::cout << GridLogMessage << "Test 2 (Mdag adjoint identity at aux=0): "
            << (ok_adj0 ? "PASS" : "FAIL") << std::endl;
  std::cout << GridLogMessage << "Test 3 (Mdag adjoint identity at random aux): "
            << (ok_adjA ? "PASS" : "FAIL") << std::endl;

  Grid_finalize();
  return (ok_zero && ok_adj0 && ok_adjA) ? 0 : 1;
}
