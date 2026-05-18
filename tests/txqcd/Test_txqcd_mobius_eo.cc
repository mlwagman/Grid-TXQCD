// TXQCDMobiusFermionEO correctness tests:
//   1. EO decomposition: full M(v) == Mooee + Meooe reconstruction.
//   2. MooeeInv refinement converges: MooeeInv(Mooee(x)) ≈ x.
//   3. M_full agrees with TXQCDMobiusOp (already validated).

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusOp.h>
#include <Grid/qcd/action/txqcd/TXQCDMobiusFermionEO.h>

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
  // aux scale via env.  Default 0.05 (real production aux norm is in this
  // range; <σ>~0.1 with std(σ)~1e-3).  Tighten the inner-CG MooeeInv solve
  // accordingly.  The test infrastructure validates the EO splitting and
  // that MooeeInv is consistent with Mooee; production efficiency at large
  // aux requires preconditioned CG (use stock MooeeInv_QCD as preconditioner).
  const char *as_env = std::getenv("AUX_SCALE");
  const RealD aux_scale = (as_env && *as_env) ? std::atof(as_env) : 0.05;
  std::cout << GridLogMessage << "[mobius_eo] AUX_SCALE=" << aux_scale << std::endl;
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

  TXQCDFermionNf v(FGrid);
  for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG5, v.f[a]);

  // ---- Test 1: EO decomposition matches full M ----
  TXQCDFermionNf Mv_full(FGrid), Mv_eo(FGrid);
  Mop.M(v, Mv_full);

  // Split v into checkerboards.
  TXQCDFermionNf v_e(FrbGrid), v_o(FrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    pickCheckerboard(Even, v_e.f[a], v.f[a]);
    pickCheckerboard(Odd,  v_o.f[a], v.f[a]);
  }
  // (Mv)_e = Mooee(v_e) + Meooe(v_o);   (Mv)_o = Mooee(v_o) + Meooe(v_e)
  TXQCDFermionNf out_e(FrbGrid), out_o(FrbGrid);
  TXQCDFermionNf tmp_e(FrbGrid), tmp_o(FrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    out_e.f[a].Checkerboard() = Even; out_o.f[a].Checkerboard() = Odd;
    tmp_e.f[a].Checkerboard() = Even; tmp_o.f[a].Checkerboard() = Odd;
  }
  Meo.Mooee(v_e, out_e);
  Meo.Meooe(v_o, tmp_e);
  for (int a = 0; a < TxqcdNf; ++a) out_e.f[a] = out_e.f[a] + tmp_e.f[a];

  Meo.Mooee(v_o, out_o);
  Meo.Meooe(v_e, tmp_o);
  for (int a = 0; a < TxqcdNf; ++a) out_o.f[a] = out_o.f[a] + tmp_o.f[a];

  for (int a = 0; a < TxqcdNf; ++a) {
    Mv_eo.f[a].Checkerboard() = Even;
    setCheckerboard(Mv_eo.f[a], out_e.f[a]);
    setCheckerboard(Mv_eo.f[a], out_o.f[a]);
  }
  RealD nrm_full = std::sqrt(norm2(Mv_full));
  RealD diff_eo = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) {
    LatticeFermion d(FGrid); d = Mv_full.f[a] - Mv_eo.f[a];
    diff_eo += norm2(d);
  }
  diff_eo = std::sqrt(diff_eo) / std::max(nrm_full, 1e-30);
  bool t1_pass = diff_eo < 1e-10;
  std::cout << GridLogMessage << "Test 1 (full M == Mooee+Meooe): rel diff="
            << diff_eo << "  " << (t1_pass ? "PASS" : "FAIL") << std::endl;

  // ---- Test 2: MooeeInv refines to convergence ----
  TXQCDFermionNf Mv_e(FrbGrid), inv_e(FrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    Mv_e.f[a].Checkerboard()  = Even;
    inv_e.f[a].Checkerboard() = Even;
  }
  Meo.Mooee(v_e, Mv_e);
  Meo.MooeeInv(Mv_e, inv_e);
  RealD diff_minv = 0.0; RealD nrm_v_e = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) {
    LatticeFermion d(FrbGrid); d.Checkerboard() = Even;
    d = inv_e.f[a] - v_e.f[a];
    diff_minv += norm2(d);
    nrm_v_e   += norm2(v_e.f[a]);
  }
  diff_minv = std::sqrt(diff_minv) / std::sqrt(std::max(nrm_v_e, 1e-30));
  bool t2_pass = diff_minv < 1e-10;
  std::cout << GridLogMessage << "Test 2 (MooeeInv(Mooee(x)) == x): rel diff="
            << diff_minv << "  " << (t2_pass ? "PASS" : "FAIL") << std::endl;

  // ---- Test 3: same for odd cb + Dag variants ----
  TXQCDFermionNf Mv_o(FrbGrid), inv_o(FrbGrid);
  for (int a = 0; a < TxqcdNf; ++a) {
    Mv_o.f[a].Checkerboard()  = Odd;
    inv_o.f[a].Checkerboard() = Odd;
  }
  Meo.MooeeDag(v_o, Mv_o);
  Meo.MooeeInvDag(Mv_o, inv_o);
  RealD diff_minvD = 0.0; RealD nrm_v_o = 0.0;
  for (int a = 0; a < TxqcdNf; ++a) {
    LatticeFermion d(FrbGrid); d.Checkerboard() = Odd;
    d = inv_o.f[a] - v_o.f[a];
    diff_minvD += norm2(d);
    nrm_v_o    += norm2(v_o.f[a]);
  }
  diff_minvD = std::sqrt(diff_minvD) / std::sqrt(std::max(nrm_v_o, 1e-30));
  bool t3_pass = diff_minvD < 1e-10;
  std::cout << GridLogMessage << "Test 3 (MooeeInvDag(MooeeDag(x)) == x, odd cb): rel diff="
            << diff_minvD << "  " << (t3_pass ? "PASS" : "FAIL") << std::endl;

  bool ok = t1_pass && t2_pass && t3_pass;
  std::cout << GridLogMessage << (ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
            << std::endl;
  Grid_finalize();
  return ok ? 0 : 1;
}
