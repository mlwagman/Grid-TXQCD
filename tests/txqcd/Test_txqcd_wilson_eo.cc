// Test the even-odd preconditioned TXQCD Wilson operator.
//
// Test 1: EO decomposition matches full-grid operator
//   M v = Mee v_e + Meo v_o (even part) and Moe v_e + Moo v_o (odd part)
//
// Test 2: Mooee · MooeeInv = Identity on both checkerboards
//
// Test 3: Schur complement CG matches full-grid CG
//   Solve M†M x = b both ways, compare solutions.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDSchurOp.h>
#include <Grid/qcd/action/txqcd/TXQCDSolvers.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  if (latt.size() == 0 || latt[0] == 0) latt = Coordinate({4, 4, 4, 8});

  GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(
      latt, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *rbgrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(grid);

  GridParallelRNG pRNG(grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4, 5});

  RealD mass = 0.5;
  int exitcode = 0;

  // Random gauge + aux fields.
  LatticeGaugeField U(grid);
  SU3::HotConfiguration(pRNG, U);

  LatticeSigmaField sigma(grid);
  LatticePiField pi(grid);
  LatticeSFieldC s(grid);
  LatticePFieldC p(grid);
  LatticeTField t(grid);

  RealD aux_scale = 0.1;
  gaussian(pRNG, sigma);
  sigma = aux_scale * 0.5 * (sigma + adj(sigma));
  gaussian(pRNG, pi);
  pi = aux_scale * 0.5 * (pi + adj(pi));
  gaussian(pRNG, s);
  s = aux_scale * 0.5 * (s + adj(s));
  gaussian(pRNG, p);
  p = aux_scale * 0.5 * (p + adj(p));
  // Antisymmetric tensor
  t = Zero();
  LatticeSFieldC tblock(grid);
  for (int mu = 0; mu < Nd; ++mu) {
    for (int nu = mu + 1; nu < Nd; ++nu) {
      gaussian(pRNG, tblock);
      tblock = aux_scale * 0.5 * (tblock + adj(tblock));
      autoView(tv, t, CpuWrite);
      autoView(bv, tblock, CpuRead);
      thread_for(ss, grid->oSites(), {
        for (int i = 0; i < Nc; ++i)
          for (int j = 0; j < Nc; ++j) {
            tv[ss]()(mu, nu)(i, j) = bv[ss]()()(i, j);
            tv[ss]()(nu, mu)(i, j) = -bv[ss]()()(i, j);
          }
      });
    }
  }

  // Build both operators.
  TXQCDWilsonOp Mfull(U, *grid, *rbgrid, mass, sigma, pi, s, p, t);
  TXQCDWilsonFermionEO Meo(U, *grid, *rbgrid, mass, sigma, pi, s, p, t);

  // ================================================================
  // Test 1: EO decomposition matches full-grid M
  // ================================================================
  std::cout << GridLogMessage << "===== Test 1: EO decomposition =====" << std::endl;
  {
    TXQCDFermionNf v(grid), Mv_full(grid), Mv_eo(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, v.f[a]);

    // Full-grid result.
    Mfull.M(v, Mv_full);

    // EO result: split v into even/odd, apply EO components, recombine.
    TXQCDFermionNf v_e(rbgrid), v_o(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      pickCheckerboard(Even, v_e.f[a], v.f[a]);
      pickCheckerboard(Odd, v_o.f[a], v.f[a]);
    }

    TXQCDFermionNf Mee_ve(rbgrid), Meo_vo(rbgrid);
    TXQCDFermionNf Moe_ve(rbgrid), Moo_vo(rbgrid);
    Meo.Mooee(v_e, Mee_ve);   // Mee * v_e
    Meo.Meooe(v_o, Meo_vo);   // Meo * v_o (odd -> even)
    Meo.Meooe(v_e, Moe_ve);   // Moe * v_e (even -> odd)
    Meo.Mooee(v_o, Moo_vo);   // Moo * v_o

    // Recombine: even part = Mee*v_e + Meo*v_o, odd part = Moe*v_e + Moo*v_o
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion even_part(rbgrid), odd_part(rbgrid);
      even_part = Mee_ve.f[a] + Meo_vo.f[a];
      odd_part = Moe_ve.f[a] + Moo_vo.f[a];
      setCheckerboard(Mv_eo.f[a], even_part);
      setCheckerboard(Mv_eo.f[a], odd_part);
    }

    // Compare.
    RealD diff = 0, ref = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(grid);
      d = Mv_full.f[a] - Mv_eo.f[a];
      diff += norm2(d);
      ref += norm2(Mv_full.f[a]);
    }
    RealD reldiff = std::sqrt(diff / ref);
    std::cout << GridLogMessage << "  ||M_full - M_eo|| / ||M_full|| = "
              << reldiff << std::endl;
    if (reldiff > 1e-12) {
      std::cout << GridLogMessage << "  FAIL (threshold 1e-12)" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }
  }

  // ================================================================
  // Test 2: Mooee · MooeeInv = Identity
  // ================================================================
  std::cout << GridLogMessage << "===== Test 2: MooeeInv =====" << std::endl;
  for (int cb = 0; cb < 2; ++cb) {
    std::string cbs = (cb == Even) ? "Even" : "Odd";
    TXQCDFermionNf v(rbgrid), tmp(rbgrid), result(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, v.f[a]);
      pickCheckerboard(cb, v.f[a], v.f[a]);
      v.f[a].Checkerboard() = cb;
    }

    // Forward: MooeeInv(Mooee(v)) = v
    Meo.Mooee(v, tmp);
    Meo.MooeeInv(tmp, result);
    RealD diff1 = 0, ref1 = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(rbgrid);
      d = result.f[a] - v.f[a];
      diff1 += norm2(d);
      ref1 += norm2(v.f[a]);
    }
    RealD rel1 = std::sqrt(diff1 / ref1);

    // Reverse: Mooee(MooeeInv(v)) = v
    Meo.MooeeInv(v, tmp);
    Meo.Mooee(tmp, result);
    RealD diff2 = 0, ref2 = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(rbgrid);
      d = result.f[a] - v.f[a];
      diff2 += norm2(d);
      ref2 += norm2(v.f[a]);
    }
    RealD rel2 = std::sqrt(diff2 / ref2);

    std::cout << GridLogMessage << "  " << cbs
              << ": MooeeInv(Mooee(v))-v rel = " << rel1
              << "  Mooee(MooeeInv(v))-v rel = " << rel2 << std::endl;
    if (rel1 > 1e-12 || rel2 > 1e-12) {
      std::cout << GridLogMessage << "  FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }

    // Also test Dag variants.
    Meo.MooeeDag(v, tmp);
    Meo.MooeeInvDag(tmp, result);
    RealD diff3 = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(rbgrid);
      d = result.f[a] - v.f[a];
      diff3 += norm2(d);
    }
    RealD rel3 = std::sqrt(diff3 / ref1);
    std::cout << GridLogMessage << "  " << cbs
              << ": MooeeInvDag(MooeeDag(v))-v rel = " << rel3 << std::endl;
    if (rel3 > 1e-12) {
      std::cout << GridLogMessage << "  FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }
  }

  // ================================================================
  // Test 3: Schur complement consistency
  // ================================================================
  std::cout << GridLogMessage << "===== Test 3: Schur complement =====" << std::endl;
  {
    TXQCDSchurOp SchurOp(Meo);
    TXQCDFermionNf v_o(rbgrid), Mpc_v(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, v_o.f[a]);
      v_o.f[a].Checkerboard() = Odd;
    }

    SchurOp.Mpc(v_o, Mpc_v);

    // Verify manually: Mpc = Moo - Moe Mee^{-1} Meo
    TXQCDFermionNf tmp_e(rbgrid), tmp2_e(rbgrid), tmp_o(rbgrid),
        Moo_v(rbgrid);
    Meo.Meooe(v_o, tmp_e);      // Meo: odd -> even
    Meo.MooeeInv(tmp_e, tmp2_e); // Mee^{-1}
    Meo.Meooe(tmp2_e, tmp_o);   // Moe: even -> odd
    Meo.Mooee(v_o, Moo_v);      // Moo
    TXQCDFermionNf manual_o(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) manual_o.f[a] = Moo_v.f[a] - tmp_o.f[a];

    RealD diff_schur = 0, ref_schur = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d(rbgrid);
      d = Mpc_v.f[a] - manual_o.f[a];
      diff_schur += norm2(d);
      ref_schur += norm2(Mpc_v.f[a]);
    }
    RealD rel_schur = std::sqrt(diff_schur / ref_schur);
    std::cout << GridLogMessage
              << "  ||Mpc - manual|| / ||Mpc|| = " << rel_schur << std::endl;
    if (rel_schur > 1e-14) {
      std::cout << GridLogMessage << "  FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }

    // Verify HermOp is Hermitian: <v, H v> should be real.
    TXQCDFermionNf Hv(rbgrid);
    SchurOp.HermOp(v_o, Hv);
    ComplexD vHv = innerProduct(v_o, Hv);
    RealD herm_imag = std::abs(vHv.imag()) / std::abs(vHv.real());
    std::cout << GridLogMessage << "  <v, HermOp v> = " << vHv
              << "  |Im|/|Re| = " << herm_imag << std::endl;
    if (herm_imag > 1e-12) {
      std::cout << GridLogMessage << "  Hermiticity FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  Hermiticity PASS" << std::endl;
    }
  }

  // ================================================================
  // Test 4: TXQCDConjugateGradient with SchurOp
  // ================================================================
  std::cout << GridLogMessage << "===== Test 4: TXQCDConjugateGradient =====" << std::endl;
  {
    TXQCDSchurOp SchurOp(Meo);
    TXQCDFermionNf b_o(rbgrid), x_o(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, b_o.f[a]);
      b_o.f[a].Checkerboard() = Odd;
    }

    TXQCDConjugateGradient CG(1e-10, 10000);
    CG(SchurOp, b_o, x_o);

    std::cout << GridLogMessage << "  iter=" << CG.IterationsToComplete
              << " true_resid=" << CG.TrueResidual << std::endl;
    if (CG.TrueResidual > 1e-8) {
      std::cout << GridLogMessage << "  FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }
  }

  // ================================================================
  // Test 5: TXQCDMultiShiftCGSchur
  // ================================================================
  std::cout << GridLogMessage << "===== Test 5: TXQCDMultiShiftCGSchur =====" << std::endl;
  {
    TXQCDSchurOp SchurOp(Meo);
    TXQCDFermionNf src(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, src.f[a]);
      src.f[a].Checkerboard() = Odd;
    }

    std::vector<RealD> shifts = {0.01, 0.1, 1.0, 10.0};
    std::vector<RealD> tols(shifts.size(), 1e-10);
    int nshift = (int)shifts.size();
    std::vector<TXQCDFermionNf> psi;
    psi.reserve(nshift);
    for (int s = 0; s < nshift; ++s) psi.emplace_back(rbgrid);

    TXQCDMultiShiftCGSchur MSCG(10000);
    MSCG(SchurOp, shifts, tols, src, psi);
    std::cout << GridLogMessage << "  iter=" << MSCG.IterationsToComplete << std::endl;

    bool all_pass = true;
    for (int s = 0; s < nshift; ++s) {
      TXQCDFermionNf Ax(rbgrid);
      SchurOp.HermOp(psi[s], Ax);
      RealD res2 = 0, ref2 = 0;
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion d(rbgrid);
        d = Ax.f[a] + shifts[s] * psi[s].f[a] - src.f[a];
        res2 += norm2(d);
        ref2 += norm2(src.f[a]);
      }
      RealD rel = std::sqrt(res2 / ref2);
      bool pass = rel < 1e-6;
      std::cout << GridLogMessage << "  shift=" << shifts[s]
                << " true_resid=" << rel
                << (pass ? "  PASS" : "  FAIL") << std::endl;
      if (!pass) all_pass = false;
    }
    if (!all_pass) exitcode = 1;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME EO TESTS FAILED" : "ALL EO TESTS PASSED")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
