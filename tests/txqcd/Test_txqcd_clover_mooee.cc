// Validate the TXQCD Wilson-Clover operator:
//
// Test 1: At aux=0, TXQCD Mooee with clover matches Grid's WilsonCloverFermion
//         Mooee per flavor to machine precision.
//
// Test 2: Mooee * MooeeInv = I with both aux fields and clover active.
//
// Test 3: Full-grid M with clover matches EO decomposition.

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>

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

  RealD mass = 0.3;
  RealD csw = 1.0;
  int exitcode = 0;

  LatticeGaugeField U(grid);
  SU3::HotConfiguration(pRNG, U);

  // ===== Test 1: aux=0, compare TXQCD Mooee vs Grid WilsonClover Mooee =====
  std::cout << GridLogMessage << "===== Test 1: Mooee at aux=0 vs WilsonClover =====" << std::endl;
  {
    LatticeSigmaField sigma(grid); sigma = Zero();
    LatticePiField pi(grid); pi = Zero();
    LatticeSFieldC s(grid); s = Zero();
    LatticePFieldC p(grid); p = Zero();
    LatticeTField t(grid); t = Zero();

    TXQCDWilsonCloverFermionEO EOp(U, *grid, *rbgrid, mass, sigma, pi, s, p, t, csw);

    WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> Dwc(
        U, *grid, *rbgrid, mass, csw, csw);

    for (int cb = 0; cb < 2; ++cb) {
      std::string cbname = (cb == Even) ? "Even" : "Odd";

      TXQCDFermionNf v(rbgrid);
      for (int a = 0; a < TxqcdNf; ++a) {
        gaussian(pRNG, v.f[a]);
        pickCheckerboard(cb, v.f[a], v.f[a]);
        v.f[a].Checkerboard() = cb;
      }

      // Apply TXQCD Mooee
      TXQCDFermionNf out_txqcd(rbgrid);
      EOp.Mooee(v, out_txqcd);

      // Apply Grid WilsonClover Mooee per flavor
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion out_wc(rbgrid);
        Dwc.Mooee(v.f[a], out_wc);

        LatticeFermion diff = out_txqcd.f[a] - out_wc;
        RealD ndiff = norm2(diff);
        RealD nref = norm2(out_wc);
        RealD rel = std::sqrt(ndiff / nref);
        std::cout << GridLogMessage << "  " << cbname << " flavor " << a
                  << ": ||TXQCD - WC|| / ||WC|| = " << rel << std::endl;
        if (rel > 1e-14) {
          std::cout << GridLogMessage << "  FAIL" << std::endl;
          exitcode = 1;
        }
      }
    }
    if (exitcode == 0)
      std::cout << GridLogMessage << "  PASS" << std::endl;

    // Diagnostic: compare at component level to find ratio
    {
      int cb = Even;
      TXQCDFermionNf v(rbgrid);
      for (int a = 0; a < TxqcdNf; ++a) {
        v.f[a] = Zero();
        v.f[a].Checkerboard() = cb;
      }
      // Set a single component: flavor 0, spin 0, color 0, site 0
      {
        typedef typename LatticeFermion::vector_object::scalar_object FermSobj;
        std::vector<FermSobj> vs;
        unvectorizeToLexOrdArray(vs, v.f[0]);
        vs[0]()(0)(0) = ComplexD(1.0, 0.0);
        vectorizeFromLexOrdArray(vs, v.f[0]);
        v.f[0].Checkerboard() = cb;
      }

      TXQCDFermionNf out_txqcd(rbgrid);
      EOp.Mooee(v, out_txqcd);

      LatticeFermion out_wc(rbgrid);
      Dwc.Mooee(v.f[0], out_wc);

      // Extract site 0 components
      typedef typename LatticeFermion::vector_object::scalar_object FermSobj;
      std::vector<FermSobj> txqcd_s, wc_s;
      unvectorizeToLexOrdArray(txqcd_s, out_txqcd.f[0]);
      unvectorizeToLexOrdArray(wc_s, out_wc);

      std::cout << GridLogMessage << "  Diagnostic site 0, flavor 0:" << std::endl;
      for (int alpha = 0; alpha < Ns; ++alpha) {
        for (int i = 0; i < Nc; ++i) {
          auto t = txqcd_s[0]()(alpha)(i);
          auto w = wc_s[0]()(alpha)(i);
          if (std::abs(std::complex<double>(t.real(), t.imag())) > 1e-10 ||
              std::abs(std::complex<double>(w.real(), w.imag())) > 1e-10) {
            std::cout << GridLogMessage << "    (alpha=" << alpha << ",i=" << i
                      << "): TXQCD=" << t << " WC=" << w;
            if (std::abs(std::complex<double>(w.real(), w.imag())) > 1e-10)
              std::cout << " ratio=" << ComplexD(t.real(),t.imag())/ComplexD(w.real(),w.imag());
            std::cout << std::endl;
          }
        }
      }
    }
  }

  // ===== Test 2: MooeeInv with aux + clover =====
  std::cout << GridLogMessage << "===== Test 2: MooeeInv with aux + clover =====" << std::endl;
  {
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

    TXQCDWilsonCloverFermionEO EOp(U, *grid, *rbgrid, mass, sigma, pi, s, p, t, csw);

    for (int cb = 0; cb < 2; ++cb) {
      std::string cbname = (cb == Even) ? "Even" : "Odd";

      TXQCDFermionNf v(rbgrid);
      for (int a = 0; a < TxqcdNf; ++a) {
        gaussian(pRNG, v.f[a]);
        pickCheckerboard(cb, v.f[a], v.f[a]);
        v.f[a].Checkerboard() = cb;
      }

      // MooeeInv(Mooee(v)) - v
      TXQCDFermionNf Mv(rbgrid), MiMv(rbgrid);
      EOp.Mooee(v, Mv);
      EOp.MooeeInv(Mv, MiMv);
      RealD nv = norm2(v);
      RealD ndiff = 0;
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion d = MiMv.f[a] - v.f[a];
        ndiff += norm2(d);
      }
      RealD rel1 = std::sqrt(ndiff / nv);

      // Mooee(MooeeInv(v)) - v
      TXQCDFermionNf Miv(rbgrid), MMiv(rbgrid);
      EOp.MooeeInv(v, Miv);
      EOp.Mooee(Miv, MMiv);
      ndiff = 0;
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion d = MMiv.f[a] - v.f[a];
        ndiff += norm2(d);
      }
      RealD rel2 = std::sqrt(ndiff / nv);

      std::cout << GridLogMessage << "  " << cbname
                << ": MooeeInv(Mooee(v))-v rel = " << rel1
                << "  Mooee(MooeeInv(v))-v rel = " << rel2 << std::endl;
      if (rel1 > 1e-13 || rel2 > 1e-13) {
        std::cout << GridLogMessage << "  FAIL" << std::endl;
        exitcode = 1;
      } else {
        std::cout << GridLogMessage << "  PASS" << std::endl;
      }

      // MooeeInvDag(MooeeDag(v)) - v
      TXQCDFermionNf Mdv(rbgrid), MidMdv(rbgrid);
      EOp.MooeeDag(v, Mdv);
      EOp.MooeeInvDag(Mdv, MidMdv);
      ndiff = 0;
      for (int a = 0; a < TxqcdNf; ++a) {
        LatticeFermion d = MidMdv.f[a] - v.f[a];
        ndiff += norm2(d);
      }
      RealD rel3 = std::sqrt(ndiff / nv);
      std::cout << GridLogMessage << "  " << cbname
                << ": MooeeInvDag(MooeeDag(v))-v rel = " << rel3 << std::endl;
      if (rel3 > 1e-13) {
        std::cout << GridLogMessage << "  FAIL" << std::endl;
        exitcode = 1;
      } else {
        std::cout << GridLogMessage << "  PASS" << std::endl;
      }
    }
  }

  // ===== Test 3: Full-grid M with clover vs EO decomposition =====
  std::cout << GridLogMessage << "===== Test 3: Full M vs EO decomposition =====" << std::endl;
  {
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

    TXQCDWilsonCloverOp fullOp(U, *grid, *rbgrid, mass, sigma, pi, s, p, t, csw);
    TXQCDWilsonCloverFermionEO EOp(U, *grid, *rbgrid, mass, sigma, pi, s, p, t, csw);

    TXQCDFermionNf v(grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, v.f[a]);

    // Full M
    TXQCDFermionNf Mv_full(grid);
    fullOp.M(v, Mv_full);

    // EO decomposition: Mv_e = Mee*v_e + Meo*v_o, Mv_o = Moe*v_e + Moo*v_o
    TXQCDFermionNf v_e(rbgrid), v_o(rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      pickCheckerboard(Even, v_e.f[a], v.f[a]);
      pickCheckerboard(Odd, v_o.f[a], v.f[a]);
    }

    TXQCDFermionNf Mee_ve(rbgrid), Meo_vo(rbgrid);
    TXQCDFermionNf Moe_ve(rbgrid), Moo_vo(rbgrid);
    EOp.Mooee(v_e, Mee_ve);
    EOp.Meooe(v_o, Meo_vo);
    EOp.Meooe(v_e, Moe_ve);
    EOp.Mooee(v_o, Moo_vo);

    // Wait — Meooe maps odd→even and even→odd. Need to be careful about
    // which checkerboard the input is on.
    // Actually, WilsonFermion::Meooe takes input on one CB and outputs on the
    // other. So Meooe(v_o) gives even output, Meooe(v_e) gives odd output.
    // But our EOp.Meooe calls per-flavor Dw_.Meooe which does the same.

    TXQCDFermionNf Mv_eo(grid);
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion res_e = Mee_ve.f[a] + Meo_vo.f[a];
      LatticeFermion res_o = Moe_ve.f[a] + Moo_vo.f[a];
      setCheckerboard(Mv_eo.f[a], res_e);
      setCheckerboard(Mv_eo.f[a], res_o);
    }

    RealD nfull = norm2(Mv_full);
    RealD ndiff = 0;
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion d = Mv_eo.f[a] - Mv_full.f[a];
      ndiff += norm2(d);
    }
    RealD rel = std::sqrt(ndiff / nfull);
    std::cout << GridLogMessage << "  ||M_eo - M_full|| / ||M_full|| = " << rel << std::endl;
    if (rel > 1e-14) {
      std::cout << GridLogMessage << "  FAIL" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage << "  PASS" << std::endl;
    }
  }

  if (exitcode == 0)
    std::cout << GridLogMessage << "ALL CLOVER TESTS PASSED" << std::endl;
  else
    std::cout << GridLogMessage << "SOME TESTS FAILED" << std::endl;

  Grid_finalize();
  return exitcode;
}
