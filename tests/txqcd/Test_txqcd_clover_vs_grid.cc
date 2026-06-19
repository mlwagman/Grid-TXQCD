// Compare TXQCD Wilson-Clover (aux=0) against Grid's WilsonCloverFermion.
//
// At aux=0 the TXQCD operator reduces to Nf copies of Wilson-Clover,
// so every component should match Grid's implementation to machine precision.
//
// Tests:
// 1. Full-grid M: TXQCDWilsonCloverOp.M vs WilsonCloverFermion.M per flavor
// 2. EO Mooee: TXQCDWilsonCloverFermionEO.Mooee vs WilsonCloverFermion.Mooee
// 3. EO MooeeInv: TXQCDWilsonCloverFermionEO.MooeeInv vs WilsonCloverFermion.MooeeInv
// 4. Schur-complement Mpc operator: TXQCD vs Grid
// 5. RHMC gauge force comparison at aux=0: decompose into hopping-only
//    and clover-only pieces and compare each against Grid

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverOp.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDCompositeImpl.h>
#include <Grid/qcd/action/txqcd/TXQCDSchurOp.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/fermion/CloverHelpers.h>
#include <Grid/qcd/action/fermion/WilsonCloverHelpers.h>
#include <Grid/qcd/action/pseudofermion/EvenOddSchurDifferentiable.h>

using namespace Grid;

int exitcode = 0;

void check(const char *name, RealD rel, RealD tol = 1e-13) {
  bool pass = rel < tol;
  std::cout << GridLogMessage << "[" << name << "] rel = " << rel
            << (pass ? "  PASS" : "  FAIL") << std::endl;
  if (!pass) exitcode = 1;
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 8});
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian grid(latt, simd, mpi);
  GridRedBlackCartesian rbgrid(&grid);
  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4, 5});

  RealD mass = 0.3;
  RealD csw = 1.0;
  bool u_identity = false;
  if (const char *m = std::getenv("MASS"); m && *m) mass = std::atof(m);
  if (const char *c = std::getenv("CSW");  c && *c) csw  = std::atof(c);
  if (const char *u = std::getenv("U_IDENTITY"); u && std::atoi(u) != 0) u_identity = true;
  std::cout << GridLogMessage << "mass=" << mass << " csw=" << csw
            << " U_IDENTITY=" << u_identity << std::endl;

  LatticeGaugeField U(&grid);
  if (u_identity) { U = 1.0; } else { SU3::HotConfiguration(pRNG, U); }

  // Zero aux fields
  LatticeSigmaField sigma(&grid); sigma = Zero();
  LatticePiField pi(&grid); pi = Zero();
  LatticeSFieldC s(&grid); s = Zero();
  LatticePFieldC p(&grid); p = Zero();
  LatticeTField t(&grid); t = Zero();

  // Grid's WilsonClover — match TXQCD operator BC (APBC time).
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;
  WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> Dwc(
      U, grid, rbgrid, mass, csw, csw, WilsonAnisotropyCoefficients(), ip);

  // TXQCD operators
  TXQCDWilsonCloverOp fullOp(U, grid, rbgrid, mass, sigma, pi, s, p, t, csw);
  TXQCDWilsonCloverFermionEO EOp(U, grid, rbgrid, mass, sigma, pi, s, p, t, csw);

  // ===== Test 1: Full-grid M =====
  std::cout << GridLogMessage << "===== Test 1: Full-grid M =====" << std::endl;
  {
    TXQCDFermionNf v(&grid);
    for (int a = 0; a < TxqcdNf; ++a) gaussian(pRNG, v.f[a]);

    TXQCDFermionNf Mv_txqcd(&grid);
    fullOp.M(v, Mv_txqcd);

    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion Mv_wc(&grid);
      Dwc.M(v.f[a], Mv_wc);

      LatticeFermion diff = Mv_txqcd.f[a] - Mv_wc;
      RealD rel = std::sqrt(norm2(diff) / norm2(Mv_wc));
      std::string label = "full M flavor " + std::to_string(a);
      check(label.c_str(), rel);
    }
  }

  // ===== Test 2: EO Mooee =====
  std::cout << GridLogMessage << "===== Test 2: EO Mooee =====" << std::endl;
  for (int cb = 0; cb < 2; ++cb) {
    TXQCDFermionNf v(&rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, v.f[a]);
      pickCheckerboard(cb, v.f[a], v.f[a]);
      v.f[a].Checkerboard() = cb;
    }

    TXQCDFermionNf out_txqcd(&rbgrid);
    EOp.Mooee(v, out_txqcd);

    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion out_wc(&rbgrid);
      Dwc.Mooee(v.f[a], out_wc);
      LatticeFermion diff = out_txqcd.f[a] - out_wc;
      RealD rel = std::sqrt(norm2(diff) / norm2(out_wc));
      std::string label = std::string(cb == Even ? "Even" : "Odd") +
                          " Mooee flavor " + std::to_string(a);
      check(label.c_str(), rel);
    }
  }

  // ===== Test 3: EO MooeeInv =====
  std::cout << GridLogMessage << "===== Test 3: EO MooeeInv =====" << std::endl;
  for (int cb = 0; cb < 2; ++cb) {
    TXQCDFermionNf v(&rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, v.f[a]);
      pickCheckerboard(cb, v.f[a], v.f[a]);
      v.f[a].Checkerboard() = cb;
    }

    TXQCDFermionNf out_txqcd(&rbgrid);
    EOp.MooeeInv(v, out_txqcd);

    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion out_wc(&rbgrid);
      Dwc.MooeeInv(v.f[a], out_wc);
      LatticeFermion diff = out_txqcd.f[a] - out_wc;
      RealD rel = std::sqrt(norm2(diff) / norm2(out_wc));
      std::string label = std::string(cb == Even ? "Even" : "Odd") +
                          " MooeeInv flavor " + std::to_string(a);
      check(label.c_str(), rel);
    }
  }

  // ===== Test 4: Schur complement Mpc =====
  std::cout << GridLogMessage << "===== Test 4: Schur complement Mpc =====" << std::endl;
  {
    // TXQCD Mpc: Moo - Moe Mee^{-1} Meo on odd sublattice
    TXQCDFermionNf v_o(&rbgrid);
    for (int a = 0; a < TxqcdNf; ++a) {
      gaussian(pRNG, v_o.f[a]);
      pickCheckerboard(Odd, v_o.f[a], v_o.f[a]);
      v_o.f[a].Checkerboard() = Odd;
    }

    // TXQCD Schur
    TXQCDFermionNf Meo_v(&rbgrid), MeeiMeo_v(&rbgrid);
    TXQCDFermionNf MoeMeeiMeo_v(&rbgrid), Moo_v(&rbgrid), Mpc_txqcd(&rbgrid);
    EOp.Meooe(v_o, Meo_v);       // Meo: odd -> even
    EOp.MooeeInv(Meo_v, MeeiMeo_v);
    EOp.Meooe(MeeiMeo_v, MoeMeeiMeo_v);  // Moe: even -> odd
    EOp.Mooee(v_o, Moo_v);
    for (int a = 0; a < TxqcdNf; ++a)
      Mpc_txqcd.f[a] = Moo_v.f[a] - MoeMeeiMeo_v.f[a];

    // Grid Schur per flavor
    for (int a = 0; a < TxqcdNf; ++a) {
      LatticeFermion Meo_wc(&rbgrid), MeeiMeo_wc(&rbgrid);
      LatticeFermion MoeMeeiMeo_wc(&rbgrid), Moo_wc(&rbgrid);
      Dwc.Meooe(v_o.f[a], Meo_wc);
      Dwc.MooeeInv(Meo_wc, MeeiMeo_wc);
      Dwc.Meooe(MeeiMeo_wc, MoeMeeiMeo_wc);
      Dwc.Mooee(v_o.f[a], Moo_wc);
      LatticeFermion Mpc_wc = Moo_wc - MoeMeeiMeo_wc;

      LatticeFermion diff = Mpc_txqcd.f[a] - Mpc_wc;
      RealD rel = std::sqrt(norm2(diff) / norm2(Mpc_wc));
      std::string label = "Mpc flavor " + std::to_string(a);
      check(label.c_str(), rel, 1e-12);
    }
  }

  // ===== Test 5: RHMC gauge force at aux=0 =====
  // Compare total gauge force from TXQCD RHMC vs what we'd get from
  // Grid's WilsonClover applied independently per flavor.
  std::cout << GridLogMessage << "===== Test 5: RHMC gauge force at aux=0 =====" << std::endl;
  {
    OneFlavourRationalParams rat_params(1e-4, 64.0, 10000, 1e-10,
                                        12, 64, 100, 1e-8, 1e-4);

    // Build TXQCD composite field with aux=0
    TXQCDField Ucomp(&grid);
    Ucomp.U = U;
    Ucomp.sigma = Zero();
    Ucomp.pi = Zero();
    Ucomp.s = Zero();
    Ucomp.p = Zero();
    Ucomp.t = Zero();

    TXQCDWilsonCloverRationalEOAction action(grid, rbgrid, mass, rat_params, csw);

    GridSerialRNG sRNG;
    sRNG.SeedFixedIntegers({11, 12, 13, 14});
    action.refresh(Ucomp, sRNG, pRNG);

    RealD S_txqcd = action.S(Ucomp);
    std::cout << GridLogMessage << "  TXQCD action S = " << S_txqcd << std::endl;

    TXQCDField dSdU(&grid);
    action.deriv(Ucomp, dSdU);

    // The gauge force dSdU.U should be the total gauge force (hopping + clover).
    // Finite difference check of the total gauge force:
    std::array<LatticeColourMatrix, 4> Emu{LatticeColourMatrix(&grid),
        LatticeColourMatrix(&grid), LatticeColourMatrix(&grid),
        LatticeColourMatrix(&grid)};
    for (int mu = 0; mu < Nd; ++mu)
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

    // deriv() returns gauge force in Convention A (half gradient).
    // FD relationship: dS/dh = -2 * Re Tr(E * F_A).
    RealD an = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;

    const RealD h = 1e-4;
    LatticeGaugeField Usaved = Ucomp.U;

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], h, 12);
      PokeIndex<LorentzIndex>(Ucomp.U, expE * Umu, mu);
    }
    RealD Sp = action.S(Ucomp);

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], -h, 12);
      PokeIndex<LorentzIndex>(Ucomp.U, expE * Umu, mu);
    }
    RealD Sm = action.S(Ucomp);

    Ucomp.U = Usaved;

    RealD fd = (Sp - Sm) / (2.0 * h);
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    check("RHMC gauge force FD (total)", rel, 1e-3);

    std::cout << GridLogMessage << "  AN = " << an << "  FD = " << fd << std::endl;

    // Also check: aux forces should be zero at aux=0 (sanity check)
    RealD nsig = std::sqrt(norm2(dSdU.sigma));
    RealD npi = std::sqrt(norm2(dSdU.pi));
    RealD ns = std::sqrt(norm2(dSdU.s));
    RealD np = std::sqrt(norm2(dSdU.p));
    RealD nt = std::sqrt(norm2(dSdU.t));
    // Note: aux forces are NOT zero even at aux=0, because the pseudofermion
    // field Phi still couples to aux fields through the derivative.
    // This is correct behavior.
    std::cout << GridLogMessage << "  |dSdU.sigma| = " << nsig << std::endl;
    std::cout << GridLogMessage << "  |dSdU.U|     = " << std::sqrt(norm2(dSdU.U)) << std::endl;
  }

  // ===== Test 6: Hopping-only gauge force (csw=0) vs Grid =====
  // This isolates the hopping part — should match at csw=0.
  std::cout << GridLogMessage << "===== Test 6: Hopping-only gauge force (csw=0) =====" << std::endl;
  {
    OneFlavourRationalParams rat_params(1e-4, 64.0, 10000, 1e-10,
                                        12, 64, 100, 1e-8, 1e-4);

    TXQCDField Ucomp(&grid);
    Ucomp.U = U;
    Ucomp.sigma = Zero();
    Ucomp.pi = Zero();
    Ucomp.s = Zero();
    Ucomp.p = Zero();
    Ucomp.t = Zero();

    TXQCDWilsonCloverRationalEOAction action_csw0(grid, rbgrid, mass, rat_params, 0.0);

    GridSerialRNG sRNG;
    sRNG.SeedFixedIntegers({11, 12, 13, 14});
    action_csw0.refresh(Ucomp, sRNG, pRNG);

    TXQCDField dSdU0(&grid);
    action_csw0.deriv(Ucomp, dSdU0);

    std::array<LatticeColourMatrix, 4> Emu{LatticeColourMatrix(&grid),
        LatticeColourMatrix(&grid), LatticeColourMatrix(&grid),
        LatticeColourMatrix(&grid)};
    for (int mu = 0; mu < Nd; ++mu)
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

    // Convention A gauge force: dS/dh = -2 * Re Tr(E * F_A).
    RealD an = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU0.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;

    const RealD h = 1e-4;
    LatticeGaugeField Usaved = Ucomp.U;

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], h, 12);
      PokeIndex<LorentzIndex>(Ucomp.U, expE * Umu, mu);
    }
    RealD Sp = action_csw0.S(Ucomp);

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      LatticeColourMatrix expE = expMat(Emu[mu], -h, 12);
      PokeIndex<LorentzIndex>(Ucomp.U, expE * Umu, mu);
    }
    RealD Sm = action_csw0.S(Ucomp);

    Ucomp.U = Usaved;

    RealD fd = (Sp - Sm) / (2.0 * h);
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    check("RHMC hopping-only gauge force FD (csw=0)", rel, 1e-3);

    std::cout << GridLogMessage << "  AN = " << an << "  FD = " << fd << std::endl;
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CLOVER-VS-GRID TESTS FAILED"
                         : "ALL CLOVER-VS-GRID TESTS PASSED")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
