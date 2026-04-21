#include <Grid/Grid.h>
#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>
#include <Grid/qcd/action/pseudofermion/QCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverAction.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  GridSerialRNG   sRNG;
  GridParallelRNG pRNG(&Grid);
  sRNG.SeedFixedIntegers({1, 2, 3, 4, 5});
  pRNG.SeedFixedIntegers({6, 7, 8, 9, 10});

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  RealD mass = -0.05;
  RealD csw = 1.0;

  typedef WilsonCloverFermion<WilsonImplR, CloverHelpers<WilsonImplR>> WCF;
  WCF Dw(U, Grid, RBGrid, mass, csw, csw);

  // ================================================================
  // Test 0: Baseline M†M force (Grid's standard test, same framework)
  // ================================================================
  std::cout << GridLogMessage << "=== Test 0: M†M WilsonClover force baseline ===" << std::endl;
  {
    LatticeFermion phi(&Grid);
    gaussian(pRNG, phi);
    LatticeFermion Mphi(&Grid);

    Dw.ImportGauge(U);
    Dw.M(phi, Mphi);
    ComplexD S_MdM_c = innerProduct(Mphi, Mphi);

    LatticeGaugeField UdSdU(&Grid), tmp_f(&Grid);
    Dw.MDeriv(tmp_f, Mphi, phi, DaggerNo);
    UdSdU = tmp_f;
    Dw.MDeriv(tmp_f, phi, Mphi, DaggerYes);
    UdSdU = UdSdU + tmp_f;

    LatticeGaugeField mom0(&Grid);
    LatticeColourMatrix mommu0(&Grid);
    for (int mu = 0; mu < Nd; mu++) {
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu0);
      PokeIndex<LorentzIndex>(mom0, mommu0, mu);
    }

    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU, mu);
      fmu = Ta(fmu) * 2.0;
      PokeIndex<LorentzIndex>(UdSdU, fmu, mu);
    }

    for (RealD dt : {1e-3, 1e-4, 1e-5}) {
      LatticeComplex dS_lc(&Grid);
      dS_lc = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU, mu);
        mommu0 = PeekIndex<LorentzIndex>(mom0, mu);
        dS_lc = dS_lc + trace(mommu0 * fmu) * dt;
      }
      ComplexD dSpred_c = sum(dS_lc);
      RealD dSpred = dSpred_c.real();

      LatticeGaugeField Uprime(&Grid);
      for (int mu = 0; mu < Nd; mu++) {
        autoView(Uprime_v, Uprime, CpuWrite);
        autoView(U_v, U, CpuRead);
        autoView(mom0_v, mom0, CpuRead);
        thread_foreach(ss, U_v, {
          Uprime_v[ss]._internal[mu] =
            ProjectOnGroup(Exponentiate(mom0_v[ss]._internal[mu], dt, 12) * U_v[ss]._internal[mu]);
        });
      }

      WCF Dw_p(Uprime, Grid, RBGrid, mass, csw, csw);
      Dw_p.ImportGauge(Uprime);
      LatticeFermion MphiP(&Grid);
      Dw_p.M(phi, MphiP);
      ComplexD Sp_c = innerProduct(MphiP, MphiP);

      RealD deltaS = (Sp_c - S_MdM_c).real();
      std::cout << GridLogMessage << "dt=" << dt
                << "  deltaS=" << deltaS
                << "  dSpred=" << dSpred
                << "  diff=" << deltaS - dSpred
                << "  |diff|/|dS|=" << std::abs(deltaS - dSpred) / std::abs(deltaS) << std::endl;
    }
  }

  // ================================================================
  // Test 1: LogDet action value (sanity check)
  // ================================================================
  std::cout << GridLogMessage << "=== Test 1: LogDet action consistency ===" << std::endl;

  QCDLogDetCloverEOAction<WilsonImplR> LogDet(Dw);
  RealD S_logdet = LogDet.S(U);
  std::cout << GridLogMessage << "S_logdet = " << S_logdet << std::endl;

  // ================================================================
  // Test 2: MDeriv = DhopDeriv + MeeDeriv + MooDeriv
  // ================================================================
  std::cout << GridLogMessage << "=== Test 2: MeeDeriv + MooDeriv = MDeriv (clover part) ===" << std::endl;

  LatticeFermion X(&Grid), Y(&Grid);
  gaussian(pRNG, X);
  gaussian(pRNG, Y);

  LatticeGaugeField force_full(&Grid), force_hop(&Grid), force_ee(&Grid), force_oo(&Grid);

  Dw.ImportGauge(U);

  Dw.MDeriv(force_full, X, Y, DaggerNo);
  Dw.DhopDeriv(force_hop, X, Y, DaggerNo);

  LatticeFermion Xe(&RBGrid), Ye(&RBGrid), Xo(&RBGrid), Yo(&RBGrid);
  pickCheckerboard(Even, Xe, X);
  pickCheckerboard(Even, Ye, Y);
  pickCheckerboard(Odd, Xo, X);
  pickCheckerboard(Odd, Yo, Y);

  Dw.MeeDeriv(force_ee, Xe, Ye, DaggerNo);
  Dw.MooDeriv(force_oo, Xo, Yo, DaggerNo);

  LatticeGaugeField force_sum = force_hop + force_ee + force_oo;
  LatticeGaugeField diff = force_full - force_sum;

  RealD diff_norm = norm2(diff);
  RealD full_norm = norm2(force_full);
  RealD rel_diff = std::sqrt(diff_norm / full_norm);

  std::cout << GridLogMessage << "||MDeriv|| = " << std::sqrt(full_norm) << std::endl;
  std::cout << GridLogMessage << "||MDeriv - (DhopDeriv+MeeDeriv+MooDeriv)|| / ||MDeriv|| = "
            << rel_diff << std::endl;

  if (rel_diff > 1e-12) {
    std::cout << GridLogMessage << "FAIL: MeeDeriv+MooDeriv decomposition" << std::endl;
    return 1;
  }
  std::cout << GridLogMessage << "PASS: MeeDeriv+MooDeriv decomposition" << std::endl;

  // ================================================================
  // Test 3: LogDet force numerical check
  //   Grid pseudofermion convention: dSpred = -deltaS (ratio → -1)
  // ================================================================
  std::cout << GridLogMessage << "=== Test 3: LogDet force numerical check ===" << std::endl;

  LatticeGaugeField UdSdU_logdet(&Grid);
  LogDet.deriv(U, UdSdU_logdet);

  LatticeGaugeField mom(&Grid);
  LatticeColourMatrix mommu(&Grid);
  for (int mu = 0; mu < Nd; mu++) {
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
    PokeIndex<LorentzIndex>(mom, mommu, mu);
  }

  for (int mu = 0; mu < Nd; mu++) {
    LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_logdet, mu);
    fmu = Ta(fmu) * 2.0;
    PokeIndex<LorentzIndex>(UdSdU_logdet, fmu, mu);
  }

  for (RealD dt : {1e-3, 1e-4, 1e-5}) {
    LatticeComplex dS_lc(&Grid);
    dS_lc = Zero();
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_logdet, mu);
      mommu = PeekIndex<LorentzIndex>(mom, mu);
      dS_lc = dS_lc + trace(mommu * fmu) * dt;
    }
    RealD dSpred = TensorRemove(sum(dS_lc)).real();

    LatticeGaugeField Uprime(&Grid);
    for (int mu = 0; mu < Nd; mu++) {
      autoView(Uprime_v, Uprime, CpuWrite);
      autoView(U_v, U, CpuRead);
      autoView(mom_v, mom, CpuRead);
      thread_foreach(ss, U_v, {
        Uprime_v[ss]._internal[mu] =
          ProjectOnGroup(Exponentiate(mom_v[ss]._internal[mu], dt, 12) * U_v[ss]._internal[mu]);
      });
    }

    RealD S0 = LogDet.S(U);
    RealD Sp = LogDet.S(Uprime);
    RealD deltaS = Sp - S0;

    std::cout << GridLogMessage << "dt=" << dt
              << "  deltaS=" << deltaS
              << "  dSpred=" << dSpred
              << "  ratio=" << dSpred / deltaS << std::endl;
  }

  // ================================================================
  // Test 4: S_logdet + S_schur vs S_nf2
  // ================================================================
  std::cout << GridLogMessage << "=== Test 4: S_logdet + S_schur vs S_nf2 ===" << std::endl;

  ConjugateGradient<LatticeFermion> CG(1e-12, 30000);

  TwoFlavourPseudoFermionAction<WilsonImplR> Nf2(Dw, CG, CG);
  TwoFlavourSchurCloverAction<WilsonImplR> SchurPF(Dw, CG, CG);

  GridSerialRNG sRNG1, sRNG2;
  GridParallelRNG pRNG1(&Grid), pRNG2(&Grid);
  sRNG1.SeedFixedIntegers({100, 200, 300, 400, 500});
  sRNG2.SeedFixedIntegers({100, 200, 300, 400, 500});
  pRNG1.SeedFixedIntegers({600, 700, 800, 900, 1000});
  pRNG2.SeedFixedIntegers({600, 700, 800, 900, 1000});

  Nf2.refresh(U, sRNG1, pRNG1);
  SchurPF.refresh(U, sRNG2, pRNG2);

  RealD S_nf2 = Nf2.S(U);
  RealD S_schur = SchurPF.S(U);

  std::cout << GridLogMessage << "S_nf2 (full) = " << S_nf2 << std::endl;
  std::cout << GridLogMessage << "S_logdet     = " << S_logdet << std::endl;
  std::cout << GridLogMessage << "S_schur (EO) = " << S_schur << std::endl;

  // ================================================================
  // Test 5: KEY TEST — Nf2.deriv vs LogDet.deriv + Schur.deriv
  //   Direct analytical comparison, no FD needed.
  //   If these match, the Schur force is correct.
  // ================================================================
  std::cout << GridLogMessage << "=== Test 5: Nf2 force = LogDet force + Schur force ===" << std::endl;
  {
    LatticeGaugeField UdSdU_nf2(&Grid);
    Nf2.deriv(U, UdSdU_nf2);

    LatticeGaugeField UdSdU_logdet(&Grid);
    LogDet.deriv(U, UdSdU_logdet);

    LatticeGaugeField UdSdU_schur(&Grid);
    SchurPF.deriv(U, UdSdU_schur);

    LatticeGaugeField UdSdU_eo = UdSdU_logdet + UdSdU_schur;
    LatticeGaugeField diff = UdSdU_nf2 - UdSdU_eo;

    RealD norm_nf2 = std::sqrt(norm2(UdSdU_nf2));
    RealD norm_eo = std::sqrt(norm2(UdSdU_eo));
    RealD norm_diff = std::sqrt(norm2(diff));
    RealD norm_logdet = std::sqrt(norm2(UdSdU_logdet));
    RealD norm_schur = std::sqrt(norm2(UdSdU_schur));

    std::cout << GridLogMessage << "||Nf2 force||    = " << norm_nf2 << std::endl;
    std::cout << GridLogMessage << "||EO force||     = " << norm_eo << std::endl;
    std::cout << GridLogMessage << "||LogDet force|| = " << norm_logdet << std::endl;
    std::cout << GridLogMessage << "||Schur force||  = " << norm_schur << std::endl;
    std::cout << GridLogMessage << "||Nf2 - EO|| / ||Nf2|| = " << norm_diff / norm_nf2 << std::endl;

    // Also compare projected forces via momentum dot product
    LatticeGaugeField mom5(&Grid);
    LatticeColourMatrix mommu5(&Grid);
    for (int mu = 0; mu < Nd; mu++) {
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu5);
      PokeIndex<LorentzIndex>(mom5, mommu5, mu);
    }

    auto contract = [&](LatticeGaugeField &F) -> RealD {
      LatticeComplex lc(&Grid);
      lc = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(F, mu);
        fmu = Ta(fmu) * 2.0;
        mommu5 = PeekIndex<LorentzIndex>(mom5, mu);
        lc = lc + trace(mommu5 * fmu);
      }
      return TensorRemove(sum(lc)).real();
    };

    RealD dS_nf2 = contract(UdSdU_nf2);
    RealD dS_eo = contract(UdSdU_eo);
    RealD dS_logdet = contract(UdSdU_logdet);
    RealD dS_schur = contract(UdSdU_schur);

    std::cout << GridLogMessage << "Tr(P*2Ta(F_nf2))    = " << dS_nf2 << std::endl;
    std::cout << GridLogMessage << "Tr(P*2Ta(F_eo))     = " << dS_eo << std::endl;
    std::cout << GridLogMessage << "Tr(P*2Ta(F_logdet)) = " << dS_logdet << std::endl;
    std::cout << GridLogMessage << "Tr(P*2Ta(F_schur))  = " << dS_schur << std::endl;
    std::cout << GridLogMessage << "logdet + schur      = " << dS_logdet + dS_schur << std::endl;
    std::cout << GridLogMessage << "(nf2 - eo)/nf2      = " << (dS_nf2 - dS_eo) / dS_nf2 << std::endl;
  }

  // ================================================================
  // Test 6: Schur force FD check at multiple dt values
  // ================================================================
  std::cout << GridLogMessage << "=== Test 6: Schur force FD check ===" << std::endl;
  {
    LatticeGaugeField UdSdU_schur(&Grid);
    SchurPF.deriv(U, UdSdU_schur);

    LatticeGaugeField UdSdU_proj(&Grid);
    UdSdU_proj = UdSdU_schur;
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_proj, mu);
      fmu = Ta(fmu) * 2.0;
      PokeIndex<LorentzIndex>(UdSdU_proj, fmu, mu);
    }

    for (RealD dt : {1e-3, 1e-4, 1e-5, 1e-6}) {
      LatticeComplex dS_lc(&Grid);
      dS_lc = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_proj, mu);
        mommu = PeekIndex<LorentzIndex>(mom, mu);
        dS_lc = dS_lc + trace(mommu * fmu) * dt;
      }
      RealD dSpred = TensorRemove(sum(dS_lc)).real();

      LatticeGaugeField Uprime(&Grid);
      for (int mu = 0; mu < Nd; mu++) {
        autoView(Uprime_v, Uprime, CpuWrite);
        autoView(U_v, U, CpuRead);
        autoView(mom_v, mom, CpuRead);
        thread_foreach(ss, U_v, {
          Uprime_v[ss]._internal[mu] =
            ProjectOnGroup(Exponentiate(mom_v[ss]._internal[mu], dt, 12) * U_v[ss]._internal[mu]);
        });
      }

      RealD Sp = SchurPF.S(Uprime);
      RealD S0 = SchurPF.S(U);
      RealD deltaS = Sp - S0;

      std::cout << GridLogMessage << "dt=" << dt
                << "  deltaS=" << deltaS
                << "  dSpred=" << dSpred
                << "  ratio=" << dSpred / deltaS << std::endl;
    }
  }

  // ================================================================
  // Test 7: Combined EO force FD check
  // ================================================================
  std::cout << GridLogMessage << "=== Test 7: Combined EO force FD check ===" << std::endl;
  {
    LatticeGaugeField UdSdU_logdet(&Grid), UdSdU_schur(&Grid);
    LogDet.deriv(U, UdSdU_logdet);
    SchurPF.deriv(U, UdSdU_schur);
    LatticeGaugeField UdSdU_eo = UdSdU_logdet + UdSdU_schur;

    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_eo, mu);
      fmu = Ta(fmu) * 2.0;
      PokeIndex<LorentzIndex>(UdSdU_eo, fmu, mu);
    }

    // Compute the Nf2 action S for FD (uses the same Phi from refresh)
    auto S_eo = [&](const LatticeGaugeField &Ug) -> RealD {
      return LogDet.S(Ug) + SchurPF.S(Ug);
    };

    for (RealD dt : {1e-3, 1e-4, 1e-5}) {
      LatticeComplex dS_lc(&Grid);
      dS_lc = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_eo, mu);
        mommu = PeekIndex<LorentzIndex>(mom, mu);
        dS_lc = dS_lc + trace(mommu * fmu) * dt;
      }
      RealD dSpred = TensorRemove(sum(dS_lc)).real();

      LatticeGaugeField Uprime(&Grid);
      for (int mu = 0; mu < Nd; mu++) {
        autoView(Uprime_v, Uprime, CpuWrite);
        autoView(U_v, U, CpuRead);
        autoView(mom_v, mom, CpuRead);
        thread_foreach(ss, U_v, {
          Uprime_v[ss]._internal[mu] =
            ProjectOnGroup(Exponentiate(mom_v[ss]._internal[mu], dt, 12) * U_v[ss]._internal[mu]);
        });
      }

      RealD deltaS = S_eo(Uprime) - S_eo(U);

      std::cout << GridLogMessage << "dt=" << dt
                << "  deltaS=" << deltaS
                << "  dSpred=" << dSpred
                << "  ratio=" << dSpred / deltaS << std::endl;
    }
  }

  // ================================================================
  // Test 8: Nf2 force FD check (reference)
  // ================================================================
  std::cout << GridLogMessage << "=== Test 8: Nf2 force FD check (reference) ===" << std::endl;
  {
    LatticeGaugeField UdSdU_nf2(&Grid);
    Nf2.deriv(U, UdSdU_nf2);

    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_nf2, mu);
      fmu = Ta(fmu) * 2.0;
      PokeIndex<LorentzIndex>(UdSdU_nf2, fmu, mu);
    }

    for (RealD dt : {1e-3, 1e-4, 1e-5}) {
      LatticeComplex dS_lc(&Grid);
      dS_lc = Zero();
      for (int mu = 0; mu < Nd; mu++) {
        LatticeColourMatrix fmu = PeekIndex<LorentzIndex>(UdSdU_nf2, mu);
        mommu = PeekIndex<LorentzIndex>(mom, mu);
        dS_lc = dS_lc + trace(mommu * fmu) * dt;
      }
      RealD dSpred = TensorRemove(sum(dS_lc)).real();

      LatticeGaugeField Uprime(&Grid);
      for (int mu = 0; mu < Nd; mu++) {
        autoView(Uprime_v, Uprime, CpuWrite);
        autoView(U_v, U, CpuRead);
        autoView(mom_v, mom, CpuRead);
        thread_foreach(ss, U_v, {
          Uprime_v[ss]._internal[mu] =
            ProjectOnGroup(Exponentiate(mom_v[ss]._internal[mu], dt, 12) * U_v[ss]._internal[mu]);
        });
      }

      RealD Sp = Nf2.S(Uprime);
      RealD S0_nf2 = Nf2.S(U);
      RealD deltaS = Sp - S0_nf2;

      std::cout << GridLogMessage << "dt=" << dt
                << "  deltaS=" << deltaS
                << "  dSpred=" << dSpred
                << "  ratio=" << dSpred / deltaS << std::endl;
    }
  }

  std::cout << GridLogMessage << "=== QCD EO Clover tests complete ===" << std::endl;

  Grid_finalize();
  return 0;
}
