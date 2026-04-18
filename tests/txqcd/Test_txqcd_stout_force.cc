// Force test for stout-smeared TXQCD actions.
//
// Verifies the smeared force chain rule by finite differences:
// perturb thin links U -> exp(hE)U, recompute S on smeared config, compare
// with analytic force from deriv() + smeared_force().
//
// Tests:
// 1. RHMC PF action with stout smearing (gauge + aux FD)
// 2. LogDet action with stout smearing (gauge FD)
// 3. Gauge action with stout smearing (gauge FD)

#include <Grid/Grid.h>
#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/qcd/action/txqcd/TXQCDLogDetCloverEOAction.h>
#include <Grid/qcd/action/txqcd/GaugeActionAdapter.h>
#include <Grid/qcd/action/txqcd/TXQCDSmearedConfiguration.h>

using namespace Grid;

static RealD HermitianTrInner(const LatticeSigmaField &E,
                              const LatticeSigmaField &F) {
  return TensorRemove(sum(trace(E * F))).real();
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  GridSerialRNG sRNG;
  pRNG.SeedFixedIntegers({11, 12, 13, 14});
  sRNG.SeedFixedIntegers({11, 12, 13, 14});

  TXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  const RealD aux_scale = 0.2;
  HermitianGaussian(pRNG, U.sigma); U.sigma = aux_scale * U.sigma;
  HermitianGaussian(pRNG, U.pi);    U.pi    = aux_scale * U.pi;
  HermitianGaussian(pRNG, U.s);     U.s     = aux_scale * U.s;
  HermitianGaussian(pRNG, U.p);     U.p     = aux_scale * U.p;
  GaussianAntisymTensor(pRNG, U.t); U.t     = aux_scale * U.t;

  RealD mass = 0.3;
  RealD csw = 1.0;
  RealD rho = 0.1;
  int Nsmear = 2;
  RealD beta = 5.6;

  Smear_Stout<PeriodicGimplR> Stout(rho);
  TXQCDSmearedConfiguration Smearer(&Grid, Nsmear, Stout);

  OneFlavourRationalParams rat_params(1e-4, 64.0, 10000, 1e-10,
                                      12, 64, 100, 1e-8, 1e-4);

  int exitcode = 0;

  auto check = [&](const char *name, RealD an, RealD fd) {
    RealD rel = std::abs(an - fd) / std::max(std::abs(fd), 1.0);
    bool pass = rel < 1e-3;
    std::cout << GridLogMessage << "[" << name << "] AN=" << an
              << " FD=" << fd << " rel=" << rel
              << (pass ? "  PASS" : "  FAIL") << std::endl;
    if (!pass) exitcode = 1;
  };

  typedef Action<TXQCDField> ActionBase;

  // Helper: compute S on smeared config from current thin links
  auto smeared_S = [&](ActionBase &action) -> RealD {
    Smearer.set_Field(U);
    return action.S(Smearer);
  };

  // Helper: gauge FD on thin links through smearing
  auto gauge_fd = [&](ActionBase &action,
                      std::array<LatticeColourMatrix, 4> &Emu) -> RealD {
    const RealD h = 1e-4;
    LatticeGaugeField Usaved = U.U;

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], h, 12) * Umu, mu);
    }
    Smearer.set_Field(U);
    RealD Sp = action.S(Smearer);

    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Umu = PeekIndex<LorentzIndex>(Usaved, mu);
      PokeIndex<LorentzIndex>(U.U, expMat(Emu[mu], -h, 12) * Umu, mu);
    }
    Smearer.set_Field(U);
    RealD Sm = action.S(Smearer);

    U.U = Usaved;
    return (Sp - Sm) / (2.0 * h);
  };

  // Random gauge-algebra perturbation
  std::array<LatticeColourMatrix, 4> Emu{
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid),
      LatticeColourMatrix(&Grid), LatticeColourMatrix(&Grid)};
  for (int mu = 0; mu < Nd; ++mu)
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, Emu[mu]);

  // ===== Test 1: RHMC PF with stout smearing =====
  std::cout << GridLogMessage
            << "===== RHMC PF with stout smearing (rho=" << rho
            << ", Nsmear=" << Nsmear << ", csw=" << csw << ") =====" << std::endl;
  {
    TXQCDWilsonCloverRationalEOAction PF(Grid, RBGrid, mass, rat_params, csw);
    PF.is_smeared = true;

    ActionBase &PFbase = PF;
    Smearer.set_Field(U);
    PFbase.refresh(Smearer, sRNG, pRNG);

    TXQCDField dSdU(&Grid);
    PFbase.deriv(Smearer, dSdU);

    // Gauge force: Convention A, dS/dh = -2 Re Tr(E F_A)
    RealD an = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;
    RealD fd = gauge_fd(PF, Emu);
    check("RHMC gauge (smeared)", an, fd);

    // Aux force: perturb sigma on thin config, S evaluated on smeared
    {
      LatticeSigmaField Esig(&Grid);
      HermitianGaussian(pRNG, Esig);
      RealD an_sig = HermitianTrInner(Esig, dSdU.sigma);
      const RealD h = 1e-4;
      auto saved = U.sigma;
      U.sigma = U.sigma + h * Esig;  RealD Sp = smeared_S(PFbase);
      U.sigma = saved;
      U.sigma = U.sigma - h * Esig;  RealD Sm = smeared_S(PFbase);
      U.sigma = saved;
      RealD fd_sig = (Sp - Sm) / (2.0 * h);
      check("RHMC sigma (smeared)", an_sig, fd_sig);
    }
  }

  // ===== Test 2: LogDet with stout smearing =====
  std::cout << GridLogMessage
            << "===== LogDet with stout smearing =====" << std::endl;
  {
    TXQCDLogDetCloverEOAction LD(Grid, RBGrid, mass, csw);
    LD.is_smeared = true;
    ActionBase &LDbase = LD;

    Smearer.set_Field(U);

    TXQCDField dSdU(&Grid);
    LDbase.deriv(Smearer, dSdU);

    RealD an = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;
    RealD fd = gauge_fd(LD, Emu);
    check("LogDet gauge (smeared)", an, fd);
  }

  // ===== Test 3: Wilson gauge action with stout smearing =====
  std::cout << GridLogMessage
            << "===== Wilson gauge with stout smearing =====" << std::endl;
  {
    GaugeActionAdapter<WilsonGaugeActionR> GA(beta);
    GA.is_smeared = true;
    ActionBase &GAbase = GA;

    Smearer.set_Field(U);

    TXQCDField dSdU(&Grid);
    GAbase.deriv(Smearer, dSdU);

    RealD an = 0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrix Fmu = PeekIndex<LorentzIndex>(dSdU.U, mu);
      an += TensorRemove(sum(trace(Emu[mu] * Fmu))).real();
    }
    an *= -2.0;
    RealD fd = gauge_fd(GA, Emu);
    check("WilsonGauge (smeared)", an, fd);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME STOUT FORCE CHECKS FAILED"
                         : "ALL STOUT FORCE CHECKS PASSED")
            << std::endl;

  Grid_finalize();
  return exitcode;
}
