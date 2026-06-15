// Test_dtxqcd_pf_realness: γ5-Hermiticity stochastic probe of the full
// doubled DTXQCD M48 operator on a loaded configuration.
//
// Rationale: signPf reweighting requires that Pf(K·M48) be real-valued
// (so it has a well-defined sign). The chain that gives this:
//   γ5·M48·γ5 = M48†                          ← γ5-Hermiticity
//   ⇒ γ5·M48 is Hermitian
//   ⇒ det(γ5·M48) is real
//   ⇒ det(M48) is real
//   ⇒ Pf(K·M48) = ±√det(M48) is real (signed)
// If step 1 fails (e.g. in the pre-2026-06-13 Hermitian-aux setup where
// the lower-block sign was wrong), Pf is complex and sign reweighting
// by signPf=(-1)^{n_neg} is the wrong operation — full complex
// Pf/|Pf| reweighting is required.
//
// The probe: draw N random doubled noise vectors η. Compute
//   A·η  with A = γ5·M48 - M48†·γ5     (should vanish when γ5-Herm.)
// and report the relative violation
//   r = ‖A·η‖ / ‖M48·η‖
// Clean if r < 1e-10.
//
// Usage:
//   MASS=0.3 CSW=0.0 N_NOISE=4 ./Test_dtxqcd_pf_realness <cfg_dir> <traj> --grid ... --mpi ...

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>

using namespace Grid;

inline void Gamma5_doubled(DTXQCDFermionDoubled &x) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    x.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.upper.f[a];
    x.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * x.lower.f[a];
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  if (argc < 3) { std::cerr << "Usage: <cfg_dir> <traj>\n"; return 1; }
  std::string cfg_dir = argv[1];
  int traj = std::atoi(argv[2]);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian        Grid_(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid_);
  GridSerialRNG sRNG;  sRNG.SeedFixedIntegers({1});
  GridParallelRNG pRNG(&Grid_);
  pRNG.SeedFixedIntegers({2, 3, 5, 7, 11});

  DTXQCDField U(&Grid_);
  DTXQCDCheckpointer::ReadConfig(U, sRNG, pRNG,
                                  cfg_dir + "/ckpoint_lat",
                                  cfg_dir + "/ckpoint_rng", traj);

  RealD mass = 0.3, csw = 0.0;
  if (const char *s = std::getenv("MASS"); s && *s) mass = std::atof(s);
  if (const char *s = std::getenv("CSW");  s && *s) csw  = std::atof(s);
  int n_noise = 4;
  if (const char *s = std::getenv("N_NOISE"); s && *s) n_noise = std::atoi(s);

  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid_, RBGrid, mass, csw,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);

  std::cout << GridLogMessage << "cfg_dir=" << cfg_dir << " traj=" << traj
            << " mass=" << mass << " csw=" << csw << std::endl;

  RealD max_r = 0.0;
  for (int h = 0; h < n_noise; ++h) {
    DTXQCDFermionDoubled eta(&Grid_), Meta(&Grid_), g5Meta(&Grid_),
                          Mdag_g5eta(&Grid_), g5eta(&Grid_), diff(&Grid_);
    for (int a = 0; a < DtxqcdNf; ++a) {
      gaussian(pRNG, eta.upper.f[a]);
      gaussian(pRNG, eta.lower.f[a]);
    }
    // M·η
    Dw.M(eta, Meta);
    // γ5·(M·η)
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5Meta.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * Meta.upper.f[a];
      g5Meta.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * Meta.lower.f[a];
    }
    // γ5·η, then M†·(γ5·η)
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5eta.upper.f[a] = Gamma(Gamma::Algebra::Gamma5) * eta.upper.f[a];
      g5eta.lower.f[a] = Gamma(Gamma::Algebra::Gamma5) * eta.lower.f[a];
    }
    Dw.Mdag(g5eta, Mdag_g5eta);
    // diff = γ5·M·η - M†·γ5·η
    for (int a = 0; a < DtxqcdNf; ++a) {
      diff.upper.f[a] = g5Meta.upper.f[a] - Mdag_g5eta.upper.f[a];
      diff.lower.f[a] = g5Meta.lower.f[a] - Mdag_g5eta.lower.f[a];
    }
    RealD nd = std::sqrt(norm2(diff));
    RealD nm = std::sqrt(norm2(Meta));
    RealD r  = (nm > 1e-30) ? nd / nm : 0.0;
    std::cout << GridLogMessage << "  noise " << h
              << ":  ‖(γ5M - M†γ5)·η‖ / ‖M·η‖ = " << r << std::endl;
    if (r > max_r) max_r = r;
  }
  std::cout << GridLogMessage << "RESULT: max γ5-Herm violation = "
            << max_r << std::endl;
  std::cout << GridLogMessage << "RESULT: "
            << ((max_r < 1e-10) ? "CLEAN  (Pf is real, signPf reweighting OK)"
                                : "DIRTY (Pf likely complex — sign reweighting"
                                   " by ±1 is NOT the correct operation)")
            << std::endl;

  Grid_finalize();
  return 0;
}
