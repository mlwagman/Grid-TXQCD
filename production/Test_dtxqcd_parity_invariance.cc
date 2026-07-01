// Test_dtxqcd_parity_invariance
//
// Verify that S = S_gauge(U) + S_aux(sigma, pi, d, n, s, p) +
//                 Phi^dagger M^dagger M Phi
// is invariant on a random 4^4 configuration under the discrete parity
//
//   P: (x, y, z, t) -> (-x, -y, -z, t)
//
// with the field transformations documented in DTXQCDParity.h:
//   Gauge  U_0(x, t) -> U_0(-x, t)
//         U_k(x, t) -> U_k(-x - k̂, t)^dagger        (k = 1, 2, 3)
//   Aux even (unchanged): sigma, s, d
//   Aux odd  (flipped):   pi,    p, n
//   Fermion chi(x, t) -> gamma_T chi(-x, t)
//
// Success threshold: all three action pieces bit-invariant to 1e-10 relative.
//
// Rationale: this test is the required correctness gate for the Z_2 heatbath
// aux flip (project_dtxqcd_low_lambda_isosinglet_condensate).  If the
// parity transformation is not an exact symmetry of the action, using it
// as a Metropolis-trivial post-accept flip would bias the ensemble.
//
// Run:
//   ./Test_dtxqcd_parity_invariance --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDParity.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <cstdlib>

using namespace Grid;

// Random gaussian on a full LatticeFermion.
static void GaussianFermion(GridParallelRNG &rng, LatticeFermion &out) {
  gaussian(rng, out);
}

static void GaussianDoubled(GridParallelRNG &rng, DTXQCDFermionDoubled &out) {
  for (int a = 0; a < DtxqcdNf; ++a) {
    GaussianFermion(rng, out.upper.f[a]);
    GaussianFermion(rng, out.lower.f[a]);
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--grid" && i + 1 < argc) {
      std::vector<int> d;
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, '.')) d.push_back(std::stoi(tok));
      if (d.size() == (size_t)Nd) latt = Coordinate(d);
    }
  }
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);

  int exitcode = 0;
  auto check = [&](const char *name, RealD value, RealD target, RealD tol) {
    RealD scale = std::max(std::abs(target), 1.0);
    bool ok = (std::abs(value - target) / scale < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << std::scientific << std::setprecision(16) << value
              << "  target " << target
              << "  rel diff " << std::abs(value - target) / scale
              << "  tol " << tol << std::endl;
    if (!ok) exitcode = 1;
  };

  // ---------- Random init: gauge + aux + pseudofermion ----------
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({601, 602, 603, 604});

  DTXQCDField F(&Grid);
  SU<Nc>::HotConfiguration(pRNG, F.U);
  DtxqcdHermitianCFGaussian(pRNG, F.sigma);
  DtxqcdHermitianCFGaussian(pRNG, F.pi);
  DtxqcdHermitianCFGaussian(pRNG, F.d);
  DtxqcdHermitianCFGaussian(pRNG, F.n);
  DtxqcdRealScalarGaussian (pRNG, F.s);
  DtxqcdRealScalarGaussian (pRNG, F.p);
  if (DtxqcdDnComplexSymmetric()) {
    DtxqcdRealSymmetricCFInPlace(F.sigma);
    DtxqcdRealSymmetricCFInPlace(F.pi);
    DtxqcdComplexSymmetricCFGaussian(pRNG, F.d);
    DtxqcdComplexSymmetricCFGaussian(pRNG, F.n);
  }

  DTXQCDFermionDoubled Phi(&Grid);
  GaussianDoubled(pRNG, Phi);

  const RealD mass = 0.4;
  const RealD csw  = 1.24930970916466;
  const RealD lambda = 0.5;
  const RealD beta   = 6.1;

  // ---------- Reference actions on (F, Phi) ----------
  auto Sgauge = [&](const DTXQCDField &G) -> RealD {
    // Wilson plaquette action:  S = beta * sum_x sum_{mu<nu} [1 - Re Tr P_{mu nu}(x)/Nc]
    // (Grid provides WilsonLoops::avgPlaquette; convert to action value.)
    RealD plaq = WilsonLoops<PeriodicGimplR>::avgPlaquette(G.U);
    const int Vd = G.U.Grid()->gSites();
    const int nPlaq = Nd * (Nd - 1) / 2;
    return beta * (1.0 - plaq) * nPlaq * Vd;
  };

  auto Saux = [&](const DTXQCDField &G) -> RealD {
    DTXQCDAuxiliaryFieldGaussianAction ag(lambda);
    return ag.S(const_cast<DTXQCDField &>(G));
  };

  auto Sferm = [&](const DTXQCDField &G,
                   const DTXQCDFermionDoubled &Ph) -> RealD {
    // Naive bilinear Phi^dag M^dag M Phi.  No CG: this is a bit-exact check
    // (up to floating-point summation order across the site loop).
    // Use a mutable copy of the gauge because the fermion op stores by
    // reference / needs a mutable handle to construct its stencils.
    LatticeGaugeField Umutable = G.U;
    DTXQCDWilsonCloverFermionEO Dw(Umutable, Grid, RBGrid, mass, csw,
                                    G.sigma, G.pi, G.d, G.n, G.s, G.p);
    DTXQCDFermionDoubled MPhi(&Grid), MdMPhi(&Grid);
    Dw.M   (Ph,   MPhi);
    Dw.Mdag(MPhi, MdMPhi);
    return real(innerProduct(Ph, MdMPhi));
  };

  const RealD Sg0 = Sgauge(F);
  const RealD Sa0 = Saux(F);
  const RealD Sf0 = Sferm(F, Phi);
  std::cout << GridLogMessage
            << "reference: Sgauge=" << Sg0
            << "  Saux=" << Sa0
            << "  Sferm=Phi^dag M^dag M Phi=" << Sf0
            << std::endl;

  // ---------- Apply parity to all fields ----------
  DTXQCDField Fp(&Grid);
  Fp = F;
  DtxqcdParity::ApplyField(Fp);

  DTXQCDFermionDoubled Phi_p(&Grid);
  Phi_p = Phi;
  DtxqcdParity::ApplyFermionDoubled(Phi_p);

  // ---------- Post-parity actions ----------
  const RealD Sg1 = Sgauge(Fp);
  const RealD Sa1 = Saux(Fp);
  const RealD Sf1 = Sferm(Fp, Phi_p);
  std::cout << GridLogMessage
            << "post-P:    Sgauge=" << Sg1
            << "  Saux=" << Sa1
            << "  Sferm=" << Sf1
            << std::endl;

  // ---------- Individual invariance checks ----------
  check("S_gauge invariant", Sg1, Sg0, 1.0e-12);
  check("S_aux   invariant", Sa1, Sa0, 1.0e-12);
  // Fermion bilinear: same operator but with reordered site sums; expect
  // agreement to O(1e-12) at 4^4 in double precision.
  check("S_ferm  invariant", Sf1, Sf0, 1.0e-10);

  // ---------- Sanity: idempotency P^2 == identity for gauge + aux ----------
  DTXQCDField Fpp(&Grid);
  Fpp = Fp;
  DtxqcdParity::ApplyField(Fpp);
  DTXQCDFermionDoubled Phi_pp(&Grid);
  Phi_pp = Phi_p;
  DtxqcdParity::ApplyFermionDoubled(Phi_pp);

  // P^2 acts as: coord identity, sign flip twice on odd aux (identity), and
  // gamma_T^2 = 1 on the fermion.  For gauge: U_k -> [U_k^dag(-x-k̂)]^dag at
  // (-x-k̂-k̂) ... which is U_k(-x-k̂-k̂-k̂) = U_k at shifted position.
  // The temporal link goes back exactly; the spatial link idempotency
  // requires 2*k̂ shift compensation, which our current definition
  // provides.  Verify by re-computing the same actions.
  const RealD Sg2 = Sgauge(Fpp);
  const RealD Sa2 = Saux(Fpp);
  const RealD Sf2 = Sferm(Fpp, Phi_pp);
  check("S_gauge P^2==id", Sg2, Sg0, 1.0e-12);
  check("S_aux   P^2==id", Sa2, Sa0, 1.0e-12);
  check("S_ferm  P^2==id", Sf2, Sf0, 1.0e-10);

  Grid_finalize();
  return exitcode;
}
