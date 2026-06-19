// Test_dtxqcd_zero_aux_doubled_qcd: when aux = 0 and csw = 0, the doubled
// DTXQCD operator decouples into two independent single-flavor Wilson
// blocks.  The hopping path uses conj(U) on the lower block (see
// DTXQCDMeooeOp.h:DtxqcdConjugateGauge), so:
//
//   M_dtxqcd[U, aux=0] (ψ_u, ψ_l) = (D_W[U] ψ_u, D_W[U*] ψ_l)
//
// where U* = conjugate(U).  Mooee and MooeeInv at aux=0, csw=0 are
// (m+4)·I per block — U-independent — so both upper and lower reduce
// to the stock Wilson Mooee on U.
//
// Run: ./tests/dtxqcd/Test_dtxqcd_zero_aux_doubled_qcd --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({601, 602, 603, 604});

  int exitcode = 0;
  auto check = [&](const char *name, RealD diff, RealD tol) {
    bool ok = (diff < tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << "  rel = " << diff << "  (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  // Random hot gauge; aux = 0 exactly.
  DTXQCDField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U.U);
  U.sigma = Zero();  U.pi = Zero();
  U.d     = Zero();  U.n  = Zero();
  U.s     = Zero();  U.p  = Zero();

  // DTXQCD operator (csw = 0 = Wilson, no clover).
  const RealD mass = 0.3;
  DTXQCDWilsonCloverFermionEO Dw(U.U, Grid, RBGrid, mass, /*csw=*/0.0,
                                  U.sigma, U.pi, U.d, U.n, U.s, U.p);

  // Stock Grid WilsonFermion on U; lower-block reference is conj(D_W[U]·conj(·))
  // applied to ψ_l (the matrix conjugate of the upper Wilson op).
  WilsonImplR::ImplParams ip;
  ip.boundary_phases.resize(Nd, 1.0);
  ip.boundary_phases[Nd - 1] = -1.0;  // APBC time matches DTXQCD default
  LatticeGaugeField U_lower(&Grid);
  U_lower = conjugate(U.U);
  WilsonFermion<WilsonImplR> Dw_upper(U.U,     Grid, RBGrid, mass, ip);
  WilsonFermion<WilsonImplR> Dw_lower(U_lower, Grid, RBGrid, mass, ip);

  // Random doubled fermion on full grid.
  DTXQCDFermionDoubled psi(&Grid), out_dtxqcd(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, psi.upper.f[a]);
    gaussian(pRNG, psi.lower.f[a]);
  }

  // ---------- check 1: full-volume M ----------
  Dw.M(psi, out_dtxqcd);
  RealD worst_rel_M = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion ref_u(&Grid), ref_l(&Grid);
    Dw_upper.M(psi.upper.f[a], ref_u);
    Dw_lower.M(psi.lower.f[a], ref_l);
    LatticeFermion du = out_dtxqcd.upper.f[a] - ref_u;
    LatticeFermion dl = out_dtxqcd.lower.f[a] - ref_l;
    RealD ru = std::sqrt(norm2(du) / std::max(norm2(ref_u), 1e-30));
    RealD rl = std::sqrt(norm2(dl) / std::max(norm2(ref_l), 1e-30));
    worst_rel_M = std::max({worst_rel_M, ru, rl});
  }
  check("DTXQCD.M[aux=0] vs (D_W[U], D_W[U*])", worst_rel_M, 1e-13);

  // ---------- check 2: EO Mooee on a CB fermion ----------
  // Build CB copies and call the EO Mooee on each side; the doubled
  // block should split into Wilson Mooee per block at aux=0.
  DTXQCDFermionDoubled psi_e(&RBGrid), out_e(&RBGrid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    pickCheckerboard(Even, psi_e.upper.f[a], psi.upper.f[a]);
    pickCheckerboard(Even, psi_e.lower.f[a], psi.lower.f[a]);
  }
  Dw.Mooee(psi_e, out_e);

  RealD worst_rel_Mooee = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion ref_u(&RBGrid), ref_l(&RBGrid);
    Dw_upper.Mooee(psi_e.upper.f[a], ref_u);
    Dw_upper.Mooee(psi_e.lower.f[a], ref_l);  // Mooee = (m+4)·I, U-independent
    LatticeFermion du = out_e.upper.f[a] - ref_u;
    LatticeFermion dl = out_e.lower.f[a] - ref_l;
    du.Checkerboard() = Even;
    dl.Checkerboard() = Even;
    RealD ru = std::sqrt(norm2(du) / std::max(norm2(ref_u), 1e-30));
    RealD rl = std::sqrt(norm2(dl) / std::max(norm2(ref_l), 1e-30));
    worst_rel_Mooee = std::max({worst_rel_Mooee, ru, rl});
  }
  check("DTXQCD.Mooee[aux=0] vs (Wilson.Mooee[U], Wilson.Mooee[U])",
        worst_rel_Mooee, 1e-13);

  // ---------- check 3: EO MooeeInv on a CB fermion ----------
  // det(Mooee) per block, so MooeeInv must also split as
  // (Wilson.MooeeInv[U], Wilson.MooeeInv[U*]).
  DTXQCDFermionDoubled inv_e(&RBGrid);
  Dw.MooeeInv(psi_e, inv_e);

  RealD worst_rel_MooeeInv = 0.0;
  for (int a = 0; a < DtxqcdNf; ++a) {
    LatticeFermion ref_u(&RBGrid), ref_l(&RBGrid);
    Dw_upper.MooeeInv(psi_e.upper.f[a], ref_u);
    Dw_upper.MooeeInv(psi_e.lower.f[a], ref_l);  // MooeeInv = 1/(m+4)·I
    LatticeFermion du = inv_e.upper.f[a] - ref_u;
    LatticeFermion dl = inv_e.lower.f[a] - ref_l;
    du.Checkerboard() = Even;
    dl.Checkerboard() = Even;
    RealD ru = std::sqrt(norm2(du) / std::max(norm2(ref_u), 1e-30));
    RealD rl = std::sqrt(norm2(dl) / std::max(norm2(ref_l), 1e-30));
    worst_rel_MooeeInv = std::max({worst_rel_MooeeInv, ru, rl});
  }
  check("DTXQCD.MooeeInv[aux=0] vs (Wilson.MooeeInv[U], Wilson.MooeeInv[U])",
        worst_rel_MooeeInv, 1e-10);

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
