// Test_dtxqcd_meooe_free_field: at U = I (identity links), the DTXQCD
// doubled Meooe must satisfy:
//   1. Upper block == lower block per flavor:  U^* = I = U, so the two
//      Wilson hoppings are bit-identical.
//   2. Each block matches what a stock single-flavor WilsonFermion gives
//      at the same gauge field — i.e. our wrapping doesn't accidentally
//      perturb the hopping kernel.
//
// At random non-trivial U we then verify:
//   3. Upper != lower (sanity that conjugate(U) actually differs from U).
//   4. The lower-block output matches stock WilsonFermion built directly
//      with conjugate(U).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_meooe_free_field --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian         Grid(latt, simd, mpi);
  GridRedBlackCartesian RBGrid(&Grid);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({201, 202, 203, 204});

  int exitcode = 0;
  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  const RealD mass = 0.3;

  // Random doubled fermion (input on even checkerboard, will be hopped to odd).
  DTXQCDFermionNf in_upper(&Grid), in_lower(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, in_upper.f[a]);
    gaussian(pRNG, in_lower.f[a]);
  }

  // Build red-black checkerboard fermions for Meooe (Grid convention).
  DTXQCDFermionNf in_upper_e(&RBGrid), in_lower_e(&RBGrid);
  DTXQCDFermionNf out_upper_o(&RBGrid), out_lower_o(&RBGrid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    pickCheckerboard(Even, in_upper_e.f[a], in_upper.f[a]);
    pickCheckerboard(Even, in_lower_e.f[a], in_lower.f[a]);
    out_upper_o.f[a].Checkerboard() = Odd;
    out_lower_o.f[a].Checkerboard() = Odd;
  }

  // ---------- 1, 2.  Free field (U = I): upper == lower on SAME input ----------
  {
    LatticeGaugeField U(&Grid);
    SU<Nc>::ColdConfiguration(U);  // U_mu(x) = I_3 everywhere

    DTXQCDMeooeDoubled meooe(U, Grid, RBGrid, mass);
    // Pass the same fermion as both upper and lower input — at U = I,
    // the two block kernels should produce bit-identical output.
    meooe.Meooe(in_upper_e, in_upper_e, out_upper_o, out_lower_o);

    RealD worst_rel = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion diff(&RBGrid);
      diff.Checkerboard() = Odd;
      diff = out_upper_o.f[a] - out_lower_o.f[a];
      RealD r = std::sqrt(norm2(diff))
              / std::max(std::sqrt(norm2(out_upper_o.f[a])), 1e-30);
      worst_rel = std::max(worst_rel, r);
    }
    check("U=I, same input: ||upper - lower|| / ||upper||", worst_rel, 1e-14);

    // Confirm wrapping matches stock single-flavor WilsonFermion.
    WilsonFermion<WilsonImplR> Dw_ref(U, Grid, RBGrid, mass);
    RealD worst_ref = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion ref_out(&RBGrid);
      ref_out.Checkerboard() = Odd;
      Dw_ref.Meooe(in_upper_e.f[a], ref_out);
      LatticeFermion diff(&RBGrid);
      diff.Checkerboard() = Odd;
      diff = out_upper_o.f[a] - ref_out;
      RealD r = std::sqrt(norm2(diff))
              / std::max(std::sqrt(norm2(ref_out)), 1e-30);
      worst_ref = std::max(worst_ref, r);
    }
    check("U=I: ||upper - stock Wilson|| / ||stock||", worst_ref, 1e-14);
  }

  // ---------- 3, 4.  Random U: upper != lower, lower matches stock-on-conjugate ----------
  {
    LatticeGaugeField U(&Grid);
    SU<Nc>::HotConfiguration(pRNG, U);

    DTXQCDMeooeDoubled meooe(U, Grid, RBGrid, mass);
    meooe.Meooe(in_upper_e, in_lower_e, out_upper_o, out_lower_o);

    // Upper vs lower should differ.
    RealD up_lo_rel = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion diff(&RBGrid);
      diff.Checkerboard() = Odd;
      diff = out_upper_o.f[a] - out_lower_o.f[a];
      RealD r = std::sqrt(norm2(diff))
              / std::max(std::sqrt(norm2(out_upper_o.f[a])), 1e-30);
      up_lo_rel = std::max(up_lo_rel, r);
    }
    std::cout << GridLogMessage
              << "Random U: ||upper - lower|| / ||upper|| (worst flavor) = "
              << up_lo_rel << std::endl;
    if (up_lo_rel < 1e-3) {
      std::cout << GridLogError
                << "[FAIL] Random U: upper and lower agree to 1e-3 — "
                << "conjugate(U) appears not to differ from U" << std::endl;
      exitcode = 1;
    } else {
      std::cout << GridLogMessage
                << "[ok] Random U: upper != lower (conjugate(U) is nontrivial)"
                << std::endl;
    }

    // Lower block must match stock WilsonFermion on conjugate(U).
    LatticeGaugeField U_conj(&Grid);
    U_conj = conjugate(U);
    WilsonFermion<WilsonImplR> Dw_ref_lower(U_conj, Grid, RBGrid, mass);
    RealD worst_lo_ref = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion ref_out(&RBGrid);
      ref_out.Checkerboard() = Odd;
      Dw_ref_lower.Meooe(in_lower_e.f[a], ref_out);
      LatticeFermion diff(&RBGrid);
      diff.Checkerboard() = Odd;
      diff = out_lower_o.f[a] - ref_out;
      RealD r = std::sqrt(norm2(diff))
              / std::max(std::sqrt(norm2(ref_out)), 1e-30);
      worst_lo_ref = std::max(worst_lo_ref, r);
    }
    check("Random U: ||lower - stock Wilson(U*)|| / ||stock||",
          worst_lo_ref, 1e-13);

    // Upper block must match stock WilsonFermion on U.
    WilsonFermion<WilsonImplR> Dw_ref_upper(U, Grid, RBGrid, mass);
    RealD worst_up_ref = 0.0;
    for (int a = 0; a < DtxqcdNf; ++a) {
      LatticeFermion ref_out(&RBGrid);
      ref_out.Checkerboard() = Odd;
      Dw_ref_upper.Meooe(in_upper_e.f[a], ref_out);
      LatticeFermion diff(&RBGrid);
      diff.Checkerboard() = Odd;
      diff = out_upper_o.f[a] - ref_out;
      RealD r = std::sqrt(norm2(diff))
              / std::max(std::sqrt(norm2(ref_out)), 1e-30);
      worst_up_ref = std::max(worst_up_ref, r);
    }
    check("Random U: ||upper - stock Wilson(U)|| / ||stock||",
          worst_up_ref, 1e-13);
  }

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
