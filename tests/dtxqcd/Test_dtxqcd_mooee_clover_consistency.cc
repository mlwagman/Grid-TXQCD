// Test_dtxqcd_mooee_clover_consistency: verify that the fermion-level
// doubled M_ee operator (DtxqcdApplyMooeeDoubled with clover enabled) and
// the dense 48x48 SiteMatrix (DTXQCDSiteMatrix.h with clover enabled)
// produce bit-identical results, site by site, when csw != 0 and a random
// anti-Hermitian F_{mu,nu} field strength is supplied.
//
// Together with the non-clover Test_dtxqcd_mooee_consistency, this nails
// down that the clover wiring (DTXQCDDeltaCloverOp + DtxqcdAddCloverToDiagBlock24)
// implements the same math in both representations.  Also prints
// ||M_upper - M_lower|| at a sample site to make explicit that the Cstar
// construction does produce distinct diagonal blocks (now both from the
// tensor-sign flip and from the clover F vs F^T transformation).
//
// Run: ./tests/dtxqcd/Test_dtxqcd_mooee_clover_consistency --grid 4.4.4.4 --mpi 1.1.1.1

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/Dtxqcd.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>

using namespace Grid;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  Coordinate latt(std::vector<int>{4, 4, 4, 4});
  Coordinate simd = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({81, 82, 83, 84});

  int exitcode = 0;

  DtxqcdSpinMatrices spin(Grid);

  // Aux fields.
  LatticeDtxqcdSigma sigma(&Grid);
  LatticeDtxqcdPi    pi(&Grid);
  LatticeDtxqcdT     t(&Grid);
  LatticeDtxqcdD     d(&Grid);
  LatticeDtxqcdN     n(&Grid);
  DtxqcdRealGaussian(pRNG, sigma);
  DtxqcdRealGaussian(pRNG, pi);
  DtxqcdGaussianAntisymTensor(pRNG, t);
  DtxqcdHermitianGaussian(pRNG, d);
  DtxqcdHermitianGaussian(pRNG, n);

  // Random anti-Hermitian field strength F_{mu,nu} per (mu<nu) pair.
  std::vector<LatticeColourMatrix> FS;
  FS.reserve(6);
  for (int k = 0; k < 6; ++k) {
    LatticeColourMatrix X(&Grid);
    gaussian(pRNG, X);
    LatticeColourMatrix F(&Grid);
    F = X - adj(X);
    FS.push_back(std::move(F));
  }

  // Random doubled fermion input.
  DTXQCDFermionNf in_upper(&Grid), in_lower(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    gaussian(pRNG, in_upper.f[a]);
    gaussian(pRNG, in_lower.f[a]);
  }

  const double mass = 0.4;
  const double csw  = 1.25;

  // Operator-level apply (with clover).
  DTXQCDFermionNf out_upper_op(&Grid), out_lower_op(&Grid);
  DtxqcdApplyMooeeDoubled(mass, sigma, pi, t, d, n,
                          in_upper, in_lower,
                          out_upper_op, out_lower_op,
                          csw, &FS);

  // SiteMatrix-level reference, site by site (with clover).
  DTXQCDFermionNf out_upper_ref(&Grid), out_lower_ref(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    out_upper_ref.f[a] = Zero();
    out_lower_ref.f[a] = Zero();
  }

  typedef typename LatticeFermion::vector_object::scalar_object SiteFerm;

  RealD worst_resid = 0.0;
  RealD upper_lower_diff_at_origin = 0.0;
  RealD upper_norm_at_origin       = 0.0;
  Coordinate gd(Grid.GlobalDimensions());
  for (int x = 0; x < gd[0]; ++x) {
    for (int y = 0; y < gd[1]; ++y) {
      for (int z = 0; z < gd[2]; ++z) {
        for (int tt = 0; tt < gd[3]; ++tt) {
          Coordinate coord(std::vector<int>{x, y, z, tt});

          DtxqcdSiteAux aux = DtxqcdSiteAux::Extract(sigma, pi, t, d, n, coord);
          DtxqcdSiteClover clover = DtxqcdSiteClover::Extract(FS, coord);

          MatrixXcd M_upper, M_lower, M_off, M48;
          DtxqcdBuildUpperBlock24(mass, aux, spin, M_upper, csw, &clover);
          DtxqcdBuildLowerBlock24(mass, aux, spin, M_lower, csw, &clover);
          DtxqcdBuildOffDiagBlock24(aux, spin, M_off);
          DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);

          // Sanity print at the origin: how different are upper and lower now?
          if (x == 0 && y == 0 && z == 0 && tt == 0) {
            upper_lower_diff_at_origin = (M_upper - M_lower).norm();
            upper_norm_at_origin       = M_upper.norm();
          }

          // Pack input fermion at this site into the 48-vector.
          VectorXcd in_vec(kDtxqcdSiteDim48);
          for (int a = 0; a < DtxqcdNf; ++a) {
            SiteFerm su, sl;
            peekSite(su, in_upper.f[a], coord);
            peekSite(sl, in_lower.f[a], coord);
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int i = 0; i < Nc; ++i) {
                int iu = DtxqcdSiteIdx24(a, alpha, i);
                int il = kDtxqcdSiteDim24 + iu;
                in_vec(iu) = ComplexD(TensorRemove(su()(alpha)(i)));
                in_vec(il) = ComplexD(TensorRemove(sl()(alpha)(i)));
              }
            }
          }
          VectorXcd out_vec = M48 * in_vec;
          for (int a = 0; a < DtxqcdNf; ++a) {
            SiteFerm su, sl;
            su = Zero();
            sl = Zero();
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int i = 0; i < Nc; ++i) {
                int iu = DtxqcdSiteIdx24(a, alpha, i);
                int il = kDtxqcdSiteDim24 + iu;
                su()(alpha)(i) = out_vec(iu);
                sl()(alpha)(i) = out_vec(il);
              }
            }
            pokeSite(su, out_upper_ref.f[a], coord);
            pokeSite(sl, out_lower_ref.f[a], coord);
          }

          // Per-site worst residual monitor.
          for (int a = 0; a < DtxqcdNf; ++a) {
            SiteFerm su_op, sl_op, su_ref, sl_ref;
            peekSite(su_op,  out_upper_op.f[a],  coord);
            peekSite(sl_op,  out_lower_op.f[a],  coord);
            peekSite(su_ref, out_upper_ref.f[a], coord);
            peekSite(sl_ref, out_lower_ref.f[a], coord);
            for (int alpha = 0; alpha < Ns; ++alpha) {
              for (int i = 0; i < Nc; ++i) {
                ComplexD du =
                    ComplexD(TensorRemove(su_op()(alpha)(i)))
                  - ComplexD(TensorRemove(su_ref()(alpha)(i)));
                ComplexD dl =
                    ComplexD(TensorRemove(sl_op()(alpha)(i)))
                  - ComplexD(TensorRemove(sl_ref()(alpha)(i)));
                worst_resid = std::max(worst_resid, std::abs(du));
                worst_resid = std::max(worst_resid, std::abs(dl));
              }
            }
          }
        }
      }
    }
  }

  // Full-lattice L2 residual (deterministic Grid reduction).
  DTXQCDFermionNf delta_upper(&Grid), delta_lower(&Grid);
  for (int a = 0; a < DtxqcdNf; ++a) {
    delta_upper.f[a] = out_upper_op.f[a] - out_upper_ref.f[a];
    delta_lower.f[a] = out_lower_op.f[a] - out_lower_ref.f[a];
  }
  RealD l2_resid_sq = norm2(delta_upper) + norm2(delta_lower);
  RealD ref_sq      = norm2(out_upper_ref) + norm2(out_lower_ref);
  RealD rel_l2 = std::sqrt(l2_resid_sq / std::max(ref_sq, 1e-30));

  auto check = [&](const char *name, RealD val, RealD tol) {
    bool ok = (val <= tol);
    std::cout << GridLogMessage << "[" << (ok ? "ok" : "FAIL") << "] " << name
              << " = " << val << " (tol " << tol << ")" << std::endl;
    if (!ok) exitcode = 1;
  };

  std::cout << GridLogMessage
            << "M_upper, M_lower @ origin: ||M_upper|| = " << upper_norm_at_origin
            << "  ||M_upper - M_lower|| = " << upper_lower_diff_at_origin
            << "  rel = "
            << (upper_lower_diff_at_origin / std::max(upper_norm_at_origin, 1.0))
            << std::endl;
  if (upper_lower_diff_at_origin < 1e-8 * std::max(upper_norm_at_origin, 1.0)) {
    std::cout << GridLogError
              << "[FAIL] M_upper and M_lower agree to 1e-8 — clover or tensor"
              << " distinction may be lost" << std::endl;
    exitcode = 1;
  } else {
    std::cout << GridLogMessage
              << "[ok] M_upper != M_lower (Cstar construction is active)"
              << std::endl;
  }
  check("op vs SiteMatrix worst per-component residual", worst_resid, 1e-11);
  check("op vs SiteMatrix full-lattice relative L2",     rel_l2,      1e-12);

  std::cout << GridLogMessage
            << (exitcode ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED")
            << std::endl;
  Grid_finalize();
  return exitcode;
}
