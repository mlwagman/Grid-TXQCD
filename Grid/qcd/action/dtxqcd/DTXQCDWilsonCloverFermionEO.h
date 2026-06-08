#pragma once
// DTXQCDWilsonCloverFermionEO: full doubled Wilson-Clover Dirac operator
// for DTXQCD, exposing the standard Grid fermion-operator interface so it
// plugs into CG / RHMC infrastructure that already works on a doubled
// fermion type.
//
// Operator structure (Cstar doubled, with corrected M_22 = C^T D^T C):
//
//   M | psi_u   = | (mass + D_W[U]_hop + Delta_diag(u) + clover_upper) psi_u
//                 + (2 d gamma_5 + 2 n) psi_l                                |
//     | psi_l   = | (mass + D_W[U^*]_hop + Delta_diag_lower(l) + clover_lower) psi_l
//                 + (2 d gamma_5 + 2 n) psi_u                                |
//
// where
//   clover_upper psi = -(csw/2) sum_{mu<nu} F_{mu,nu}    (sigma_{mu,nu} psi)
//   clover_lower psi = +(csw/2) sum_{mu<nu} F^T_{mu,nu}  (sigma_{mu,nu} psi)
//
// Field strength F_{mu,nu} is rebuilt from U via Grid's WilsonLoops at
// ImportGauge time.  Per-site 48x48 LU factorizations of Mooee are computed
// on demand by MooeeInv (v1: per-call rebuild — to be cached in a
// follow-up).
//
// gamma_5-Hermiticity: gamma_5 M gamma_5 = M^dag (validated in
// Test_dtxqcd_gamma5_herm_full).  Mdag is implemented via the identity.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/utils/WilsonLoops.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDWilsonCloverFermionEO {
 public:
  typedef WilsonImplR Impl;
  typedef DTXQCDFermionDoubled Field;
  typedef LatticeGaugeField GaugeField;

  DTXQCDWilsonCloverFermionEO(GaugeField &U,
                              GridCartesian &grid,
                              GridRedBlackCartesian &rbgrid,
                              RealD mass,
                              RealD csw,
                              const LatticeDtxqcdSigma &sigma,
                              const LatticeDtxqcdPi    &pi,
                              const LatticeDtxqcdT     &t,
                              const LatticeDtxqcdD     &d,
                              const LatticeDtxqcdN     &n)
      : mass_(mass),
        csw_(csw),
        sigma_(sigma),
        pi_(pi),
        t_(t),
        d_(d),
        n_(n),
        grid_(grid),
        rbgrid_(rbgrid),
        meooe_(U, grid, rbgrid, mass),
        FS_(),
        spin_(grid) {
    BuildFieldStrength(U);
  }

  // Refresh internal state after the caller updates U (e.g. HMC integrator).
  void ImportGauge(GaugeField &U) {
    meooe_.ImportGauge(U);
    BuildFieldStrength(U);
  }

  // -------- Full-volume apply --------

  // M psi: full operator on a full-volume doubled fermion.  Uses
  // WilsonFermion::M per block (mass + hopping) plus site-local Delta + cross
  // + clover.  Matches the assembly validated in Test_dtxqcd_gamma5_herm_full.
  void M(const Field &in, Field &out) {
    // mass + hopping per block, per flavor.
    for (int a = 0; a < DtxqcdNf; ++a) {
      meooe_.UpperWilson().M(in.upper.f[a], out.upper.f[a]);
      meooe_.LowerWilson().M(in.lower.f[a], out.lower.f[a]);
    }
    AddDiagAndCrossAndClover(in, out);
  }

  // Mdag psi via gamma_5 M gamma_5.  Cheaper than wiring a separate apply
  // and consistent with the validated gamma_5-Hermiticity.
  void Mdag(const Field &in, Field &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5_in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5_in.upper.f[a] = g5 * in.upper.f[a];
      g5_in.lower.f[a] = g5 * in.lower.f[a];
    }
    M(g5_in, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
    }
  }

  // -------- EO-decomposed pieces --------

  // Mooee psi: site-local diagonal block of M on the input checkerboard.
  // Adds mass*in + Delta_diag (upper) / Delta_diag_lower (lower) + d, n cross
  // + clover.  No hopping.
  void Mooee(const Field &in, Field &out) {
    DtxqcdApplyMooeeDoubled(mass_, sigma_, pi_, t_, d_, n_,
                            in.upper, in.lower, out.upper, out.lower,
                            csw_, (csw_ != 0.0 ? &FS_ : nullptr));
  }

  // MooeeDag psi via gamma_5 Mooee gamma_5 (same trick as Mdag; Mooee is
  // itself gamma_5-Hermitian by the per-block analysis).
  void MooeeDag(const Field &in, Field &out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5_in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5_in.upper.f[a] = g5 * in.upper.f[a];
      g5_in.lower.f[a] = g5 * in.lower.f[a];
    }
    Mooee(g5_in, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
    }
  }

  // Meooe psi: hopping piece (cross-checkerboard).  Forwards to
  // DTXQCDMeooeDoubled which wraps two WilsonFermion engines (one with U
  // for upper, one with conjugate(U) for lower).
  void Meooe(const Field &in, Field &out) {
    meooe_.Meooe(in.upper, in.lower, out.upper, out.lower);
  }
  void MeooeDag(const Field &in, Field &out) {
    meooe_.MeooeDag(in.upper, in.lower, out.upper, out.lower);
  }

  // MooeeInv psi: site-local inverse of Mooee on the input checkerboard.
  // v1 implementation: at every site of the input lattice, build the 48x48
  // dense site matrix from the aux fields + clover, LU-factor, solve, and
  // poke the result back.  Correct but slow — production needs cached LU
  // factors per site (TODO; mirrors TXQCDSiteMatrix's vectorized iMatrix
  // tensor storage).
  void MooeeInv(const Field &in, Field &out) {
    SiteWiseSolve(in, out, /*dag=*/false);
  }
  void MooeeInvDag(const Field &in, Field &out) {
    SiteWiseSolve(in, out, /*dag=*/true);
  }

  // -------- Accessors / introspection (mainly for tests) --------

  RealD Mass() const { return mass_; }
  RealD Csw()  const { return csw_; }
  const std::vector<LatticeColourMatrix> &FieldStrength() const { return FS_; }
  DTXQCDMeooeDoubled &MeooeEngine() { return meooe_; }

 private:
  // Build F_{mu,nu} for the 6 (mu<nu) pairs from U via WilsonLoops.
  // Stored in the order matching SigmaMuNuAlgebra / DtxqcdSiteClover.
  void BuildFieldStrength(GaugeField &U) {
    FS_.clear();
    FS_.reserve(6);
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(U.Grid());
        WilsonLoops<WilsonImplR>::FieldStrength(F, U, mu, nu);
        FS_.push_back(std::move(F));
      }
    }
  }

  // Site-local Delta + cross + clover, added on top of (mass + hopping) which
  // WilsonFermion::M already produced into `out`.  Used by M; not by the EO
  // pieces (Mooee handles those itself via DtxqcdApplyMooeeDoubled).
  void AddDiagAndCrossAndClover(const Field &in, Field &out) {
    GridBase *grid = in.Grid();

    DTXQCDFermionNf delta_u(grid), delta_l(grid);
    DtxqcdApplyDeltaDiag(sigma_, pi_, t_, in.upper, delta_u);
    DtxqcdApplyDeltaDiagLower(sigma_, pi_, t_, in.lower, delta_l);

    DTXQCDFermionNf cross_u(grid), cross_l(grid);
    DtxqcdApplyDnCross(d_, n_, in.lower, cross_u);
    DtxqcdApplyDnCross(d_, n_, in.upper, cross_l);

    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = out.upper.f[a] + delta_u.f[a] + cross_u.f[a];
      out.lower.f[a] = out.lower.f[a] + delta_l.f[a] + cross_l.f[a];
    }

    if (csw_ != 0.0) {
      DTXQCDFermionNf clov_u(grid), clov_l(grid);
      DtxqcdApplyCloverUpper(csw_, FS_, in.upper, clov_u);
      DtxqcdApplyCloverLower(csw_, FS_, in.lower, clov_l);
      for (int a = 0; a < DtxqcdNf; ++a) {
        out.upper.f[a] = out.upper.f[a] + clov_u.f[a];
        out.lower.f[a] = out.lower.f[a] + clov_l.f[a];
      }
    }
  }

  // Per-site 48x48 LU solve for MooeeInv / MooeeInvDag.  Iterates every
  // lattice site, builds the doubled site matrix M48 from the aux + clover
  // values at that site, packs the input doubled fermion into a 48-vector,
  // solves M48 x = b (or M48^dag x = b for dag), unpacks back.
  void SiteWiseSolve(const Field &in, Field &out, bool dag) {
    typedef typename LatticeFermion::vector_object::scalar_object SiteFerm;
    Coordinate gd(grid_.GlobalDimensions());
    Coordinate coord(Nd, 0);

    // Initialize out to zero (we'll pokeSite into the right CB sites).
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = Zero();
      out.lower.f[a] = Zero();
    }

    for (int x = 0; x < gd[0]; ++x) {
      for (int y = 0; y < gd[1]; ++y) {
        for (int z = 0; z < gd[2]; ++z) {
          for (int s = 0; s < gd[3]; ++s) {
            coord = Coordinate(std::vector<int>{x, y, z, s});

            DtxqcdSiteAux aux =
                DtxqcdSiteAux::Extract(sigma_, pi_, t_, d_, n_, coord);
            Eigen::MatrixXcd M_upper, M_lower, M_off, M48;
            if (csw_ != 0.0) {
              DtxqcdSiteClover clover = DtxqcdSiteClover::Extract(FS_, coord);
              DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper, csw_, &clover);
              DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower, csw_, &clover);
            } else {
              DtxqcdBuildUpperBlock24(mass_, aux, spin_, M_upper);
              DtxqcdBuildLowerBlock24(mass_, aux, spin_, M_lower);
            }
            DtxqcdBuildOffDiagBlock24(aux, spin_, M_off);
            DtxqcdAssembleDoubled48(M_upper, M_lower, M_off, M48);

            Eigen::VectorXcd b(kDtxqcdSiteDim48);
            for (int a = 0; a < DtxqcdNf; ++a) {
              SiteFerm su, sl;
              peekSite(su, in.upper.f[a], coord);
              peekSite(sl, in.lower.f[a], coord);
              for (int alpha = 0; alpha < Ns; ++alpha) {
                for (int i = 0; i < Nc; ++i) {
                  int iu = DtxqcdSiteIdx24(a, alpha, i);
                  int il = kDtxqcdSiteDim24 + iu;
                  b(iu) = ComplexD(TensorRemove(su()(alpha)(i)));
                  b(il) = ComplexD(TensorRemove(sl()(alpha)(i)));
                }
              }
            }

            Eigen::VectorXcd xv;
            if (dag) {
              Eigen::MatrixXcd M48d = M48.adjoint();
              xv = M48d.partialPivLu().solve(b);
            } else {
              xv = M48.partialPivLu().solve(b);
            }

            for (int a = 0; a < DtxqcdNf; ++a) {
              SiteFerm su, sl;
              su = Zero();
              sl = Zero();
              for (int alpha = 0; alpha < Ns; ++alpha) {
                for (int i = 0; i < Nc; ++i) {
                  int iu = DtxqcdSiteIdx24(a, alpha, i);
                  int il = kDtxqcdSiteDim24 + iu;
                  su()(alpha)(i) = xv(iu);
                  sl()(alpha)(i) = xv(il);
                }
              }
              pokeSite(su, out.upper.f[a], coord);
              pokeSite(sl, out.lower.f[a], coord);
            }
          }
        }
      }
    }
  }

  RealD mass_;
  RealD csw_;
  const LatticeDtxqcdSigma &sigma_;
  const LatticeDtxqcdPi    &pi_;
  const LatticeDtxqcdT     &t_;
  const LatticeDtxqcdD     &d_;
  const LatticeDtxqcdN     &n_;
  GridCartesian            &grid_;
  GridRedBlackCartesian    &rbgrid_;

  DTXQCDMeooeDoubled meooe_;
  std::vector<LatticeColourMatrix> FS_;
  DtxqcdSpinMatrices spin_;
};

NAMESPACE_END(Grid);
