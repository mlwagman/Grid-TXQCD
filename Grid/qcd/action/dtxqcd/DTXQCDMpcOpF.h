#pragma once
// Single-precision Schur-complement (Mpc) operator on the odd sub-lattice for
// the DTXQCD doubled Wilson-Clover fermion -- the inner-iteration matrix
// multiply of the reliable-update mixed-precision multishift CG
// (DTXQCDMultiShiftCGMixedPrec.h).
//
// Numerically a faithful single-precision twin of DTXQCDMpcOp:
//
//   Mpc psi_o = Mooee psi_o - Meooe (MooeeInv (Meooe psi_o))
//             = (M_oo - M_oe M_ee^{-1} M_eo) psi_o
//
// Built directly from a fully-imported double-precision
// DTXQCDWilsonCloverFermionEO: the SP per-site 48x48 forward / inverse SIMD
// caches are precisionChange downcasts of the DP caches (built once by
// DTXQCDWilsonCloverFermionEO::EnableSinglePrec()), and the doubled Wilson
// hopping runs on two stock Grid WilsonFermionF engines fed the SP gauge
// field U (upper) and conj(U) (lower).  No DTXQCD-specific assembly is
// re-derived here -- only the gauge / cache downcasts -- so all the
// gate-defended sigmaHerm / clover-sign / d-n conventions stay single-source
// in the DP operator.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubledF.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>
#include <Grid/qcd/action/fermion/WilsonFermion.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDMpcOpF {
 public:
  typedef DTXQCDFermionDoubledF Field;
  typedef WilsonImplF Impl;
  typedef WilsonFermion<Impl> WilsonOpF;
  typedef DTXQCDWilsonCloverFermionEO::InvFieldF InvFieldF;

  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  // Construct the SP operator from a DP operator that has already imported
  // gauge + aux and has SP caches enabled (EnableSinglePrec()).  Holds SP
  // single-precision gauge fields and two WilsonFermionF hopping engines.
  explicit DTXQCDMpcOpF(DTXQCDWilsonCloverFermionEO &Dw)
      : grid_f_(Dw.Grid().FullDimensions(),
                GridDefaultSimd(Dw.Grid().Nd(), vComplexF::Nsimd()),
                Dw.Grid().ProcessorGrid()),
        rbgrid_f_(&grid_f_),
        U_f_(&grid_f_),
        U_conj_f_(&grid_f_),
        Dw_upper_f_(U_f_, grid_f_, rbgrid_f_, Dw.Mass(),
                    DTXQCDMeooeDoubled::DefaultImplParams()),
        Dw_lower_f_(U_conj_f_, grid_f_, rbgrid_f_, Dw.Mass(),
                    DTXQCDMeooeDoubled::DefaultImplParams()),
        inv_f_e_(&rbgrid_f_), inv_f_o_(&rbgrid_f_),
        fwd_f_e_(&rbgrid_f_), fwd_f_o_(&rbgrid_f_),
        mass_(Dw.Mass()) {
    GRID_ASSERT(Dw.SinglePrecEnabled() &&
                "DTXQCDMpcOpF requires DTXQCDWilsonCloverFermionEO::EnableSinglePrec()");
    // Downcast gauge (cross-grid precisionChange: DP grid -> SP grid) + import
    // the SP Wilson hopping engines.
    precisionChange(U_f_, Dw.Gauge());
    precisionChange(U_conj_f_, Dw.ConjugatedGauge());
    Dw_upper_f_.ImportGauge(U_f_);
    Dw_lower_f_.ImportGauge(U_conj_f_);
    // Downcast the DP 48x48 SIMD caches (built on Dw's DP rbgrid) into the SP
    // caches on this operator's SP rbgrid.  precisionChange handles the SIMD
    // re-layout + precision drop across the two grids.
    precisionChange(inv_f_e_, Dw.InvCache(Even));
    precisionChange(inv_f_o_, Dw.InvCache(Odd));
    precisionChange(fwd_f_e_, Dw.FwdCache(Even));
    precisionChange(fwd_f_o_, Dw.FwdCache(Odd));
    inv_f_e_.Checkerboard() = Even;  inv_f_o_.Checkerboard() = Odd;
    fwd_f_e_.Checkerboard() = Even;  fwd_f_o_.Checkerboard() = Odd;
  }

  GridCartesian         &Grid()   { return grid_f_; }
  GridRedBlackCartesian &RbGrid() { return rbgrid_f_; }

  // -------- EO pieces (SP) --------

  void Meooe(const Field &in, Field &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      Dw_upper_f_.Meooe(in.upper.f[a], out.upper.f[a]);
      Dw_lower_f_.Meooe(in.lower.f[a], out.lower.f[a]);
    }
  }
  void MeooeDag(const Field &in, Field &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      Dw_upper_f_.MeooeDag(in.upper.f[a], out.upper.f[a]);
      Dw_lower_f_.MeooeDag(in.lower.f[a], out.lower.f[a]);
    }
  }

  void Mooee(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    ApplySimd((cb == Even) ? fwd_f_e_ : fwd_f_o_, in, out, cb);
  }
  void MooeeDag(const Field &in, Field &out) {
    GammaWrap(in, out, /*fwd=*/true);
  }
  void MooeeInv(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    ApplySimd((cb == Even) ? inv_f_e_ : inv_f_o_, in, out, cb);
  }
  void MooeeInvDag(const Field &in, Field &out) {
    GammaWrap(in, out, /*fwd=*/false);
  }

  // -------- Mpc / Mpc^dag (SP) --------

  void M(const Field &in, Field &out) {
    GridBase *grid = in.Grid();
    Field tmp_e(grid), tmp_e2(grid), tmp_o(grid);
    Meooe(in, tmp_e);          // M_eo psi_o  (odd -> even)
    MooeeInv(tmp_e, tmp_e2);   // M_ee^{-1} (...)
    Meooe(tmp_e2, tmp_o);      // M_oe (...)
    Mooee(in, out);            // M_oo psi_o
    SubInPlace(out, tmp_o);
  }

  void Mdag(const Field &in, Field &out) {
    GridBase *grid = in.Grid();
    Field tmp_e(grid), tmp_e2(grid), tmp_o(grid);
    MeooeDag(in, tmp_e);
    MooeeInvDag(tmp_e, tmp_e2);
    MeooeDag(tmp_e2, tmp_o);
    MooeeDag(in, out);
    SubInPlace(out, tmp_o);
  }

 private:
  static void SubInPlace(Field &x, const Field &y) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      x.upper.f[a] = x.upper.f[a] - y.upper.f[a];
      x.lower.f[a] = x.lower.f[a] - y.lower.f[a];
    }
  }

  // MooeeDag / MooeeInvDag via gamma_5 (forward/inverse) gamma_5, exactly
  // mirroring the DP operator's per-block gamma_5-Hermiticity wrapper.
  void GammaWrap(const Field &in, Field &out, bool fwd) {
    int cb = in.upper.f[0].Checkerboard();
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5in.upper.f[a] = g5 * in.upper.f[a];
      g5in.lower.f[a] = g5 * in.lower.f[a];
      g5in.upper.f[a].Checkerboard() = cb;
      g5in.lower.f[a].Checkerboard() = cb;
    }
    const InvFieldF &mat = fwd ? ((cb == Even) ? fwd_f_e_ : fwd_f_o_)
                               : ((cb == Even) ? inv_f_e_ : inv_f_o_);
    ApplySimd(mat, g5in, tmp, cb);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }
  }

 public:
  // Single-precision per-oSite 48x48 gemv applying a cached SIMD matrix to a
  // doubled SP fermion.  Direct SP port of
  // DTXQCDWilsonCloverFermionEO::ApplyInverseSimd (same index layout: rows
  // 0..23 upper(a,alpha,i), rows 24..47 lower).  Public for the same nvcc
  // extended-lambda reason as the DP twin.
  void ApplySimd(const InvFieldF &mat, const Field &in, Field &out, int cb) {
    GridBase *fg = in.upper.f[0].Grid();
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }
    autoView(mat_v, mat, AcceleratorRead);
    auto in_up_v  = DTXQCDWilsonCloverFermionEO::MakeFermViewsRead<DtxqcdNf>(
        in.upper, std::make_index_sequence<DtxqcdNf>{});
    auto in_lo_v  = DTXQCDWilsonCloverFermionEO::MakeFermViewsRead<DtxqcdNf>(
        in.lower, std::make_index_sequence<DtxqcdNf>{});
    auto out_up_v = DTXQCDWilsonCloverFermionEO::MakeFermViewsWrite<DtxqcdNf>(
        out.upper, std::make_index_sequence<DtxqcdNf>{});
    auto out_lo_v = DTXQCDWilsonCloverFermionEO::MakeFermViewsWrite<DtxqcdNf>(
        out.lower, std::make_index_sequence<DtxqcdNf>{});

    typedef decltype(coalescedRead(in_up_v[0][0])) FermSitePerLane;
    const int Nsimd = LatticeFermionF::vector_object::Nsimd();

    accelerator_for(s, fg->oSites(), Nsimd, {
      auto Mlane = mat_v(s);
      FermSitePerLane in_up[DtxqcdNf], in_lo[DtxqcdNf];
      for (int a = 0; a < DtxqcdNf; ++a) {
        in_up[a] = in_up_v[a](s);
        in_lo[a] = in_lo_v[a](s);
      }
      FermSitePerLane out_up[DtxqcdNf], out_lo[DtxqcdNf];
      typedef typename std::remove_reference<decltype(Mlane()()(0, 0))>::type MEl;

      for (int r_blk = 0; r_blk < 2; ++r_blk) {
        for (int r_a = 0; r_a < DtxqcdNf; ++r_a) {
          for (int r_alpha = 0; r_alpha < Ns; ++r_alpha) {
            for (int r_i = 0; r_i < Nc; ++r_i) {
              int r = r_blk * kDim24 + r_a * Ns * Nc + r_alpha * Nc + r_i;
              MEl sum;
              zeroit(sum);
              for (int c_blk = 0; c_blk < 2; ++c_blk) {
                for (int c_a = 0; c_a < DtxqcdNf; ++c_a) {
                  for (int c_alpha = 0; c_alpha < Ns; ++c_alpha) {
                    for (int c_i = 0; c_i < Nc; ++c_i) {
                      int c = c_blk * kDim24 + c_a * Ns * Nc + c_alpha * Nc + c_i;
                      auto Mrc = Mlane()()(r, c);
                      auto vc = (c_blk == 0)
                                ? in_up[c_a]()(c_alpha)(c_i)
                                : in_lo[c_a]()(c_alpha)(c_i);
                      sum = sum + Mrc * vc;
                    }
                  }
                }
              }
              if (r_blk == 0) out_up[r_a]()(r_alpha)(r_i) = sum;
              else            out_lo[r_a]()(r_alpha)(r_i) = sum;
            }
          }
        }
      }
      for (int a = 0; a < DtxqcdNf; ++a) {
        coalescedWrite(out_up_v[a][s], out_up[a]);
        coalescedWrite(out_lo_v[a][s], out_lo[a]);
      }
    });

    for (int a = 0; a < DtxqcdNf; ++a) {
      in_up_v[a].ViewClose();
      in_lo_v[a].ViewClose();
      out_up_v[a].ViewClose();
      out_lo_v[a].ViewClose();
    }
  }

 private:
  GridCartesian          grid_f_;
  GridRedBlackCartesian  rbgrid_f_;
  LatticeGaugeFieldF     U_f_;
  LatticeGaugeFieldF     U_conj_f_;
  WilsonOpF              Dw_upper_f_;
  WilsonOpF              Dw_lower_f_;
  InvFieldF              inv_f_e_, inv_f_o_;
  InvFieldF              fwd_f_e_, fwd_f_o_;
  RealD                  mass_;
};

NAMESPACE_END(Grid);
