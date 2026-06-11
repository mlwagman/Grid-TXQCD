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
// gamma_5-Hermiticity: gamma_5 M gamma_5 = M^dag (validated in
// Test_dtxqcd_gamma5_herm_full); Mdag uses this identity.
//
// EO production caches.  ImportFields() (called from the constructor and
// from ImportGauge) builds:
//
//   1. Per-CB aux + F_{mu,nu} copies (LatticeDtxqcd{Sigma,Pi,T,D,N} on
//      rbgrid_): forward Mooee on a CB fermion uses these so the
//      DtxqcdApplyMooeeDoubled lattice op gets conformable inputs (its
//      aux refs and the fermion live on the SAME grid).
//
//   2. Per-CB cached per-site 48x48 Mooee^{-1} stored as
//      Lattice<iScalar<iScalar<iMatrix<vComplex, 48>>>> on rbgrid_:
//      MooeeInv on a CB fermion is a single SIMD-vectorized gemv per
//      oSite, replacing the v1 per-call Eigen LU rebuild that dominated
//      multi-shift CG runtime.
//
// Caches must be rebuilt whenever the gauge field or aux fields change
// externally.  ImportGauge(U) rebuilds everything (gauge + aux + caches);
// callers mutating only aux should call ImportFields() afterwards.

#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMeooeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDMooeeOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaCloverOp.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDSiteMatrix.h>
#include <Grid/qcd/utils/WilsonLoops.h>

NAMESPACE_BEGIN(Grid);

// Cached 48x48 per-site matrix type (Mooee^{-1}).  SIMD-vectorized over
// outer sites; one such matrix per CB site stored as an iMatrix tensor.
template <class vtype>
using DtxqcdSiteInvMat = iScalar<iScalar<iMatrix<vtype, kDtxqcdSiteDim48>>>;

class DTXQCDWilsonCloverFermionEO {
 public:
  typedef WilsonImplR Impl;
  typedef DTXQCDFermionDoubled Field;
  typedef LatticeGaugeField GaugeField;
  typedef typename LatticeFermion::vector_object::vector_type FermVtype;
  typedef DtxqcdSiteInvMat<FermVtype> SiteInvMat;
  typedef Lattice<SiteInvMat> InvField;

  static constexpr int kDim24 = kDtxqcdSiteDim24;
  static constexpr int kDim48 = kDtxqcdSiteDim48;

  DTXQCDWilsonCloverFermionEO(GaugeField &U,
                              GridCartesian &grid,
                              GridRedBlackCartesian &rbgrid,
                              RealD mass,
                              RealD csw,
                              const LatticeDtxqcdSigma &sigma,
                              const LatticeDtxqcdPi    &pi,
                              const LatticeDtxqcdD     &d,
                              const LatticeDtxqcdN     &n,
                              const LatticeDtxqcdS     &s,
                              const LatticeDtxqcdP     &p)
      : mass_(mass),
        csw_(csw),
        sigma_(sigma),
        pi_(pi),
        d_(d),
        n_(n),
        s_(s),
        p_(p),
        grid_(grid),
        rbgrid_(rbgrid),
        Umu_(U),
        meooe_(U, grid, rbgrid, mass),
        sigma_e_(&rbgrid), sigma_o_(&rbgrid),
        pi_e_(&rbgrid),    pi_o_(&rbgrid),
        d_e_(&rbgrid),     d_o_(&rbgrid),
        n_e_(&rbgrid),     n_o_(&rbgrid),
        s_e_(&rbgrid),     s_o_(&rbgrid),
        p_e_(&rbgrid),     p_o_(&rbgrid),
        inv_e_(&rbgrid),   inv_o_(&rbgrid),
        spin_(grid) {
    ImportFields();
  }

  // Refresh internal state after the caller updates U (e.g. HMC integrator).
  // Rebuilds the doubled-Wilson hopping engine, the field strength, the
  // per-CB aux copies, and the per-CB Mooee^{-1} caches.
  void ImportGauge(GaugeField &U) {
    meooe_.ImportGauge(U);
    ImportFields();
  }

  // Refresh per-CB aux copies, field strength (gauge unchanged here), and
  // the per-CB Mooee^{-1} caches.  Callers that mutate only aux fields
  // (not gauge) should call this after each mutation; ImportGauge calls
  // this internally.
  void ImportFields() {
    // Per-CB aux copies (pickCheckerboard from the full-grid refs).
    pickCheckerboard(Even, sigma_e_, sigma_);
    pickCheckerboard(Odd,  sigma_o_, sigma_);
    pickCheckerboard(Even, pi_e_,    pi_);
    pickCheckerboard(Odd,  pi_o_,    pi_);
    pickCheckerboard(Even, d_e_,     d_);
    pickCheckerboard(Odd,  d_o_,     d_);
    pickCheckerboard(Even, n_e_,     n_);
    pickCheckerboard(Odd,  n_o_,     n_);
    pickCheckerboard(Even, s_e_,     s_);
    pickCheckerboard(Odd,  s_o_,     s_);
    pickCheckerboard(Even, p_e_,     p_);
    pickCheckerboard(Odd,  p_o_,     p_);

    // F_{mu,nu} from U (full grid) + per-CB copies (csw != 0 only).
    BuildFieldStrength();

    // Mooee^{-1} caches per CB.
    BuildInverseCacheCB(Even, inv_e_, inv_lex_e_);
    BuildInverseCacheCB(Odd,  inv_o_, inv_lex_o_);
  }

  // -------- Full-volume apply --------

  // M psi: full operator on a full-volume doubled fermion.  Uses
  // WilsonFermion::M per block (mass + hopping) plus site-local Delta + cross
  // + clover.  Matches the assembly validated in Test_dtxqcd_gamma5_herm_full.
  void M(const Field &in, Field &out) {
    for (int a = 0; a < DtxqcdNf; ++a) {
      meooe_.UpperWilson().M(in.upper.f[a], out.upper.f[a]);
      meooe_.LowerWilson().M(in.lower.f[a], out.lower.f[a]);
    }
    AddDiagAndCrossAndClover(in, out);
  }

  // Mdag psi via gamma_5 M gamma_5.
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
  // CB input dispatches to DtxqcdApplyMooeeDoubled with the matching per-CB
  // aux / FS copies (cached at ImportFields time).  Full-grid input keeps
  // the full-grid path.
  void Mooee(const Field &in, Field &out) {
    if (in.upper.f[0].Grid() == &rbgrid_) {
      int cb = in.upper.f[0].Checkerboard();
      const auto &sig = (cb == Even) ? sigma_e_ : sigma_o_;
      const auto &pp  = (cb == Even) ? pi_e_    : pi_o_;
      const auto &dd  = (cb == Even) ? d_e_     : d_o_;
      const auto &nn  = (cb == Even) ? n_e_     : n_o_;
      const auto &ss  = (cb == Even) ? s_e_     : s_o_;
      const auto &qq  = (cb == Even) ? p_e_     : p_o_;
      const auto *fs  = (csw_ != 0.0)
                         ? ((cb == Even) ? &FS_e_ : &FS_o_)
                         : nullptr;
      DtxqcdApplyMooeeDoubled(mass_, sig, pp, dd, nn, ss, qq,
                              in.upper, in.lower, out.upper, out.lower,
                              csw_, fs);
    } else {
      DtxqcdApplyMooeeDoubled(mass_, sigma_, pi_, d_, n_, s_, p_,
                              in.upper, in.lower, out.upper, out.lower,
                              csw_, (csw_ != 0.0 ? &FS_ : nullptr));
    }
  }

  // MooeeDag psi via gamma_5 Mooee gamma_5 (per-block gamma_5-Hermiticity).
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

  // MooeeInv psi via the cached SIMD per-site 48x48 inverse.  CB-only path.
  void MooeeInv(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    GRID_ASSERT(in.upper.f[0].Grid() == &rbgrid_);
    ApplyInverseSimd(in, out, cb);
  }
  // MooeeInvDag psi via gamma_5 MooeeInv gamma_5 (per-block gamma_5-Hermiticity
  // of Mooee carries over to its inverse).
  void MooeeInvDag(const Field &in, Field &out) {
    int cb = in.upper.f[0].Checkerboard();
    GRID_ASSERT(in.upper.f[0].Grid() == &rbgrid_);
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5in.upper.f[a] = g5 * in.upper.f[a];
      g5in.lower.f[a] = g5 * in.lower.f[a];
      g5in.upper.f[a].Checkerboard() = cb;
      g5in.lower.f[a].Checkerboard() = cb;
    }
    ApplyInverseSimd(g5in, tmp, cb);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }
  }

  // -------- Multi-RHS variants for batched-gemm hot loops --------
  //
  // MooeeInvN: applies Mooee^{-1} to NRHS fermions in one threaded sweep
  // over CB sites.  Each thread does a single Eigen 48 x NRHS gemm instead
  // of NRHS separate 48-component gemvs -- the per-site matrix is loaded
  // once per site and reused across all RHS columns.  For NRHS = (multi-shift
  // degree) this gives the matrix-load savings that the cuBLAS-batched
  // production path on GPU provides.  CPU benefit scales with NRHS up to
  // the point where the 48 x NRHS Eigen matrix overflows L2.
  //
  // Used by DTXQCDWilsonCloverRationalEOAction::deriv()'s post-CG pole loop
  // (Npole independent Mpc applies on the same gauge configuration).
  void MooeeInvN(const std::vector<const Field *> &ins,
                 const std::vector<Field *> &outs) {
    ApplyInverseLex(ins, outs, /*dag=*/false);
  }

  void MooeeInvDagN(const std::vector<const Field *> &ins,
                    const std::vector<Field *> &outs) {
    ApplyInverseLex(ins, outs, /*dag=*/true);
  }

  // -------- Accessors / introspection (mainly for tests) --------

  RealD Mass() const { return mass_; }
  RealD Csw()  const { return csw_; }
  const std::vector<LatticeColourMatrix> &FieldStrength() const { return FS_; }
  DTXQCDMeooeDoubled &MeooeEngine() { return meooe_; }

 private:
  // Build F_{mu,nu} for the 6 (mu<nu) pairs from U via WilsonLoops, plus
  // pickCheckerboard into per-CB copies.  No-op when csw == 0.
  void BuildFieldStrength() {
    FS_.clear();
    FS_e_.clear();
    FS_o_.clear();
    if (csw_ == 0.0) return;
    FS_.reserve(6);
    FS_e_.reserve(6);
    FS_o_.reserve(6);
    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        LatticeColourMatrix F(&grid_);
        WilsonLoops<WilsonImplR>::FieldStrength(F, Umu_, mu, nu);
        LatticeColourMatrix F_e(&rbgrid_), F_o(&rbgrid_);
        pickCheckerboard(Even, F_e, F);
        pickCheckerboard(Odd,  F_o, F);
        FS_.push_back(std::move(F));
        FS_e_.push_back(std::move(F_e));
        FS_o_.push_back(std::move(F_o));
      }
    }
  }

  // Site-local Delta + cross + clover, added on top of (mass + hopping) which
  // WilsonFermion::M already produced into `out`.  Used by M() only.
  void AddDiagAndCrossAndClover(const Field &in, Field &out) {
    GridBase *grid = in.Grid();

    DTXQCDFermionNf delta_u(grid), delta_l(grid);
    DtxqcdApplyDeltaDiag     (sigma_, pi_, s_, p_, in.upper, delta_u);
    DtxqcdApplyDeltaDiagLower(sigma_, pi_, s_, p_, in.lower, delta_l);

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

  // Build the per-CB 48x48 Mooee^{-1} cache.  Iterates sites of the chosen
  // CB, assembles M48 from the full-grid aux + clover at that site (Eigen),
  // computes M48^{-1} via partialPivLu, and pokes the result into the
  // SIMD-vectorized lattice slot.  thread_for-parallel over sites; the
  // SIMD repack is hidden inside Grid's pokeSite.
  void BuildInverseCacheCB(int cb, InvField &inv,
                            std::vector<Eigen::MatrixXcd> &inv_lex) {
    typedef typename InvField::vector_object::scalar_object SmatSobj;
    inv = Zero();
    inv.Checkerboard() = cb;

    Coordinate gd(grid_.GlobalDimensions());

    // Flatten the 4D grid into a 1D index list for the requested CB so we
    // can thread_for over it.  Inverse construction is per-site independent.
    std::vector<Coordinate> coords;
    coords.reserve(grid_.lSites() / 2);
    for (int x = 0; x < gd[0]; ++x)
      for (int y = 0; y < gd[1]; ++y)
        for (int z = 0; z < gd[2]; ++z)
          for (int s = 0; s < gd[3]; ++s) {
            int parity = (x + y + z + s) & 1;
            if (parity != cb) continue;
            coords.push_back(Coordinate(std::vector<int>{x, y, z, s}));
          }

    const uint64_t Nsite = coords.size();
    std::vector<SmatSobj> sobjs(Nsite);

    thread_for(idx, Nsite, {
      const Coordinate &coord = coords[idx];
      DtxqcdSiteAux aux =
          DtxqcdSiteAux::Extract(sigma_, pi_, d_, n_, s_, p_, coord);
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

      Eigen::MatrixXcd Minv = M48.inverse();

      SmatSobj sobj;
      sobj = Zero();
      for (int r = 0; r < kDim48; ++r)
        for (int c = 0; c < kDim48; ++c)
          sobj()()(r, c) = Minv(r, c);
      sobjs[idx] = sobj;
    });

    // pokeSite is not thread-safe on the same lattice; do the pokes serially
    // (much cheaper than the per-site inversion above).
    for (uint64_t idx = 0; idx < Nsite; ++idx) {
      pokeSite(sobjs[idx], inv, coords[idx]);
    }

    // Multi-RHS path also needs a lex-ordered std::vector<Eigen::MatrixXcd>
    // matching Grid's CB-lex convention (the same order unvectorizeToLexOrdArray
    // produces on the input fermion).  Build by unvectorizing the SIMD field
    // we just populated -- this guarantees index alignment with the fermion
    // pack/unpack in ApplyInverseLex below, regardless of how my (x,y,z,t)
    // iteration order above maps onto Grid's CB-lex enumeration.
    std::vector<SmatSobj> lex_sobjs;
    unvectorizeToLexOrdArray(lex_sobjs, inv);
    inv_lex.clear();
    inv_lex.resize(lex_sobjs.size());
    thread_for(s, lex_sobjs.size(), {
      Eigen::MatrixXcd M(kDim48, kDim48);
      for (int r = 0; r < kDim48; ++r)
        for (int c = 0; c < kDim48; ++c)
          M(r, c) = ComplexD(lex_sobjs[s]()()(r, c));
      inv_lex[s] = std::move(M);
    });
  }

  // SIMD per-oSite gemv applying the cached 48x48 inverse to a doubled
  // fermion.  Reads inv_v(s) as a single iMatrix lane and the four
  // LatticeFermion views (upper × DtxqcdNf, lower × DtxqcdNf) as the input
  // vector lanes.  Direct port of TXQCD's ApplyMooeeInvSimd pattern with
  // an outer block (upper, lower) dimension added.
  void ApplyInverseSimd(const Field &in, Field &out, int cb) {
    InvField &inv = (cb == Even) ? inv_e_ : inv_o_;
    GridBase *fg = in.upper.f[0].Grid();
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a].Checkerboard() = cb;
      out.lower.f[a].Checkerboard() = cb;
    }

    autoView(inv_v, inv, AcceleratorRead);
    auto in_up_v  = MakeFermViewsRead<DtxqcdNf>(in.upper,  std::make_index_sequence<DtxqcdNf>{});
    auto in_lo_v  = MakeFermViewsRead<DtxqcdNf>(in.lower,  std::make_index_sequence<DtxqcdNf>{});
    auto out_up_v = MakeFermViewsWrite<DtxqcdNf>(out.upper, std::make_index_sequence<DtxqcdNf>{});
    auto out_lo_v = MakeFermViewsWrite<DtxqcdNf>(out.lower, std::make_index_sequence<DtxqcdNf>{});

    typedef decltype(coalescedRead(in_up_v[0][0])) FermSitePerLane;
    const int Nsimd = LatticeFermion::vector_object::Nsimd();

    accelerator_for(s, fg->oSites(), Nsimd, {
      auto Mlane = inv_v(s);
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
              int r = r_blk * kDim24
                    + r_a * Ns * Nc + r_alpha * Nc + r_i;
              MEl sum;
              zeroit(sum);
              for (int c_blk = 0; c_blk < 2; ++c_blk) {
                for (int c_a = 0; c_a < DtxqcdNf; ++c_a) {
                  for (int c_alpha = 0; c_alpha < Ns; ++c_alpha) {
                    for (int c_i = 0; c_i < Nc; ++c_i) {
                      int c = c_blk * kDim24
                            + c_a * Ns * Nc + c_alpha * Nc + c_i;
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

  // Multi-RHS apply via per-site Eigen 48 x NRHS gemm.  Mirrors TXQCD's
  // ApplyMooeeInvScalar pattern with an outer NRHS dimension and the
  // doubled (upper, lower) block layout.  Threaded across CB sites; each
  // thread holds a stack-resident 48 x NRHS input + output Eigen matrix.
  //
  //   dag = false:  out_k = inv_lex[site] *      input_k
  //   dag = true :  out_k = inv_lex[site]^dag *  input_k
  //                       = (inv^dag is the inverse of M^dag = gamma_5 M gamma_5,
  //                          so we wrap in gamma_5 outside and call dag=false here).
  //
  // The dag path is unused on the inside (we provide MooeeInvDagN above as a
  // gamma_5 wrapper), but kept symmetric in case a caller wants the raw
  // adjoint apply.
  void ApplyInverseLex(const std::vector<const Field *> &ins,
                        const std::vector<Field *> &outs,
                        bool dag) {
    const int NRHS = static_cast<int>(ins.size());
    GRID_ASSERT(NRHS > 0);
    GRID_ASSERT(static_cast<int>(outs.size()) == NRHS);

    const int cb = ins[0]->upper.f[0].Checkerboard();
    GRID_ASSERT(ins[0]->upper.f[0].Grid() == &rbgrid_);

    if (dag) {
      // Wrap in gamma_5 on each RHS and recurse with dag=false.
      Gamma g5(Gamma::Algebra::Gamma5);
      std::vector<Field> g5_in;       g5_in.reserve(NRHS);
      std::vector<Field> tmp;         tmp.reserve(NRHS);
      std::vector<const Field *> g5_in_p(NRHS);
      std::vector<Field *>       tmp_p(NRHS);
      for (int k = 0; k < NRHS; ++k) {
        g5_in.emplace_back(&rbgrid_);
        tmp.emplace_back(&rbgrid_);
        for (int a = 0; a < DtxqcdNf; ++a) {
          g5_in[k].upper.f[a] = g5 * ins[k]->upper.f[a];
          g5_in[k].lower.f[a] = g5 * ins[k]->lower.f[a];
          g5_in[k].upper.f[a].Checkerboard() = cb;
          g5_in[k].lower.f[a].Checkerboard() = cb;
        }
        g5_in_p[k] = &g5_in[k];
        tmp_p[k]   = &tmp[k];
      }
      ApplyInverseLex(g5_in_p, tmp_p, /*dag=*/false);
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          outs[k]->upper.f[a] = g5 * tmp[k].upper.f[a];
          outs[k]->lower.f[a] = g5 * tmp[k].lower.f[a];
          outs[k]->upper.f[a].Checkerboard() = cb;
          outs[k]->lower.f[a].Checkerboard() = cb;
        }
      }
      return;
    }

    // Forward (no-dag) batched apply.
    const auto &inv_lex = (cb == Even) ? inv_lex_e_ : inv_lex_o_;
    typedef typename LatticeFermion::vector_object::scalar_object SiteFerm;

    // Unvectorize each RHS into per-(rhs, flavor, block) lex arrays.  The
    // lex order Grid uses here matches the order inv_lex was populated in
    // (via the same unvectorize on the SIMD InvField in BuildInverseCacheCB),
    // so per-site indices line up directly.
    std::vector<std::vector<std::vector<SiteFerm>>> in_up_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> in_lo_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> out_up_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    std::vector<std::vector<std::vector<SiteFerm>>> out_lo_lex(NRHS,
        std::vector<std::vector<SiteFerm>>(DtxqcdNf));
    for (int k = 0; k < NRHS; ++k) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        unvectorizeToLexOrdArray(in_up_lex[k][a], ins[k]->upper.f[a]);
        unvectorizeToLexOrdArray(in_lo_lex[k][a], ins[k]->lower.f[a]);
        out_up_lex[k][a].resize(in_up_lex[k][a].size());
        out_lo_lex[k][a].resize(in_lo_lex[k][a].size());
      }
    }
    const uint64_t Nsite = in_up_lex[0][0].size();
    GRID_ASSERT(Nsite == inv_lex.size());

    thread_for(s, Nsite, {
      Eigen::MatrixXcd B(kDim48, NRHS);
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int i = 0; i < Nc; ++i) {
              int r24 = a * Ns * Nc + alpha * Nc + i;
              B(r24,           k) = ComplexD(in_up_lex[k][a][s]()(alpha)(i));
              B(kDim24 + r24,  k) = ComplexD(in_lo_lex[k][a][s]()(alpha)(i));
            }
          }
        }
      }
      Eigen::MatrixXcd C = inv_lex[s] * B;
      for (int k = 0; k < NRHS; ++k) {
        for (int a = 0; a < DtxqcdNf; ++a) {
          for (int alpha = 0; alpha < Ns; ++alpha) {
            for (int i = 0; i < Nc; ++i) {
              int r24 = a * Ns * Nc + alpha * Nc + i;
              out_up_lex[k][a][s]()(alpha)(i) = C(r24,          k);
              out_lo_lex[k][a][s]()(alpha)(i) = C(kDim24 + r24, k);
            }
          }
        }
      }
    });

    for (int k = 0; k < NRHS; ++k) {
      for (int a = 0; a < DtxqcdNf; ++a) {
        vectorizeFromLexOrdArray(out_up_lex[k][a], outs[k]->upper.f[a]);
        vectorizeFromLexOrdArray(out_lo_lex[k][a], outs[k]->lower.f[a]);
        outs[k]->upper.f[a].Checkerboard() = cb;
        outs[k]->lower.f[a].Checkerboard() = cb;
      }
    }
  }

  // std::array-of-LatticeView helpers (LatticeView has no default ctor; the
  // index_sequence trick aggregate-initialises the array).  Mirror of
  // TXQCD's MakeFermViewsRead/Write.
  template <std::size_t N, class FieldT, std::size_t... Is>
  static auto MakeFermViewsRead(const FieldT &fld, std::index_sequence<Is...>)
      -> std::array<decltype(fld.f[0].View(AcceleratorRead)), N> {
    return {{ fld.f[Is].View(AcceleratorRead)... }};
  }
  template <std::size_t N, class FieldT, std::size_t... Is>
  static auto MakeFermViewsWrite(FieldT &fld, std::index_sequence<Is...>)
      -> std::array<decltype(fld.f[0].View(AcceleratorWrite)), N> {
    return {{ fld.f[Is].View(AcceleratorWrite)... }};
  }

  RealD mass_;
  RealD csw_;
  const LatticeDtxqcdSigma &sigma_;
  const LatticeDtxqcdPi    &pi_;
  const LatticeDtxqcdD     &d_;
  const LatticeDtxqcdN     &n_;
  const LatticeDtxqcdS     &s_;
  const LatticeDtxqcdP     &p_;
  GridCartesian            &grid_;
  GridRedBlackCartesian    &rbgrid_;
  GaugeField               &Umu_;

  DTXQCDMeooeDoubled       meooe_;
  std::vector<LatticeColourMatrix> FS_;
  std::vector<LatticeColourMatrix> FS_e_, FS_o_;

  LatticeDtxqcdSigma sigma_e_, sigma_o_;
  LatticeDtxqcdPi    pi_e_,    pi_o_;
  LatticeDtxqcdD     d_e_,     d_o_;
  LatticeDtxqcdN     n_e_,     n_o_;
  LatticeDtxqcdS     s_e_,     s_o_;
  LatticeDtxqcdP     p_e_,     p_o_;

  InvField inv_e_, inv_o_;

  // Multi-RHS scratch: per-CB lex-ordered Eigen 48x48 inverses.  Built by
  // unvectorizing the SIMD InvField so the lex index matches what
  // unvectorizeToLexOrdArray produces on input fermions in ApplyInverseLex.
  std::vector<Eigen::MatrixXcd> inv_lex_e_, inv_lex_o_;

  DtxqcdSpinMatrices spin_;
};

NAMESPACE_END(Grid);
