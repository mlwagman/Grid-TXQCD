#pragma once
// QUDA-accelerated Nf=2 light-quark force for Wilson-Clover, EVEN-parity
// Schur (matches PyQUDA's QUDA_MATPC_EVEN_EVEN_ASYMMETRIC convention).
//
// Mirrors the Phase 7 strange-quark class (OneFlavourSchurCloverQudaForceRationalActionMP)
// but with single-shift solve and nvector=1 (no rational expansion).
//
// Action:  S = PhiEven† (Mpc_ee†Mpc_ee)^{-1} PhiEven    (Nf=2 weighting via det²)
// deriv:   solve X = (Mpc_ee†Mpc_ee)^{-1} PhiEven via MP CG, then call QUDA's
//          computeCloverForceQuda (X, dagger=YES, multiplicity=0, single coeff=1)
//          which fuses the Wilson-hop + σ_μν·F_μν kernel chain into one GPU call.
//
// Validated convention (Phase D D.5, hot 4⁴): dagger=YES + EVEN_EVEN_ASYMMETRIC
// gives cos(Ta(A_grid), B_quda)=1.000000 and FD ratio 9.9999999e-01 vs Path A.
// LogDet (σ-trace) is handled separately by Grid's QCDLogDetCloverEOAction —
// pass multiplicity=0 to skip QUDA's σ-trace term and avoid double-count.
//
// Activation:  QUDA_FORCE_LIGHT=1 in gen_qcd_cfgs_2plus1.cc opts in. Inherits
// from TwoFlavourSchurCloverActionEven so the action S, refresh, and Path-A
// deriv all live on EVEN parity, matching what QUDA expects.

#include <Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverActionEven.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaInit.h>

#include <quda.h>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class TwoFlavourSchurCloverQudaForceActionMP
    : public TwoFlavourSchurCloverActionEven<ImplD, FermOpD_> {
 public:
  typedef TwoFlavourSchurCloverActionEven<ImplD, FermOpD_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  TwoFlavourSchurCloverQudaForceActionMP(
      typename Base::FermionOperator &opD,
      FermOpF &opF,
      OperatorFunction<typename Base::FermionField> &DS,
      OperatorFunction<typename Base::FermionField> &AS,
      const QudaCloverParams &qp)
      : Base(opD, DS, AS), opF_(opF), qp_(qp) {
    Quda::initialize();
    QudaCloverMultiShiftSpec spec;
    // EVEN parity throughout — matches PyQUDA's CloverWilsonAction convention
    // and the Phase 7 strange (OneFlavourSchurCloverQudaForceRationalActionMP)
    // which inherits from the Even rational base class.
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    spec.shifts     = std::vector<RealD>(1, 0.0);
    spec.tols       = std::vector<RealD>(1, 1e-8);
    quda_loader_.reset(new QudaCloverMultiShiftInverter(opD.GaugeGrid(), qp_, spec));
    std::cout << GridLogMessage
              << "[TwoFlavourSchurCloverQudaForceActionMP] built, "
              << "matpc=EVEN_EVEN_ASYMMETRIC, dagger=YES (PyQUDA convention)"
              << std::endl;
  }

  std::string action_name() override {
    return "TwoFlavourSchurCloverQudaForceActionMP";
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp = this->FermOp;

    // Refresh SP operator from the raw gauge field (mirrors the existing
    // TwoFlavourSchurCloverActionMP pattern in gen_qcd_cfgs_2plus1.cc).
    typename ImplF::GaugeField UmuF(opF_.GaugeGrid());
    typename ImplD::GaugeLinkField U_d(U.Grid());
    typename ImplF::GaugeLinkField U_f(opF_.GaugeGrid());
    for (int mu = 0; mu < Nd; ++mu) {
      U_d = PeekIndex<LorentzIndex>(U, mu);
      precisionChange(U_f, U_d);
      PokeIndex<LorentzIndex>(UmuF, U_f, mu);
    }
    opF_.ImportGauge(UmuF);
    FermOp.ImportGauge(U);
    quda_loader_->SetGauge(U);

    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    GridBase *ggrid   = FermOp.GaugeGrid();
    FermionField X(fcbgrid);
    SchurDifferentiableOperator<ImplD> Mpc(FermOp);

    // Solve X = (Mpc_ee†Mpc_ee)^{-1} PhiEven.  Default path: MP CG via the
    // base class's DerivativeSolver.  USE_HMC_MG=1 swaps in QUDA multigrid
    // via the two-solve Schur reduction trick:
    //   y_full = M†^{-1}·(PhiEven, 0)            (full-volume MG, dagger=YES)
    //   y_e = even part of y_full = Mpc^{-†}·PhiEven
    //   X_full = M^{-1}·(y_e, 0)                  (full-volume MG, dagger=NO)
    //   X = even part of X_full = (Mpc†·Mpc)^{-1}·PhiEven   ✓
    // The trick: setting the odd half of the source to zero reduces M^{-1}
    // (full) to Mpc^{-1} on the even sublattice (see Schur algebra).  Both
    // MG solves are at the same expensive light mass, but each is many
    // times faster than vanilla CG.
    X = Zero();
    const bool use_hmc_mg = std::getenv("USE_HMC_MG") != nullptr;
    if (use_hmc_mg) {
      // Lazy-init the MG inverter on first deriv() call.  Reuses gauge via
      // QudaCloverInverter::SetGauge thin-update (newMultigridQuda only on
      // first SetGauge, updateMultigridQuda thereafter).
      if (!mg_inv_) {
        QudaCloverParams qp_mg = qp_;
        qp_mg.use_multigrid = true;
        // 2 levels by default (4⁴ / 16³×48 tests); HMC_MG_NLEVEL=3 for
        // production 48³×96.  HMC_MG_BLOCK_L0 etc override blocks.
        const char *nlv = std::getenv("HMC_MG_NLEVEL");
        qp_mg.mg.n_level = nlv ? std::atoi(nlv) : 2;
        auto parse_block = [](const char *s, std::array<int,4> dflt) {
          if (!s || !*s) return dflt;
          std::array<int,4> b = dflt;
          std::sscanf(s, "%d %d %d %d", &b[0], &b[1], &b[2], &b[3]);
          return b;
        };
        std::array<int,4> b0 = parse_block(std::getenv("HMC_MG_BLOCK_L0"), {4,4,4,4});
        std::array<int,4> b1 = parse_block(std::getenv("HMC_MG_BLOCK_L1"), {2,2,2,2});
        qp_mg.mg.geo_block_size = (qp_mg.mg.n_level >= 3)
            ? std::vector<std::array<int,4>>{b0, b1}
            : std::vector<std::array<int,4>>{b0};
        mg_inv_.reset(new QudaCloverInverter(ggrid, qp_mg));
        std::cout << GridLogMessage
                  << "[TwoFlavourSchurCloverQudaForceActionMP] USE_HMC_MG=1 — "
                  << "built MG inverter (" << qp_mg.mg.n_level << " levels)"
                  << std::endl;
      }
      mg_inv_->SetGauge(U);

      FermionField src_full(ggrid), y_full(ggrid), X_full(ggrid);
      FermionField y_e(fcbgrid);
      // src_full = (PhiEven, 0).  setCheckerboard places PhiEven on even sites;
      // odd sites are zero from the Zero() above (the Zero() at line 162 is
      // for X — set a separate zero for src_full).
      src_full = Zero();
      setCheckerboard(src_full, this->PhiEven);

      // Step 1: y_full = M†^{-1} · src_full
      QudaInvertParam &mg_iparam = mg_inv_->InvertParam();
      QudaDagType saved_mg_dagger = mg_iparam.dagger;
      mg_iparam.dagger = QUDA_DAG_YES;
      (*mg_inv_)(Mpc, src_full, y_full);

      // Extract EVEN part of y_full → y_e, embed in fresh src_full with odd = 0
      y_e.Checkerboard() = Even;
      pickCheckerboard(Even, y_e, y_full);
      src_full = Zero();
      setCheckerboard(src_full, y_e);

      // Step 2: X_full = M^{-1} · src_full
      mg_iparam.dagger = QUDA_DAG_NO;
      (*mg_inv_)(Mpc, src_full, X_full);
      X.Checkerboard() = Even;
      pickCheckerboard(Even, X, X_full);

      mg_iparam.dagger = saved_mg_dagger;
    } else {
      this->DerivativeSolver(Mpc, this->PhiEven, X);
    }

    // QUDA_FORCE_KERNEL_COMPARE=1: compute Path A force and emit Ta-projected
    // cos(PathA, PathB) comparator. Mirrors Phase 7's diagnostic at line 406+.
    bool path_a_compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;
    GaugeField dSdU_pathA(ggrid);
    if (path_a_compare) {
      // EVEN-parity Path A formula (mirrors OneFlavourSchurCloverRationalActionEven::deriv):
      //   dSdU = MpcDeriv(Y, X) + MpcDagDeriv(X, Y)            // hopping
      //        + MeeDeriv(Y, X, NO) + MeeDeriv(X, Y, YES)      // diagonal on EVEN
      //        + MooDeriv(Z_o, W_o, NO) + MooDeriv(W_o, Z_o, YES)
      //        with W_o = M_oo^{-1} M_eo X,  Z_o = M_oo^{-†} M_oe^† Y.
      FermionField Y(fcbgrid);
      Mpc.Mpc(X, Y);
      GaugeField tmp(ggrid);
      dSdU_pathA = Zero();
      Mpc.MpcDeriv(tmp, Y, X);     dSdU_pathA = dSdU_pathA + tmp;
      Mpc.MpcDagDeriv(tmp, X, Y);  dSdU_pathA = dSdU_pathA + tmp;
      FermOp.MeeDeriv(tmp, Y, X, DaggerNo);   dSdU_pathA = dSdU_pathA + tmp;
      FermOp.MeeDeriv(tmp, X, Y, DaggerYes);  dSdU_pathA = dSdU_pathA + tmp;
      FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);
      FermOp.Meooe(X, tmp1);     FermOp.MooeeInv(tmp1, W_o);
      FermOp.MeooeDag(Y, tmp1);  FermOp.MooeeInvDag(tmp1, Z_o);
      FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);  dSdU_pathA = dSdU_pathA + tmp;
      FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes); dSdU_pathA = dSdU_pathA + tmp;
      std::cout << GridLogMessage
                << "[TwoFlavourQudaForce] PathA dSdU norm2=" << norm2(dSdU_pathA)
                << std::endl;
    }

    // Pack X (single vector, EVEN-parity slot) into host buffer for QUDA.
    int V_eo = Quda::local_volume(ggrid) / 2;
    using SiteSpinor = typename FermionField::scalar_object;
    static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                  "expected 24 doubles/site for fermion");
    std::vector<double> x_buf(24 * V_eo);
    {
      std::vector<SiteSpinor> sv;
      unvectorizeToLexOrdArray(sv, X);
      std::memcpy(x_buf.data(), sv.data(), V_eo * 24 * sizeof(double));
    }
    void *x_ptr = x_buf.data();
    void *p_ptr = nullptr;  // unused — QUDA orchestrator builds Y internally

    int V = Quda::local_volume(ggrid);
    constexpr int MOM_RECON = 10;
    std::vector<double> mom_buf(V * 4 * MOM_RECON, 0.0);

    QudaInvertParam &inv_param = quda_loader_->InvertParam();
    int saved_use_resident = inv_param.use_resident_solution;
    QudaDagType saved_dagger = inv_param.dagger;
    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    inv_param.use_resident_solution = 0;
    // PyQUDA's CloverWilsonAction.force() sets dagger=YES; validated
    // cos=1.0 element-wise vs Path A in Phase D D.5 (4⁴ hot, EE-asymmetric).
    inv_param.dagger        = std::getenv("QUDA_FORCE_DAGGER_NO") ? QUDA_DAG_NO
                                                                  : QUDA_DAG_YES;
    inv_param.twist_flavor  = QUDA_TWIST_NO;

    QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.overwrite_mom     = 1;
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;

    const double kappa  = inv_param.kappa;
    const double kappa2 = -kappa * kappa;
    const double ck     = -inv_param.clover_csw * kappa / 8.0;
    const double dt     = 1.0;
    const double mult   = 0.0;  // skip σ-trace LogDet (Grid handles separately)
    std::vector<double> coeff = {1.0};

    computeCloverForceQuda(mom_buf.data(),
                           dt,
                           &x_ptr,
                           &p_ptr,
                           coeff.data(),
                           kappa2, ck,
                           /*nvector=*/1, mult,
                           /*gauge=*/nullptr,
                           &force_gauge_param,
                           &inv_param);

    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist;

    // Unpack mom_buf (MILC anti-Hermitian 10-real per site/dir) → Grid GaugeField.
    Coordinate lc = ggrid->LocalDimensions();
    std::vector<std::vector<double>> dir_eo_18(4, std::vector<double>(18 * V));
    for (int mu = 0; mu < 4; ++mu) {
      double *dst = dir_eo_18[mu].data();
      for (int site = 0; site < V; ++site) {
        const double *m = &mom_buf[(site * 4 + mu) * MOM_RECON];
        double r01 = m[0], i01 = m[1];
        double r02 = m[2], i02 = m[3];
        double r12 = m[4], i12 = m[5];
        double a0  = m[6], a1  = m[7], a2 = m[8];
        double *d = &dst[site * 18];
        d[ 0] = 0.0;   d[ 1] = a0;
        d[ 2] = r01;   d[ 3] = i01;
        d[ 4] = r02;   d[ 5] = i02;
        d[ 6] = -r01;  d[ 7] = i01;
        d[ 8] = 0.0;   d[ 9] = a1;
        d[10] = r12;   d[11] = i12;
        d[12] = -r02;  d[13] = i02;
        d[14] = -r12;  d[15] = i12;
        d[16] = 0.0;   d[17] = a2;
      }
    }
    std::vector<std::vector<double>> dir_lex_18(4, std::vector<double>(18 * V));
    double *lex_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      Quda::eo_to_lex_permute(dir_eo_18[mu].data(), dir_lex_18[mu].data(),
                              V, 18, lc);
      lex_ptrs[mu] = dir_lex_18[mu].data();
    }
    Quda::lex_buffers_to_gauge(lex_ptrs, dSdU);

    // Phase 7 (Nf=1 RHMC, 20-pole) validated -1/(8κ²) gives cos=1.0 vs Path A
    // with EVEN_EVEN_ASYMMETRIC matpc + dagger=YES. The same factor should
    // apply to TwoFlavour single-shift (coeff=[1.0]) once parity matches.
    // The earlier 1/9.3 empirical was a workaround for the ODD-parity bug.
    // Override via env QUDA_FORCE_TWOFL_RESCALE if FD scan finds otherwise.
    double rescale = 1.0;
    if (const char *r = std::getenv("QUDA_FORCE_TWOFL_RESCALE"); r && *r)
      rescale = std::atof(r);
    const double quda_to_grid_factor = rescale * (-1.0 / (8.0 * kappa * kappa));
    dSdU = quda_to_grid_factor * dSdU;
    std::cout << GridLogMessage
              << "[TwoFlavourQudaForce] κ=" << kappa
              << " rescale=" << rescale
              << " quda_to_grid_factor=" << quda_to_grid_factor
              << " norm2(dSdU_PathB)=" << norm2(dSdU)
              << std::endl;

    if (path_a_compare) {
      GaugeField TaA(ggrid);
      for (int mu = 0; mu < Nd; ++mu) {
        auto Amu = PeekIndex<LorentzIndex>(dSdU_pathA, mu);
        auto TaAmu = Ta(Amu);
        PokeIndex<LorentzIndex>(TaA, TaAmu, mu);
      }
      double n2A_full = norm2(dSdU_pathA);
      double n2A_Ta = norm2(TaA);
      double n2B = norm2(dSdU);
      auto inner_full = innerProduct(dSdU_pathA, dSdU);
      auto inner_Ta   = innerProduct(TaA, dSdU);
      std::cout << GridLogMessage
                << "[TwoFlavourQudaForce] |PathA|²(full)=" << n2A_full
                << " |Ta(PathA)|²=" << n2A_Ta
                << " |PathB|²=" << n2B
                << " ⟨A,B⟩(full)=" << real(inner_full)
                << " ⟨Ta(A),B⟩=" << real(inner_Ta)
                << " factor Ta(A)/B=" << real(inner_Ta) / n2B
                << " cos(Ta(A),B)=" << real(inner_Ta) / std::sqrt(n2A_Ta * n2B)
                << std::endl;
      // In compare mode, return PathA's correct force.
      dSdU = dSdU_pathA;
    }
  }

 private:
  FermOpF &opF_;
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_loader_;
  // USE_HMC_MG=1 path: lazy-init MG inverter for the (Mpc†·Mpc)^{-1}·PhiEven
  // solve via the Schur 2-solve trick.  Reuses gauge across MD steps via
  // QudaCloverInverter::SetGauge thin-update.
  mutable std::unique_ptr<QudaCloverInverter> mg_inv_;
};

}  // namespace Grid
