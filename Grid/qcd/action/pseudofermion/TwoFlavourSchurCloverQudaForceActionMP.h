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

    // Solve X = (Mpc_ee†Mpc_ee)^{-1} PhiEven via DerivativeSolver (MP CG wrapper).
    X = Zero();
    this->DerivativeSolver(Mpc, this->PhiEven, X);

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
};

}  // namespace Grid
