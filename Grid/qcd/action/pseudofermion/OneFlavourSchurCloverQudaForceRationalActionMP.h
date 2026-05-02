#pragma once
// QUDA-accelerated rational action with QUDA-computed gauge force.
//
// This is the "Phase 6" target: replace not just the inner multishift
// CG (Phase 5) but the entire per-pole gauge-derivative chain with one
// QUDA call to computeCloverForceQuda — which batches the Wilson hop
// derivative AND the σ_μν·F_μν clover derivative across all rational
// poles in a single fused GPU kernel chain.
//
// Built on top of OneFlavourSchurCloverRationalActionEven because
// computeCloverForceQuda hardcodes EVEN_EVEN_ASYMMETRIC matpc and
// expects EVEN-parity X_k inputs.
//
// Refresh and S inherit from the Grid-side base — Phase 5 already
// validated those paths against the FD test.  Only deriv() is overridden.

#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class OneFlavourSchurCloverQudaForceRationalActionMP
    : public OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> {
 public:
  typedef OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  OneFlavourSchurCloverQudaForceRationalActionMP(
      FermOpD &opD, FermOpF & /*opF*/, GridBase * /*sp_rbgrid*/,
      OneFlavourRationalParams &p,
      const QudaCloverParams &qp,
      int /*reliable_update_freq*/ = 50)
    : Base(opD, p), qp_(qp) {

    QudaCloverMultiShiftSpec spec;
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    auto &poles = this->PowerNegHalf.poles;
    spec.shifts.resize(poles.size());
    spec.tols.assign(poles.size(), p.tolerance);
    for (size_t k = 0; k < poles.size(); ++k) spec.shifts[k] = poles[k];

    Quda::initialize();
    quda_ms_.reset(new QudaCloverMultiShiftInverter(
        opD.GaugeGrid(), qp_, spec));
    std::cout << GridLogMessage
              << "[OneFlavourSchurCloverQudaForceRationalActionMP] "
              << "built with " << poles.size()
              << " rational shifts, matpc=EVEN_EVEN_ASYMMETRIC" << std::endl;
  }

  // Override: solve via QUDA multishift, then compute the gauge force
  // via computeCloverForceQuda (single fused QUDA kernel chain).
  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp = this->FermOp;
    auto &PhiEven = this->PhiEven;
    auto &PowerNegHalf = this->PowerNegHalf;
    const int Npole = PowerNegHalf.poles.size();
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    GridBase *ggrid   = FermOp.GaugeGrid();

    FermOp.ImportGauge(U);
    quda_ms_->SetGauge(U);

    std::vector<FermionField> MPhi_k(Npole, fcbgrid);
    quda_ms_->solve_rb_even(PhiEven, MPhi_k);

    // Pack each MPhi_k (Even RB grid) into a flat half-volume host buffer
    // for QUDA.  The packing reuses the same Grid-RB-cb-site → flat-V/2
    // mapping that QudaCloverMultiShiftInverter::solve_rb_even uses.
    int V_eo = Quda::local_volume(ggrid) / 2;
    using SiteSpinor = typename FermionField::scalar_object;
    static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                  "expected 24 doubles/site for fermion");
    std::vector<std::vector<double>> x_bufs(Npole, std::vector<double>(24 * V_eo));
    std::vector<void *> x_ptrs(Npole);
    for (int k = 0; k < Npole; ++k) {
      std::vector<SiteSpinor> scalars;
      unvectorizeToLexOrdArray(scalars, MPhi_k[k]);
      std::memcpy(x_bufs[k].data(), scalars.data(),
                  V_eo * 24 * sizeof(double));
      x_ptrs[k] = x_bufs[k].data();
    }

    // Force buffer: QUDA expects QDP gauge order (4 per-direction blocks
    // of 18 doubles per site), in QUDA_MILC_GAUGE_ORDER ... actually the
    // wrapper uses gauge_param.gauge_order, so we keep QUDA_QDP_GAUGE_ORDER.
    int V = Quda::local_volume(ggrid);
    std::vector<double> mom_buf(4 * V * 18, 0.0);  // momentum, written by QUDA

    // QUDA's gauge_param/inv_param come from the multishift inverter — same
    // gauge upload, same κ/csw, same EVEN_EVEN_ASYMMETRIC matpc.
    QudaGaugeParam &gauge_param = quda_ms_->GaugeParam();
    QudaInvertParam &inv_param  = quda_ms_->InvertParam();

    // Coefficients: PowerNegHalf.residues[k] are the rational coefficients.
    // computeCloverForceQuda multiplies internally by 2·dt·coeff·kappa²
    // (see milc_interface.cpp:2655).  We want force only (no integration
    // step), so dt = 1.0 and the residues go in directly.
    std::vector<double> coeff(Npole);
    for (int k = 0; k < Npole; ++k) coeff[k] = PowerNegHalf.residues[k];

    const double kappa = inv_param.kappa;
    const double kappa2 = -kappa * kappa;
    const double ck = -inv_param.clover_csw * kappa / 8.0;

    // Need a flat host gauge buffer too — pass the same one that's loaded
    // in QUDA (resident).  We don't have direct access to it; QUDA reads
    // the resident gaugePrecise.  computeCloverForceQuda accepts gauge =
    // nullptr and uses the resident.  Let's pass nullptr.
    // (MILC passes a real buffer but it's unused if resident.)

    // p (second array of vectors): the function signature ignores it
    // ("void**" — see lib/interface_quda.cpp:4488 the second array param
    // is unused in the symmetric clover case).  Pass a dummy.
    std::vector<void *> p_ptrs(Npole, nullptr);

    computeCloverForceQuda(mom_buf.data(),
                           /*dt=*/1.0,
                           x_ptrs.data(),
                           p_ptrs.data(),
                           coeff.data(),
                           kappa2,
                           ck,
                           Npole,
                           /*multiplicity=*/1.0,
                           /*gauge=*/nullptr,
                           &gauge_param,
                           &inv_param);

    // Unpack mom_buf (4×V×18 doubles, QUDA_QDP_GAUGE_ORDER + EO site
    // order) into Grid's GaugeField.  This reverses the gauge load
    // packing in QudaCloverMultiShiftInverter::SetGauge.
    Coordinate lc = ggrid->LocalDimensions();
    std::vector<std::vector<double>> mom_eo_dirs(4, std::vector<double>(18 * V));
    for (int mu = 0; mu < 4; ++mu) {
      std::memcpy(mom_eo_dirs[mu].data(),
                  &mom_buf[mu * V * 18],
                  V * 18 * sizeof(double));
    }

    std::vector<std::vector<double>> mom_lex_dirs(4, std::vector<double>(18 * V));
    double *mom_lex_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      Quda::eo_to_lex_permute(mom_eo_dirs[mu].data(), mom_lex_dirs[mu].data(),
                              V, 18, lc);
      mom_lex_ptrs[mu] = mom_lex_dirs[mu].data();
    }

    Quda::lex_buffers_to_gauge(mom_lex_ptrs, dSdU);
  }

 private:
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_ms_;
};

}  // namespace Grid
