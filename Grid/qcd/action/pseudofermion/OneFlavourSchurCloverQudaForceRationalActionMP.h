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

  // Override: solve via QUDA multishift.  When QUDA_FORCE_KERNEL=1 (env)
  // also use computeCloverForceQuda for the gauge-deriv chain; otherwise
  // (default during debugging) fall back to Grid's per-pole deriv chain
  // for the gauge force, which validates the EVEN-parity multishift
  // independently of the QUDA force routine.
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

    // ------------------------------------------------------------------
    // Path A: Grid deriv chain (no QUDA force).  Same as the EVEN action's
    // base deriv() — validates the EVEN-parity multishift in isolation.
    // ------------------------------------------------------------------
    if (std::getenv("QUDA_FORCE_KERNEL") == nullptr) {
      SchurDifferentiableOperator<ImplD> Mpc(FermOp);
      FermionField X(fcbgrid), Y(fcbgrid);
      GaugeField tmp(ggrid);
      dSdU = Zero();
      for (int k = 0; k < Npole; ++k) {
        RealD ak = PowerNegHalf.residues[k];
        X = MPhi_k[k];
        Mpc.Mpc(X, Y);

        Mpc.MpcDeriv(tmp, Y, X);     dSdU = dSdU + ak * tmp;
        Mpc.MpcDagDeriv(tmp, X, Y);  dSdU = dSdU + ak * tmp;

        FermOp.MeeDeriv(tmp, Y, X, DaggerNo);    dSdU = dSdU + ak * tmp;
        FermOp.MeeDeriv(tmp, X, Y, DaggerYes);   dSdU = dSdU + ak * tmp;

        FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);
        FermOp.Meooe(X, tmp1);          FermOp.MooeeInv(tmp1, W_o);
        FermOp.MeooeDag(Y, tmp1);       FermOp.MooeeInvDag(tmp1, Z_o);
        FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);   dSdU = dSdU + ak * tmp;
        FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes);  dSdU = dSdU + ak * tmp;
      }
      return;
    }
    // ------------------------------------------------------------------
    // Path B: QUDA's computeCloverForceQuda — fused force routine.
    // ------------------------------------------------------------------

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

    // Force / momentum buffer: QUDA writes ASQTAD_MOM_LINKS, reconstruct=10
    // (anti-Hermitian traceless 3×3 packed in 10 reals per site/dir).
    // With gauge_order = QUDA_QDP_GAUGE_ORDER, the host buffer is 4 per-dir
    // blocks of V × 10 doubles each.
    int V = Quda::local_volume(ggrid);
    constexpr int MOM_RECON = 10;
    std::vector<double> mom_buf(4 * V * MOM_RECON, 0.0);

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

    // Unpack QUDA's 10-real anti-Hermitian-traceless momentum format into
    // Grid's GaugeField (3×3 complex matrix per site/dir).
    //
    // QUDA's mom layout (per-direction QDP order, EO site order, 10
    // doubles/site/dir).  The 10 reals encode the anti-Hermitian traceless
    // SU(3) algebra element packed as:
    //     [Im(M_00 - M_11)/2,   Im(M_11 - M_22)/2,        // 2 diag (3rd is fixed by traceless)
    //      Re(M_01), Im(M_01),
    //      Re(M_02), Im(M_02),
    //      Re(M_12), Im(M_12),
    //      <2 padding reals>]
    // (See QUDA gauge_field_order.h Reconstructor<10>.)  The Lie-algebra
    // element T satisfies T = -T†, tr T = 0.  We expand back to the full
    // 3×3 antihermitian matrix.
    Coordinate lc = ggrid->LocalDimensions();
    std::vector<std::vector<double>> dir_eo_18(4, std::vector<double>(18 * V));
    for (int mu = 0; mu < 4; ++mu) {
      const double *src = &mom_buf[mu * V * MOM_RECON];
      double *dst = dir_eo_18[mu].data();
      for (int site = 0; site < V; ++site) {
        const double *m = &src[site * MOM_RECON];
        // Anti-hermitian traceless reconstruction.
        // Diagonal:  iM[0][0] = m[0],  iM[1][1] = m[1],  iM[2][2] = -m[0]-m[1]
        // (i.e. M[k][k] is purely imaginary, real parts zero.)
        // Off-diag:  M[0][1] = m[2] + i*m[3], M[1][0] = -conj(M[0][1])
        //           M[0][2] = m[4] + i*m[5], M[2][0] = -conj(M[0][2])
        //           M[1][2] = m[6] + i*m[7], M[2][1] = -conj(M[1][2])
        double a0 = m[0], a1 = m[1];           // imag diag entries [0][0], [1][1]
        double a2 = -(a0 + a1);                // imag diag [2][2] (traceless)
        double r01 = m[2], i01 = m[3];
        double r02 = m[4], i02 = m[5];
        double r12 = m[6], i12 = m[7];
        double *d = &dst[site * 18];
        // Row 0
        d[ 0] = 0.0;       d[ 1] = a0;
        d[ 2] = r01;       d[ 3] = i01;
        d[ 4] = r02;       d[ 5] = i02;
        // Row 1
        d[ 6] = -r01;      d[ 7] =  i01;
        d[ 8] = 0.0;       d[ 9] = a1;
        d[10] = r12;       d[11] = i12;
        // Row 2
        d[12] = -r02;      d[13] =  i02;
        d[14] = -r12;      d[15] =  i12;
        d[16] = 0.0;       d[17] = a2;
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
  }

 private:
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_ms_;
};

}  // namespace Grid
