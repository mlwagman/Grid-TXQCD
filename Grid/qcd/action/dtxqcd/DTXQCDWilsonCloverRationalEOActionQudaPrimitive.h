#pragma once
// Phase 3: DTXQCD light Nf=2 RHMC with the Wilson HOPPING gauge force offloaded
// to QUDA, mirroring TXQCD's Phase H (computeCloverWilsonForceWithSchurFields).
//
// The DTXQCD 48x48 doubling has upper block = M_QCD[U] and lower block =
// M_QCD[U*], via the identity  -C M_QCD[U]^T C = M_QCD[U]^*  (verified
// numerically).  The Wilson hopping gauge force is therefore TWO copies of
// TXQCD's single-block force:
//   - upper: standard QUDA Wilson force on gauge U
//   - lower: standard QUDA Wilson force on gauge conj(U); the result is
//            dS/dU_conj, mapped to dS/dU by entry-wise conjugation (chain rule
//            for U_conj = conj(U)).
// This reproduces DTXQCDWilsonCloverRationalEOAction::AccumulateHoppingForce
// (the per-pole MoeDeriv/MeoDeriv reference) but batches every (flavor, pole)
// rhs into one QUDA call per block.  Because conjugation is linear, conjugating
// the pole-summed QUDA result equals the per-pole conjugation of the reference.
//
// The aux-field forces, the LogDet force, and the site/clover-sigma forces stay
// in Grid/cuBLAS -- QUDA's clover is spin-only and cannot represent the
// spin x flavor Delta insertion.  Only the Wilson dslash-derivative (the
// dominant force-assembly cost, ~4.5 s/eval at 16^3x48) moves to QUDA.
//
// Activated by env DTXQCD_QUDA_HYBRID=1.  csw != 0 only: the QUDA clover loader
// needs a resident clover field, so csw == 0 falls back to the Grid reference.
//
// Conventions are byte-for-byte TXQCD Phase H (validated 2026-05-05 on plain
// QCD via bench_clover_oprod): X scale 1, Y mass-form scale 1, W/Z off-parity
// scale 2, dagger=YES, kappa2 = +kappa^2, force unpack -1/(8 kappa^2).

#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>  // gauge/clover loader
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaPackGpu.h>

#include <cstdlib>
#include <memory>

NAMESPACE_BEGIN(Grid);

class DTXQCDWilsonCloverRationalEOActionQudaPrimitive
    : public DTXQCDWilsonCloverRationalEOAction {
 public:
  using Base   = DTXQCDWilsonCloverRationalEOAction;
  using Params = Base::Params;

  DTXQCDWilsonCloverRationalEOActionQudaPrimitive(
      GridCartesian &grid, GridRedBlackCartesian &rbgrid,
      RealD mass, Params &p, RealD csw = 0.0)
      : Base(grid, rbgrid, mass, p, csw) {
    Quda::initialize();
    // Multishift inverter used only as a gauge+clover loader for the force
    // primitive (dummy single shift).  DTXQCD's light flavours are degenerate,
    // so the single light mass parameterizes the clover the same way TXQCD does.
    QudaCloverParams qp;
    qp.mass = mass;
    qp.csw  = csw;
    QudaCloverMultiShiftSpec spec;
    spec.shifts     = std::vector<RealD>(1, 0.0);
    spec.tols       = std::vector<RealD>(1, 1e-8);
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;  // DTXQCD keeps ODD parity
    quda_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        &this->grid_, qp, spec);
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalEOActionQudaPrimitive";
  }

 protected:
  // Override the all-poles hopping force: batch both doubled blocks through QUDA.
  void AccumulateHoppingForceAllPoles(
      const DTXQCDField &U,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      DTXQCDWilsonCloverFermionEO &Dw,
      DTXQCDField &dSdU) override {
    static int hybrid = []() {
      const char *e = std::getenv("DTXQCD_QUDA_HYBRID");
      return (e && *e) ? std::atoi(e) : 0;
    }();
    // csw == 0 has no resident clover field for the loader -> Grid reference.
    if (!hybrid || this->csw_ == 0.0) {
      Base::AccumulateHoppingForceAllPoles(U, Xk, Yk, Wek, Zek, Dw, dSdU);
      return;
    }

    // ---- Upper block: QUDA Wilson force on gauge U ----
    LatticeGaugeField gf_upper(&this->grid_);
    QudaHoppingBlock(U.U, /*lower=*/false, Xk, Yk, Wek, Zek, gf_upper);

    // ---- Lower block: QUDA Wilson force on conj(U); conjugate the result ----
    LatticeGaugeField Uconj(&this->grid_);
    Uconj = conjugate(U.U);
    LatticeGaugeField gf_lower(&this->grid_);
    QudaHoppingBlock(Uconj, /*lower=*/true, Xk, Yk, Wek, Zek, gf_lower);
    gf_lower = conjugate(gf_lower);  // dS/dU_conj -> dS/dU

    dSdU.U = dSdU.U + gf_upper + gf_lower;
  }

 private:
  // One doubled-block Wilson hopping force: pack all (flavor, pole) rhs, one
  // batched computeCloverWilsonForceWithSchurFields, unpack into out_force.
  void QudaHoppingBlock(const LatticeGaugeField &gauge, bool lower,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      LatticeGaugeField &out_force) {
    const int Npole = static_cast<int>(Xk.size());
    quda_loader_->SetGauge(gauge);

    const int Nrhs = DtxqcdNf * Npole;
    const int V_eo = Quda::local_volume(&this->grid_) / 2;
    using SiteSpinor = typename LatticeFermion::scalar_object;
    static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                  "expected 24 doubles/site for fermion");

    if (!pack_lex_built_) {
      Quda::BuildLexTable(&this->rbgrid_, pack_lex_table_dev_);
      pack_lex_built_ = true;
    }

    const uint64_t per_rhs  = uint64_t(V_eo) * 24;
    const uint64_t per_slot = uint64_t(Nrhs) * per_rhs;
    const uint64_t total    = 4 * per_slot;
    if (pack_dev_scratch_.size() < total) pack_dev_scratch_.resize(total);

    std::vector<void *> x_ptrs(Nrhs), y_ptrs(Nrhs), w_ptrs(Nrhs), z_ptrs(Nrhs);
    std::vector<double> coeff_rhs(Nrhs);

    QudaInvertParam &inv_param = quda_loader_->InvertParam();
    const double off_scale_wilson = 2.0;  // (1+-gamma)/2 projector compensation

    const int *lex_p = &pack_lex_table_dev_[0];
    double    *dev_p = &pack_dev_scratch_[0];
    const uint64_t off_x = 0 * per_slot;
    const uint64_t off_y = 1 * per_slot;
    const uint64_t off_w = 2 * per_slot;
    const uint64_t off_z = 3 * per_slot;

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = this->PowerNegQuarter.residues[k];
      for (int a = 0; a < DtxqcdNf; ++a) {
        const int i = k * DtxqcdNf + a;
        const LatticeFermion &Xf = lower ? Xk[k].lower.f[a]  : Xk[k].upper.f[a];
        const LatticeFermion &Yf = lower ? Yk[k].lower.f[a]  : Yk[k].upper.f[a];
        const LatticeFermion &Wf = lower ? Wek[k].lower.f[a] : Wek[k].upper.f[a];
        const LatticeFermion &Zf = lower ? Zek[k].lower.f[a] : Zek[k].upper.f[a];
        Quda::GpuPackFermionRbLex(Xf, 1.0,
                                   dev_p + off_x + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Yf, 1.0,               // mass-form Y
                                   dev_p + off_y + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Wf, off_scale_wilson,
                                   dev_p + off_w + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Zf, off_scale_wilson,
                                   dev_p + off_z + i * per_rhs, lex_p);
        coeff_rhs[i] = ak;
      }
    }
    for (int i = 0; i < Nrhs; ++i) {
      x_ptrs[i] = dev_p + off_x + i * per_rhs;
      y_ptrs[i] = dev_p + off_y + i * per_rhs;
      w_ptrs[i] = dev_p + off_w + i * per_rhs;
      z_ptrs[i] = dev_p + off_z + i * per_rhs;
    }

    const int V = Quda::local_volume(&this->grid_);
    constexpr int MOM_RECON = 10;
    if (mom_buf_dev_.size() < uint64_t(V) * 4 * MOM_RECON)
      mom_buf_dev_.resize(uint64_t(V) * 4 * MOM_RECON);
    if (!unpack_eo_built_) {
      Coordinate lc = this->grid_.LocalDimensions();
      Quda::BuildEoTable(&this->grid_, lc, unpack_eo_table_dev_);
      unpack_eo_built_ = true;
    }

    const double kappa = inv_param.kappa;
    int saved_use_resident = inv_param.use_resident_solution;
    QudaDagType saved_dagger = inv_param.dagger;
    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    QudaFieldLocation saved_input_loc = inv_param.input_location;
    inv_param.use_resident_solution = 0;
    inv_param.dagger         = QUDA_DAG_YES;              // Phase H validated
    inv_param.twist_flavor   = QUDA_TWIST_NO;
    inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;  // device pack

    QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION;
    force_gauge_param.overwrite_mom     = 1;
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;

    const double kappa2 = +kappa * kappa;   // Phase H validated (positive)
    const double ck     = -inv_param.clover_csw * kappa / 8.0;
    const double dt     = 1.0;

    Quda::computeCloverWilsonForceWithSchurFields(
        &mom_buf_dev_[0],
        x_ptrs.data(), y_ptrs.data(),
        w_ptrs.data(), z_ptrs.data(),
        Nrhs, coeff_rhs,
        kappa2, ck, dt,
        &force_gauge_param, &inv_param);

    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist;
    inv_param.input_location        = saved_input_loc;

    const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
    out_force = Zero();
    Quda::GpuUnpackMomToGauge(&mom_buf_dev_[0], &unpack_eo_table_dev_[0],
                               out_force, quda_to_grid_factor);
  }

  std::unique_ptr<QudaCloverMultiShiftInverter> quda_loader_;
  // Persistent device scratch (lazy, grows once): lex table + 4-slot pack buffer
  // (X, Y, W, Z) x Nrhs x V_eo x 24 doubles + full-grid mom buffer + EO table.
  deviceVector<int>    pack_lex_table_dev_;
  bool                 pack_lex_built_{false};
  deviceVector<double> pack_dev_scratch_;
  deviceVector<int>    unpack_eo_table_dev_;
  bool                 unpack_eo_built_{false};
  deviceVector<double> mom_buf_dev_;
};

NAMESPACE_END(Grid);
