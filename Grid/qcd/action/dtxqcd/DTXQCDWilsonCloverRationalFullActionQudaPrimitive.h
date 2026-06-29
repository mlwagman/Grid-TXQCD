#pragma once
// Phase W.1: DTXQCD non-EO Wilson HOPPING gauge force offloaded to QUDA.
//
// Sibling of DTXQCDWilsonCloverRationalEOActionQudaPrimitive, but for the
// non-EO (USE_FULL_PF=1) action.  The parent DTXQCDWilsonCloverRationalFullAction
// runs the full-volume multishift CG (X_k, Y_k = M X_k) entirely on the
// full doubled M; only the Wilson dslash-derivative is changed here.
//
// Convention W.3 (closed via 4⁴ FD sweep, bit-exact vs Grid Wilson hop):
//   - Same QUDA force kernel: computeCloverWilsonForceWithSchurFields
//   - Same kappa form: kappa2 = +κ², dt = 1.0, dagger = YES,
//     ck = -csw·κ/8, unpack scale = -1/(8κ²)
//   - off_scale_wilson = -2.0
//     The EO sibling uses off_scale=+2.0 (Schur (1±γ)/2 compensator).  Non-EO
//     mass-form X, Y are raw pickCheckerboard slices of the full-volume X,Y,
//     so the projector compensator is absent — but the QUDA kernel was wired
//     for the Schur-form sign convention.  The W.1 hypothesis (drop the 2× to
//     1.0 since there's no projector) failed cos=-0.30; 4⁴ FD-bisected to
//     off_scale=-2.0, which lands bit-exact (|B|/|A|=1, cos=1 across all μ).
//     Net effect: off-parity bilinear contribution to dS/dU enters with the
//     same magnitude as the EO sibling but the OPPOSITE sign vs the par slot.
//
// Parity-split feeding:
//   The QUDA kernel expects (x_par, p_par) on PAR=ODD parity and
//   (x_other, p_other) on the EVEN half (matpc = ODD_ODD_ASYMMETRIC).
//   For non-EO we just  pickCheckerboard(Odd, ...)/pickCheckerboard(Even, ...)
//   on the full-volume X.upper.f[a] / Y.upper.f[a] (and same for .lower) and
//   pack each RB half into the corresponding slot.  No Schur completion
//   chain; no W, Z derivation.
//
// Gating: in production driver via USE_FULL_PF_QUDA_WILSON=1.  Default off.
//
// Aux + clover-gauge stay on the parent (no σ-force for non-EO Wilson port
// in W.1 — Wilson hop is the dominant cost).  A later phase can extend
// to σ-clover via parity-split + R4-7 if W.1 lands.

#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaPackGpu.h>

#include <cstdlib>
#include <memory>

NAMESPACE_BEGIN(Grid);

class DTXQCDWilsonCloverRationalFullActionQudaPrimitive
    : public DTXQCDWilsonCloverRationalFullAction {
 public:
  using Base   = DTXQCDWilsonCloverRationalFullAction;
  using Params = Base::Params;

  DTXQCDWilsonCloverRationalFullActionQudaPrimitive(
      GridCartesian &grid, GridRedBlackCartesian &rbgrid,
      RealD mass, Params &p, RealD csw = 0.0)
      : Base(grid, rbgrid, mass, p, csw) {
    Quda::initialize();
    QudaCloverParams qp;
    qp.mass = mass;
    qp.csw  = csw;
    QudaCloverMultiShiftSpec spec;
    spec.shifts     = std::vector<RealD>(1, 0.0);
    spec.tols       = std::vector<RealD>(1, 1e-8);
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
    quda_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        &this->grid_, qp, spec);
  }

  std::string action_name() override {
    return "DTXQCDWilsonCloverRationalFullActionQudaPrimitive";
  }

 protected:
  // ----------------------------------------------------------------------
  // Override the all-poles Wilson hopping force: batch both doubled blocks
  // (upper, lower) through QUDA's computeCloverWilsonForceWithSchurFields.
  // csw == 0 has no resident clover field for the QUDA loader -> Grid fallback.
  // USE_FULL_PF_QUDA_WILSON=0 also falls back (default OFF, escape hatch).
  // ----------------------------------------------------------------------
  void AccumulateHoppingForceAllPoles(
      const DTXQCDField &U,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      DTXQCDWilsonCloverFermionEO &Dw,
      DTXQCDField &dSdU) override {
    auto env_on = [](const char *k) {
      const char *e = std::getenv(k);
      return (e && *e) ? std::atoi(e) : 0;
    };
    const int gate = env_on("USE_FULL_PF_QUDA_WILSON");
    if (!gate || this->csw_ == 0.0) {
      Base::AccumulateHoppingForceAllPoles(U, Xk, Yk, Dw, dSdU);
      return;
    }

    // ---- Upper block: QUDA Wilson force on gauge U ----
    LatticeGaugeField gf_upper(&this->grid_);
    QudaHoppingBlockFull(U.U, /*lower=*/false, Xk, Yk, gf_upper);

    // ---- Lower block: QUDA Wilson force on conj(U); conjugate the result ----
    LatticeGaugeField Uconj(&this->grid_);
    Uconj = conjugate(U.U);
    LatticeGaugeField gf_lower(&this->grid_);
    QudaHoppingBlockFull(Uconj, /*lower=*/true, Xk, Yk, gf_lower);
    gf_lower = conjugate(gf_lower);  // dS/dU_conj -> dS/dU

    dSdU.U = dSdU.U + gf_upper + gf_lower;
  }

 private:
  // One doubled-block Wilson hopping force on FULL-volume X, Y inputs.
  // Parity-split each (flavor, pole) into RB halves and feed
  // computeCloverWilsonForceWithSchurFields.  Mirrors the EO sibling's
  // QudaHoppingBlock with off_scale_wilson = 1.0 instead of 2.0.
  void QudaHoppingBlockFull(const LatticeGaugeField &gauge, bool lower,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
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

    // W.3 closed: off_scale_wilson = -2.0 (see header docstring).  Knobs
    // W2_OFF_SCALE / W2_ZERO_OTHER retained for future convention probes.
    auto env_int = [](const char *k, int dflt) {
      const char *e = std::getenv(k);
      return (e && *e) ? std::atoi(e) : dflt;
    };
    auto env_dbl = [](const char *k, double dflt) {
      const char *e = std::getenv(k);
      return (e && *e) ? std::atof(e) : dflt;
    };
    const int    zero_other       = env_int("W2_ZERO_OTHER", 0);
    const double off_scale_wilson = zero_other ? 0.0 : env_dbl("W2_OFF_SCALE", -2.0);

    const int *lex_p = &pack_lex_table_dev_[0];
    double    *dev_p = &pack_dev_scratch_[0];
    const uint64_t off_x = 0 * per_slot;
    const uint64_t off_y = 1 * per_slot;
    const uint64_t off_w = 2 * per_slot;
    const uint64_t off_z = 3 * per_slot;

    // Scratch RB halves for parity-split (reused per RHS).
    LatticeFermion Xo(&this->rbgrid_), Xe(&this->rbgrid_);
    LatticeFermion Yo(&this->rbgrid_), Ye(&this->rbgrid_);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = this->PowerNegQuarter.residues[k];
      for (int a = 0; a < DtxqcdNf; ++a) {
        const int i = k * DtxqcdNf + a;
        const LatticeFermion &Xf = lower ? Xk[k].lower.f[a] : Xk[k].upper.f[a];
        const LatticeFermion &Yf = lower ? Yk[k].lower.f[a] : Yk[k].upper.f[a];

        // Parity-split full-vol X, Y into RB halves.  ODD = par, EVEN = other.
        pickCheckerboard(Odd,  Xo, Xf);
        pickCheckerboard(Even, Xe, Xf);
        pickCheckerboard(Odd,  Yo, Yf);
        pickCheckerboard(Even, Ye, Yf);

        // Pack into 4 buffers: (x_par, p_par) at scale=1, (x_other, p_other)
        // at off_scale_wilson.  Layout, scaling and lex-table all mirror the
        // EO sibling so the QUDA call signature is identical.
        Quda::GpuPackFermionRbLex(Xo, 1.0,
                                   dev_p + off_x + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Yo, 1.0,
                                   dev_p + off_y + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Xe, off_scale_wilson,
                                   dev_p + off_w + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Ye, off_scale_wilson,
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
    // W.2-diag dagger probe knob.
    inv_param.dagger         = env_int("W2_DAGGER_NO", 0) ? QUDA_DAG_NO : QUDA_DAG_YES;
    inv_param.twist_flavor   = QUDA_TWIST_NO;
    inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;

    QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION;
    force_gauge_param.overwrite_mom     = 1;
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;

    const double kappa2 = +kappa * kappa;
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

  // QUDA gauge+clover loader (single dummy shift; used only as a loader).
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_loader_;

  // Device-resident pack/unpack scratch and lex/eo tables.  Lazy-built on
  // first QudaHoppingBlockFull call; reused across all subsequent calls.
  deviceVector<int>    pack_lex_table_dev_;
  bool                 pack_lex_built_ = false;
  deviceVector<double> pack_dev_scratch_;
  deviceVector<int>    unpack_eo_table_dev_;
  bool                 unpack_eo_built_ = false;
  deviceVector<double> mom_buf_dev_;
};

NAMESPACE_END(Grid);
