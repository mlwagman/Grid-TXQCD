#pragma once
// Phase C: TXQCD light Nf=2 RHMC with hybrid Grid + QUDA σ-piece force.
//
// Mirrors TXQCDWilsonCloverRationalEOAction's deriv() exactly, but replaces
// the per-pole Cmunu sigma-contraction loop (Phase B-prime equivalent of
// Path A's MeeDeriv+MooDeriv) with a single QUDA computeCloverSigmaOprod +
// cloverDerivative call that batches all (flavor, pole) rhs.
//
// TXQCD's Δ-aware multishift, aux-field force, and Wilson-hop gauge force
// stay in Grid host code (Δ doesn't enter the Wilson hop).  The clover
// gauge force from M_ee = (1+T) + Δ has ∂_U(M_ee) = ∂_U(1+T) (since Δ has
// no U dependence), so QUDA's σ_μν·F_μν kernel produces it correctly when
// fed Δ-aware (X, Y, W_e, Z_e) bilinear inputs from TXQCDWilsonCloverFermionEO.
//
// Activated by env var TXQCD_QUDA_HYBRID=1 in production.

#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalEOAction.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>  // gauge/clover loader
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaPackGpu.h>

#include <cstdlib>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonCloverRationalEOActionQudaPrimitive
    : public TXQCDWilsonCloverRationalEOAction {
 public:
  using Base = TXQCDWilsonCloverRationalEOAction;
  using Params = Base::Params;

  TXQCDWilsonCloverRationalEOActionQudaPrimitive(
      GridCartesian &grid,
      GridRedBlackCartesian &rbgrid,
      const std::array<RealD, TxqcdNf> &mass,
      Params &p, RealD csw = 0.0)
      : Base(grid, rbgrid, mass, p, csw) {
    Quda::initialize();
    // Build a multishift inverter just for QUDA gauge+clover loading.
    // QudaCloverParams uses the light quark mass (mass[0]; assume degenerate
    // for now since that's what TXQCDWilsonCloverRationalEOAction targets).
    QudaCloverParams qp;
    qp.mass = mass[0];
    qp.csw  = csw;
    QudaCloverMultiShiftSpec spec;
    spec.shifts = std::vector<RealD>(1, 0.0);  // dummy single shift
    spec.tols   = std::vector<RealD>(1, 1e-8);
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
    quda_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        &this->grid_, qp, spec);
  }

  TXQCDWilsonCloverRationalEOActionQudaPrimitive(
      GridCartesian &grid,
      GridRedBlackCartesian &rbgrid,
      RealD mass, Params &p, RealD csw = 0.0)
      : TXQCDWilsonCloverRationalEOActionQudaPrimitive(
            grid, rbgrid,
            TXQCDSiteMatrixUtil::MassArray(mass), p, csw) {}

  std::string action_name() override {
    return "TXQCDWilsonCloverRationalEOActionQudaPrimitive";
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    // ----------------------------------------------------------------------
    // Phase A (Grid TXQCD path) — same as parent.
    // ----------------------------------------------------------------------
    auto EOp = this->MakeEOp(U);
    TXQCDCloverSchurOp SchurOp(EOp);
    const int Npole = static_cast<int>(this->PowerNegHalf.poles.size());

    n_deriv_++;
    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&this->rbgrid_);
    std::vector<RealD> md_tol(Npole, this->param.mdtolerance);
    TXQCDMultiShiftCGSchur MSCG(this->param.MaxIter);
    auto t_cg0 = usecond();
    MSCG(SchurOp, this->PowerNegHalf.poles, md_tol, this->Phi, Xk);
    t_cg_us_ += usecond() - t_cg0;

    dSdU.sigma = Zero();
    dSdU.pi    = Zero();
    dSdU.s     = Zero();
    dSdU.p     = Zero();
    dSdU.t     = Zero();
    dSdU.U     = Zero();

    const ComplexD inv_sqrt2(1.0 / std::sqrt(2.0), 0.0);
    SpinTable Id{IdentitySpinMatrix()};
    SpinTable G5{Gamma5Matrix()};
    Gamma g5(Gamma::Algebra::Gamma5);

    LatticeGaugeField gforce(&this->grid_);
    LatticeGaugeField gtmp(&this->grid_);

    // Stash per-pole Y, W_e, Z_e for the QUDA σ-piece batch call below.
    std::vector<TXQCDFermionNf> Yk;       Yk.reserve(Npole);
    std::vector<TXQCDFermionNf> Wek;      Wek.reserve(Npole);
    std::vector<TXQCDFermionNf> Zek;      Zek.reserve(Npole);

    auto t_pp0 = usecond();
    for (int k = 0; k < Npole; ++k) {
      const RealD ak = this->PowerNegHalf.residues[k];
      TXQCDFermionNf &X = Xk[k];
      Yk.emplace_back(&this->rbgrid_);
      Wek.emplace_back(&this->rbgrid_);
      Zek.emplace_back(&this->rbgrid_);
      TXQCDFermionNf &Y   = Yk.back();
      TXQCDFermionNf &W_e = Wek.back();
      TXQCDFermionNf &Z_e = Zek.back();

      auto t_pp_mpc0 = usecond();
      SchurOp.Mpc(X, Y);
      t_perpole_mpc_us_ += usecond() - t_pp_mpc0;
      auto t_pp_wz0 = usecond();
      TXQCDFermionNf tmp_e(&this->rbgrid_);
      EOp.Meooe(X, tmp_e);        EOp.MooeeInv(tmp_e, W_e);
      EOp.MeooeDag(Y, tmp_e);     EOp.MooeeInvDag(tmp_e, Z_e);
      t_perpole_wezee_us_ += usecond() - t_pp_wz0;

      // ---- Aux-field forces (same as parent) ----
      auto t_pp_aux0 = usecond();
      this->AccumulateAuxForce(dSdU, ak, Y,   X,   g5, inv_sqrt2, Id, G5);
      this->AccumulateAuxForce(dSdU, ak, Z_e, W_e, g5, inv_sqrt2, Id, G5);
      t_perpole_aux_us_ += usecond() - t_pp_aux0;

      // ---- Gauge force (Wilson hopping) ----
      // TXQCD_QUDA_FULL=1 skips this loop and lets QUDA's
      // computeCloverForceWithGridY handle the Wilson+σ kernel below.
      if (std::getenv("TXQCD_QUDA_FULL") == nullptr) {
        gforce = Zero();
        LatticeGaugeField ForceO(&this->rbgrid_), ForceE(&this->rbgrid_);
        for (int a = 0; a < TxqcdNf; ++a) {
          auto t_h0 = usecond();
          EOp.Wilson().MoeDeriv(ForceO, Y.f[a], W_e.f[a], DaggerNo);
          EOp.Wilson().MeoDeriv(ForceE, Z_e.f[a], X.f[a], DaggerNo);
          t_hop_us_ += usecond() - t_h0;
          n_hop_++;
          setCheckerboard(gtmp, ForceO);
          setCheckerboard(gtmp, ForceE);
          gforce = gforce - gtmp;

          auto t_h1 = usecond();
          EOp.Wilson().MoeDeriv(ForceO, X.f[a], Z_e.f[a], DaggerYes);
          EOp.Wilson().MeoDeriv(ForceE, W_e.f[a], Y.f[a], DaggerYes);
          t_hop_us_ += usecond() - t_h1;
          n_hop_++;
          setCheckerboard(gtmp, ForceO);
          setCheckerboard(gtmp, ForceE);
          gforce = gforce - gtmp;
        }
        dSdU.U = dSdU.U + ak * gforce;
      }
    }

    // ----------------------------------------------------------------------
    // Phase H (TXQCD_QUDA_FULL=1): Wilson-piece QUDA force.  Replaces Grid's
    // per-pole MoeDeriv+MeoDeriv chain (skipped via the FULL guard above)
    // with QUDA's computeCloverOprod fed Schur-completed off-parity inputs.
    //
    // CONVENTIONS (validated 2026-05-05 on plain QCD via bench_clover_oprod
    // at cos=1.0, factor=1.0 on 4⁴/8⁴/12⁴):
    //   - Y in MASS-form (no 2κ rescale).  σ-piece uses kappa-form — that's
    //     why this is split off from computeCloverFullForceWithSchurFields.
    //   - W_e, Z_e off-parity scale = 2.0  (compensates the 1/2 implicit in
    //     QUDA's (1±γ_μ)/2 .project().reconstruct() chirality projector).
    //     σ-piece uses √(2κ) — different normalization.
    //   - inv_param.dagger = QUDA_DAG_YES.
    //   - kappa2 = +κ·κ  (POSITIVE — Phase D.1 used negative which gave
    //     cos=-1.0; flipping the sign gives perfect agreement).
    //
    // The σ-piece branch below runs unchanged (its convention is already
    // validated by the working HYBRID path), and accumulates additively.
    t_perpole_us_ += usecond() - t_pp0;
    if (std::getenv("TXQCD_QUDA_FULL") != nullptr && this->csw_ != 0.0) {
      auto t_wpack0 = usecond();
      quda_loader_->SetGauge(EOp.Gauge());
      const int Nrhs = TxqcdNf * Npole;
      int V_eo = Quda::local_volume(&this->grid_) / 2;
      using SiteSpinor = typename LatticeFermion::scalar_object;
      static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                    "expected 24 doubles/site for fermion");

      // Build lex table once per action lifetime (depends only on RB grid).
      if (!pack_lex_built_) {
        Quda::BuildLexTable(&this->rbgrid_, pack_lex_table_dev_);
        pack_lex_built_ = true;
      }

      // Allocate combined device + host scratch: 4 slots × Nrhs × V_eo × 24.
      const uint64_t per_rhs = uint64_t(V_eo) * 24;
      const uint64_t per_slot = uint64_t(Nrhs) * per_rhs;
      const uint64_t total    = 4 * per_slot;
      if (pack_dev_scratch_.size() < total) {
        pack_dev_scratch_.resize(total);
        pack_host_scratch_.resize(total);
      }

      std::vector<void *> x_ptrs(Nrhs), y_ptrs(Nrhs), w_ptrs(Nrhs), z_ptrs(Nrhs);
      std::vector<double> coeff_rhs(Nrhs);

      QudaInvertParam &inv_param = quda_loader_->InvertParam();
      const double kappa = inv_param.kappa;
      const double off_scale_wilson = 2.0;  // Phase H validated.

      const int *lex_p   = &pack_lex_table_dev_[0];
      double    *dev_p   = &pack_dev_scratch_[0];
      double    *host_p  = pack_host_scratch_.data();
      const uint64_t off_x = 0 * per_slot;
      const uint64_t off_y = 1 * per_slot;
      const uint64_t off_w = 2 * per_slot;
      const uint64_t off_z = 3 * per_slot;

      // GPU pack: Nrhs × 4 = 160 kernels writing into one device buffer.
      for (int k = 0; k < Npole; ++k) {
        const RealD ak = this->PowerNegHalf.residues[k];
        for (int a = 0; a < TxqcdNf; ++a) {
          const int i = k * TxqcdNf + a;
          Quda::GpuPackFermionRbLex(Xk[k].f[a],  1.0,
                                     dev_p + off_x + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Yk[k].f[a],  1.0,                // mass-form Y
                                     dev_p + off_y + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Wek[k].f[a], off_scale_wilson,   // 2.0
                                     dev_p + off_w + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Zek[k].f[a], off_scale_wilson,
                                     dev_p + off_z + i * per_rhs, lex_p);
          coeff_rhs[i] = ak;
        }
      }
      // Pass DEVICE pointers directly to QUDA — CSF construction will treat
      // them as device-resident, avoiding H2D entirely.  Set
      // inv_param->input_location accordingly below.
      for (int i = 0; i < Nrhs; ++i) {
        x_ptrs[i] = dev_p + off_x + i * per_rhs;
        y_ptrs[i] = dev_p + off_y + i * per_rhs;
        w_ptrs[i] = dev_p + off_w + i * per_rhs;
        z_ptrs[i] = dev_p + off_z + i * per_rhs;
      }
      t_wilson_pack_us_ += usecond() - t_wpack0;

      int V = Quda::local_volume(&this->grid_);
      constexpr int MOM_RECON = 10;
      // Resize device mom buffer + build EO table once.
      if (mom_buf_dev_.size() < uint64_t(V) * 4 * MOM_RECON) {
        mom_buf_dev_.resize(uint64_t(V) * 4 * MOM_RECON);
      }
      if (!unpack_eo_built_) {
        Coordinate lc = this->grid_.LocalDimensions();
        Quda::BuildEoTable(&this->grid_, lc, unpack_eo_table_dev_);
        unpack_eo_built_ = true;
      }

      int saved_use_resident = inv_param.use_resident_solution;
      QudaDagType saved_dagger = inv_param.dagger;
      QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
      QudaFieldLocation saved_input_loc = inv_param.input_location;
      inv_param.use_resident_solution = 0;
      inv_param.dagger        = QUDA_DAG_YES;       // Phase H validated
      inv_param.twist_flavor  = QUDA_TWIST_NO;
      inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;  // device pack

      QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
      force_gauge_param.type        = QUDA_GENERAL_LINKS;
      force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
      force_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION; // device mom
      force_gauge_param.overwrite_mom     = 1;
      force_gauge_param.use_resident_mom  = 0;
      force_gauge_param.make_resident_mom = 0;
      force_gauge_param.return_result_mom = 1;

      const double kappa2 = +kappa * kappa;   // Phase H validated (positive)
      const double ck     = -inv_param.clover_csw * kappa / 8.0;
      const double dt     = 1.0;

      auto t_wcall0 = usecond();
      Quda::computeCloverWilsonForceWithSchurFields(
          &mom_buf_dev_[0],
          x_ptrs.data(), y_ptrs.data(),
          w_ptrs.data(), z_ptrs.data(),
          Nrhs, coeff_rhs,
          kappa2, ck, dt,
          &force_gauge_param,
          &inv_param);
      t_wilson_call_us_ += usecond() - t_wcall0;

      inv_param.use_resident_solution = saved_use_resident;
      inv_param.dagger                = saved_dagger;
      inv_param.twist_flavor          = saved_twist;
      inv_param.input_location        = saved_input_loc;

      auto t_wunpack0 = usecond();
      const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
      LatticeGaugeField wilson_gforce(&this->grid_);
      Quda::GpuUnpackMomToGauge(&mom_buf_dev_[0], &unpack_eo_table_dev_[0],
                                 wilson_gforce, quda_to_grid_factor);
      dSdU.U = dSdU.U + wilson_gforce;
      t_wilson_unpack_us_ += usecond() - t_wunpack0;
      // Fall through to σ branch below (additive).
    }

    // ----------------------------------------------------------------------
    // Phase C: QUDA σ-piece batched call replacing parent's clover Cmunu loop.
    //
    // Pack all (flavor a, pole k) rhs as N = TxqcdNf * Npole entries.  For
    // each rhs feed:
    //   x_par   = Xk[k].f[a]  (ODD parity slot, multishift solution)
    //   p_par   = Yk[k].f[a]  (ODD, M_pc·X)
    //   x_other = Wek[k].f[a] (EVEN, Schur completion W_e = Mee^{-1}·Meo·X)
    //   p_other = Zek[k].f[a] (EVEN, Z_e = Mee^{-1†}·Moe†·Y)
    //
    // matpc_type = ODD_ODD_ASYMMETRIC (TXQCD's kept parity is ODD).
    // Per Phase B-prime: uniform coeff on both parities, scale_off = √(2κ)
    // on W_e and Z_e to compensate kappa-form vs mass-form normalization.
    // ----------------------------------------------------------------------
    if (this->csw_ != 0.0) {
      auto t_spack0 = usecond();
      quda_loader_->SetGauge(EOp.Gauge());

      const int Nrhs = TxqcdNf * Npole;
      int V_eo = Quda::local_volume(&this->grid_) / 2;
      using SiteSpinor = typename LatticeFermion::scalar_object;
      static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                    "expected 24 doubles/site for fermion");

      // Build lex table once per action lifetime (shared with FULL branch).
      if (!pack_lex_built_) {
        Quda::BuildLexTable(&this->rbgrid_, pack_lex_table_dev_);
        pack_lex_built_ = true;
      }

      const uint64_t per_rhs = uint64_t(V_eo) * 24;
      const uint64_t per_slot = uint64_t(Nrhs) * per_rhs;
      const uint64_t total    = 4 * per_slot;
      if (pack_dev_scratch_.size() < total) {
        pack_dev_scratch_.resize(total);
        pack_host_scratch_.resize(total);
      }

      std::vector<void *> x_ptrs(Nrhs), y_ptrs(Nrhs), w_ptrs(Nrhs), z_ptrs(Nrhs);
      std::vector<double> coeff_rhs(Nrhs);

      QudaInvertParam &inv_param = quda_loader_->InvertParam();
      const double kappa = inv_param.kappa;
      const double two_kappa = 2.0 * kappa;
      const double sqrt_two_kappa = std::sqrt(two_kappa);
      RealD scale_off = sqrt_two_kappa;
      if (const char *s = std::getenv("QUDA_FORCE_SCHUR_SCALE"); s && *s)
        scale_off = std::atof(s);

      const int *lex_p  = &pack_lex_table_dev_[0];
      double    *dev_p  = &pack_dev_scratch_[0];
      double    *host_p = pack_host_scratch_.data();
      const uint64_t off_x = 0 * per_slot;
      const uint64_t off_y = 1 * per_slot;
      const uint64_t off_w = 2 * per_slot;
      const uint64_t off_z = 3 * per_slot;

      for (int k = 0; k < Npole; ++k) {
        const RealD ak = this->PowerNegHalf.residues[k];
        for (int a = 0; a < TxqcdNf; ++a) {
          const int i = k * TxqcdNf + a;
          // Parity slot (ODD): X̂, kappa-form Y = (2κ)·M_pc·X̂
          Quda::GpuPackFermionRbLex(Xk[k].f[a],  1.0,
                                     dev_p + off_x + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Yk[k].f[a],  two_kappa,
                                     dev_p + off_y + i * per_rhs, lex_p);
          // Off-parity slot (EVEN): W_e, Z_e with √(2κ) scale per Phase B-prime.
          Quda::GpuPackFermionRbLex(Wek[k].f[a], scale_off,
                                     dev_p + off_w + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Zek[k].f[a], scale_off,
                                     dev_p + off_z + i * per_rhs, lex_p);
          coeff_rhs[i] = ak;
        }
      }
      // Device pointers passed directly — no D2H, no H2D.
      for (int i = 0; i < Nrhs; ++i) {
        x_ptrs[i] = dev_p + off_x + i * per_rhs;
        y_ptrs[i] = dev_p + off_y + i * per_rhs;
        w_ptrs[i] = dev_p + off_w + i * per_rhs;
        z_ptrs[i] = dev_p + off_z + i * per_rhs;
      }

      int V = Quda::local_volume(&this->grid_);
      constexpr int MOM_RECON = 10;
      // Resize device mom buffer + EO table (shared with FULL branch).
      if (mom_buf_dev_.size() < uint64_t(V) * 4 * MOM_RECON) {
        mom_buf_dev_.resize(uint64_t(V) * 4 * MOM_RECON);
      }
      if (!unpack_eo_built_) {
        Coordinate lc = this->grid_.LocalDimensions();
        Quda::BuildEoTable(&this->grid_, lc, unpack_eo_table_dev_);
        unpack_eo_built_ = true;
      }

      // QUDA force-call setup (mirrors strange RHMC's Phase B-prime).
      int saved_use_resident = inv_param.use_resident_solution;
      QudaDagType saved_dagger = inv_param.dagger;
      QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
      QudaFieldLocation saved_input_loc = inv_param.input_location;
      inv_param.use_resident_solution = 0;
      inv_param.dagger        = QUDA_DAG_NO;
      inv_param.twist_flavor  = QUDA_TWIST_NO;
      inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;  // device pack

      QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
      force_gauge_param.type        = QUDA_GENERAL_LINKS;
      force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
      force_gauge_param.location    = QUDA_CUDA_FIELD_LOCATION; // device mom
      force_gauge_param.overwrite_mom     = 1;
      force_gauge_param.use_resident_mom  = 0;
      force_gauge_param.make_resident_mom = 0;
      force_gauge_param.return_result_mom = 1;

      const double kappa2 = -kappa * kappa;
      const double ck     = -inv_param.clover_csw * kappa / 8.0;
      const double dt     = 1.0;
      const double sigma_trace_coeff = 0.0;
      t_sigma_pack_us_ += usecond() - t_spack0;

      auto t_scall0 = usecond();
      Quda::computeCloverSigmaForceWithSchurFields(
          &mom_buf_dev_[0],
          x_ptrs.data(), y_ptrs.data(),
          w_ptrs.data(), z_ptrs.data(),
          Nrhs, coeff_rhs,
          kappa2, ck, dt,
          sigma_trace_coeff,
          &force_gauge_param,
          &inv_param);
      t_sigma_call_us_ += usecond() - t_scall0;

      inv_param.use_resident_solution = saved_use_resident;
      inv_param.dagger                = saved_dagger;
      inv_param.twist_flavor          = saved_twist;
      inv_param.input_location        = saved_input_loc;

      auto t_sunpack0 = usecond();
      const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
      // Convention-A correction (-1/2) applied by parent after Cmunu; QUDA
      // already returns convention-A force, so just scale by -1/(8κ²).
      LatticeGaugeField clover_gforce(&this->grid_);
      Quda::GpuUnpackMomToGauge(&mom_buf_dev_[0], &unpack_eo_table_dev_[0],
                                 clover_gforce, quda_to_grid_factor);
      dSdU.U = dSdU.U + clover_gforce;
      t_sigma_unpack_us_ += usecond() - t_sunpack0;
    }
  }

 private:
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_loader_;
  // Timer for the per-pole Wilson hop loop (MoeDeriv+MeoDeriv pairs).
  mutable uint64_t t_hop_us_{0};
  mutable uint64_t n_hop_{0};
  // Phase H profile timers (FULL stack).
  mutable uint64_t t_cg_us_{0};
  mutable uint64_t t_perpole_us_{0};
  mutable uint64_t t_perpole_mpc_us_{0};      // Mpc(X) → Y
  mutable uint64_t t_perpole_wezee_us_{0};    // W_e/Z_e Schur completions
  mutable uint64_t t_perpole_aux_us_{0};      // 2× AccumulateAuxForce
  mutable uint64_t t_wilson_pack_us_{0};
  mutable uint64_t t_wilson_call_us_{0};
  mutable uint64_t t_wilson_unpack_us_{0};
  mutable uint64_t t_sigma_pack_us_{0};
  mutable uint64_t t_sigma_call_us_{0};
  mutable uint64_t t_sigma_unpack_us_{0};
  mutable uint64_t n_deriv_{0};
  // Phase H GPU pack: lex table (oSite,lane → V_eo lex index) cached once.
  mutable deviceVector<int> pack_lex_table_dev_;
  mutable bool pack_lex_built_{false};
  // Persistent device + host scratch for batched pack: 4 buffer slots
  // (X, Y, W, Z) × Nrhs × V_eo × 24 doubles, allocated lazily.
  mutable deviceVector<double> pack_dev_scratch_;
  mutable std::vector<double>  pack_host_scratch_;
  // Phase H GPU unpack: full-grid EO source-site table + device mom buffer.
  mutable deviceVector<int>    unpack_eo_table_dev_;
  mutable bool                 unpack_eo_built_{false};
  mutable deviceVector<double> mom_buf_dev_;

 public:
  void PrintHopTimer(const char *tag = "") const {
    if (n_hop_ == 0 && n_deriv_ == 0) return;
    if (n_hop_ > 0) {
      std::cout << GridLogMessage
                << "[TXQCDWilsonClover.hop/" << tag << "] "
                << n_hop_ << " hop pairs, "
                << double(t_hop_us_) * 1e-6 << " s total ("
                << double(t_hop_us_) * 1e-3 / n_hop_ << " ms/pair)" << std::endl;
    }
    if (n_deriv_ > 0) {
      std::cout << GridLogMessage
                << "[TXQCDWilsonClover.profile/" << tag << "] " << n_deriv_
                << " deriv calls (s total / ms/call):"
                << "  CG="  << double(t_cg_us_) * 1e-6 << " / "
                                << double(t_cg_us_) * 1e-3 / n_deriv_
                << "  perpole=" << double(t_perpole_us_) * 1e-6 << " / "
                                << double(t_perpole_us_) * 1e-3 / n_deriv_
                << "  Wpack=" << double(t_wilson_pack_us_) * 1e-6 << " / "
                                << double(t_wilson_pack_us_) * 1e-3 / n_deriv_
                << "  Wcall=" << double(t_wilson_call_us_) * 1e-6 << " / "
                                << double(t_wilson_call_us_) * 1e-3 / n_deriv_
                << "  Wunpack=" << double(t_wilson_unpack_us_) * 1e-6 << " / "
                                  << double(t_wilson_unpack_us_) * 1e-3 / n_deriv_
                << "  σpack=" << double(t_sigma_pack_us_) * 1e-6 << " / "
                                << double(t_sigma_pack_us_) * 1e-3 / n_deriv_
                << "  σcall=" << double(t_sigma_call_us_) * 1e-6 << " / "
                                << double(t_sigma_call_us_) * 1e-3 / n_deriv_
                << "  σunpack=" << double(t_sigma_unpack_us_) * 1e-6 << " / "
                                  << double(t_sigma_unpack_us_) * 1e-3 / n_deriv_
                << std::endl;
      std::cout << GridLogMessage
                << "[TXQCDWilsonClover.profile/" << tag << "] perpole breakdown (ms/call):"
                << "  Mpc=" << double(t_perpole_mpc_us_) * 1e-3 / n_deriv_
                << "  W_e/Z_e=" << double(t_perpole_wezee_us_) * 1e-3 / n_deriv_
                << "  Aux=" << double(t_perpole_aux_us_) * 1e-3 / n_deriv_
                << std::endl;
    }
  }
  ~TXQCDWilsonCloverRationalEOActionQudaPrimitive() { PrintHopTimer("dtor"); }
};

NAMESPACE_END(Grid);
