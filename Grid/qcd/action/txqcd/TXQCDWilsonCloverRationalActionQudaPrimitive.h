#pragma once
// QUDA σ-piece variant of the non-EO TXQCDWilsonCloverRationalAction.
//
// Mirrors TXQCDWilsonCloverRationalEOActionQudaPrimitive's structure but on
// the full-volume PF action: aux + Wilson-hop force stay in Grid, the
// per-pole clover σ_{μν} contraction is batched out to QUDA's
// computeCloverSigmaForceWithSchurFields kernel.  Because (X, Y) live on the
// full grid in the non-EO path, we parity-split each into (X_odd, X_even)
// and (Y_odd, Y_even) and feed them as (x_par, p_par, x_other, p_other).
//
// Convention note: the EO version uses asymmetric off-parity scaling
// (W_e/Z_e scale = √(2κ)) because W_e is a Schur completion of X with an
// implicit M_ee^{-1} factor.  For non-EO X_even / Y_even are just the EVEN
// parity halves of the full-volume solution / M·X — same kappa/mass form as
// the ODD halves — so the off-parity scaling should equal the par-parity
// scaling.  We use (X: 1.0, Y: 2κ) on BOTH parities.  Bit-equivalence vs
// the Grid backend in Step 1 is the validation gate; override the off-parity
// scale via FULL_PF_QUDA_SCHUR_SCALE if needed.
//
// Activated by env var TXQCD_QUDA_HYBRID=1 alongside USE_FULL_PF=1.
//
// **CONVENTION STATUS: VALIDATED 2026-06-09 at 4⁴.**  The parity-symmetric
// scaling (X: 1.0 both parities, Y: 2κ both parities, NO √(2κ) on the EVEN
// slot) bit-matches the Grid backend's force average to 8 sig figs on the
// first deriv() call at λ=2.0 4⁴ NO_METROP NTRAJ=1: 17.0571842 both backends.
// Total H also matches bit-exact (26351.3718501015).
//
// If higher-precision validation is needed at production scale, sweep
// FULL_PF_QUDA_SCHUR_SCALE_{X,Y} env knobs.

#include <Grid/qcd/action/txqcd/TXQCDWilsonCloverRationalAction.h>
#ifdef GRID_HAVE_QUDA
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaPackGpu.h>

#include <cstdlib>

NAMESPACE_BEGIN(Grid);

class TXQCDWilsonCloverRationalActionQudaPrimitive
    : public TXQCDWilsonCloverRationalAction {
 public:
  using Base   = TXQCDWilsonCloverRationalAction;
  using Params = Base::Params;

  TXQCDWilsonCloverRationalActionQudaPrimitive(
      GridCartesian &grid,
      GridRedBlackCartesian &rbgrid,
      const std::array<RealD, TxqcdNf> &mass,
      Params &p, RealD csw = 0.0)
      : Base(grid, rbgrid, mass, p, csw) {
    Quda::initialize();
    // QUDA loader for σ-piece kernel only (we don't drive QUDA's solver).
    QudaCloverParams qp;
    qp.mass = mass[0];   // light/degenerate; mirror EO QudaPrimitive
    qp.csw  = csw;
    QudaCloverMultiShiftSpec spec;
    spec.shifts = std::vector<RealD>(1, 0.0);
    spec.tols   = std::vector<RealD>(1, 1e-8);
    spec.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
    quda_loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        &this->grid_, qp, spec);
  }

  TXQCDWilsonCloverRationalActionQudaPrimitive(
      GridCartesian &grid,
      GridRedBlackCartesian &rbgrid,
      RealD mass, Params &p, RealD csw = 0.0)
      : TXQCDWilsonCloverRationalActionQudaPrimitive(
            grid, rbgrid, TXQCDSiteMatrixUtil::MassArray(mass), p, csw) {}

  std::string action_name() override {
    return "TXQCDWilsonCloverRationalActionQudaPrimitive";
  }

  void deriv(const TXQCDField &U, TXQCDField &dSdU) override {
    auto Mop = this->MakeOp(U);
    const int Npole = static_cast<int>(this->PowerNegHalf.poles.size());

    // Multishift CG: full-volume, parent's path (QUDA-accelerated M·v via
    // WilsonFermion's QUDA routing when TXQCD_QUDA_FULL=1).
    std::vector<TXQCDFermionNf> Xk;
    Xk.reserve(Npole);
    for (int k = 0; k < Npole; ++k) Xk.emplace_back(&this->grid_);
    std::vector<RealD> md_tol(Npole, this->param.mdtolerance);
    TXQCDMultiShiftCG(Mop, this->PowerNegHalf.poles, md_tol, this->Phi, Xk,
                      this->param.MaxIter);

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

    Mop.Wilson().ImportGauge(U.U);
    LatticeGaugeField gtmp(&this->grid_);
    LatticeGaugeField gforce(&this->grid_);

    // Stash per-pole Y for the σ-piece batch call below.
    std::vector<TXQCDFermionNf> Yk;
    Yk.reserve(Npole);

    for (int k = 0; k < Npole; ++k) {
      const RealD ak = this->PowerNegHalf.residues[k];

      TXQCDFermionNf &X = Xk[k];
      Yk.emplace_back(&this->grid_);
      TXQCDFermionNf &Y = Yk.back();
      Mop.M(X, Y);

      // ---- Aux-field forces (parent helper, full-grid path) ----
      this->AccumulateAuxForce(dSdU, ak, Y, X, g5, inv_sqrt2, Id, G5);

      // ---- Gauge hopping force (per-flavor Wilson MDeriv) ----
      gforce = Zero();
      for (int a = 0; a < TxqcdNf; ++a) {
        Mop.Wilson().MDeriv(gtmp, Y.f[a], X.f[a], DaggerNo);
        gforce = gforce + gtmp;
        Mop.Wilson().MDeriv(gtmp, X.f[a], Y.f[a], DaggerYes);
        gforce = gforce + gtmp;
      }
      dSdU.U = dSdU.U + ak * gforce;
    }

    // --------------------------------------------------------------------
    // QUDA σ-piece: batched all-pole, all-flavor call replaces the per-pole
    // Cmunu loop in the Grid backend.  Parity-split each (X, Y) into
    // (X_odd, X_even, Y_odd, Y_even) and feed via Schur kernel.
    // --------------------------------------------------------------------
    if (this->csw_ != 0.0) {
      quda_loader_->SetGauge(U.U);
      const int Nrhs = TxqcdNf * Npole;
      int V_eo = Quda::local_volume(&this->grid_) / 2;
      using SiteSpinor = typename LatticeFermion::scalar_object;
      static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                    "expected 24 doubles/site for fermion");

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
      // Non-EO off-parity scale: by default, same as par-parity scale, since
      // X_even / Y_even are not Schur completions.  Override via env if the
      // bit-equivalence test against the Grid backend fails.
      RealD scale_off_x = 1.0;
      RealD scale_off_y = two_kappa;
      if (const char *s = std::getenv("FULL_PF_QUDA_SCHUR_SCALE_X"); s && *s)
        scale_off_x = std::atof(s);
      if (const char *s = std::getenv("FULL_PF_QUDA_SCHUR_SCALE_Y"); s && *s)
        scale_off_y = std::atof(s);

      const int *lex_p = &pack_lex_table_dev_[0];
      double *dev_p    = &pack_dev_scratch_[0];
      const uint64_t off_x = 0 * per_slot;
      const uint64_t off_y = 1 * per_slot;
      const uint64_t off_w = 2 * per_slot;
      const uint64_t off_z = 3 * per_slot;

      LatticeFermion X_odd(&this->rbgrid_), X_even(&this->rbgrid_);
      LatticeFermion Y_odd(&this->rbgrid_), Y_even(&this->rbgrid_);
      for (int k = 0; k < Npole; ++k) {
        const RealD ak = this->PowerNegHalf.residues[k];
        for (int a = 0; a < TxqcdNf; ++a) {
          const int i = k * TxqcdNf + a;
          pickCheckerboard(Odd,  X_odd,  Xk[k].f[a]);
          pickCheckerboard(Even, X_even, Xk[k].f[a]);
          pickCheckerboard(Odd,  Y_odd,  Yk[k].f[a]);
          pickCheckerboard(Even, Y_even, Yk[k].f[a]);
          // Par slot (ODD): X mass-form, Y kappa-form (×2κ)
          Quda::GpuPackFermionRbLex(X_odd, 1.0,
                                     dev_p + off_x + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Y_odd, two_kappa,
                                     dev_p + off_y + i * per_rhs, lex_p);
          // Other slot (EVEN): same kappa/mass scales (non-Schur)
          Quda::GpuPackFermionRbLex(X_even, scale_off_x,
                                     dev_p + off_w + i * per_rhs, lex_p);
          Quda::GpuPackFermionRbLex(Y_even, scale_off_y,
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

      int V = Quda::local_volume(&this->grid_);
      constexpr int MOM_RECON = 10;
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
      inv_param.dagger        = QUDA_DAG_NO;
      inv_param.twist_flavor  = QUDA_TWIST_NO;
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

      const double kappa2 = -kappa * kappa;
      const double ck     = -inv_param.clover_csw * kappa / 8.0;
      const double dt     = 1.0;
      const double sigma_trace_coeff = 0.0;

      Quda::computeCloverSigmaForceWithSchurFields(
          &mom_buf_dev_[0],
          x_ptrs.data(), y_ptrs.data(),
          w_ptrs.data(), z_ptrs.data(),
          Nrhs, coeff_rhs,
          kappa2, ck, dt,
          sigma_trace_coeff,
          &force_gauge_param,
          &inv_param);

      inv_param.use_resident_solution = saved_use_resident;
      inv_param.dagger                = saved_dagger;
      inv_param.twist_flavor          = saved_twist;
      inv_param.input_location        = saved_input_loc;

      const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
      LatticeGaugeField clover_gforce(&this->grid_);
      Quda::GpuUnpackMomToGauge(&mom_buf_dev_[0], &unpack_eo_table_dev_[0],
                                 clover_gforce, quda_to_grid_factor);
      dSdU.U = dSdU.U + clover_gforce;
    }
  }

 private:
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_loader_;
  mutable deviceVector<int> pack_lex_table_dev_;
  mutable bool pack_lex_built_{false};
  mutable deviceVector<double> pack_dev_scratch_;
  mutable std::vector<double>  pack_host_scratch_;
  mutable deviceVector<int>    unpack_eo_table_dev_;
  mutable bool                 unpack_eo_built_{false};
  mutable deviceVector<double> mom_buf_dev_;
};

NAMESPACE_END(Grid);
#endif  // GRID_HAVE_QUDA
