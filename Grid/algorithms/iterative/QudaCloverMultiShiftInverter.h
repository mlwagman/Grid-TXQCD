#pragma once
// QUDA-backed multi-shift Wilson-clover inverter.
//
// Drop-in replacement for ConjugateGradientMultiShiftMixedPrec<LatticeFermion>
// in any code path that calls a Grid multishift CG via the
// OperatorMultiFunction interface — same SetGauge semantics as the
// single-shift QudaCloverInverter.
//
// QUDA's invertMultiShiftQuda solves (A + offset[k]) x_k = b for k = 0..N-1
// where A is the preconditioned squared operator (NORMOP_PC_SOLVE: M_pc^†M_pc).
// Grid's multishift uses the same convention, so shift values pass through
// unchanged.

#include <Grid/Grid.h>
#include <Grid/algorithms/LinearOperator.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>  // QudaCloverParams
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#ifndef GRID_HAVE_QUDA
#  error "QudaCloverMultiShiftInverter requires --with-quda"
#endif
#include <quda.h>
#include <quda_constants.h>

NAMESPACE_BEGIN(Grid);

// Shifts and per-shift tolerances must be supplied at construction.  In
// HMC, the rational-approximation shifts/residues are fixed for the
// lifetime of the action, so this matches usage cleanly.
struct QudaCloverMultiShiftSpec {
  std::vector<RealD> shifts;
  std::vector<RealD> tols;     // per-shift; if empty, all use overall tol
};

class QudaCloverMultiShiftInverter
    : public OperatorMultiFunction<LatticeFermion> {
public:
  QudaCloverMultiShiftInverter(GridBase *grid,
                               const QudaCloverParams &p,
                               const QudaCloverMultiShiftSpec &spec)
    : grid_(grid), params_(p), spec_(spec), gauge_loaded_(false) {
    assert(spec_.shifts.size() > 0);
    assert(spec_.shifts.size() <= QUDA_MAX_MULTI_SHIFT);
    if (!spec_.tols.empty())
      assert(spec_.tols.size() == spec_.shifts.size());
    setup_params_();
  }

  void SetGauge(const LatticeGaugeField &U) {
    int V = Quda::local_volume(grid_);
    Coordinate lc = grid_->LocalDimensions();
    std::vector<std::vector<double>> lex_bufs(4, std::vector<double>(18 * V));
    double *lex_ptrs[4] = {lex_bufs[0].data(), lex_bufs[1].data(),
                           lex_bufs[2].data(), lex_bufs[3].data()};
    Quda::gauge_to_lex_buffers(U, lex_ptrs);

    // Bake antiperiodic time phase into U_t at t = Lt-1 (matches Grid's
    // WilsonImpl convention; see QudaCloverInverter::SetGauge for full
    // commentary).
    if (params_.anti_periodic_t) {
      const int Lt = lc[3];
      int V_per_t = lc[0] * lc[1] * lc[2];
      int site_lo = (Lt - 1) * V_per_t;
      int site_hi = Lt * V_per_t;
      for (int site = site_lo; site < site_hi; ++site) {
        double *u_t = &lex_bufs[3][18 * site];
        for (int k = 0; k < 18; ++k) u_t[k] = -u_t[k];
      }
    }

    eo_bufs_ = std::vector<std::vector<double>>(4, std::vector<double>(18 * V));
    void *gauge_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      Quda::lex_to_eo_permute(lex_bufs[mu].data(), eo_bufs_[mu].data(),
                              V, 18, lc);
      gauge_ptrs[mu] = eo_bufs_[mu].data();
    }
    loadGaugeQuda(gauge_ptrs, &gauge_param_);
    loadCloverQuda(nullptr, nullptr, &inv_param_);
    gauge_loaded_ = true;
  }

  // OperatorMultiFunction interface: solve (A + shift[k]) x_k = src for all k.
  void operator()(LinearOperatorBase<LatticeFermion> &Linop,
                  const LatticeFermion &src,
                  std::vector<LatticeFermion> &out) override {
    (void)Linop;
    if (!gauge_loaded_) {
      assert(false && "QudaCloverMultiShiftInverter: SetGauge() not called");
    }
    int N = (int)spec_.shifts.size();
    assert((int)out.size() == N);

    int V = Quda::local_volume(grid_);
    Coordinate lc = grid_->LocalDimensions();

    std::vector<double> src_lex(24 * V), src_eo(24 * V);
    Quda::fermion_to_lex_buffer(src, src_lex.data());
    Quda::lex_to_eo_permute(src_lex.data(), src_eo.data(), V, 24, lc);

    // QUDA needs an array of solution pointers.
    std::vector<std::vector<double>> sol_eo(N, std::vector<double>(24 * V, 0.0));
    std::vector<void *> sol_ptrs(N);
    for (int k = 0; k < N; ++k) sol_ptrs[k] = sol_eo[k].data();

    invertMultiShiftQuda(sol_ptrs.data(), src_eo.data(), &inv_param_);

    for (int k = 0; k < N; ++k) {
      std::vector<double> sol_lex(24 * V);
      Quda::eo_to_lex_permute(sol_eo[k].data(), sol_lex.data(), V, 24, lc);
      Quda::lex_buffer_to_fermion(sol_lex.data(), out[k]);
    }

    last_iter_ = inv_param_.iter;
    last_secs_ = inv_param_.secs;
    last_res_per_shift_.assign(N, 0.0);
    for (int k = 0; k < N; ++k) last_res_per_shift_[k] = inv_param_.true_res_offset[k];
  }

  int LastIter() const { return last_iter_; }
  double LastSecs() const { return last_secs_; }
  const std::vector<double> &LastResPerShift() const {
    return last_res_per_shift_;
  }

private:
  void setup_params_() {
    Coordinate lc = grid_->LocalDimensions();

    // gauge_param same as single-shift inverter.
    gauge_param_ = newQudaGaugeParam();
    for (int d = 0; d < 4; ++d) gauge_param_.X[d] = lc[d];
    gauge_param_.anisotropy = 1.0;
    gauge_param_.type = QUDA_WILSON_LINKS;
    gauge_param_.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gauge_param_.t_boundary = params_.anti_periodic_t
                                ? QUDA_ANTI_PERIODIC_T
                                : QUDA_PERIODIC_T;
    gauge_param_.cpu_prec = QUDA_DOUBLE_PRECISION;
    gauge_param_.cuda_prec = params_.cuda_prec;
    gauge_param_.cuda_prec_sloppy = params_.cuda_prec_sloppy;
    gauge_param_.cuda_prec_precondition = params_.cuda_prec_sloppy;
    gauge_param_.cuda_prec_refinement_sloppy = params_.cuda_prec_sloppy;
    gauge_param_.reconstruct = params_.recon;
    gauge_param_.reconstruct_sloppy = params_.recon_sloppy;
    gauge_param_.reconstruct_precondition = params_.recon_sloppy;
    gauge_param_.reconstruct_refinement_sloppy = params_.recon_sloppy;
    gauge_param_.gauge_fix = QUDA_GAUGE_FIXED_NO;
    int x_face = lc[1] * lc[2] * lc[3] / 2;
    int y_face = lc[0] * lc[2] * lc[3] / 2;
    int z_face = lc[0] * lc[1] * lc[3] / 2;
    int t_face = lc[0] * lc[1] * lc[2] / 2;
    gauge_param_.ga_pad = std::max({x_face, y_face, z_face, t_face});
    gauge_param_.struct_size = sizeof(gauge_param_);

    // inv_param: multi-shift CG on the preconditioned squared operator.
    inv_param_ = newQudaInvertParam();
    inv_param_.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
    inv_param_.kappa  = 1.0 / (2.0 * (4.0 + params_.mass));
    inv_param_.mass   = params_.mass;
    inv_param_.Ls     = 1;

    inv_param_.clover_csw    = params_.csw;
    inv_param_.clover_coeff  = params_.csw * inv_param_.kappa;
    inv_param_.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param_.clover_cuda_prec = params_.cuda_prec;
    inv_param_.clover_cuda_prec_sloppy = params_.cuda_prec_sloppy;
    inv_param_.clover_cuda_prec_precondition = params_.cuda_prec_sloppy;
    inv_param_.clover_cuda_prec_refinement_sloppy = params_.cuda_prec_sloppy;
    inv_param_.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param_.compute_clover = 1;
    inv_param_.compute_clover_inverse = 1;

    inv_param_.inv_type        = QUDA_CG_INVERTER;
    inv_param_.solution_type   = QUDA_MATPCDAG_MATPC_SOLUTION;  // multishift on M_pc^† M_pc
    inv_param_.solve_type      = QUDA_NORMOP_PC_SOLVE;
    inv_param_.matpc_type      = QUDA_MATPC_EVEN_EVEN;
    inv_param_.dagger          = QUDA_DAG_NO;
    inv_param_.mass_normalization   = QUDA_MASS_NORMALIZATION;
    inv_param_.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

    inv_param_.tol      = params_.tol;
    inv_param_.maxiter  = params_.max_iter;
    inv_param_.reliable_delta = 1e-1;
    inv_param_.use_sloppy_partial_accumulator = 0;
    inv_param_.solution_accumulator_pipeline = 1;
    inv_param_.pipeline = 0;
    inv_param_.gcrNkrylov = 10;
    inv_param_.tol_restart = 0.0005;
    inv_param_.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
    inv_param_.tol_hq = 0.0;

    // Multishift-specific fields.
    int N = (int)spec_.shifts.size();
    inv_param_.num_offset = N;
    for (int k = 0; k < N; ++k) {
      inv_param_.offset[k] = spec_.shifts[k];
      inv_param_.tol_offset[k] =
          spec_.tols.empty() ? params_.tol : spec_.tols[k];
      inv_param_.tol_hq_offset[k] = 0.0;
    }

    inv_param_.cpu_prec  = QUDA_DOUBLE_PRECISION;
    inv_param_.cuda_prec = params_.cuda_prec;
    inv_param_.cuda_prec_sloppy = params_.cuda_prec_sloppy;
    inv_param_.cuda_prec_refinement_sloppy = params_.cuda_prec_sloppy;
    inv_param_.cuda_prec_precondition = params_.cuda_prec_sloppy;
    inv_param_.preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inv_param_.gamma_basis  = params_.gamma_basis;
    inv_param_.dirac_order  = QUDA_DIRAC_ORDER;
    inv_param_.input_location  = QUDA_CPU_FIELD_LOCATION;
    inv_param_.output_location = QUDA_CPU_FIELD_LOCATION;

    inv_param_.verbosity = QUDA_SUMMARIZE;
    inv_param_.struct_size = sizeof(inv_param_);
  }

  GridBase *grid_;
  QudaCloverParams params_;
  QudaCloverMultiShiftSpec spec_;
  QudaGaugeParam gauge_param_{};
  QudaInvertParam inv_param_{};
  std::vector<std::vector<double>> eo_bufs_;
  bool gauge_loaded_;
  int    last_iter_ = 0;
  double last_secs_ = 0.0;
  std::vector<double> last_res_per_shift_;
};

NAMESPACE_END(Grid);
