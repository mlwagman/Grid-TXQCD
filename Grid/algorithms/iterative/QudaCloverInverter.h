#pragma once
// Single-shift QUDA Wilson-clover inverter, presented as a Grid
// OperatorFunction<LatticeFermion>.
//
// Drop-in replacement for ConjugateGradient<LatticeFermion> in any code path
// that calls a Grid CG via the OperatorFunction interface.  The Linop
// argument is ignored — QUDA always operates against the gauge/clover it
// last loaded via loadGauge/loadCloverQuda.  Caller MUST invoke
// SetGauge(U) after every smearing update so the GPU copy stays in sync.
//
// Build requires --with-quda=PATH (defines GRID_HAVE_QUDA).  Without that,
// instantiating this class is a hard error — callers should branch on
// Grid::Quda::available() and fall back to ConjugateGradient.

#include <Grid/Grid.h>
#include <Grid/algorithms/LinearOperator.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaMultigridConfig.h>
#include <chrono>
#include <algorithm>

#ifndef GRID_HAVE_QUDA
#  error "QudaCloverInverter requires GRID_HAVE_QUDA — configure --with-quda"
#endif

#include <quda.h>

NAMESPACE_BEGIN(Grid);

struct QudaCloverParams {
  double mass;          // bare quark mass (Wilson convention; m=-0.245 for our prod ensemble)
  double csw;           // unrenormalised clover coefficient
  bool   anti_periodic_t = true;
  double tol = 1e-10;
  int    max_iter = 5000;
  // Grid uses chiral gamma basis; chroma/QUDA-IO conventionally use
  // DEGRAND_ROSSI.  Override at construction if the empirical
  // gamma-basis check selects something else.
  QudaGammaBasis gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  QudaPrecision  cuda_prec  = QUDA_DOUBLE_PRECISION;
  QudaPrecision  cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  QudaReconstructType recon = QUDA_RECONSTRUCT_NO;
  QudaReconstructType recon_sloppy = QUDA_RECONSTRUCT_12;

  // Multigrid preconditioning.  Off by default — opt-in for light-mass
  // inversions where CG convergence is slow.  When enabled, the outer
  // solver switches from CG to GCR with a QUDA-side MG preconditioner.
  // See Grid/util/QudaMultigridConfig.h for the per-level knobs.
  bool use_multigrid = false;
  QudaMgUserParams mg;
};

class QudaCloverInverter : public OperatorFunction<LatticeFermion> {
public:
  QudaCloverInverter(GridBase *grid, const QudaCloverParams &p)
    : grid_(grid), params_(p), gauge_loaded_(false) {
    setup_params_();
  }

  ~QudaCloverInverter() {
    if (mg_preconditioner_ != nullptr) {
      destroyMultigridQuda(mg_preconditioner_);
      mg_preconditioner_ = nullptr;
    }
    // The rest of QUDA-side cleanup is handled by Grid::Quda::finalize().
  }

  // Re-upload gauge + clover to the GPU.  Call after every smearing update.
  void SetGauge(const LatticeGaugeField &U) {
    int V = Quda::local_volume(grid_);
    Coordinate lc = grid_->LocalDimensions();
    // Pack Grid LatticeGaugeField → 4 lex per-direction host buffers
    // → EO permute → loadGaugeQuda (QUDA_QDP_GAUGE_ORDER, EO site order).
    std::vector<std::vector<double>> lex_bufs(4, std::vector<double>(18 * V));
    double *lex_ptrs[4] = {lex_bufs[0].data(), lex_bufs[1].data(),
                           lex_bufs[2].data(), lex_bufs[3].data()};
    Quda::gauge_to_lex_buffers(U, lex_ptrs);

    // Antiperiodic time BC: bake the −1 phase into U_t at t = Lt-1 BEFORE
    // sending to QUDA, and tell QUDA the gauge is periodic.  Going through
    // QUDA's own t_boundary does *not* match Grid's WilsonImpl phase
    // application (empirically: 0.37 residual at MASS_NORMALIZATION + DR
    // basis).  Pre-baking is robust and self-consistent.
    if (params_.anti_periodic_t) {
      // Only the rank holding the GLOBAL last-t timeslice flips its
      // last-LOCAL t.  In a single-rank run this is trivially true; with
      // mpi=1.1.1.N, only rank N-1 (in t) does the flip.  Otherwise the
      // flip is double-applied at intra-rank t boundaries and produces
      // wrong forces (~30% off in 1-traj FD).
      Coordinate proc_coor = grid_->ThisProcessorCoor();
      Coordinate procs = grid_->ProcessorGrid();
      bool last_t_rank = (proc_coor[3] == procs[3] - 1);
      if (last_t_rank) {
        const int Lt = lc[3];
        int V_per_t = lc[0] * lc[1] * lc[2];
        int site_lo = (Lt - 1) * V_per_t;
        int site_hi = Lt * V_per_t;
        for (int site = site_lo; site < site_hi; ++site) {
          double *u_t = &lex_bufs[3][18 * site];
          for (int k = 0; k < 18; ++k) u_t[k] = -u_t[k];
        }
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

    // Compute clover internally (h_clover = h_clovinv = NULL): QUDA derives
    // F_μν · σ_μν · csw·κ on-device, sidestepping any Grid-vs-QUDA clover
    // sign-convention mismatch.
    loadCloverQuda(nullptr, nullptr, &inv_param_);
    // Every load REPLACES QUDA's resident gauge/clover fields, so any MG
    // preconditioner built earlier (by ANY instance) now dereferences freed
    // memory until it is thin-updated.  Bump the process-wide generation so
    // stale handles can be detected (EnsureMgCurrent).
    ++resident_field_generation_;

    // ---- Multigrid setup --------------------------------------------------
    // Must happen AFTER loadGauge + loadClover.  If MG is already built (this
    // is a re-SetGauge after smearing update), update it in place rather
    // than rebuild from scratch — `updateMultigridQuda` re-uses the existing
    // null vectors when the gauge has changed only mildly (matches the
    // chroma+QUDA "thin update" recipe).
    if (params_.use_multigrid) {
      auto t0 = std::chrono::steady_clock::now();
      if (mg_preconditioner_ == nullptr) {
        buildMgInnerInvertParam(mg_inv_param_, inv_param_, params_.mg);
        buildMultigridParam(mg_param_, params_.mg);
        mg_param_.invert_param = &mg_inv_param_;
        mg_preconditioner_ = newMultigridQuda(&mg_param_);
        inv_param_.preconditioner = mg_preconditioner_;
        last_mg_setup_secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << GridLogMessage
                  << "QudaCloverInverter: MG preconditioner built ("
                  << params_.mg.n_level << " levels) setup=" << last_mg_setup_secs_ << " s"
                  << std::endl;
      } else {
        // Cadence (default = legacy pure thin-update; non-HMC callers unaffected):
        //   setup_maxiter_refresh > 0 alone => refresh null vectors on EVERY update
        //     (thin_update_only=FALSE) so the near-null space tracks an evolving
        //     HMC gauge.
        //   + refresh_every = N > 0 => refresh only every Nth update, thin-update
        //     otherwise (fixed cadence).
        //   + threshold_count > 0 => refresh on the update FOLLOWING any solve
        //     whose outer iteration count reached the threshold (adaptive; flag
        //     set in operator()).  OR-combined with refresh_every.
        //   rebuild_every = N > 0 => full destroy+newMultigridQuda every N
        //     updates (unconditional backstop; supersedes a pending refresh).
        const int  rebuild_every = params_.mg.rebuild_every;
        const int  refresh_every = params_.mg.refresh_every;
        const bool cadence_gated = (refresh_every > 0 || params_.mg.threshold_count > 0);
        ++mg_setgauge_count_;
        const bool counter_due =
            (refresh_every > 0 && mg_setgauge_count_ % refresh_every == 0);
        // Adaptive soft-tier trigger fired: an iter-threshold flag (set in
        // operator()) or an every-N counter.
        const bool adaptive_due = cadence_gated && (mg_refresh_pending_ || counter_due);
        // Route the adaptive trigger to a full REBUILD instead of an in-place
        // refresh when HMC_MG_THRESHOLD_REBUILD is set (4-node-safe; refresh
        // needs 8 -- see QudaMultigridConfig.h). Capture the trigger reason
        // before mg_refresh_pending_ is cleared below.
        const bool adaptive_rebuild_due = params_.mg.threshold_action_rebuild && adaptive_due;
        const bool was_iter_threshold  = mg_refresh_pending_;
        bool refresh = (params_.mg.setup_maxiter_refresh > 0);
        const char *trigger = "";
        if (refresh && cadence_gated && !adaptive_rebuild_due) {
          if (mg_refresh_pending_)  trigger = " [trigger: iter-threshold]";
          else if (counter_due)     trigger = " [trigger: every-N]";
          refresh = adaptive_due;
        }
        if ((rebuild_every > 0 && (++mg_update_count_ % rebuild_every == 0))
            || adaptive_rebuild_due) {
          rebuild_mg_();
          mg_refresh_pending_ = false;
          last_mg_setup_secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::cout << GridLogMessage << "QudaCloverInverter: MG full-rebuilt ";
          if (adaptive_rebuild_due)
            std::cout << (was_iter_threshold ? "(iter-threshold)" : "(every-N adaptive)");
          else
            std::cout << "(every " << rebuild_every << ")";
          std::cout << " setup=" << last_mg_setup_secs_ << " s" << std::endl;
        } else {
          mg_param_.thin_update_only = refresh ? QUDA_BOOLEAN_FALSE : QUDA_BOOLEAN_TRUE;
          updateMultigridQuda(mg_preconditioner_, &mg_param_);
          mg_param_.thin_update_only = QUDA_BOOLEAN_FALSE;
          if (refresh) mg_refresh_pending_ = false;
          last_mg_setup_secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::cout << GridLogMessage << "QudaCloverInverter: MG "
                    << (refresh ? "refreshed (null-vectors)" : "thin-updated")
                    << trigger
                    << " setup=" << last_mg_setup_secs_ << " s" << std::endl;
        }
      }
      // Whichever path ran (build / rebuild / update), the MG is now
      // consistent with the fields this SetGauge just loaded.
      mg_resident_gen_ = resident_field_generation_;
    }

    gauge_loaded_ = true;
  }

  // OperatorFunction<LatticeFermion> interface.  Linop is intentionally
  // unused; QUDA inverts against its own loaded gauge/clover.
  void operator()(LinearOperatorBase<LatticeFermion> &Linop,
                  const LatticeFermion &src,
                  LatticeFermion &sol) override {
    (void)Linop;
    if (!gauge_loaded_) {
      assert(false && "QudaCloverInverter::operator() called before SetGauge()");
    }
    int V = Quda::local_volume(grid_);

    // Fused unvectorize + EO permute directly into the QUDA EO buffer.
    std::vector<double> src_eo(24 * V), sol_eo(24 * V, 0.0);
    Quda::fermion_to_eo_buffer(src, src_eo.data());

    invertQuda(sol_eo.data(), src_eo.data(), &inv_param_);

    last_iter_ = inv_param_.iter;
    last_residual_ = inv_param_.true_res[0];
    last_secs_ = inv_param_.secs;

    if (params_.use_multigrid) {
      // Hard tier (Chroma RsdToleranceFactor): the solve came back without
      // meeting tolerance — a force computed from it would be silently wrong.
      // Rebuild the MG subspace from scratch against the loaded gauge and
      // re-solve once from a zero guess; abort if even that fails.
      const double rsd_fac = params_.mg.rsd_tolerance_factor;
      if (rsd_fac > 0.0 && last_residual_ > rsd_fac * params_.tol) {
        std::cout << GridLogMessage
                  << "QudaCloverInverter: MG HARD FAILURE — true residual "
                  << last_residual_ << " > " << rsd_fac << " x tol " << params_.tol
                  << " (outer iters " << last_iter_
                  << "); rebuilding subspace + re-solving" << std::endl;
        auto t0 = std::chrono::steady_clock::now();
        rebuild_mg_();
        last_mg_setup_secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << GridLogMessage
                  << "QudaCloverInverter: MG full-rebuilt (hard-failure recovery) setup="
                  << last_mg_setup_secs_ << " s" << std::endl;
        std::fill(sol_eo.begin(), sol_eo.end(), 0.0);
        invertQuda(sol_eo.data(), src_eo.data(), &inv_param_);
        last_iter_ = inv_param_.iter;
        last_residual_ = inv_param_.true_res[0];
        last_secs_ = inv_param_.secs;
        if (last_residual_ > rsd_fac * params_.tol) {
          std::cout << GridLogError
                    << "QudaCloverInverter: MG hard-failure recovery FAILED — residual "
                    << last_residual_ << " still > " << rsd_fac << " x tol "
                    << params_.tol << " after subspace rebuild; aborting."
                    << std::endl;
          assert(false && "QudaCloverInverter: MG solve failed tolerance even after subspace rebuild");
        }
      }
      // Soft tier (Chroma ThresholdCount): solve converged but worked too
      // hard — null vectors are going stale.  Flag a refresh for the next
      // SetGauge so the next solve is healthy again.
      const int thr = params_.mg.threshold_count;
      if (thr > 0 && last_iter_ >= thr && !mg_refresh_pending_) {
        mg_refresh_pending_ = true;
        std::cout << GridLogMessage
                  << "QudaCloverInverter: MG refresh flagged (outer iters "
                  << last_iter_ << " >= threshold " << thr << ")" << std::endl;
      }
    }

    Quda::eo_buffer_to_fermion(sol_eo.data(), sol);
  }

  int    LastIter()     const { return last_iter_; }
  double LastResidual() const { return last_residual_; }
  double LastSecs()     const { return last_secs_; }
  // Wall time of the most recent MG build/rebuild/thin-update/refresh inside
  // SetGauge (0 if use_multigrid=false, or before the first SetGauge call).
  double LastMgSetupSecs() const { return last_mg_setup_secs_; }

  QudaInvertParam &InvertParam() { return inv_param_; }
  QudaGaugeParam  &GaugeParam()  { return gauge_param_; }

  // The MG preconditioner handle this instance owns (null before the first
  // SetGauge, or if use_multigrid=false).  Exposed so ANOTHER inverter can
  // reuse this setup as an external preconditioner (chroma's shared
  // SubspaceID pattern: one MG built at the lightest mass preconditions GCR
  // solves at heavier masses -- correctness lives in the outer solve, the
  // mass mismatch only costs outer iterations).  Callers must re-fetch
  // before every solve: rebuild_mg_ (hard-failure recovery / rebuild_every)
  // REPLACES the handle.
  void *MgPreconditioner() const { return mg_preconditioner_; }

  // Thin-update the MG against QUDA's CURRENTLY RESIDENT gauge/clover if any
  // instance has loaded fields since this MG was last built/updated (loadGauge/
  // loadCloverQuda replace the resident fields, leaving the MG's internal
  // operators dangling -- borrowing the handle without this re-sync aborts in
  // DiracClover::checkParitySpinor with a garbage checkerboard volume, seen
  // 2026-07-09).  Sharees MUST call this right before every borrowed-handle
  // solve; no-op when already current.  The re-coarsening runs with this
  // (donor) instance's invert_param mass against the sharee's resident fields;
  // that mismatch only affects preconditioner quality -- correctness lives in
  // the sharee's outer GCR at its own mass.
  void EnsureMgCurrent() {
    if (mg_preconditioner_ == nullptr ||
        mg_resident_gen_ == resident_field_generation_) return;
    auto t0 = std::chrono::steady_clock::now();
    mg_param_.thin_update_only = QUDA_BOOLEAN_TRUE;
    updateMultigridQuda(mg_preconditioner_, &mg_param_);
    mg_param_.thin_update_only = QUDA_BOOLEAN_FALSE;
    mg_resident_gen_ = resident_field_generation_;
    last_mg_setup_secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << GridLogMessage
              << "QudaCloverInverter: MG thin-updated [trigger: resident-field sync] setup="
              << last_mg_setup_secs_ << " s" << std::endl;
  }

private:
  // Destroy + rebuild the MG preconditioner against the currently loaded
  // gauge/clover.  Shared by the rebuild_every cadence and the post-solve
  // hard-failure recovery.
  void rebuild_mg_() {
    destroyMultigridQuda(mg_preconditioner_);
    mg_preconditioner_ = newMultigridQuda(&mg_param_);
    inv_param_.preconditioner = mg_preconditioner_;
    // Rebuild runs post-solve, with this instance's fields still resident.
    mg_resident_gen_ = resident_field_generation_;
  }

  void setup_params_() {
    Coordinate lc = grid_->LocalDimensions();

    // ---- gauge_param --------------------------------------------------------
    gauge_param_ = newQudaGaugeParam();
    for (int d = 0; d < 4; ++d) gauge_param_.X[d] = lc[d];
    gauge_param_.anisotropy = 1.0;
    gauge_param_.type = QUDA_WILSON_LINKS;
    gauge_param_.gauge_order = QUDA_QDP_GAUGE_ORDER;
    // Match the agrebe/chroma convention: pre-bake the antiperiodic phase
    // into U_t in SetGauge AND tell QUDA the BC is antiperiodic (this is
    // metadata for QUDA — the actual phase is in the gauge).  Empirically
    // this is what chroma does and it works on production cfgs.
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
    gauge_param_.ga_pad = max_face_pad_();
    gauge_param_.struct_size = sizeof(gauge_param_);

    // ---- inv_param ----------------------------------------------------------
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
    inv_param_.return_clover = 0;
    inv_param_.return_clover_inverse = 0;

    // Outer solver: vanilla CG by default; GCR-with-MG-preconditioner when
    // params_.use_multigrid is set.  The MG preconditioner handle is
    // installed in SetGauge() after loadGaugeQuda/loadCloverQuda — until
    // then inv_param_.preconditioner stays null.
    if (params_.use_multigrid) {
      inv_param_.inv_type           = QUDA_GCR_INVERTER;
      inv_param_.inv_type_precondition = QUDA_MG_INVERTER;
      // QUDA's MG-as-preconditioner currently REQUIRES QUDA_DIRECT_SOLVE
      // on the outer solve (not _PC_); the MG preconditioner handles
      // the EO preconditioning internally via mg.smoother_solve_type.
      // (Mismatch ⇒ "Outer MG solver can only use QUDA_DIRECT_SOLVE")
      inv_param_.solve_type         = QUDA_DIRECT_SOLVE;
      inv_param_.schwarz_type       = QUDA_INVALID_SCHWARZ;
      inv_param_.precondition_cycle = 1;
      inv_param_.tol_precondition   = 1e-1;
      inv_param_.maxiter_precondition = 1;
    } else {
      inv_param_.inv_type           = QUDA_CG_INVERTER;
      inv_param_.solve_type         = QUDA_NORMOP_PC_SOLVE;
    }
    // Full M^-1 solve: take a full-volume source, return full-volume
    // solution; QUDA does EO preconditioning internally and reconstructs
    // the odd half from the even solution.
    inv_param_.solution_type   = QUDA_MAT_SOLUTION;
    inv_param_.matpc_type      = QUDA_MATPC_EVEN_EVEN;
    inv_param_.dagger          = QUDA_DAG_NO;
    // Grid's M is mass-form: (m+4) - 0.5·D_W - 0.5·c_sw·σF (no κ scaling).
    // QUDA's MASS_NORMALIZATION matches that.
    inv_param_.mass_normalization   = QUDA_MASS_NORMALIZATION;
    inv_param_.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

    inv_param_.tol      = params_.tol;
    inv_param_.maxiter  = params_.max_iter;
    // Tight reliable update — chroma's QUDA wrapper uses 1e-3.  1e-1 is
    // too loose for our well-conditioned propagator inversion, especially
    // when sloppy is single-precision.
    inv_param_.reliable_delta = 1e-3;
    inv_param_.use_sloppy_partial_accumulator = 0;
    inv_param_.solution_accumulator_pipeline = 1;
    inv_param_.pipeline = 0;
    inv_param_.gcrNkrylov = 10;
    inv_param_.tol_restart = 0.0005;
    inv_param_.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
    inv_param_.tol_hq = 0.0;
    // Multiple reliable-update / refinement passes if the iterated and
    // true residual diverge.  Plenty for double-prec target tol.
    inv_param_.Nsteps = 5;

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

  int max_face_pad_() const {
    Coordinate lc = grid_->LocalDimensions();
    int x_face = lc[1] * lc[2] * lc[3] / 2;
    int y_face = lc[0] * lc[2] * lc[3] / 2;
    int z_face = lc[0] * lc[1] * lc[3] / 2;
    int t_face = lc[0] * lc[1] * lc[2] / 2;
    return std::max({x_face, y_face, z_face, t_face});
  }

  GridBase *grid_;
  QudaCloverParams params_;
  QudaGaugeParam gauge_param_{};
  QudaInvertParam inv_param_{};
  std::vector<std::vector<double>> eo_bufs_;  // 4 per-dir gauge buffers, kept alive
  bool gauge_loaded_;
  int    last_iter_     = 0;
  double last_residual_ = 0.0;
  double last_secs_     = 0.0;
  double last_mg_setup_secs_ = 0.0;
  int    mg_update_count_ = 0;  // gauge updates since build, for rebuild_every cadence
                                // (NOTE: only advances when rebuild_every > 0 — the
                                // increment is short-circuited away otherwise)
  int    mg_setgauge_count_ = 0;   // ALL post-build SetGauge calls, for refresh_every
  bool   mg_refresh_pending_ = false;  // set post-solve by the threshold_count tier
  // Multigrid state — populated only if params_.use_multigrid (else null).
  QudaMultigridParam mg_param_{};
  QudaInvertParam    mg_inv_param_{};
  void              *mg_preconditioner_ = nullptr;
  // Resident-field staleness tracking for shared-MG (EnsureMgCurrent):
  // process-wide count of loadGauge/loadClover calls vs the value this MG was
  // last built/updated at.  QUDA is driven single-threaded, so a plain long
  // suffices.
  long mg_resident_gen_ = -1;
  inline static long resident_field_generation_ = 0;
};

NAMESPACE_END(Grid);
