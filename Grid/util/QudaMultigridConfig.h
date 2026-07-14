#pragma once
// QUDA multigrid configuration helpers for Grid's QudaCloverInverter.
//
// Builds a QudaMultigridParam + companion inner QudaInvertParam from a small
// number of user-facing tunables.  Defaults match the NPLQCD production
// MG parameters used by `invert_test_wrapper-48-phys-4` on the same
// cl21_48_96_b6p3_m0p2416_m0p2050-djm-3 ensemble:
//
//   --mg-levels 3
//   --mg-block-size 0 4 4 4 4   1 2 2 2 2
//   --mg-setup-inv 0 cg 1 cg
//   --mg-nvec 0 24 1 24
//   --mg-smoother 0 ca-gcr 1 ca-gcr
//   --mg-coarse-solver-tol 1 0.1  2 0.1
//   --mg-setup-maxiter 0 2000 1 2000
//   --mg-omega 0.95
//   --mg-nu-post 0 8 1 8
//   --mg-nu-pre  0 0 1 0
//   --prec-precondition half
//
// To use with smaller lattices (e.g., 16^3 x 48 unit tests), pass `levels=2`
// and adjust block sizes via `geo_block_size`.

#include <Grid/util/QudaInit.h>
#include <quda.h>
#include <vector>
#include <array>
#include <cstdlib>

NAMESPACE_BEGIN(Grid);

struct QudaMgUserParams {
  // Number of MG levels (typically 2 for 16^3 tests, 3 for production).
  int n_level = 3;

  // Geometric block size per level (level 0 = fine, level n_level-1 = coarsest).
  // Default matches NPLQCD's 48^3 x 96 production params.
  // Per-level vector of length 4 (x, y, z, t).
  // Only first (n_level-1) entries are used; coarsest level has no block.
  std::vector<std::array<int, 4>> geo_block_size = {
      {4, 4, 4, 4},   // level 0 → level 1
      {1, 2, 2, 2},   // level 1 → level 2 (only used if n_level == 3)
  };

  // Number of null-space vectors per level (default 24, NPLQCD prod value).
  int n_vec = 24;

  // Optional per-level null-vector counts (Chroma: 24 32). Empty => use the
  // single `n_vec` for every level. Set by the HMC force header from
  // HMC_MG_NVEC='24 32' (length up to n_level-1; coarsest level unused).
  std::vector<int> n_vec_levels;

  // Setup solver: CG (matches --mg-setup-inv cg).
  QudaInverterType setup_inv = QUDA_CG_INVERTER;
  int setup_maxiter = 2000;
  double setup_tol = 5e-6;

  // --- HMC evolving-gauge cadence. Defaults (0,0) = legacy pure thin-update,
  //     so fixed-gauge callers (compute_vev, propagators) are unaffected. The
  //     HMC force header sets these from env. (Chroma uses adaptive refresh,
  //     MaxIterSubspaceRefresh=500, not a static thin-update — see cfg metadata.)
  // Null-vector refresh iters/level on updateMultigridQuda; >0 => refresh each update.
  int setup_maxiter_refresh = 0;
  // Full destroy+newMultigridQuda every N SetGauge calls (0 = never).
  int rebuild_every = 0;
  // Refresh only on every Nth post-build SetGauge, thin-update otherwise
  // (0 = legacy: refresh on EVERY SetGauge when setup_maxiter_refresh > 0).
  // Env: HMC_MG_REFRESH_EVERY.
  int refresh_every = 0;
  // Adaptive soft tier (Chroma ThresholdCount): after any solve whose outer
  // iteration count reaches this, flag a null-vector refresh for the next
  // SetGauge (0 = off).  OR-combined with refresh_every.
  // Env: HMC_MG_THRESHOLD_COUNT.
  int threshold_count = 0;
  // Route the adaptive soft-tier trigger (threshold_count / refresh_every) to a
  // full destroy+rebuild instead of an in-place refresh.  Rebuild's lower memory
  // peak fits 4 nodes at 48^3, whereas refresh needs 8 (measured 2026-07-06,
  // OOMs Grid's device allocs on 4 nodes).  Default false = legacy refresh
  // action.  When true, setup_maxiter_refresh need NOT be set (the trigger fires
  // a rebuild, which does not use the refresh machinery).  Env: HMC_MG_THRESHOLD_REBUILD.
  bool threshold_action_rebuild = false;
  // Hard tier (Chroma RsdToleranceFactor): after each solve, if the true
  // residual exceeds factor x requested tol, rebuild the subspace from
  // scratch and re-solve once from a zero guess; abort if it still fails
  // (0 = off, i.e. no post-solve residual check).
  // Env: HMC_MG_RSD_TOL_FACTOR.
  double rsd_tolerance_factor = 0.0;

  // Smoother (matches --mg-smoother ca-gcr).
  QudaInverterType smoother = QUDA_CA_GCR_INVERTER;
  int nu_pre  = 0;
  int nu_post = 8;
  double omega = 0.95;
  double smoother_tol = 0.25;

  // Coarse-grid solver tol (matches --mg-coarse-solver-tol 0.1).
  double coarse_solver_tol = 0.1;
  int coarse_solver_maxiter = 16;

  // Precision for preconditioner.  Reference NPLQCD MG uses half ("--prec-precondition half"),
  // but our QudaCloverInverter loads the gauge at gauge_param_.cuda_prec_precondition =
  // params_.cuda_prec_sloppy (single by default), so HALF here would reference an
  // unloaded gauge copy → garbage volume.  Keep at single until SetGauge teaches
  // QUDA about the half-precision gauge copy too.
  QudaPrecision precondition_prec = QUDA_SINGLE_PRECISION;

  // Whether to run QUDA's MG verification at setup (slow but instructive on
  // first try).  Default off for production runs.
  bool run_verify = false;

  // Verbosity for MG setup (QUDA_SUMMARIZE default, QUDA_VERBOSE for debug).
  QudaVerbosity verbosity = QUDA_SUMMARIZE;
};

// Parse the HMC MG-cadence env knobs (HMC_MG_REFRESH_EVERY,
// HMC_MG_THRESHOLD_COUNT, HMC_MG_RSD_TOL_FACTOR) into an existing
// QudaMgUserParams.  Complements — does not replace — the per-driver parsing
// of HMC_MG_REFRESH / HMC_MG_REBUILD_EVERY; call once wherever those are read.
inline void applyHmcMgCadenceEnv(QudaMgUserParams &mg) {
  if (const char *r = std::getenv("HMC_MG_REFRESH_EVERY"))
    mg.refresh_every = std::atoi(r);
  if (const char *r = std::getenv("HMC_MG_THRESHOLD_COUNT"))
    mg.threshold_count = std::atoi(r);
  if (const char *r = std::getenv("HMC_MG_THRESHOLD_REBUILD"))
    mg.threshold_action_rebuild = (std::atoi(r) != 0);
  if (const char *r = std::getenv("HMC_MG_RSD_TOL_FACTOR"))
    mg.rsd_tolerance_factor = std::atof(r);
}

// Build the inner QudaInvertParam used by MG (passed to mg_param.invert_param).
// The "outer" QudaInvertParam (held by QudaCloverInverter) handles the
// user-facing solve and references this one via `inv_param.preconditioner`.
inline void buildMgInnerInvertParam(QudaInvertParam &mg_inv,
                                    const QudaInvertParam &outer,
                                    const QudaMgUserParams &mg_user) {
  // Start from a clean newQudaInvertParam(); copy only the physics fields
  // (kappa, csw, mass_normalization, precisions, gamma_basis, ordering)
  // and reset solver/control fields explicitly.  Inheriting wholesale
  // from the outer invert_param leads to mg internals seeing the outer's
  // preconditioner pointer / state and reporting garbage volumes.
  mg_inv = newQudaInvertParam();

  // Operator definition (must match outer + the loaded gauge/clover)
  mg_inv.dslash_type     = outer.dslash_type;
  mg_inv.kappa           = outer.kappa;
  mg_inv.mass            = outer.mass;
  mg_inv.Ls              = outer.Ls;
  mg_inv.clover_csw      = outer.clover_csw;
  mg_inv.clover_coeff    = outer.clover_coeff;
  mg_inv.compute_clover  = 0;            // already computed on outer load
  mg_inv.compute_clover_inverse = 0;
  // QUDA's MG implementation expects KAPPA_NORMALIZATION on its inner
  // invert_param, even when the outer uses MASS_NORMALIZATION (the QUDA
  // tests' reference setMultigridParam hardcodes KAPPA here).  Mixing
  // doesn't trigger an explicit error but yields a Dirac whose internal
  // gauge metadata is uninitialized — manifests as a "Spinor volume X
  // doesn't match gauge volume Y" error (Y = garbage) during the solve.
  mg_inv.mass_normalization   = QUDA_KAPPA_NORMALIZATION;
  mg_inv.solver_normalization = outer.solver_normalization;
  mg_inv.gamma_basis     = outer.gamma_basis;
  mg_inv.dirac_order     = outer.dirac_order;
  mg_inv.input_location  = outer.input_location;
  mg_inv.output_location = outer.output_location;
  mg_inv.matpc_type      = outer.matpc_type;
  mg_inv.dagger          = QUDA_DAG_NO;

  // Precisions
  mg_inv.cpu_prec               = outer.cpu_prec;
  mg_inv.cuda_prec              = outer.cuda_prec;
  mg_inv.cuda_prec_sloppy       = outer.cuda_prec_sloppy;
  mg_inv.cuda_prec_refinement_sloppy = outer.cuda_prec_refinement_sloppy;
  mg_inv.cuda_prec_precondition = mg_user.precondition_prec;
  mg_inv.clover_cpu_prec        = outer.clover_cpu_prec;
  mg_inv.clover_cuda_prec       = outer.clover_cuda_prec;
  mg_inv.clover_cuda_prec_sloppy = outer.clover_cuda_prec_sloppy;
  mg_inv.clover_cuda_prec_refinement_sloppy = outer.clover_cuda_prec_refinement_sloppy;
  mg_inv.clover_cuda_prec_precondition = mg_user.precondition_prec;
  mg_inv.clover_order           = outer.clover_order;

  // Inner Krylov used by MG (this gets overridden per-level by mg_param)
  mg_inv.inv_type        = QUDA_GCR_INVERTER;
  mg_inv.tol             = 1e-10;
  mg_inv.maxiter         = 1000;
  mg_inv.reliable_delta  = 1e-5;
  mg_inv.gcrNkrylov      = 10;
  // Inner MG invert_param: QUDA's multigrid_solver constructor REQUIRES
  // QUDA_DIRECT_SOLVE here (else: "Outer MG solver can only use
  // QUDA_DIRECT_SOLVE at present" from interface_quda.cpp:2870).
  mg_inv.solve_type      = QUDA_DIRECT_SOLVE;
  mg_inv.solution_type   = QUDA_MAT_SOLUTION;
  mg_inv.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  mg_inv.preconditioner  = nullptr;  // explicitly null — MG is not nested

  mg_inv.verbosity_precondition = QUDA_SILENT;
  mg_inv.verbosity       = mg_user.verbosity;
  mg_inv.struct_size     = sizeof(mg_inv);
}

// Build the QudaMultigridParam.  Caller must have built `mg_inv_param` first
// and set `mg_param.invert_param = &mg_inv_param` before calling this.
inline void buildMultigridParam(QudaMultigridParam &mg,
                                const QudaMgUserParams &mg_user) {
  mg = newQudaMultigridParam();
  // mg.invert_param is set by the caller.
  mg.n_level = mg_user.n_level;
  assert(mg_user.n_level <= QUDA_MAX_MG_LEVEL);
  assert((int)mg_user.geo_block_size.size() >= mg_user.n_level - 1);

  for (int l = 0; l < mg_user.n_level; ++l) {
    // Geometric block size for going from level l → l+1.  Coarsest level has
    // no block; we still write the trailing entry so QUDA doesn't read stale.
    if (l < mg_user.n_level - 1) {
      for (int d = 0; d < 4; ++d)
        mg.geo_block_size[l][d] = mg_user.geo_block_size[l][d];
    } else {
      for (int d = 0; d < 4; ++d) mg.geo_block_size[l][d] = 1;
    }
    // QUDA_MAX_DIM (= 6) extra dims must be 1 — otherwise QUDA computes
    // garbage coarse-grid volume from uninitialized memory.
    for (int d = 4; d < QUDA_MAX_DIM; ++d) mg.geo_block_size[l][d] = 1;
    // Wilson MG: chiral spin-halving (4 → 2 spins) only at the fine level;
    // coarser levels keep the 2-spin block-orthonormal structure.
    mg.spin_block_size[l] = (l == 0) ? 2 : 1;

    // Null vector count: per-level if n_vec_levels is set (Chroma 24 32),
    // else the single n_vec for every level.
    mg.n_vec[l] = (l < (int)mg_user.n_vec_levels.size())
                      ? mg_user.n_vec_levels[l]
                      : mg_user.n_vec;
    mg.precision_null[l] = QUDA_HALF_PRECISION;
    mg.n_block_ortho[l] = 1;
    mg.block_ortho_two_pass[l] = QUDA_BOOLEAN_TRUE;

    // Setup
    mg.setup_inv_type[l] = mg_user.setup_inv;
    mg.num_setup_iter[l] = 1;
    mg.setup_maxiter[l] = mg_user.setup_maxiter;
    mg.setup_maxiter_refresh[l] = mg_user.setup_maxiter_refresh;
    mg.setup_tol[l] = mg_user.setup_tol;
    mg.setup_ca_basis[l] = QUDA_POWER_BASIS;
    mg.setup_ca_basis_size[l] = 4;
    mg.setup_ca_lambda_min[l] = 0.0;
    mg.setup_ca_lambda_max[l] = -1.0;
    mg.n_vec_batch[l] = 1;

    // Coarse solver
    mg.coarse_solver[l] = QUDA_GCR_INVERTER;
    mg.coarse_solver_tol[l] = mg_user.coarse_solver_tol;
    mg.coarse_solver_maxiter[l] = mg_user.coarse_solver_maxiter;
    mg.coarse_solver_ca_basis[l] = QUDA_POWER_BASIS;
    mg.coarse_solver_ca_basis_size[l] = 4;
    mg.coarse_solver_ca_lambda_min[l] = 0.0;
    mg.coarse_solver_ca_lambda_max[l] = -1.0;

    // Smoother
    mg.smoother[l] = mg_user.smoother;
    mg.smoother_tol[l] = mg_user.smoother_tol;
    mg.nu_pre[l] = mg_user.nu_pre;
    mg.nu_post[l] = mg_user.nu_post;
    mg.omega[l] = mg_user.omega;
    // QUDA promotes coarse-level link storage to HALF internally; halo
    // precision must match.  At level 0 (fine) the halo follows the
    // precondition precision; at coarse levels it must be HALF.
    mg.smoother_halo_precision[l] = (l == 0) ? mg_user.precondition_prec
                                              : QUDA_HALF_PRECISION;
    mg.smoother_schwarz_type[l] = QUDA_INVALID_SCHWARZ;
    mg.smoother_schwarz_cycle[l] = 1;
    mg.smoother_solver_ca_basis[l] = QUDA_POWER_BASIS;
    mg.smoother_solver_ca_lambda_min[l] = 0.0;
    mg.smoother_solver_ca_lambda_max[l] = -1.0;

    // Cycle
    mg.cycle_type[l] = QUDA_MG_CYCLE_RECURSIVE;
    // Solution type passed to the next coarse level: full Schur on the
    // intermediate, full M on the coarsest (no further coarsening).
    mg.coarse_grid_solution_type[l] = (l == mg_user.n_level - 1)
                                          ? QUDA_MAT_SOLUTION
                                          : QUDA_MATPC_SOLUTION;
    mg.smoother_solve_type[l] = QUDA_DIRECT_PC_SOLVE;
    mg.global_reduction[l] = QUDA_BOOLEAN_TRUE;
    mg.location[l] = QUDA_CUDA_FIELD_LOCATION;
    mg.setup_location[l] = QUDA_CUDA_FIELD_LOCATION;
    mg.use_eig_solver[l] = QUDA_BOOLEAN_FALSE;
    mg.verbosity[l] = mg_user.verbosity;
    mg.setup_use_mma[l] = QUDA_BOOLEAN_FALSE;
    mg.dslash_use_mma[l] = QUDA_BOOLEAN_FALSE;
    mg.transfer_use_mma[l] = QUDA_BOOLEAN_FALSE;
    mg.vec_load[l] = QUDA_BOOLEAN_FALSE;
    mg.vec_store[l] = QUDA_BOOLEAN_FALSE;
    std::snprintf(mg.vec_infile[l],  sizeof(mg.vec_infile[l]),  "%s", "");
    std::snprintf(mg.vec_outfile[l], sizeof(mg.vec_outfile[l]), "%s", "");
  }

  mg.setup_type = QUDA_NULL_VECTOR_SETUP;
  mg.pre_orthonormalize = QUDA_BOOLEAN_FALSE;
  mg.post_orthonormalize = QUDA_BOOLEAN_TRUE;
  mg.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
  mg.generate_all_levels = QUDA_BOOLEAN_TRUE;
  mg.run_verify          = mg_user.run_verify ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  mg.run_low_mode_check  = QUDA_BOOLEAN_FALSE;
  mg.run_oblique_proj_check = QUDA_BOOLEAN_FALSE;

  mg.coarse_guess        = QUDA_BOOLEAN_FALSE;
  mg.preserve_deflation  = QUDA_BOOLEAN_FALSE;
  mg.allow_truncation    = QUDA_BOOLEAN_FALSE;
  mg.staggered_kd_dagger_approximation = QUDA_BOOLEAN_FALSE;
  mg.thin_update_only    = QUDA_BOOLEAN_FALSE;

  mg.struct_size = sizeof(mg);
}

NAMESPACE_END(Grid);
