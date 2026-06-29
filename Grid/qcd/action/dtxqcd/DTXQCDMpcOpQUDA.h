#pragma once
// M-wrap.4-v2: full-volume DTXQCD doubled Wilson-Clover M operator with QUDA's
// stock Wilson-clover Mat as the per-block kernel.
//
// Lower-block identity used (per DTXQCDDeltaCloverOp.h:8-27 derivation):
//
//     M_22 · ψ_lower = C^T · M_QCD^T · C · ψ_lower
//                    = C   · M_QCD^T · C · ψ_lower      (C^T = -C in DR; the
//                                                        two minus signs
//                                                        from the sandwich
//                                                        cancel.)
//
// And  M_QCD^T · w = conj(M_QCD^† · conj(w)).  So we get the lower block via
// QUDA's MatQuda(dagger=YES) wrapped with conj-then-C on each side.  C = γ_2·γ_4
// in DR basis (a pure spinor permutation + sign, no complex conjugation in C).
//
// History:
//   M-wrap.4 v1 used a γ_2·conj sandwich (no transpose on M_QCD).  That
//   identity holds for plain Wilson but BREAKS for Wilson-clover — the σ_μν
//   factor in the clover term has an `i` prefactor that flips under conj but
//   not under the basis similarity, so the sandwich gets the clover sign
//   wrong.  v2 uses the transposed M_QCD form, which fixes the clover sign
//   automatically (σ_μν^T·F^T from the transpose, +/- signs from C·C
//   sandwich; works out to exactly +(csw/2)·F^T·σ_μν for the lower block).
//
// Apply protocol (M):
//   1. Pack each of in.upper.f[0,1] from Grid SIMD -> flat double EO host
//      buffer via Quda::fermion_to_eo_buffer.
//   2. For each of in.lower.f[0,1]: pack -> flat host buffer -> apply_C ->
//      conj_inplace.
//   3. Call MatQuda four times:
//        upper.f[a]:        dagger from apply_mat_ arg (NO for forward M)
//        lower-frame[a]:    dagger=YES (gives M_QCD^† · ..., which the
//                           surrounding conj·...·conj turns into M_QCD^T · ...).
//   4. For the two lower outputs, conj_inplace -> apply_C in place.
//   5. Unpack each EO buffer -> LatticeFermion in out (upper / lower).
//   6. Call DtxqcdQudaAuxKernel::ApplyFusedDtxqcdAuxKernel(...) to ADD the
//      DTXQCD aux contribution (σ + π + s + p + d + n) on top of the Mat
//      results — production defaults transpose_aux=true, use_dn_conj=true,
//      accumulate=true.
//
// What we trust from upstream lockout:
//   - QUDA's Wilson-clover Mat with csw·F clover is the full upper-block
//     Wilson + clover piece (= mass + hop(U) + (-csw/2)·F·σ_μν).
//   - C·M_QCD^T·C with C = γ_2·γ_4 reproduces M_22 = lower block of M_DTXQCD,
//     including the +(csw/2)·F^T·σ_μν clover sign + F-color-transpose that
//     distinguish lower from upper.
//   - Antiperiodic-T BC is baked into U_t at t = Lt-1 BEFORE loadGaugeQuda
//     (the QudaCloverInverter recipe); both blocks see the same BC.
//
// Mdag is γ_5·M·γ_5 — γ5-hermiticity (validated in Test_dtxqcd_gamma5_herm_full).
// Implemented exactly like DTXQCDWilsonCloverFermionEO::Mdag: Grid-side γ5
// rotation on the doubled fermion before/after a forward M call — sidesteps
// any Grid-vs-QUDA γ5 basis question.
//
// THIS FILE PRESENTS THE GRID-LAYOUT API ONLY (M, Mdag take/return
// DTXQCDFermionDoubled).  Per-iter device-resident pointers + custom
// blas_quda inner loop are M-wrap.5's concern.

#include <Grid/Grid.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDFermionDoubled.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDDeltaOp.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_apply_C.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_device.h>
// Session 3d header-split: pull in only the thin DECL headers for the 4 native
// kernels here.  The full inline impls in `_native[_v2].h` transitively include
// QUDA's trove externals whose global `detail` namespace collides with
// `Grid::detail` when a TU does `using namespace Grid;` (every production
// driver does).  The strong out-of-line definitions live in
// dtxqcd_quda_<name>_native_impl.cc compiled with a clean include scope.
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_csf_helpers.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_post_mat_lower_native_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma5_native_decl.h>
// Phase β Session B — SP native kernel decls for the MP-CG cleanup path.
// Activated only when DTXQCD_MP_CG_CLEANUP=1.  Pure additions (DP-Style C
// behaviour unchanged).
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel_native_v2_sp_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_pre_mat_lower_native_sp_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_post_mat_lower_native_sp_decl.h>
#include <Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma5_native_sp_decl.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#ifndef GRID_HAVE_QUDA
#  error "DTXQCDMpcOpQUDA requires GRID_HAVE_QUDA — configure --with-quda"
#endif

#include <quda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <memory>

NAMESPACE_BEGIN(Grid);

class DTXQCDMpcOpQUDA {
 public:
  typedef DTXQCDFermionDoubled Field;

  DTXQCDMpcOpQUDA(LatticeGaugeField &U,
                  GridCartesian &grid,
                  GridRedBlackCartesian &rbgrid,
                  RealD mass,
                  RealD csw,
                  const LatticeDtxqcdSigma &sigma,
                  const LatticeDtxqcdPi    &pi,
                  const LatticeDtxqcdD     &d,
                  const LatticeDtxqcdN     &n,
                  const LatticeDtxqcdS     &s,
                  const LatticeDtxqcdP     &p,
                  bool anti_periodic_t = true,
                  QudaGammaBasis gb = QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      : grid_(&grid),
        rbgrid_(&rbgrid),
        mass_(mass),
        csw_(csw),
        sigma_(sigma), pi_(pi), d_(d), n_(n), s_(s), p_(p),
        Umu_(U),
        anti_periodic_t_(anti_periodic_t),
        gamma_basis_(gb),
        gauge_loaded_(false) {
    setup_params_();
    SetGauge(U);
  }

  // -------- main API --------

  // Re-pack U → load to QUDA's gauge + clover.  Call after every gauge update
  // (e.g. each HMC integrator step / smearing chain refresh).  Aux fields live
  // in Grid only and are referenced by const-ref — no QUDA state change for
  // aux mutation.
  void SetGauge(LatticeGaugeField &U) {
    int V = Quda::local_volume(grid_);
    Coordinate lc = grid_->LocalDimensions();
    std::vector<std::vector<double>> lex_bufs(4, std::vector<double>(18 * V));
    double *lex_ptrs[4] = {lex_bufs[0].data(), lex_bufs[1].data(),
                           lex_bufs[2].data(), lex_bufs[3].data()};
    Quda::gauge_to_lex_buffers(U, lex_ptrs);

    // Bake the antiperiodic-t phase into U_t at t = Lt-1 (LAST rank in t only).
    // Same recipe as QudaCloverInverter; this is what avoids the empirical
    // mismatch between QUDA's t_boundary handling and Grid's WilsonImpl phase.
    if (anti_periodic_t_) {
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

    eo_gauge_bufs_.assign(4, std::vector<double>(18 * V));
    void *gauge_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      Quda::lex_to_eo_permute(lex_bufs[mu].data(), eo_gauge_bufs_[mu].data(),
                              V, 18, lc);
      gauge_ptrs[mu] = eo_gauge_bufs_[mu].data();
    }
    loadGaugeQuda(gauge_ptrs, &gauge_param_);
    // QUDA computes the clover (-csw·κ form internally) from the loaded gauge.
    loadCloverQuda(nullptr, nullptr, &inv_param_);

    gauge_loaded_ = true;
  }

  // Forward M.  Per the protocol described at file top.
  void M(const Field &in, Field &out) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::M called before SetGauge");
    apply_mat_(in, out, /*dagger=*/false);
  }

  // Mdag via γ_5-hermiticity.  Mirrors DTXQCDWilsonCloverFermionEO::Mdag.
  void Mdag(const Field &in, Field &out) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::Mdag called before SetGauge");
    Gamma g5(Gamma::Algebra::Gamma5);
    Field g5_in(in.Grid()), tmp(in.Grid());
    for (int a = 0; a < DtxqcdNf; ++a) {
      g5_in.upper.f[a] = g5 * in.upper.f[a];
      g5_in.lower.f[a] = g5 * in.lower.f[a];
    }
    M(g5_in, tmp);
    for (int a = 0; a < DtxqcdNf; ++a) {
      out.upper.f[a] = g5 * tmp.upper.f[a];
      out.lower.f[a] = g5 * tmp.lower.f[a];
    }
  }

  // M-wrap.5a: device-resident apply.  Caller owns the 4 in + 4 out device
  // buffers; each is a flat double EO buffer of size 24*V (one per flavor per
  // upper/lower, mirroring the host-mode work buffers in apply_mat_).
  //
  // The aux step uses Option α: download device buffers → host, run the
  // existing Grid-SIMD aux kernel against a Grid LatticeFermion, upload the
  // accumulated result.  This is a temporary measure for M-wrap.5a — the
  // aux kernel can be ported to QUDA SoA layout in M-wrap.5a.5 if needed.
  // For correctness validation (M_device ≡ M_host) the host roundtrip is
  // fine; the per-iter perf characterization will land in M-wrap.5b/.7.
  //
  // Dagger flag for the upper-block MatQuda follows `dagger` arg.  Lower
  // block always uses QUDA_DAG_YES (the C·M_QCD^T·C identity).
  void M_device(double *in_upper_d[DtxqcdNf],
                double *in_lower_d[DtxqcdNf],
                double *out_upper_d[DtxqcdNf],
                double *out_lower_d[DtxqcdNf],
                double *scratch_lower_d[DtxqcdNf],
                bool dagger = false) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::M_device called before SetGauge");
    int V = Quda::local_volume(grid_);
    size_t bytes = 24 * V * sizeof(double);

    // Stash + override CPU<->GPU location knobs for this call; restore on exit.
    QudaFieldLocation saved_in_loc  = inv_param_.input_location;
    QudaFieldLocation saved_out_loc = inv_param_.output_location;
    QudaDagType saved_dagger_flag   = inv_param_.dagger;
    inv_param_.input_location  = QUDA_CUDA_FIELD_LOCATION;
    inv_param_.output_location = QUDA_CUDA_FIELD_LOCATION;

    // Step 1: lower-block pre-MatQuda frame correction.  scratch = conj(C·in_lower)
    // via the fused PreMatLowerKernel (single device pass).  Leaves in_lower_d
    // untouched.
    for (int a = 0; a < DtxqcdNf; ++a) {
      Quda::PreMatLowerKernel(scratch_lower_d[a], in_lower_d[a], V);
    }

    // Step 2: per-Mat call.
    //   Style A (default): MatQuda(out_d, in_d, &inv_param_) — bears full
    //     per-call alloc/repack/Dirac-create overhead (~23 ms/call at 16³×48
    //     mpi=1.1.1.4, observed 2026-06-27 Stage B Option β).
    //   Style B (env DTXQCD_STAGEB_STYLE=B): persistent native-layout CSF
    //     scratches + persistent Dirac, with `dirac->M`/`dirac->Mdag` instead
    //     of MatQuda.  Scout-validated 0.091 ms/dirac->M call + ~2-3 ms/copy.
    //     MatQuda applies `ax(0.5/kappa)` internally, but `dirac->M` does not
    //     — we apply the rescale via quda::blas::ax to match.
    if (style_b_()) {
      init_style_b_();
      double scale = 0.5 / inv_param_.kappa;
      for (int a = 0; a < DtxqcdNf; ++a) {
        // Upper block: caller's `dagger` flag.
        apply_dirac_op_style_b_(in_upper_d[a], out_upper_d[a],
                                /*dagger=*/dagger, scale, V);
        // Lower-frame block: always dagger=YES (the M_QCD^T sandwich).
        apply_dirac_op_style_b_(scratch_lower_d[a], out_lower_d[a],
                                /*dagger=*/true, scale, V);
      }
      // QUDA's CSF copy / dirac->M / blas::ax run on QUDA-managed CUDA streams
      // independent of Grid's accelerator_for stream.  The subsequent
      // PostMatLowerKernel (Grid) reads out_lower_d, and the aux step
      // (acceleratorCopyFromDevice) reads all in/out buffers — both via
      // Grid's stream.  Stage A's MatQuda has its own internal sync between
      // its compute and the user's view of the output buffer; Style B's
      // direct dirac->M does not.  One barrier per M_device call is enough.
      cudaDeviceSynchronize();
    } else {
      for (int a = 0; a < DtxqcdNf; ++a) {
        inv_param_.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
        MatQuda(out_upper_d[a], in_upper_d[a], &inv_param_);
        inv_param_.dagger = QUDA_DAG_YES;
        MatQuda(out_lower_d[a], scratch_lower_d[a], &inv_param_);
      }
    }

    // Step 3: lower-block post-MatQuda frame correction.  out_lower = -C·conj(out_lower)
    // via the fused PostMatLowerKernel (in-place safe with per-site local regs).
    // No scratch ping-pong, no device-to-device copy.
    for (int a = 0; a < DtxqcdNf; ++a) {
      Quda::PostMatLowerKernel(out_lower_d[a], V);
    }

    // Step 4: aux contribution.
    //   - Option α (default): download → Grid aux → upload.  Per-call host
    //     roundtrip of all 4 in + 4 out spinors — ~120 ms/iter at 16³×48,
    //     dominates Stage B per-iter cost.
    //   - Option β (env DTXQCD_STAGEB_AUX_OPTION=B): device-side aux kernel
    //     operating on the same flat-EO buffers M_device already owns.  One
    //     pack+upload of aux fields per call (~1-2 ms at 16³×48) instead.
    if (aux_option_b_()) {
      // Ensure cache allocated + (re-)pack aux to device.  Aux is const-ref
      // captured; we re-pack on every call (cheap — host pack + ~14 MB H→D).
      // Caller can invalidate explicitly by calling ResetAuxCache() if needed.
      if (!aux_cache_.allocated()) {
        DtxqcdQudaAuxKernelDevice::allocate_aux_cache(aux_cache_, (std::size_t)V);
      }
      DtxqcdQudaAuxKernelDevice::pack_aux_to_device(sigma_, pi_, d_, n_, s_, p_,
                                                    aux_cache_);
      DtxqcdQudaAuxKernelDevice::ApplyAuxKernel(
          aux_cache_,
          in_upper_d[0], in_upper_d[1], in_lower_d[0], in_lower_d[1],
          out_upper_d[0], out_upper_d[1], out_lower_d[0], out_lower_d[1],
          (std::size_t)V,
          /*transpose_aux=*/true, /*use_dn_conj=*/true);
    } else {
      // Option α (legacy host roundtrip).
      Field in_grid(grid_), out_grid(grid_);
      std::vector<double> host_buf(24 * V);
      for (int a = 0; a < DtxqcdNf; ++a) {
        acceleratorCopyFromDevice(in_upper_d[a], host_buf.data(), bytes);
        Quda::eo_buffer_to_fermion(host_buf.data(), in_grid.upper.f[a]);
        acceleratorCopyFromDevice(in_lower_d[a], host_buf.data(), bytes);
        Quda::eo_buffer_to_fermion(host_buf.data(), in_grid.lower.f[a]);
        acceleratorCopyFromDevice(out_upper_d[a], host_buf.data(), bytes);
        Quda::eo_buffer_to_fermion(host_buf.data(), out_grid.upper.f[a]);
        acceleratorCopyFromDevice(out_lower_d[a], host_buf.data(), bytes);
        Quda::eo_buffer_to_fermion(host_buf.data(), out_grid.lower.f[a]);
      }
      DtxqcdQudaAuxKernel::ApplyFusedDtxqcdAuxKernel(sigma_, pi_, d_, n_, s_, p_,
                                                     in_grid, out_grid,
                                                     /*transpose_aux=*/true,
                                                     /*use_dn_conj=*/true,
                                                     /*accumulate=*/true);
      for (int a = 0; a < DtxqcdNf; ++a) {
        Quda::fermion_to_eo_buffer(out_grid.upper.f[a], host_buf.data());
        acceleratorCopyToDevice(host_buf.data(), out_upper_d[a], bytes);
        Quda::fermion_to_eo_buffer(out_grid.lower.f[a], host_buf.data());
        acceleratorCopyToDevice(host_buf.data(), out_lower_d[a], bytes);
      }
    }

    // Restore inv_param state.
    inv_param_.input_location  = saved_in_loc;
    inv_param_.output_location = saved_out_loc;
    inv_param_.dagger          = saved_dagger_flag;
  }

  // M-wrap.5b.2 Stage B: device-resident Mdag via γ_5·M·γ_5.  Same buffer
  // convention as M_device; γ_5 is applied in-place by a small device kernel
  // (per-site sign flip on the first two spin components in Grid's DR
  // convention).  The sign variant is selectable via env `DTXQCD_STAGEB_G5_SIGN`
  // = 0 (negate spin 0,1; default) or 1 (negate spin 2,3) so we can probe
  // the convention at FD if needed.
  void Mdag_device(double *in_upper_d[DtxqcdNf],
                   double *in_lower_d[DtxqcdNf],
                   double *out_upper_d[DtxqcdNf],
                   double *out_lower_d[DtxqcdNf],
                   double *scratch_lower_d[DtxqcdNf]) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::Mdag_device called before SetGauge");
    int V = Quda::local_volume(grid_);
    int g5_variant = g5_variant_();

    // Apply γ_5 in-place to inputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaStageB::ApplyGamma5Inplace(in_upper_d[a], V, g5_variant);
      DtxqcdQudaStageB::ApplyGamma5Inplace(in_lower_d[a], V, g5_variant);
    }
    // Forward M_device.
    M_device(in_upper_d, in_lower_d, out_upper_d, out_lower_d,
             scratch_lower_d, /*dagger=*/false);
    // Restore γ_5 on inputs (they were mutated in place).
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaStageB::ApplyGamma5Inplace(in_upper_d[a], V, g5_variant);
      DtxqcdQudaStageB::ApplyGamma5Inplace(in_lower_d[a], V, g5_variant);
    }
    // Apply γ_5 to outputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaStageB::ApplyGamma5Inplace(out_upper_d[a], V, g5_variant);
      DtxqcdQudaStageB::ApplyGamma5Inplace(out_lower_d[a], V, g5_variant);
    }
  }

  static int g5_variant_() {
    static int v = []() {
      const char *e = std::getenv("DTXQCD_STAGEB_G5_SIGN");
      return (e && *e) ? std::atoi(e) : 0;
    }();
    return v;
  }

  // ------------------------------------------------------------------------
  // M-wrap.5b.4 / Style C: device-resident M/Mdag on caller-owned native CSFs.
  //
  // Style A (M_device, M_device with raw flat-EO double*): each Mat call goes
  // through MatQuda (input_location=CUDA → QUDA repacks internally) plus
  // optional Option α/β aux paths.  Per-call ~23 ms at 16³×48 multi-rank.
  // Style B (M_device with DTXQCD_STAGEB_STYLE=B): persistent Dirac, but
  // per-call csf.copy(ref↔native) at each apply_dirac_op_style_b_ — measured
  // ~30 ms each at 16³, more expensive than the MatQuda overhead it
  // replaces.  Style C: caller's CG state ALREADY lives in native CSF; no
  // per-call csf.copy.  Inner kernels (PreMat/PostMat/Gamma5/aux) all run
  // via FloatNOrder accessors on the same native storage.  csf.copy happens
  // only at CG entry/exit (Grid spinor → flat-24V → native), not per Mat.
  //
  // Caller responsibilities:
  //   - Allocate CSFs via MakeNativeCsfParam() → quda::ColorSpinorField ctor.
  //   - 4 inputs (upper.f[0,1] + lower.f[0,1]), 4 outputs, 2 lower-scratch.
  //   - Pre-populate via host buffer + csf.copy ONCE at CG entry.
  // ------------------------------------------------------------------------
  quda::ColorSpinorParam MakeNativeCsfParam() {
    init_style_b_();
    return style_b_state_.cuda_param_tmpl;
  }

  // SP variant — used by DoubledStateCSFSp.allocate().  Caller-owned SP CSFs
  // (FloatNOrder<float,4,3,4>, UKQCD basis, halo-padded) hold the inner-loop
  // SP cleanup state.  Lazy-builds the SP persistent Dirac on first call.
  quda::ColorSpinorParam MakeNativeCsfParamSp() {
    init_style_c_sp_();
    return style_c_sp_state_.cuda_param_tmpl;
  }

  void M_device_csf(quda::ColorSpinorField *in_upper[DtxqcdNf],
                    quda::ColorSpinorField *in_lower[DtxqcdNf],
                    quda::ColorSpinorField *out_upper[DtxqcdNf],
                    quda::ColorSpinorField *out_lower[DtxqcdNf],
                    quda::ColorSpinorField *scratch_lower[DtxqcdNf],
                    bool dagger = false) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::M_device_csf called before SetGauge");
    init_style_b_();
    init_aux_perm_();

    int V = Quda::local_volume(grid_);

    // SYNC: caller may have just written in_upper / in_lower on Grid's
    // accelerator_for stream (e.g. Gamma5InplaceNative inside Mdag_device_csf
    // wrap).  dirac->M/Mdag below reads them on QUDA's stream — need explicit
    // barrier or the kernels race.
    cudaDeviceSynchronize();

    // Step 1: PreMatLowerNative on in_lower → scratch_lower.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPreMatLowerNative::ApplyPreMatLowerNative(
          *in_lower[a], *scratch_lower[a]);
    }

    // SYNC: PreMatLowerNative writes scratch_lower on Grid's stream;
    // dirac->Mdag below reads it on QUDA's stream.
    cudaDeviceSynchronize();

    // Step 2: dirac->M/Mdag on persistent Dirac, direct on caller's CSFs.
    // Followed by quda::blas::ax(0.5/kappa) to match MatQuda's implicit
    // kappa rescale.
    double scale = 0.5 / inv_param_.kappa;
    quda::vector<quda::Complex> a_coeff{quda::Complex(scale, 0.0)};
    for (int a = 0; a < DtxqcdNf; ++a) {
      // Upper block: caller's `dagger` flag.
      if (dagger) {
        style_b_state_.dirac->Mdag(*out_upper[a], *in_upper[a]);
      } else {
        style_b_state_.dirac->M(*out_upper[a], *in_upper[a]);
      }
      // Lower block: always dagger=YES (M_QCD^T sandwich).
      style_b_state_.dirac->Mdag(*out_lower[a], *scratch_lower[a]);

      quda::vector_ref<quda::ColorSpinorField> y_ref_u{*out_upper[a]};
      quda::vector_ref<quda::ColorSpinorField> y_ref_l{*out_lower[a]};
      quda::blas::ax(a_coeff, y_ref_u);
      quda::blas::ax(a_coeff, y_ref_l);
    }

    // SYNC: dirac->M/Mdag and blas::ax write out_upper/out_lower on QUDA's
    // stream; Step 3/4 native kernels read on Grid's stream.
    cudaDeviceSynchronize();

    // Step 3: PostMatLowerNative (in place).
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPostMatLowerNative::ApplyPostMatLowerNative(*out_lower[a]);
    }

    // Step 4: aux contribution.  Re-pack aux to device on every call (cheap),
    // then dispatch the native kernel which reads aux via the perm table.
    if (!aux_cache_.allocated()) {
      DtxqcdQudaAuxKernelDevice::allocate_aux_cache(aux_cache_, (std::size_t)V);
    }
    DtxqcdQudaAuxKernelDevice::pack_aux_to_device(sigma_, pi_, d_, n_, s_, p_,
                                                   aux_cache_);
    DtxqcdQudaAuxKernelNativeV2::ApplyAuxKernelNative(
        aux_cache_, aux_perm_d_,
        *in_upper[0], *in_upper[1], *in_lower[0], *in_lower[1],
        *out_upper[0], *out_upper[1], *out_lower[0], *out_lower[1],
        /*transpose_aux=*/true, /*use_dn_conj=*/true);

    // Final barrier — outputs are now visible to all subsequent callers
    // regardless of which stream they use.
    cudaDeviceSynchronize();
  }

  void Mdag_device_csf(quda::ColorSpinorField *in_upper[DtxqcdNf],
                       quda::ColorSpinorField *in_lower[DtxqcdNf],
                       quda::ColorSpinorField *out_upper[DtxqcdNf],
                       quda::ColorSpinorField *out_lower[DtxqcdNf],
                       quda::ColorSpinorField *scratch_lower[DtxqcdNf]) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::Mdag_device_csf called before SetGauge");
    int g5_variant = g5_variant_();
    // Apply γ_5 in place to inputs (native variant).
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_upper[a], g5_variant);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_lower[a], g5_variant);
    }
    M_device_csf(in_upper, in_lower, out_upper, out_lower, scratch_lower,
                 /*dagger=*/false);
    // Restore γ_5 on inputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_upper[a], g5_variant);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_lower[a], g5_variant);
    }
    // Apply γ_5 to outputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*out_upper[a], g5_variant);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*out_lower[a], g5_variant);
    }
  }

  // ------------------------------------------------------------------------
  // Phase β Session B — SP-precision M_device_csf_sp / Mdag_device_csf_sp.
  // Identical structure to the DP version above but with SP native kernels
  // and SP persistent Dirac (cuda_prec=QUDA_SINGLE_PRECISION).  The aux fields
  // and perm-table are precision-agnostic and shared with the DP path.
  //
  // Aux field accumulation runs as `aux_kernel + ax(scale)` with the same
  // 0.5/kappa rescale as the DP path.  SP scratch CSFs live in the caller's
  // DoubledStateCSFSp arrays; this routine does no allocation.
  // ------------------------------------------------------------------------
  void M_device_csf_sp(quda::ColorSpinorField *in_upper[DtxqcdNf],
                      quda::ColorSpinorField *in_lower[DtxqcdNf],
                      quda::ColorSpinorField *out_upper[DtxqcdNf],
                      quda::ColorSpinorField *out_lower[DtxqcdNf],
                      quda::ColorSpinorField *scratch_lower[DtxqcdNf],
                      bool dagger = false) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::M_device_csf_sp called before SetGauge");
    init_style_b_();
    init_style_c_sp_();
    init_aux_perm_();

    cudaDeviceSynchronize();

    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPreMatLowerNativeSp::ApplyPreMatLowerNativeSp(
          *in_lower[a], *scratch_lower[a]);
    }

    cudaDeviceSynchronize();

    double scale = 0.5 / inv_param_.kappa;
    quda::vector<quda::Complex> a_coeff{quda::Complex(scale, 0.0)};
    for (int a = 0; a < DtxqcdNf; ++a) {
      if (dagger) {
        style_c_sp_state_.dirac->Mdag(*out_upper[a], *in_upper[a]);
      } else {
        style_c_sp_state_.dirac->M(*out_upper[a], *in_upper[a]);
      }
      style_c_sp_state_.dirac->Mdag(*out_lower[a], *scratch_lower[a]);

      quda::vector_ref<quda::ColorSpinorField> y_ref_u{*out_upper[a]};
      quda::vector_ref<quda::ColorSpinorField> y_ref_l{*out_lower[a]};
      quda::blas::ax(a_coeff, y_ref_u);
      quda::blas::ax(a_coeff, y_ref_l);
    }

    cudaDeviceSynchronize();

    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPostMatLowerNativeSp::ApplyPostMatLowerNativeSp(*out_lower[a]);
    }

    int V = Quda::local_volume(grid_);
    if (!aux_cache_.allocated()) {
      DtxqcdQudaAuxKernelDevice::allocate_aux_cache(aux_cache_, (std::size_t)V);
    }
    DtxqcdQudaAuxKernelDevice::pack_aux_to_device(sigma_, pi_, d_, n_, s_, p_,
                                                   aux_cache_);
    DtxqcdQudaAuxKernelNativeV2Sp::ApplyAuxKernelNativeSp(
        aux_cache_, aux_perm_d_,
        *in_upper[0], *in_upper[1], *in_lower[0], *in_lower[1],
        *out_upper[0], *out_upper[1], *out_lower[0], *out_lower[1],
        /*transpose_aux=*/true, /*use_dn_conj=*/true);

    cudaDeviceSynchronize();
  }

  void Mdag_device_csf_sp(quda::ColorSpinorField *in_upper[DtxqcdNf],
                         quda::ColorSpinorField *in_lower[DtxqcdNf],
                         quda::ColorSpinorField *out_upper[DtxqcdNf],
                         quda::ColorSpinorField *out_lower[DtxqcdNf],
                         quda::ColorSpinorField *scratch_lower[DtxqcdNf]) {
    GRID_ASSERT(gauge_loaded_ && "DTXQCDMpcOpQUDA::Mdag_device_csf_sp called before SetGauge");
    int g5_variant = g5_variant_();
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*in_upper[a], g5_variant);
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*in_lower[a], g5_variant);
    }
    M_device_csf_sp(in_upper, in_lower, out_upper, out_lower, scratch_lower,
                    /*dagger=*/false);
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*in_upper[a], g5_variant);
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*in_lower[a], g5_variant);
    }
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*out_upper[a], g5_variant);
      DtxqcdQudaGamma5NativeSp::ApplyGamma5InplaceNativeSp(*out_lower[a], g5_variant);
    }
  }

  // Env DTXQCD_STAGEB_STYLE = "B" or "b" → use persistent CSF + persistent
  // Dirac + dirac->M / dirac->Mdag in M_device (Style B, M-wrap.5b.3).
  // Anything else → MatQuda per call (Style A, M-wrap.5b.2 Option α/β baseline).
  // Locked at first call to keep behavior stable across CG iters.
  static bool style_b_() {
    static bool v = []() {
      const char *e = std::getenv("DTXQCD_STAGEB_STYLE");
      return (e && *e && (*e == 'B' || *e == 'b'));
    }();
    return v;
  }

  // Env DTXQCD_STAGEB_AUX_OPTION = "B" → device aux kernel; anything else → α.
  static bool aux_option_b_() {
    static bool v = []() {
      const char *e = std::getenv("DTXQCD_STAGEB_AUX_OPTION");
      return (e && *e && (*e == 'B' || *e == 'b' || *e == '1'));
    }();
    return v;
  }

  // Caller may invalidate the device aux cache to force re-pack on next call.
  // (Safe to call even when cache not allocated.)
  void ResetAuxCache() {
    if (aux_cache_.allocated()) {
      DtxqcdQudaAuxKernelDevice::free_aux_cache(aux_cache_);
    }
  }

  ~DTXQCDMpcOpQUDA() {
    ResetAuxCache();
    // Task #223: skip QUDA persistent-handle teardown.  QUDA's atexit cleanup
    // tears down its global state before our destructor runs at program exit;
    // calling delete on Dirac* or .reset() on a ColorSpinorField unique_ptr
    // at this point crashes inside QUDA's destructor (it accesses already-
    // freed internal state).  Persistent QUDA handles are program-lifetime
    // objects; the OS reclaims their memory at exit cleanly.  Drop ownership
    // without invoking destructors.
    style_b_state_.dirac = nullptr;
    (void)style_b_state_.in_native.release();
    (void)style_b_state_.out_native.release();
    style_b_state_.initialized = false;
    style_c_sp_state_.dirac = nullptr;
    style_c_sp_state_.initialized = false;
    if (aux_perm_d_) {
      acceleratorFreeDevice(aux_perm_d_);
      aux_perm_d_ = nullptr;
    }
  }

  // Phase β Session B — env DTXQCD_MP_CG_CLEANUP=1 → enable SP cleanup tail.
  // NOT cached so tests can toggle between runs via setenv/unsetenv.
  static bool sp_cleanup_enabled_() {
    const char *e = std::getenv("DTXQCD_MP_CG_CLEANUP");
    return (e && *e && std::atoi(e) != 0);
  }

  GridCartesian         *Grid()   { return grid_; }
  GridRedBlackCartesian *RbGrid() { return rbgrid_; }
  QudaInvertParam       &InvertParam() { return inv_param_; }
  QudaGaugeParam        &GaugeParam()  { return gauge_param_; }

 private:
  // Core apply: handles M and (via dagger flag for the QUDA call) M† for the
  // Wilson-clover piece.  The aux contribution and γ_2·conj transformation
  // are unchanged either way at the wrapper level.  Mdag here is exposed via
  // γ_5-hermiticity rather than QUDA's dagger flag — kept for completeness.
  void apply_mat_(const Field &in, Field &out, bool dagger) {
    int V = Quda::local_volume(grid_);
    inv_param_.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

    // Per-call work buffers (4 in, 4 out = 8 × 24V doubles).  Sized once per
    // call — at multishift inner loop in M-wrap.5 these'll become device-
    // resident reusables.
    std::array<std::vector<double>, DtxqcdNf> in_u_eo, in_l_eo, out_u_eo, out_l_eo;
    for (int a = 0; a < DtxqcdNf; ++a) {
      in_u_eo[a].assign(24 * V, 0.0);
      in_l_eo[a].assign(24 * V, 0.0);
      out_u_eo[a].assign(24 * V, 0.0);
      out_l_eo[a].assign(24 * V, 0.0);
    }

    // Pack: upper as-is, lower with C·conj() prepended (= conj(C·v) given
    // C is purely real spinor permutation+sign).  We apply C first, then
    // conj_inplace -- both orderings give the same result since C is real.
    for (int a = 0; a < DtxqcdNf; ++a) {
      Quda::fermion_to_eo_buffer(in.upper.f[a], in_u_eo[a].data());
      // Pack to a temp, then C → conj into the in_l_eo[a] slot we feed QUDA.
      std::vector<double> tmp(24 * V);
      Quda::fermion_to_eo_buffer(in.lower.f[a], tmp.data());
      Quda::ApplyCKernelHost(in_l_eo[a].data(), tmp.data(), V);
      Quda::ConjInplaceHost(in_l_eo[a].data(), V);
    }

    // Upper-block MatQuda: dagger from the apply_mat_ argument.
    // Lower-block MatQuda: dagger=YES UNCONDITIONALLY.  The surrounding
    // conj·MatQuda(dagger=YES)·conj sequence is the M_QCD^T transposition
    // (M^T = conj(M^†)), which combined with the C·...·C sandwich gives
    // the lower block of M_DTXQCD.  (For Mdag/M_22 the outer γ5 wrap in
    // Mdag() handles the dagger; this routine's `dagger` argument applies
    // only to the upper block.)
    for (int a = 0; a < DtxqcdNf; ++a) {
      inv_param_.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
      MatQuda(out_u_eo[a].data(), in_u_eo[a].data(), &inv_param_);
      inv_param_.dagger = QUDA_DAG_YES;
      MatQuda(out_l_eo[a].data(), in_l_eo[a].data(), &inv_param_);
    }
    inv_param_.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

    // Lower-block frame correction (in place): conj → C → negate.  Together
    // with the pre-MatQuda C → conj, this implements
    //   M_22 = C^T · M_QCD^T · C = -C · M_QCD^T · C   (C^T = -C in DR)
    // The negate at the end realizes the explicit "-" sign — equivalently
    // we could fold it into a "-C" kernel; doing it as a post-multiply keeps
    // the apply_C kernel itself unsigned.
    for (int a = 0; a < DtxqcdNf; ++a) {
      Quda::ConjInplaceHost(out_l_eo[a].data(), V);
      std::vector<double> tmp(24 * V);
      Quda::ApplyCKernelHost(tmp.data(), out_l_eo[a].data(), V);
      for (int k = 0; k < 24 * V; ++k) out_l_eo[a][k] = -tmp[k];
    }

    // Unpack back to Grid layout.  The aux kernel will then accumulate
    // σ+π+s+p (diagonal) + d+n (cross) on top of these Wilson-clover Mat
    // results.
    for (int a = 0; a < DtxqcdNf; ++a) {
      Quda::eo_buffer_to_fermion(out_u_eo[a].data(), out.upper.f[a]);
      Quda::eo_buffer_to_fermion(out_l_eo[a].data(), out.lower.f[a]);
    }

    // Production-locked flags (transpose_aux=true, use_dn_conj=true) match the
    // DTXQCD_SIGMA_PI_HERMITIAN_ONLY + DTXQCD_DN_COMPLEX_SYMMETRIC constexpr
    // path that Test_dtxqcd_quda_aux_kernel validated bit-exact.
    DtxqcdQudaAuxKernel::ApplyFusedDtxqcdAuxKernel(sigma_, pi_, d_, n_, s_, p_,
                                                   in, out,
                                                   /*transpose_aux=*/true,
                                                   /*use_dn_conj=*/true,
                                                   /*accumulate=*/true);
  }

  void setup_params_() {
    Coordinate lc = grid_->LocalDimensions();

    // ---- gauge_param --------------------------------------------------------
    gauge_param_ = newQudaGaugeParam();
    for (int d = 0; d < 4; ++d) gauge_param_.X[d] = lc[d];
    gauge_param_.anisotropy = 1.0;
    gauge_param_.type = QUDA_WILSON_LINKS;
    gauge_param_.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gauge_param_.t_boundary = anti_periodic_t_ ? QUDA_ANTI_PERIODIC_T
                                                : QUDA_PERIODIC_T;
    gauge_param_.cpu_prec = QUDA_DOUBLE_PRECISION;
    gauge_param_.cuda_prec = QUDA_DOUBLE_PRECISION;
    // Phase β Session B — SP sloppy gauge for the optional SP cleanup path
    // (DTXQCD_MP_CG_CLEANUP=1).  QUDA's loadGaugeQuda allocates both DP and
    // SP gauge copies when sloppy != main; no extra calls needed.
    gauge_param_.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    gauge_param_.cuda_prec_precondition = QUDA_SINGLE_PRECISION;
    gauge_param_.cuda_prec_refinement_sloppy = QUDA_SINGLE_PRECISION;
    gauge_param_.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param_.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param_.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
    gauge_param_.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param_.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gauge_param_.ga_pad = max_face_pad_();
    gauge_param_.struct_size = sizeof(gauge_param_);

    // ---- inv_param ----------------------------------------------------------
    inv_param_ = newQudaInvertParam();
    inv_param_.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
    inv_param_.kappa  = 1.0 / (2.0 * (4.0 + mass_));
    inv_param_.mass   = mass_;
    inv_param_.Ls     = 1;

    inv_param_.clover_csw    = csw_;
    inv_param_.clover_coeff  = csw_ * inv_param_.kappa;
    inv_param_.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param_.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    // Phase β Session B — SP sloppy clover for SP cleanup path.
    inv_param_.clover_cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    inv_param_.clover_cuda_prec_precondition = QUDA_SINGLE_PRECISION;
    inv_param_.clover_cuda_prec_refinement_sloppy = QUDA_SINGLE_PRECISION;
    inv_param_.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param_.compute_clover = 1;
    inv_param_.compute_clover_inverse = 1;
    inv_param_.return_clover = 0;
    inv_param_.return_clover_inverse = 0;

    // MatQuda usage: solution_type / solve_type are for invertQuda; we only
    // need dslash_type, kappa, mass, clover params + dagger flag.  Setting
    // them to sane defaults anyway for the host-side ColorSpinorParam ctor
    // that MatQuda uses internally.
    inv_param_.solution_type   = QUDA_MAT_SOLUTION;
    inv_param_.solve_type      = QUDA_DIRECT_SOLVE;
    inv_param_.matpc_type      = QUDA_MATPC_EVEN_EVEN;
    inv_param_.dagger          = QUDA_DAG_NO;
    inv_param_.mass_normalization   = QUDA_MASS_NORMALIZATION;
    inv_param_.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

    inv_param_.cpu_prec  = QUDA_DOUBLE_PRECISION;
    inv_param_.cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param_.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param_.cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param_.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
    inv_param_.preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inv_param_.gamma_basis  = gamma_basis_;
    inv_param_.dirac_order  = QUDA_DIRAC_ORDER;
    inv_param_.input_location  = QUDA_CPU_FIELD_LOCATION;
    inv_param_.output_location = QUDA_CPU_FIELD_LOCATION;

    inv_param_.verbosity = QUDA_SILENT;
    inv_param_.struct_size = sizeof(inv_param_);
  }

  // M-wrap.5b.3 Style B: persistent native-layout CSF scratches + persistent
  // Dirac.  Lazy-allocated on first M_device call when DTXQCD_STAGEB_STYLE=B.
  // The cpu ColorSpinorParam template is precomputed; each call ref-wraps the
  // caller-owned raw `double*` device buffer via cpu_param_tmpl_.v = ptr.
  struct StyleBPersistent {
    quda::Dirac *dirac = nullptr;
    quda::ColorSpinorParam cpu_param_tmpl;   // ref-style template, set .v per call
    quda::ColorSpinorParam cuda_param_tmpl;  // native-layout template (used to ctor scratches)
    std::unique_ptr<quda::ColorSpinorField> in_native;
    std::unique_ptr<quda::ColorSpinorField> out_native;
    bool initialized = false;
  };
  mutable StyleBPersistent style_b_state_;

  void init_style_b_() {
    if (style_b_state_.initialized) return;
    // cpu (ref-style) template — wraps a raw double* device buffer in
    // DIRAC_ORDER+EVEN_ODD layout matching fermion_to_eo_buffer's output.
    // pc=false (we operate on the full-volume Wilson-clover M, not Schur Mpc).
    quda::lat_dim_t X;
    for (int d = 0; d < 4; ++d) X[d] = gauge_param_.X[d];
    for (int d = 4; d < QUDA_MAX_DIM; ++d) X[d] = 1;
    style_b_state_.cpu_param_tmpl = quda::ColorSpinorParam(
        nullptr, inv_param_, X, /*pc=*/false, QUDA_CUDA_FIELD_LOCATION);
    // cuda (native) template — promoted from cpu via the dedicated ctor that
    // forces fieldOrder=NATIVE + create=NULL_FIELD_CREATE.
    style_b_state_.cuda_param_tmpl = quda::ColorSpinorParam(
        style_b_state_.cpu_param_tmpl, inv_param_, QUDA_CUDA_FIELD_LOCATION);
    style_b_state_.cuda_param_tmpl.create = QUDA_NULL_FIELD_CREATE;
    // Persistent native scratches reused across all Mat calls.
    style_b_state_.in_native = std::make_unique<quda::ColorSpinorField>(
        style_b_state_.cuda_param_tmpl);
    style_b_state_.out_native = std::make_unique<quda::ColorSpinorField>(
        style_b_state_.cuda_param_tmpl);
    // Persistent Dirac — Wilson-clover M operator built from current
    // inv_param_ (kappa, csw, anti-periodic-t baked).
    quda::DiracParam dParam;
    quda::setDiracParam(dParam, &inv_param_, /*pc=*/false);
    style_b_state_.dirac = quda::Dirac::create(dParam);
    style_b_state_.initialized = true;
  }

  void shutdown_style_b_() {
    if (!style_b_state_.initialized) return;
    delete style_b_state_.dirac;
    style_b_state_.dirac = nullptr;
    style_b_state_.in_native.reset();
    style_b_state_.out_native.reset();
    style_b_state_.initialized = false;
  }

  // Phase β Session B — persistent SP Dirac + SP CSF allocation template.
  // Lazy-allocated on first M_device_csf_sp / MakeNativeCsfParamSp call.
  struct StyleCSpPersistent {
    quda::Dirac *dirac = nullptr;
    quda::ColorSpinorParam cuda_param_tmpl;  // native SP CSF template
    QudaInvertParam inv_param_sp;            // SP-precision shadow of inv_param_
    bool initialized = false;
  };
  mutable StyleCSpPersistent style_c_sp_state_;

  void init_style_c_sp_() {
    if (style_c_sp_state_.initialized) return;
    init_style_b_();  // ensures Dirac DP infra ready (gauge already loaded)
    // Build an SP-precision shadow of inv_param_ — only the precision fields
    // differ.  gauge_param_ is shared; QUDA picks up SP sloppy gauge if
    // configured, otherwise the DP gauge is downcast on-the-fly per call.
    style_c_sp_state_.inv_param_sp = inv_param_;
    style_c_sp_state_.inv_param_sp.cpu_prec  = QUDA_SINGLE_PRECISION;
    // Sloppy slots SP; cuda_prec stays DP for the Dirac construction path
    // (gaugePrecise check inside setDiracSloppyParam's inner setDiracParam
    // requires gaugePrecise->Precision() == inv_param->cuda_prec == DP).
    style_c_sp_state_.inv_param_sp.cuda_prec = QUDA_SINGLE_PRECISION;  // tmp for CSF param
    style_c_sp_state_.inv_param_sp.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    style_c_sp_state_.inv_param_sp.cuda_prec_refinement_sloppy = QUDA_SINGLE_PRECISION;
    style_c_sp_state_.inv_param_sp.cuda_prec_precondition = QUDA_SINGLE_PRECISION;
    style_c_sp_state_.inv_param_sp.clover_cuda_prec = QUDA_SINGLE_PRECISION;  // tmp
    style_c_sp_state_.inv_param_sp.clover_cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    style_c_sp_state_.inv_param_sp.clover_cuda_prec_precondition = QUDA_SINGLE_PRECISION;
    style_c_sp_state_.inv_param_sp.clover_cuda_prec_refinement_sloppy = QUDA_SINGLE_PRECISION;

    // SP cuda CSF template: built with cuda_prec=SP so the CSF allocation
    // is genuinely SP.  Construction uses inv_param.cuda_prec for the
    // LatticeFieldParam precision (color_spinor_field.h:240).
    auto cpu_sp = style_b_state_.cpu_param_tmpl;
    cpu_sp.setPrecision(QUDA_SINGLE_PRECISION);
    style_c_sp_state_.cuda_param_tmpl = quda::ColorSpinorParam(
        cpu_sp, style_c_sp_state_.inv_param_sp, QUDA_CUDA_FIELD_LOCATION);
    style_c_sp_state_.cuda_param_tmpl.create = QUDA_NULL_FIELD_CREATE;

    // Now switch inv_param_sp.cuda_prec / clover_cuda_prec to DP so the
    // inner setDiracParam check inside setDiracSloppyParam passes
    // (gaugePrecise + cloverPrecise are DP).
    style_c_sp_state_.inv_param_sp.cuda_prec = QUDA_DOUBLE_PRECISION;
    style_c_sp_state_.inv_param_sp.clover_cuda_prec = QUDA_DOUBLE_PRECISION;

    // Persistent SP Dirac — use setDiracSloppyParam (interface_quda.cpp:1554)
    // which overrides diracParam.gauge to gaugeSloppy (SP, pre-allocated by
    // loadGaugeQuda because gauge_param_.cuda_prec_sloppy=SP).
    quda::DiracParam dParam;
    quda::setDiracSloppyParam(dParam, &style_c_sp_state_.inv_param_sp, /*pc=*/false);
    style_c_sp_state_.dirac = quda::Dirac::create(dParam);
    style_c_sp_state_.initialized = true;
  }

  void shutdown_style_c_sp_() {
    if (!style_c_sp_state_.initialized) return;
    delete style_c_sp_state_.dirac;
    style_c_sp_state_.dirac = nullptr;
    style_c_sp_state_.initialized = false;
  }

  // Style C perm table: (parity, x_cb) → our_eo_idx (production EO ordering).
  // Required by ApplyAuxKernelNative — aux fields are packed in production EO
  // order, but the native CSF accessor presents (parity, x_cb).  Build once
  // per DTXQCDMpcOpQUDA lifetime (depends only on QUDA's csf.copy convention
  // and lattice dims, both fixed at ctor time).
  void init_aux_perm_() {
    if (aux_perm_built_) return;
    init_style_b_();  // needs in_native scratch
    int X_full[4];
    for (int d = 0; d < 4; ++d) X_full[d] = gauge_param_.X[d];
    std::vector<int> perm = DtxqcdQudaAuxKernelNativeV2::build_perm_table(
        *style_b_state_.in_native, inv_param_, X_full, grid_);
    size_t bytes = perm.size() * sizeof(int);
    aux_perm_d_ = (int*)acceleratorAllocDevice(bytes);
    acceleratorCopyToDevice(perm.data(), aux_perm_d_, bytes);
    aux_perm_size_ = perm.size();
    aux_perm_built_ = true;
  }

  // Single Mat or Mdag call via persistent CSF + Dirac, with MatQuda's
  // implicit `ax(0.5/kappa)` rescale applied via quda::blas::ax.
  // Inputs: raw 24·V device pointers (in_d -> out_d) in DIRAC_ORDER+EVEN_ODD
  // layout.  Per-call cost = copy(ref→native) + dirac->M + copy(native→ref).
  void apply_dirac_op_style_b_(double *in_d, double *out_d,
                                bool dagger, double scale, int /*V*/) {
    // Wrap caller's raw buffers as ref-style CSFs (no allocation).
    auto cpuParamIn = style_b_state_.cpu_param_tmpl;
    cpuParamIn.v = in_d;
    quda::ColorSpinorField in_ref(cpuParamIn);
    auto cpuParamOut = style_b_state_.cpu_param_tmpl;
    cpuParamOut.v = out_d;
    quda::ColorSpinorField out_ref(cpuParamOut);

    // Layout convert: ref(SPACE_SPIN_COLOR+EO) → native(FLOAT2-padded).
    style_b_state_.in_native->copy(in_ref);

    // Forward or dagger.
    if (dagger) {
      style_b_state_.dirac->Mdag(*style_b_state_.out_native,
                                 *style_b_state_.in_native);
    } else {
      style_b_state_.dirac->M(*style_b_state_.out_native,
                              *style_b_state_.in_native);
    }
    // MatQuda's implicit kappa rescale (out *= 0.5/kappa).  `quda::blas::ax`
    // takes complex coefficients; pass scale + 0i.  cvector_ref / cvector are
    // brace-initialised vector wrappers per QUDA's batched API.
    {
      quda::vector_ref<quda::ColorSpinorField> y_ref{*style_b_state_.out_native};
      quda::vector<quda::Complex> a_coeff{quda::Complex(scale, 0.0)};
      quda::blas::ax(a_coeff, y_ref);
    }

    // Layout convert back: native → ref(SPACE_SPIN_COLOR+EO) → out_d.
    out_ref.copy(*style_b_state_.out_native);
  }

  int max_face_pad_() const {
    Coordinate lc = grid_->LocalDimensions();
    int x_face = lc[1] * lc[2] * lc[3] / 2;
    int y_face = lc[0] * lc[2] * lc[3] / 2;
    int z_face = lc[0] * lc[1] * lc[3] / 2;
    int t_face = lc[0] * lc[1] * lc[2] / 2;
    return std::max({x_face, y_face, z_face, t_face});
  }

  GridCartesian         *grid_;
  GridRedBlackCartesian *rbgrid_;
  RealD mass_, csw_;
  const LatticeDtxqcdSigma &sigma_;
  const LatticeDtxqcdPi    &pi_;
  const LatticeDtxqcdD     &d_;
  const LatticeDtxqcdN     &n_;
  const LatticeDtxqcdS     &s_;
  const LatticeDtxqcdP     &p_;
  LatticeGaugeField &Umu_;
  bool anti_periodic_t_;
  QudaGammaBasis gamma_basis_;
  QudaGaugeParam gauge_param_{};
  QudaInvertParam inv_param_{};
  std::vector<std::vector<double>> eo_gauge_bufs_;
  bool gauge_loaded_;

  // Option β device aux cache (lazily allocated on first M_device call when
  // DTXQCD_STAGEB_AUX_OPTION=B).  See aux_option_b_() / ResetAuxCache().
  DtxqcdQudaAuxKernelDevice::DeviceAuxCache aux_cache_;

  // Style C aux perm table (lazy-built on first M_device_csf call).  Maps
  // QUDA's native (parity, x_cb) → production EO order so the native aux
  // kernel can index aux fields packed in production EO layout.
  int *aux_perm_d_ = nullptr;
  std::size_t aux_perm_size_ = 0;
  bool aux_perm_built_ = false;
};

NAMESPACE_END(Grid);
