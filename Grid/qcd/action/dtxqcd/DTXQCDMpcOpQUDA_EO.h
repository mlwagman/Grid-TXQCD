#pragma once
// DTXQCDMpcOpQUDA_EO — device-resident odd-CB Schur (Mpc) operator for the
// DTXQCD doubled Wilson-Clover fermion.  EO Style C port (2026-07-02, task
// #239 / HANDOFF_EO_STYLE_C_PORT.md).
//
//   Mpc psi_o = M_oo psi_o - M_oe M_ee^{-1} M_eo psi_o
//
// composed entirely on device against native-CSF CG state:
//
//   hop  (M_eo / M_oe): QUDA parity Dslash on the persistent Dirac.  Upper
//        block direct; lower block via the C·M^T·C sandwich restricted to
//        hopping — PreMatLowerNative -> Dslash(dagger) -> PostMatLowerNative
//        (the same frame kernels the full-volume Style C path uses; they are
//        elementwise over the CSF's own sites, so parity fields work as-is).
//        Both blocks scaled by hop_scale = -1/2 (Grid Meooe = -(1/2)·D_quda;
//        the full-M 0.5/kappa MatQuda rescale identity gives exactly this).
//        Env DTXQCD_EO_HOP_SCALE overrides for convention bisection.
//   site (M_ee^{-1} on Even, M_oo on Odd): cuBLAS gemmBatched against the
//        fermion class's persistent device caches (InvDevCache(Even) /
//        FwdDevCache(Odd), column-major [Nsite*48*48], RB-lex == cb_site site
//        order).  The CSF <-> flat-48 conversion is: quda csf.copy (native ->
//        production-EO layout + DR basis, device-to-device) followed by a
//        strided cudaMemcpy2D interleave of the 4 (block,flavor) fields into
//        contiguous per-site 48-vectors (rows r = blk*24 + a*12 + alpha*3 + i,
//        matching the cache build).  No new device kernels.
//
// Mdag = Gamma5 Mpc Gamma5 (Mpc inherits gamma5-Hermiticity from the Schur
// factorization of the gamma5-Hermitian M48; blockwise proof in the EO-port
// handoff discussion).  Uses the existing Gamma5 native kernel.
//
// The class subclasses DTXQCDMpcOpQUDA to reuse gauge load, persistent Dirac,
// params and the g5 machinery.  All shadowed methods are NON-virtual: the
// StyleC solvers are templated on the concrete Mop type, so the derived
// methods bind statically (no vtable, no dispatch cost).
//
// Aux fields: NOT applied separately — sigma/pi/d/n/s/p enter Mpc only
// through M_ee/M_oo, which the batched caches already contain.  The caches
// are rebuilt by DTXQCDWilsonCloverFermionEO::ImportFields on every aux/gauge
// change (MakeEOp per refresh/S/deriv), so cache staleness follows exactly
// the existing EO action lifecycle.

#include <Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h>
#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h>

NAMESPACE_BEGIN(Grid);

class DTXQCDMpcOpQUDA_EO : public DTXQCDMpcOpQUDA {
 public:
  typedef DTXQCDFermionDoubled Field;
  static constexpr int kNField = 2 * DtxqcdNf;  // (upper,lower) x flavor

  DTXQCDMpcOpQUDA_EO(DTXQCDWilsonCloverFermionEO &Dw,
                     LatticeGaugeField &U,
                     GridCartesian &grid,
                     GridRedBlackCartesian &rbgrid,
                     RealD mass, RealD csw,
                     const LatticeDtxqcdSigma &sigma,
                     const LatticeDtxqcdPi    &pi,
                     const LatticeDtxqcdD     &d,
                     const LatticeDtxqcdN     &n,
                     const LatticeDtxqcdS     &s,
                     const LatticeDtxqcdP     &p,
                     bool anti_periodic_t = true)
      : DTXQCDMpcOpQUDA(U, grid, rbgrid, mass, csw, sigma, pi, d, n, s, p,
                        anti_periodic_t),
        Dw_(Dw) {}

  ~DTXQCDMpcOpQUDA_EO() {
    // Mirror the base: QUDA teardown order makes destructors unsafe at exit;
    // drop ownership of persistent QUDA handles without destroying.
    eo_state_.dirac_dag = nullptr;
    for (auto &u : eo_state_.even_a) (void)u.release();
    for (auto &u : eo_state_.even_b) (void)u.release();
    for (auto &u : eo_state_.even_scratch) (void)u.release();
    eo_state_.initialized = false;
    if (eo_state_.flat_in)  { acceleratorFreeDevice(eo_state_.flat_in);  eo_state_.flat_in  = nullptr; }
    if (eo_state_.flat_out) { acceleratorFreeDevice(eo_state_.flat_out); eo_state_.flat_out = nullptr; }
  }

  // ---- parity-CSF allocation templates (shadow the base's full-volume ones)
  quda::ColorSpinorParam MakeNativeCsfParam() {
    init_eo_state_();
    return eo_state_.cuda_param_pc;
  }
  quda::ColorSpinorParam MakeNativeCsfParamSp() {
    // SP path not ported (MP multishift variants are default-OFF and were
    // ruled out as net-loss; see HANDOFF_EO_STYLE_C_PORT.md "what NOT to
    // port").  Fail loudly if something routes here.
    GRID_ASSERT(0 && "DTXQCDMpcOpQUDA_EO: SP CSF path not ported");
    return eo_state_.cuda_param_pc;
  }

  // ---- Grid-layout Mpc (odd-CB doubled fermions).  Used by Gate-1 tests and
  // available for host-side fallbacks; the CG hot path uses M_device_csf.
  void M(const Field &in, Field &out) {
    DTXQCDMpcOp mpc(Dw_);
    mpc.M(in, out);
  }
  void Mdag(const Field &in, Field &out) {
    DTXQCDMpcOp mpc(Dw_);
    mpc.Mdag(in, out);
  }

  // ---- device-resident Mpc on native parity CSFs --------------------------
  // in/out: 4 CSFs (upper.f[0,1], lower.f[0,1]) on the ODD parity.
  // scratch_lower: 2 CSFs, ODD parity (lower-frame scratch for hop1).
  void M_device_csf(quda::ColorSpinorField *in_upper[DtxqcdNf],
                    quda::ColorSpinorField *in_lower[DtxqcdNf],
                    quda::ColorSpinorField *out_upper[DtxqcdNf],
                    quda::ColorSpinorField *out_lower[DtxqcdNf],
                    quda::ColorSpinorField *scratch_lower[DtxqcdNf],
                    bool dagger = false) {
    // dagger flag routes through Mdag_device_csf's gamma5 wrap instead; the
    // base used it to flip the QUDA Mat, but Mpc's dagger is NOT a plain flag
    // flip.  Assert against silent misuse.
    GRID_ASSERT(!dagger && "DTXQCDMpcOpQUDA_EO: use Mdag_device_csf");
    init_eo_state_();
    cudaDeviceSynchronize();

    quda::ColorSpinorField *t_e [kNField];  // hop1 result (even)
    quda::ColorSpinorField *t_e2[kNField];  // Mee^{-1} result (even)
    for (int k = 0; k < kNField; ++k) {
      t_e [k] = eo_state_.even_a[k].get();
      t_e2[k] = eo_state_.even_b[k].get();
    }
    quda::ColorSpinorField *in_all[kNField] = {
        in_upper[0], in_upper[1], in_lower[0], in_lower[1]};
    quda::ColorSpinorField *out_all[kNField] = {
        out_upper[0], out_upper[1], out_lower[0], out_lower[1]};

    // ---- hop1: t_e = M_eo psi_o  (odd -> even) ----
    HopDevice_(in_all, t_e, /*out_parity=*/QUDA_EVEN_PARITY,
               scratch_lower);

    // ---- site1: t_e2 = M_ee^{-1} t_e  (even) ----
    SiteMatDevice_(t_e, t_e2, Even, /*use_inverse=*/true);

    // ---- hop2: out = M_oe t_e2  (even -> odd) ----
    HopDevice_(t_e2, out_all, /*out_parity=*/QUDA_ODD_PARITY,
               eo_state_.even_scratch_ptrs());

    // ---- site2 + subtract: out = M_oo psi_o - out ----
    // Reuse t_e/t_e2 slots' flat buffers via SiteMatDevice_ into t_e (odd
    // values live fine in the even-template CSFs: same volume/geometry).
    SiteMatDevice_(in_all, t_e, Odd, /*use_inverse=*/false);
    // out = t_e - out  ==>  out = (-1)*out + 1*t_e  via axpby.
    {
      quda::vector<double> a{1.0}, b{-1.0};
      for (int k = 0; k < kNField; ++k) {
        quda::vector_ref<const quda::ColorSpinorField> xr{*t_e[k]};
        quda::vector_ref<quda::ColorSpinorField> yr{*out_all[k]};
        quda::blas::axpby(a, xr, b, yr);
      }
    }
    cudaDeviceSynchronize();
  }

  void Mdag_device_csf(quda::ColorSpinorField *in_upper[DtxqcdNf],
                       quda::ColorSpinorField *in_lower[DtxqcdNf],
                       quda::ColorSpinorField *out_upper[DtxqcdNf],
                       quda::ColorSpinorField *out_lower[DtxqcdNf],
                       quda::ColorSpinorField *scratch_lower[DtxqcdNf]) {
    int g5v = g5_variant_();
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_upper[a], g5v);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_lower[a], g5v);
    }
    M_device_csf(in_upper, in_lower, out_upper, out_lower, scratch_lower);
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_upper[a], g5v);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*in_lower[a], g5v);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*out_upper[a], g5v);
      DtxqcdQudaGamma5Native::ApplyGamma5InplaceNative(*out_lower[a], g5v);
    }
  }

  // Exposed for the Gate-1 piecewise tests (hop / site-inverse / site-forward
  // in isolation).  Same contracts as the private chain steps.
  void TestHopDevice(quda::ColorSpinorField *in_all[kNField],
                     quda::ColorSpinorField *out_all[kNField],
                     QudaParity out_parity,
                     quda::ColorSpinorField *scr_lo[DtxqcdNf]) {
    init_eo_state_();
    cudaDeviceSynchronize();
    HopDevice_(in_all, out_all, out_parity, scr_lo);
    cudaDeviceSynchronize();
  }
  void TestSiteMatDevice(quda::ColorSpinorField *in_all[kNField],
                         quda::ColorSpinorField *out_all[kNField],
                         int cb, bool use_inverse) {
    init_eo_state_();
    cudaDeviceSynchronize();
    SiteMatDevice_(in_all, out_all, cb, use_inverse);
    cudaDeviceSynchronize();
  }

  DTXQCDWilsonCloverFermionEO &Fermion() { return Dw_; }

 protected:
  // -------------------------------------------------------------------------
  struct EOPersistent {
    quda::ColorSpinorParam cpu_param_pc;    // ref-style, parity flat buffer
    quda::ColorSpinorParam cuda_param_pc;   // native parity CSF template
    quda::Dirac *dirac_dag = nullptr;       // persistent dagger-Dirac (lower hop)
    // persistent even-parity scratch (native CSF): two 4-field sets + 2 for
    // the lower-frame scratch of hop2.
    std::array<std::unique_ptr<quda::ColorSpinorField>, kNField> even_a;
    std::array<std::unique_ptr<quda::ColorSpinorField>, kNField> even_b;
    std::array<std::unique_ptr<quda::ColorSpinorField>, DtxqcdNf> even_scratch;
    quda::ColorSpinorField *even_scratch_p[DtxqcdNf] = {nullptr, nullptr};
    // flat device buffers for the site-matrix path: [V_cb x 48] ComplexD each,
    // used both as the 4x12 per-field staging area and the interleaved B/C.
    ComplexD *flat_in  = nullptr;   // interleaved input  B
    ComplexD *flat_out = nullptr;   // interleaved output C
    // per-field staging flat buffers (24 doubles/site each, production-EO
    // half-slab layout) carved out of one allocation:
    ComplexD *stage = nullptr;      // kNField x 12 x V_cb ComplexD
    // cuBLAS per-site pointer arrays (rebuilt when cache generation changes)
    deviceVector<ComplexD*> Amk_inv, Amk_fwd, Bkn, Cmn;
    uint64_t inv_gen_seen = ~0ull, fwd_gen_seen = ~0ull;
    uint64_t nsite = 0;
    bool initialized = false;

    quda::ColorSpinorField **even_scratch_ptrs() {
      even_scratch_p[0] = even_scratch[0].get();
      even_scratch_p[1] = even_scratch[1].get();
      return even_scratch_p;
    }
  };
  mutable EOPersistent eo_state_;

  static RealD HopScale_() {
    static RealD v = []() {
      if (const char *e = std::getenv("DTXQCD_EO_HOP_SCALE"); e && *e)
        return (RealD)std::atof(e);
      return (RealD)(-0.5);   // Grid Meooe = -(1/2) * D_quda
    }();
    return v;
  }

  void init_eo_state_() {
    if (eo_state_.initialized) return;
    init_style_b_();   // persistent (forward) Dirac + gauge assumed loaded

    // Parity CSF templates: pc=true halves the X[0] geometry per QUDA
    // convention; native promotion mirrors init_style_b_.
    quda::lat_dim_t X;
    for (int d_ = 0; d_ < 4; ++d_) X[d_] = gauge_param_.X[d_];
    for (int d_ = 4; d_ < QUDA_MAX_DIM; ++d_) X[d_] = 1;
    eo_state_.cpu_param_pc = quda::ColorSpinorParam(
        nullptr, inv_param_, X, /*pc=*/true, QUDA_CUDA_FIELD_LOCATION);
    eo_state_.cuda_param_pc = quda::ColorSpinorParam(
        eo_state_.cpu_param_pc, inv_param_, QUDA_CUDA_FIELD_LOCATION);
    eo_state_.cuda_param_pc.create = QUDA_NULL_FIELD_CREATE;

    for (int k = 0; k < kNField; ++k) {
      eo_state_.even_a[k] = std::make_unique<quda::ColorSpinorField>(
          eo_state_.cuda_param_pc);
      eo_state_.even_b[k] = std::make_unique<quda::ColorSpinorField>(
          eo_state_.cuda_param_pc);
    }
    for (int a = 0; a < DtxqcdNf; ++a) {
      eo_state_.even_scratch[a] = std::make_unique<quda::ColorSpinorField>(
          eo_state_.cuda_param_pc);
    }

    // Persistent dagger-Dirac for the lower-block hop sandwich (avoids
    // toggling the forward Dirac's dagger flag mid-chain).
    {
      QudaInvertParam inv_dag = inv_param_;
      inv_dag.dagger = QUDA_DAG_YES;
      quda::DiracParam dParam;
      quda::setDiracParam(dParam, &inv_dag, /*pc=*/false);
      eo_state_.dirac_dag = quda::Dirac::create(dParam);
    }

    // Flat device buffers.
    const uint64_t Vcb = (uint64_t)Quda::local_volume(grid_) / 2;
    eo_state_.nsite = Vcb;
    eo_state_.flat_in  = (ComplexD *)acceleratorAllocDevice(
        Vcb * 48 * sizeof(ComplexD));
    eo_state_.flat_out = (ComplexD *)acceleratorAllocDevice(
        Vcb * 48 * sizeof(ComplexD));
    eo_state_.stage    = (ComplexD *)acceleratorAllocDevice(
        (uint64_t)kNField * 12 * Vcb * sizeof(ComplexD));

    eo_state_.Bkn.resize(Vcb);
    eo_state_.Cmn.resize(Vcb);
    {
      std::vector<ComplexD *> B_h(Vcb), C_h(Vcb);
      for (uint64_t s_ = 0; s_ < Vcb; ++s_) {
        B_h[s_] = eo_state_.flat_in  + s_ * 48;
        C_h[s_] = eo_state_.flat_out + s_ * 48;
      }
      acceleratorCopyToDevice((void *)B_h.data(), (void *)&eo_state_.Bkn[0],
                              Vcb * sizeof(ComplexD *));
      acceleratorCopyToDevice((void *)C_h.data(), (void *)&eo_state_.Cmn[0],
                              Vcb * sizeof(ComplexD *));
    }
    eo_state_.initialized = true;
  }

  // Rebuild the A-pointer array for a cache when its generation changed.
  void ensure_A_ptrs_(int cb, bool use_inverse) {
    typedef DTXQCDWilsonCloverFermionEO F;
    const uint64_t Vcb = eo_state_.nsite;
    if (use_inverse) {
      GRID_ASSERT(F::InvDevN(cb) == Vcb &&
                  "EO Mop: inverse device cache missing/mismatched — need "
                  "DTXQCD_PRECOMPUTE_GPU=1 (CUDA default) + ImportFields");
      if (eo_state_.inv_gen_seen == F::InvDevGen(cb) &&
          eo_state_.Amk_inv.size() == Vcb) return;
      eo_state_.Amk_inv.resize(Vcb);
      std::vector<ComplexD *> A_h(Vcb);
      ComplexD *base = &F::InvDevCache(cb)[0];
      for (uint64_t s_ = 0; s_ < Vcb; ++s_) A_h[s_] = base + s_ * 48 * 48;
      acceleratorCopyToDevice((void *)A_h.data(), (void *)&eo_state_.Amk_inv[0],
                              Vcb * sizeof(ComplexD *));
      eo_state_.inv_gen_seen = F::InvDevGen(cb);
    } else {
      GRID_ASSERT(F::FwdDevN(cb) == Vcb &&
                  "EO Mop: forward device cache missing/mismatched — need "
                  "DTXQCD_PRECOMPUTE_GPU=1 (CUDA default) + ImportFields");
      if (eo_state_.fwd_gen_seen == F::FwdDevGen(cb) &&
          eo_state_.Amk_fwd.size() == Vcb) return;
      eo_state_.Amk_fwd.resize(Vcb);
      std::vector<ComplexD *> A_h(Vcb);
      ComplexD *base = &F::FwdDevCache(cb)[0];
      for (uint64_t s_ = 0; s_ < Vcb; ++s_) A_h[s_] = base + s_ * 48 * 48;
      acceleratorCopyToDevice((void *)A_h.data(), (void *)&eo_state_.Amk_fwd[0],
                              Vcb * sizeof(ComplexD *));
      eo_state_.fwd_gen_seen = F::FwdDevGen(cb);
    }
  }

  // ---- hopping: out_all(parity P) = Meooe in_all(parity ~P) ---------------
  // upper: dirac->Dslash; lower: PreMatLowerNative -> dagger-Dslash ->
  // PostMatLowerNative.  Both scaled by HopScale_().
  void HopDevice_(quda::ColorSpinorField *in_all[kNField],
                  quda::ColorSpinorField *out_all[kNField],
                  QudaParity out_parity,
                  quda::ColorSpinorField *scr_lo[DtxqcdNf]) {
    // Lower-frame pre on the two lower inputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPreMatLowerNative::ApplyPreMatLowerNative(
          *in_all[DtxqcdNf + a], *scr_lo[a]);
    }
    cudaDeviceSynchronize();

    quda::vector<quda::Complex> hs{quda::Complex(HopScale_(), 0.0)};
    for (int a = 0; a < DtxqcdNf; ++a) {
      // upper block: forward Dslash.
      style_b_state_.dirac->Dslash(*out_all[a], *in_all[a], out_parity);
      // lower block: dagger Dslash on the pre-rotated frame.
      eo_state_.dirac_dag->Dslash(*out_all[DtxqcdNf + a], *scr_lo[a],
                                  out_parity);
      quda::vector_ref<quda::ColorSpinorField> yu{*out_all[a]};
      quda::vector_ref<quda::ColorSpinorField> yl{*out_all[DtxqcdNf + a]};
      quda::blas::ax(hs, yu);
      quda::blas::ax(hs, yl);
    }
    cudaDeviceSynchronize();

    // Lower-frame post on the two lower outputs.
    for (int a = 0; a < DtxqcdNf; ++a) {
      DtxqcdQudaPostMatLowerNative::ApplyPostMatLowerNative(
          *out_all[DtxqcdNf + a]);
    }
    cudaDeviceSynchronize();
  }

  // ---- site-matrix apply: out_all = A(cb) . in_all -------------------------
  //   use_inverse=true : A = M_ee^{-1} cache (cb must be Even in the chain)
  //   use_inverse=false: A = M_oo forward cache (cb Odd in the chain)
  // CSF -> flat (csf.copy, device) -> interleave (cudaMemcpy2D) -> gemmBatched
  // -> deinterleave -> flat -> CSF.
  void SiteMatDevice_(quda::ColorSpinorField *in_all[kNField],
                      quda::ColorSpinorField *out_all[kNField],
                      int cb, bool use_inverse) {
    const uint64_t Vcb = eo_state_.nsite;
    ensure_A_ptrs_(cb, use_inverse);

    quda::lat_dim_t X;
    for (int d_ = 0; d_ < 4; ++d_) X[d_] = gauge_param_.X[d_];
    for (int d_ = 4; d_ < QUDA_MAX_DIM; ++d_) X[d_] = 1;
    QudaInvertParam inv = inv_param_;
    inv.input_location  = QUDA_CUDA_FIELD_LOCATION;
    inv.output_location = QUDA_CUDA_FIELD_LOCATION;

    // (1) native CSF -> per-field flat stage (device csf.copy layout convert;
    //     SPACE_SPIN_COLOR order, DR basis — the layout the caches contract).
    for (int k = 0; k < kNField; ++k) {
      double *stage_k = (double *)(eo_state_.stage + (uint64_t)k * 12 * Vcb);
      quda::ColorSpinorParam cpuParam(stage_k, inv, X, /*pc=*/true,
                                      QUDA_CUDA_FIELD_LOCATION);
      quda::ColorSpinorField flat_csf(cpuParam);
      flat_csf.copy(*in_all[k]);
    }
    cudaDeviceSynchronize();

    // (2) interleave: B[s*48 + k*12 + c12] = stage_k[s*12 + c12]
    for (int k = 0; k < kNField; ++k) {
      cudaMemcpy2D(/*dst=*/eo_state_.flat_in + (uint64_t)k * 12,
                   /*dpitch=*/48 * sizeof(ComplexD),
                   /*src=*/eo_state_.stage + (uint64_t)k * 12 * Vcb,
                   /*spitch=*/12 * sizeof(ComplexD),
                   /*width=*/12 * sizeof(ComplexD),
                   /*height=*/Vcb, cudaMemcpyDeviceToDevice);
    }

    // (3) batched gemv: C[48] = A[48x48] . B[48] per site (column-major).
    {
      GridBLAS blas;
      blas.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                       48, 1, 48,
                       ComplexD(1.0, 0.0),
                       use_inverse ? eo_state_.Amk_inv : eo_state_.Amk_fwd,
                       eo_state_.Bkn,
                       ComplexD(0.0, 0.0),
                       eo_state_.Cmn);
      blas.synchronise();
    }

    // (4) deinterleave + (5) flat -> native CSF.
    for (int k = 0; k < kNField; ++k) {
      cudaMemcpy2D(eo_state_.stage + (uint64_t)k * 12 * Vcb,
                   12 * sizeof(ComplexD),
                   eo_state_.flat_out + (uint64_t)k * 12,
                   48 * sizeof(ComplexD),
                   12 * sizeof(ComplexD),
                   Vcb, cudaMemcpyDeviceToDevice);
      double *stage_k = (double *)(eo_state_.stage + (uint64_t)k * 12 * Vcb);
      quda::ColorSpinorParam cpuParam(stage_k, inv, X, /*pc=*/true,
                                      QUDA_CUDA_FIELD_LOCATION);
      quda::ColorSpinorField flat_csf(cpuParam);
      out_all[k]->copy(flat_csf);
    }
    cudaDeviceSynchronize();
  }

 private:
  DTXQCDWilsonCloverFermionEO &Dw_;
};

NAMESPACE_END(Grid);
