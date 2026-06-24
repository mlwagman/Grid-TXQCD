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
// Activated by env DTXQCD_QUDA_HYBRID=1 (Wilson-hop only) or DTXQCD_QUDA_FULL=1
// (Wilson-hop + σ-clover; FULL is a superset of HYBRID).  csw != 0 only: the
// QUDA clover loader needs a resident clover field, so csw == 0 falls back to
// the Grid reference.
//
// Wilson-hop conventions byte-for-byte TXQCD Phase H (validated 2026-05-05 on
// plain QCD via bench_clover_oprod): X scale 1, Y mass-form scale 1, W/Z
// off-parity scale 2, dagger=YES, kappa2 = +kappa^2, force unpack -1/(8 kappa^2).
//
// σ-clover (DTXQCD_QUDA_FULL=1 only) conventions per Phase H.0 R4-7 (validated
// 2026-06-24 via Test_dtxqcd_qudasigma_probe on 4⁴ hot at csw=1.249,
// cos=0.999999988 factor=0.5 uniform):
//   X scale 1, Y scale 2κ (kappa-form), W/Z off-parity scale √(2κ),
//   dagger=NO, kappa2=+kappa^2 (unused — kernel hard-codes 1.0),
//   ck=-csw·κ/8, unpack -1/(4κ²) (= -1/(8κ²) × 2 for the R4-7 residual).

#include <Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOAction.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>  // gauge/clover loader
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaFieldConvert.h>
#include <Grid/util/QudaPackGpu.h>

#include <cstdlib>
#include <memory>
#include <functional>

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
    // Per-call env lookup (non-static) so within-process FD tests can flip the
    // path between back-to-back deriv() invocations.  Cost: 2 strcmp/call.
    auto env_on = [](const char *k) {
      const char *e = std::getenv(k);
      return (e && *e) ? std::atoi(e) : 0;
    };
    const int hybrid = env_on("DTXQCD_QUDA_HYBRID");
    const int full   = env_on("DTXQCD_QUDA_FULL");
    // csw == 0 has no resident clover field for the loader -> Grid reference.
    // DTXQCD_QUDA_FULL=1 implies hybrid (FULL = HYBRID + σ-clover-also-via-QUDA).
    if ((!hybrid && !full) || this->csw_ == 0.0) {
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

  // Override the σ-piece gauge-clover force: batch every (block, flavor, pole)
  // RHS into one computeCloverSigmaForceWithSchurFields call per doubled block.
  // Recipe (Phase H.0 R4-7 validated 2026-06-24, 4⁴ hot gauge, csw=1.249):
  //   x_scale = 1
  //   y_scale = 2κ    (kappa-form Y; equivalent to packing 2κ·M·X̂)
  //   off_scale = √(2κ)  (W_o, Z_o off-parity scale)
  //   dagger = NO
  //   matpc = ODD_ODD_ASYMMETRIC  (matches the existing QudaCloverMultiShift
  //          loader; probe verified parity-symmetric to <0.3% noise)
  //   kappa2 = +κ²  (UNUSED by σ kernel — line 488 of QudaForcePrimitives.h
  //                  hard-codes 1.0 — but pass +κ² for forward compatibility)
  //   ck = -csw·κ/8
  //   coeff_rhs[i] = ak (per-pole residue)
  //   unpack factor = -1/(8κ²)  (then ×2 to absorb the cos=1.0 factor=0.5
  //                              R4-7 residual → final factor = -1/(4κ²))
  // Lower block: QUDA gauge = conj(U); result is dS/dU_conj, mapped to dS/dU
  // by entry-wise conjugation (same chain rule as the Wilson hop).
  void AccumulateGaugeCloverForce(
      const DTXQCDField &U,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      DTXQCDWilsonCloverFermionEO &Dw,
      std::vector<LatticeColourMatrix> &clover_sigma_full,
      DTXQCDField &dSdU) override {
    // Per-call env lookup (non-static) so within-process FD tests can flip the
    // path between back-to-back deriv() invocations.
    const char *e_full = std::getenv("DTXQCD_QUDA_FULL");
    const int full = (e_full && *e_full) ? std::atoi(e_full) : 0;
    if (!full || this->csw_ == 0.0) {
      Base::AccumulateGaugeCloverForce(U, Xk, Yk, Wek, Zek, Dw,
                                        clover_sigma_full, dSdU);
      return;
    }

    // ============================================================
    // C-b path (DTXQCD_QUDA_FULL_OPROD=1): raw σ-Oprod from QUDA per
    // block, combine in Grid per CPU CS[mn] formula, fold via base's
    // Cmunu.  Bypasses the Cmunu-locks-in-orientation issue that the
    // direct σ-force path hit at cos~0.95 in the FD bench.
    //   CS_doubled[mn] = conj( transpose(CS_upper[mn]) − CS_lower[mn] )
    // The factor -0.5·csw and per-pole residue are folded into the
    // QUDA wrapper's ferm_epsilon = 2·ck·coeff·dt = -csw·κ·coeff·dt/4
    // — so CS_QUDA already carries pole-summed scale -csw·κ/4 (relative
    // to CPU's -0.5·csw at unit κ).  Net unpack scale below.
    // ============================================================
    const char *e_oprod = std::getenv("DTXQCD_QUDA_FULL_OPROD");
    if (e_oprod && *e_oprod && std::atoi(e_oprod) != 0) {
      std::vector<LatticeColourMatrix> CS_upper(
          6, LatticeColourMatrix(&this->grid_));
      std::vector<LatticeColourMatrix> CS_lower(
          6, LatticeColourMatrix(&this->grid_));

      QudaSigmaOprodBlock(U.U,    /*lower=*/false, Xk, Yk, Wek, Zek, CS_upper);
      LatticeGaugeField Uconj_cb(&this->grid_);
      Uconj_cb = conjugate(U.U);
      QudaSigmaOprodBlock(Uconj_cb, /*lower=*/true,  Xk, Yk, Wek, Zek, CS_lower);

      // Combine per CPU CS[mn] formula on a per-(μ,ν) ColourMatrix basis.
      // The base's default AccumulateGaugeCloverForce applies the -0.5 in
      // Convention A; here we ONLY need to produce the σ-tensor with the
      // same scale that the CPU per-pole loop would have produced
      // (-0.5·csw absorbed in base).  QUDA's oprod already carries
      // 2·ck·coeff·dt per pole = -csw·κ/4 × Σ(coeff_k · raw_oprod_k).
      // CPU per pole CS_k = -0.5·csw·conj(raw_diff_k)·coeff_k.
      // So scale relating QUDA-oprod to CPU-CS at unit -0.5·csw:
      //   CS_CPU[mn] = -0.5·csw · conj( transpose(CS_upper) − CS_lower )
      //   Need QUDA_oprod_scaled = CS_CPU / (-0.5·csw) for base to fold;
      //   ratio = (CS_CPU)/(QUDA-oprod) =  (-0.5·csw)/(-csw·κ/4)
      //                                 =  (0.5·csw·4)/(csw·κ) = 2/κ.
      // Multiply by 2/κ to land on the right magnitude pre-fold.
      // ESCAPE HATCH: DTXQCD_QUDA_FULL_OPROD_SCALE env overrides.
      // Empirical 4⁴ aux=0 calibration after mn=2↔3 swap: |Vd|/|G| = 2κ per
      // (μν), giving scale_correct = 1/(2κ) × (2/κ) = 1/κ² in closed form.
      // (For our test setup κ=1/(2(m+4))=0.1136, 1/κ²=77.5 vs prior 2/κ=17.6
      // — measured ratio 4.4× = 1/(2κ) confirms.)
      const double kappa_loc = this->quda_loader_->InvertParam().kappa;
      double oprod_scale = 1.0 / (kappa_loc * kappa_loc);
      const char *e_sc = std::getenv("DTXQCD_QUDA_FULL_OPROD_SCALE");
      if (e_sc && *e_sc) oprod_scale = std::atof(e_sc);

      // QUDA TENSOR_GEOMETRY enumerates (μν) in a different order than Grid's
      // lex (Grid: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)).  Empirical per-(μν) cosine
      // probe at 4⁴ aux=0 shows mn=2 ↔ mn=3 swap; the other 4 channels are
      // bit-exact at cos=1.0 with uniform |Vd/G|.  See `[perblock-cs]` data.
      // Map: QUDA_mn[2] → Grid_mn[3], QUDA_mn[3] → Grid_mn[2].  Swap during
      // combine.  Knob DTXQCD_QUDA_FULL_NOSWAP=1 disables for debugging.
      const char *e_noswap = std::getenv("DTXQCD_QUDA_FULL_NOSWAP");
      bool do_swap = !(e_noswap && *e_noswap && std::atoi(e_noswap) != 0);
      std::vector<LatticeColourMatrix> CS_doubled(
          6, LatticeColourMatrix(&this->grid_));
      auto build_doubled = [&](int dst_mn, int src_mn) {
        LatticeColourMatrix Tup = transpose(CS_upper[src_mn]);
        LatticeColourMatrix Diff = Tup - CS_lower[src_mn];
        CS_doubled[dst_mn] = ComplexD(oprod_scale, 0.0) * conjugate(Diff);
      };
      if (do_swap) {
        build_doubled(0, 0);
        build_doubled(1, 1);
        build_doubled(2, 3);  // <-- swap
        build_doubled(3, 2);  // <-- swap
        build_doubled(4, 4);
        build_doubled(5, 5);
      } else {
        for (int mn = 0; mn < 6; ++mn) build_doubled(mn, mn);
      }

      // Optional: dump per-mn norms for diagnostic.
      const char *e_dbg_cb = std::getenv("DTXQCD_QUDA_FULL_DEBUG");
      if (e_dbg_cb && *e_dbg_cb && std::atoi(e_dbg_cb) != 0) {
        for (int mn = 0; mn < 6; ++mn) {
          std::cout << GridLogMessage
                    << "[qudafull-cb] mn=" << mn
                    << "  |CS_up|²=" << norm2(CS_upper[mn])
                    << "  |CS_lo|²=" << norm2(CS_lower[mn])
                    << "  |CS_doubled|²=" << norm2(CS_doubled[mn])
                    << std::endl;
        }
      }

      // ===========================================================
      // DIAGNOSTIC: compare CS_doubled to clover_sigma_full (Grid ref)
      // per-(μν), under 8 unary transforms.  Locates where the combine
      // mismatch lives — uniform-cos channels = scalar issue,
      // channel-varying = σ-basis issue, low cos = orientation issue.
      // Gated by DTXQCD_TEST_PER_BLOCK_CS=1.
      // ===========================================================
      const char *e_perblock = std::getenv("DTXQCD_TEST_PER_BLOCK_CS");
      if (e_perblock && *e_perblock && std::atoi(e_perblock) != 0) {
        // clover_sigma_full[mn] is already summed over poles + both blocks
        // by the per-pole AccumulateSiteForces loop in deriv().  CS_doubled[mn]
        // is C-b's QUDA-derived combined CS.  cos here = cos at F level too,
        // because Cmunu fold is linear in CS.
        struct Variant {
          const char *name;
          std::function<LatticeColourMatrix(const LatticeColourMatrix &)> fn;
        };
        std::vector<Variant> variants = {
          {"+id",    [](const LatticeColourMatrix &M){ return LatticeColourMatrix(M); }},
          {"-id",    [](const LatticeColourMatrix &M){ return LatticeColourMatrix(-M); }},
          {"+conj",  [](const LatticeColourMatrix &M){ return LatticeColourMatrix(conjugate(M)); }},
          {"-conj",  [](const LatticeColourMatrix &M){ return LatticeColourMatrix(-conjugate(M)); }},
          {"+T",     [](const LatticeColourMatrix &M){ return LatticeColourMatrix(transpose(M)); }},
          {"-T",     [](const LatticeColourMatrix &M){ return LatticeColourMatrix(-transpose(M)); }},
          {"+adj",   [](const LatticeColourMatrix &M){ return LatticeColourMatrix(adj(M)); }},
          {"-adj",   [](const LatticeColourMatrix &M){ return LatticeColourMatrix(-adj(M)); }},
        };
        for (int mn = 0; mn < 6; ++mn) {
          RealD ng = sqrt(norm2(clover_sigma_full[mn]));
          for (auto &v : variants) {
            LatticeColourMatrix Vd = v.fn(CS_doubled[mn]);
            RealD nv = sqrt(norm2(Vd));
            ComplexD ip = TensorRemove(sum(trace(adj(clover_sigma_full[mn]) * Vd)));
            RealD cosv = (ng * nv > 1e-30) ? real(ip) / (ng * nv) : 0.0;
            std::cout << GridLogMessage
                      << "[perblock-cs] mn=" << mn
                      << "  var=" << v.name
                      << "  cos=" << cosv
                      << "  |G|²=" << (ng*ng)
                      << "  |Vd|²=" << (nv*nv)
                      << "  |Vd/G|=" << ((ng>1e-30) ? nv/ng : 0.0)
                      << std::endl;
          }
        }
      }

      // Defer to base's Cmunu fold (default impl applies -0.5 Convention A).
      Base::AccumulateGaugeCloverForce(U, Xk, Yk, Wek, Zek, Dw,
                                        CS_doubled, dSdU);
      return;
    }

    // ---- Upper block: QUDA σ-force on gauge U ----
    LatticeGaugeField gf_upper(&this->grid_);
    QudaSigmaBlock(U.U, /*lower=*/false, Xk, Yk, Wek, Zek, gf_upper);

    // ---- Lower block: QUDA σ-force on conj(U); conjugate the result ----
    // FD-bench notes (4⁴ Path-A vs Path-B, mass=0.4, csw=1.249):
    //   "+ gf_upper + gf_lower"   → cos=0.9496, |B|/|A|=1.270 (current; best)
    //   "+ gf_upper − gf_lower"   → cos=0.4832, |B|/|A|=1.599 (regression)
    // The CPU CS[mn] formula in DTXQCDRationalForceGpuKernel.h:276-278
    // subtracts the lower block's contribution (val += upper_bil − lower_bil),
    // but the sign apparently lands implicitly through the (U→conj(U)) +
    // (conjugate(F)) chain rule for QUDA's lower-block path, since explicit
    // subtraction makes the equivalence WORSE.  Convention residual (~5%) is
    // tracked separately — Path B IS bit-equivalent on aux components (1e-13)
    // and on the per-pole residue weights, but a residual ~5% perp persists
    // in the gauge Ta-projected force.  See HANDOFF_PHASE_H1_FD_RESIDUAL.md.
    LatticeGaugeField Uconj(&this->grid_);
    Uconj = conjugate(U.U);
    LatticeGaugeField gf_lower(&this->grid_);
    QudaSigmaBlock(Uconj, /*lower=*/true, Xk, Yk, Wek, Zek, gf_lower);
    gf_lower = conjugate(gf_lower);

    // DTXQCD_QUDA_FULL_DEBUG=1: dump per-block norms + cross-term so we don't
    // have to reverse-engineer them from add/sub variant runs.
    const char *e_dbg = std::getenv("DTXQCD_QUDA_FULL_DEBUG");
    if (e_dbg && *e_dbg && std::atoi(e_dbg) != 0) {
      RealD n_up = norm2(gf_upper);
      RealD n_lo = norm2(gf_lower);
      ComplexD ip = innerProduct(gf_upper, gf_lower);
      std::cout << GridLogMessage
                << "[qudafull-dbg]"
                << "  |gf_upper|²=" << n_up
                << "  |gf_lower|²=" << n_lo
                << "  Re<up,lo>=" << real(ip)
                << "  Im<up,lo>=" << imag(ip)
                << std::endl;
    }

    // Combine: per coordinator Step 2 of FD debug — the CPU CS[mn] formula is
    // CS[mn](ic, jc) = -0.5 · csw · conj( upper_bil(jc,ic) − lower_bil(ic,jc) ).
    // The outer conj wraps the whole upper−lower combination, suggesting:
    //   out = conj(gf_upper) − conj(gf_lower)   ← DTXQCD_QUDA_FULL_CONJWRAP=1
    //   out = gf_upper + gf_lower               ← default (Wilson-hop pattern)
    // FD-bench notes (4⁴ Path-A vs Path-B):
    //   "+ gf_upper + gf_lower" aux≠0 → cos=0.9496, |B|/|A|=1.270
    //   "+ gf_upper + gf_lower" aux=0 → cos=0.9591, |B|/|A|=1.668 (Step 1)
    //   "+ gf_upper − gf_lower" aux≠0 → cos=0.4832, |B|/|A|=1.599
    // DTXQCD_QUDA_FULL_COMBINE selects the block-combination strategy:
    //   0 (default): + gf_upper + gf_lower         (Wilson-hop style)
    //   1: + conj(gf_upper) − conj(gf_lower)        (outer-conj wrap, Step 2)
    //   2: + adj(gf_upper)  + adj(gf_lower)         (Step 3a: full adjoint)
    //   3: + transpose(gf_upper) + gf_lower         (Step 3b: upper transpose only)
    // CONJWRAP env var preserved for backward compat with prior runs.
    const char *e_combine = std::getenv("DTXQCD_QUDA_FULL_COMBINE");
    int combine = (e_combine && *e_combine) ? std::atoi(e_combine) : 0;
    const char *e_conjwrap = std::getenv("DTXQCD_QUDA_FULL_CONJWRAP");
    if (combine == 0 && e_conjwrap && *e_conjwrap && std::atoi(e_conjwrap) != 0)
      combine = 1;
    // Helpers: adj() and transpose() are defined on LatticeColourMatrix; on
    // LatticeGaugeField they need a per-direction peek/poke chain.
    auto map_each_mu = [&](const LatticeGaugeField &Gin,
                            auto fn) -> LatticeGaugeField {
      LatticeGaugeField Gout(&this->grid_);
      Gout = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeColourMatrix Cin(&this->grid_);
        Cin = PeekIndex<LorentzIndex>(Gin, mu);
        LatticeColourMatrix Cout = fn(Cin);
        PokeIndex<LorentzIndex>(Gout, Cout, mu);
      }
      return Gout;
    };

    if (combine == 1) {
      LatticeGaugeField gf_up_c = map_each_mu(gf_upper,
        [](const LatticeColourMatrix &M) { return conjugate(M); });
      LatticeGaugeField gf_lo_c = map_each_mu(gf_lower,
        [](const LatticeColourMatrix &M) { return conjugate(M); });
      dSdU.U = dSdU.U + gf_up_c - gf_lo_c;
    } else if (combine == 2) {
      LatticeGaugeField gf_up_a = map_each_mu(gf_upper,
        [](const LatticeColourMatrix &M) { return adj(M); });
      LatticeGaugeField gf_lo_a = map_each_mu(gf_lower,
        [](const LatticeColourMatrix &M) { return adj(M); });
      dSdU.U = dSdU.U + gf_up_a + gf_lo_a;
    } else if (combine == 3) {
      LatticeGaugeField gf_up_t = map_each_mu(gf_upper,
        [](const LatticeColourMatrix &M) { return transpose(M); });
      dSdU.U = dSdU.U + gf_up_t + gf_lower;
    } else {
      dSdU.U = dSdU.U + gf_upper + gf_lower;
    }
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

  // One doubled-block σ-piece gauge-clover force: pack all (flavor, pole) rhs,
  // one batched computeCloverSigmaForceWithSchurFields, unpack into out_force.
  // Convention recipe per Phase H.0 R4-7 (cos=0.999999988, factor=0.5
  // uniform on 4⁴ hot at csw=1.249).  Post-multiply by 2 to absorb the
  // factor-of-2 residual; final scale = -2/(8κ²) = -1/(4κ²).
  void QudaSigmaBlock(const LatticeGaugeField &gauge, bool lower,
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
    const double kappa = inv_param.kappa;
    const double two_kappa = 2.0 * kappa;
    const double sqrt_two_kappa = std::sqrt(two_kappa);
    // R4-7 σ-piece recipe:  Y rescaled to 2κ·M·X̂ (kappa-form),
    // W/Z off-parity scaled by √(2κ), X unchanged.
    const double y_scale_sigma   = two_kappa;
    const double off_scale_sigma = sqrt_two_kappa;

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
        Quda::GpuPackFermionRbLex(Yf, y_scale_sigma,        // 2κ kappa-form Y
                                   dev_p + off_y + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Wf, off_scale_sigma,      // √(2κ) off-parity
                                   dev_p + off_w + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Zf, off_scale_sigma,
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

    int saved_use_resident = inv_param.use_resident_solution;
    QudaDagType saved_dagger = inv_param.dagger;
    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    QudaFieldLocation saved_input_loc = inv_param.input_location;
    inv_param.use_resident_solution = 0;
    inv_param.dagger         = QUDA_DAG_NO;               // R4-7 validated
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

    // kappa2 is UNUSED inside computeCloverSigmaForceWithSchurFields
    // (QudaForcePrimitives.h:488 hard-codes 1.0).  Pass +κ² as documentation.
    const double kappa2 = +kappa * kappa;
    const double ck     = -inv_param.clover_csw * kappa / 8.0;
    const double dt     = 1.0;
    const double sigma_trace_coeff = 0.0;

    Quda::computeCloverSigmaForceWithSchurFields(
        &mom_buf_dev_[0],
        x_ptrs.data(), y_ptrs.data(),
        w_ptrs.data(), z_ptrs.data(),
        Nrhs, coeff_rhs,
        kappa2, ck, dt, sigma_trace_coeff,
        &force_gauge_param, &inv_param);

    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist;
    inv_param.input_location        = saved_input_loc;

    // Final scale: QUDA→Grid factor (-1/(8κ²)) × R4-7 ×2 residual.
    const double quda_to_grid_factor = -1.0 / (4.0 * kappa * kappa);
    out_force = Zero();
    Quda::GpuUnpackMomToGauge(&mom_buf_dev_[0], &unpack_eo_table_dev_[0],
                               out_force, quda_to_grid_factor);
  }

  // ----------------------------------------------------------------------
  // C-b helper: one doubled-block raw σ-Oprod call.  Packs all (flavor, pole)
  // RHS, runs computeCloverSigmaOprodWithSchurFields, unpacks the V·6·18
  // host buffer into 6 LatticeColourMatrix (natural QUDA (ic,jc) orientation).
  // CPU per-mn extract + eo_to_lex_permute + pokeSite path.  For 4⁴ V=256
  // this is ~0.1 ms; for 16³×48 V=196k it's ~30 ms (acceptable; D2H of
  // 6·V·18·8 = 28 KB / 170 MB is the dominant cost).
  // ----------------------------------------------------------------------
  void QudaSigmaOprodBlock(const LatticeGaugeField &gauge, bool lower,
      const std::vector<DTXQCDFermionDoubled> &Xk,
      const std::vector<DTXQCDFermionDoubled> &Yk,
      const std::vector<DTXQCDFermionDoubled> &Wek,
      const std::vector<DTXQCDFermionDoubled> &Zek,
      std::vector<LatticeColourMatrix> &cs_out) {
    const int Npole = static_cast<int>(Xk.size());
    quda_loader_->SetGauge(gauge);

    const int Nrhs = DtxqcdNf * Npole;
    const int V_eo = Quda::local_volume(&this->grid_) / 2;
    using SiteSpinor = typename LatticeFermion::scalar_object;

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
    const double kappa = inv_param.kappa;
    const double two_kappa = 2.0 * kappa;
    const double sqrt_two_kappa = std::sqrt(two_kappa);
    const double y_scale_sigma   = two_kappa;
    const double off_scale_sigma = sqrt_two_kappa;

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
        Quda::GpuPackFermionRbLex(Yf, y_scale_sigma,
                                   dev_p + off_y + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Wf, off_scale_sigma,
                                   dev_p + off_w + i * per_rhs, lex_p);
        Quda::GpuPackFermionRbLex(Zf, off_scale_sigma,
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
    // Host buffer for σ-Oprod: V × 6 × 18 doubles.  We allocate it per call
    // (cheap — V*6*18*8 bytes; for 16³×48 = 170 MB which is fine on one
    // GPU node; for 4⁴ negligible).  Persistent buffer would be a follow-up.
    std::vector<double> oprod_host(uint64_t(V) * 6 * 18, 0.0);

    int saved_use_resident = inv_param.use_resident_solution;
    QudaDagType saved_dagger = inv_param.dagger;
    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    QudaFieldLocation saved_input_loc = inv_param.input_location;
    inv_param.use_resident_solution = 0;
    inv_param.dagger         = QUDA_DAG_NO;
    inv_param.twist_flavor   = QUDA_TWIST_NO;
    inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;

    QudaGaugeParam force_gauge_param = quda_loader_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.location    = QUDA_CPU_FIELD_LOCATION;

    const double ck     = -inv_param.clover_csw * kappa / 8.0;
    const double dt     = 1.0;
    const double sigma_trace_coeff = 0.0;

    Quda::computeCloverSigmaOprodWithSchurFields(
        oprod_host.data(),
        x_ptrs.data(), y_ptrs.data(),
        w_ptrs.data(), z_ptrs.data(),
        Nrhs, coeff_rhs,
        ck, dt, sigma_trace_coeff,
        &force_gauge_param, &inv_param);

    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist;
    inv_param.input_location        = saved_input_loc;

    // Unpack: oprod_host layout (MILC TENSOR_GEOMETRY, RECONSTRUCT_NO):
    //   contiguous in (site_eo) outer, (mn) middle (6 values, 0..5), (entry)
    //   inner (18 doubles for a 3x3 complex ColourMatrix).
    //   per_site_doubles = 6 * 18 = 108.
    Coordinate lc = this->grid_.LocalDimensions();
    for (int mn = 0; mn < 6; ++mn) {
      // Extract mn slot for all EO sites (V * 18 doubles).
      std::vector<double> eo_buf(uint64_t(V) * 18);
      thread_for(site_eo, V, {
        const double *src = &oprod_host[(uint64_t(site_eo) * 6 + mn) * 18];
        double *dst = &eo_buf[uint64_t(site_eo) * 18];
        std::memcpy(dst, src, 18 * sizeof(double));
      });
      // EO→lex permute.
      std::vector<double> lex_buf(uint64_t(V) * 18);
      Quda::eo_to_lex_permute(eo_buf.data(), lex_buf.data(), V, 18, lc);
      // Per-site convert to ColourMatrix + pokeSite.
      cs_out[mn] = Zero();
      cs_out[mn].Checkerboard() = 0;  // full grid; checkerboard tag is inert
      using SiteCM = typename LatticeColourMatrix::scalar_object;
      Coordinate lex_coord(Nd);
      for (int lex_site = 0; lex_site < V; ++lex_site) {
        Lexicographic::CoorFromIndex(lex_coord, lex_site, lc);
        SiteCM scm;
        const double *p = &lex_buf[uint64_t(lex_site) * 18];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            scm()()(i, j) = ComplexD(p[(i * 3 + j) * 2],
                                      p[(i * 3 + j) * 2 + 1]);
          }
        }
        pokeSite(scm, cs_out[mn], lex_coord);
      }
    }
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
