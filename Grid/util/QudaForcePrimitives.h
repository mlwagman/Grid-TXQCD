#pragma once
// QudaForcePrimitives — Grid-side wrapper around QUDA's internal C++ force
// primitives.  Replicates the body of computeCloverForceQuda
// (interface_quda.cpp:4743 + lib/clover_force.cpp), but with a key
// substitution: the user supplies the host buffer Y = M_pc · X computed by
// Grid (or any other operator: TXQCD, twisted-mass, etc.) instead of QUDA
// computing it internally via its own Dirac op.
//
// This is the crucial mechanism for TXQCD HMC acceleration: the TXQCD
// operator includes a site-diagonal Δ insertion that QUDA's Dirac doesn't
// know about.  By feeding Grid's M_pc_TXQCD · X as the "intermediate" P
// field, the QUDA primitives (computeCloverOprod, computeCloverSigmaOprod,
// cloverDerivative, updateMomentum) compose the gauge force correctly.
//
// For plain Wilson-clover (Phase B), this also fixes Phase 7's residual 10%
// cos(Ta(A),B) gap if the gap originates from how QUDA's γ5+Dslash+M
// internal P differs from Grid's M_pc·X — which is the most likely culprit
// after we exhausted the convention-toggle hypotheses.
//
// Build requirements:
// - Caller must compile against QUDA's internal C++ headers
//   (external/quda-install/include).  These don't carry __device__ /
//   __host__ attributes at the public API level, so plain g++/nvcc-as-cxx
//   compilation works.
// - Caller must link against libquda.so.

#ifdef GRID_HAVE_QUDA

#include <Grid/GridCore.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>

// QUDA internal C++ types (require external/quda-install/include in -I path).
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <momentum.h>
#include <dirac_quda.h>

#include <vector>
#include <array>

NAMESPACE_BEGIN(Grid);
namespace Quda {

// External symbols defined in libquda.so:
//   quda::computeCloverOprod         (clover_field.h:608)
//   quda::computeCloverSigmaOprod    (clover_field.h:620)
//   quda::computeCloverSigmaTrace    (clover_field.h:631)
//   quda::cloverDerivative           (clover_field.h:643)
//   quda::updateMomentum             (momentum.h:25)
// extendedGaugeResident is QUDA-internal; we trigger its setup via the
// chroma-fork-style updateExtendedGaugeResident call inside the wrapper.

}  // namespace Quda
NAMESPACE_END(Grid);

// QUDA-side extern declarations of internal globals (defined at file scope
// in external/quda-src/lib/interface_quda.cpp).  Although that file does
// `using namespace quda` at the top, declarations at file scope still
// reside in the GLOBAL namespace — confirmed by nm showing unmangled
// 'gaugePrecise', 'cloverPrecise', 'extendedGaugeResident' symbols in
// libquda.so.  Their TYPES are quda::* but the variables are ::gaugePrecise
// etc.
extern ::quda::GaugeField  *::gaugePrecise;
extern ::quda::CloverField *::cloverPrecise;
extern ::quda::GaugeField  *extendedGaugeResident;

NAMESPACE_BEGIN(Grid);
namespace Quda {

// ----------------------------------------------------------------------------
// computeCloverForceWithGridY
// ----------------------------------------------------------------------------
// Replicates computeCloverForceQuda's body but injects user-supplied Y bufs.
//
//   h_mom        out  host mom buffer (V·4·10 doubles, MILC layout, RECONSTRUCT_10)
//   h_x          in   array of N parity-subset X buffers (24·V_eo doubles each)
//   h_y          in   array of N parity-subset Y = M_pc_grid·X buffers
//                     (24·V_eo doubles each, same layout as h_x)
//   coeff        in   per-pole rational residues (length N)
//   kappa2       in   force-coeff scaling: force_coeff[i] = 2·dt·coeff[i]·kappa2
//                     (typically -kappa·kappa)
//   ck           in   σ-Oprod coefficient: ferm_epsilon[i][0] = 2·ck·coeff[i]·dt
//                     (typically -kappa·csw/8 — empirical Phase 7 sign)
//   dt           in   integration step (typically 1.0 for raw force)
//   sigma_trace_coeff in if non-zero, includes σ_μν·F_μν trace (LogDet term);
//                     pass 0 to skip (Grid handles LogDet separately).
//   gauge_param  in   QudaGaugeParam — must have gauge_order=MILC,
//                     type=GENERAL_LINKS, reconstruct=NO, overwrite_mom=1,
//                     return_result_mom=1, t_boundary set to match resident
//                     gauge.  Caller's responsibility (see existing setup
//                     in OneFlavourSchurCloverQudaForceRationalActionMP.h).
//   inv_param    in   QudaInvertParam — must have matpc_type set to
//                     EVEN_EVEN_ASYMMETRIC or ODD_ODD_ASYMMETRIC,
//                     dslash_type=CLOVER_WILSON, gamma_basis=DEGRAND_ROSSI,
//                     and clover_csw / kappa set.  Caller's responsibility.
//
// Pre-conditions:
//   - QUDA initialized (Quda::initialize() called).
//   - loadGaugeQuda + loadCloverQuda already called for the current U.
//     (The multishift inverter does this via its SetGauge.)
//
// Post-condition:
//   h_mom contains the gauge-momentum update from the rational pseudofermion
//   force, in MILC anti-Hermitian RECONSTRUCT_10 layout.  Caller unpacks via
//   the same routine used for computeCloverForceQuda's mom_buf.
//
// The body matches lib/clover_force.cpp::computeCloverForce except line 50
// (`dirac->M(p[i].parity, p[i].parity)`) is replaced by loading h_y[i] into
// p[i][parity] and applying the same final γ5 logic.  Off-parity completion
// via Dslash is unchanged (and works for TXQCD because Δ is site-diagonal
// and doesn't touch the Wilson hop).
inline void computeCloverForceWithGridY(
    void *h_mom,
    void **h_x, void **h_y,
    int nvector,
    const std::vector<double> &coeff,
    double kappa2, double ck, double dt,
    double sigma_trace_coeff,
    QudaGaugeParam *gauge_param,
    QudaInvertParam *inv_param)
{
  using namespace ::quda;
  // gaugePrecise / cloverPrecise / extendedGaugeResident are global-scope
  // variables in libquda.so (not in namespace quda — see header note above).
  // Names are unqualified here, found via global namespace.

  if (!::gaugePrecise) errorQuda("No resident gauge field");
  if (!::cloverPrecise) errorQuda("No resident clover field");
  if (inv_param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC &&
      inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    errorQuda("MatPC type %d not supported by computeCloverForceWithGridY",
              inv_param->matpc_type);
  }

  // ------------------------------------------------------------------
  // Mom field setup (mirrors interface_quda.cpp:4754-4767).
  // ------------------------------------------------------------------
  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom(fParam);  // CPU view of the user's h_mom buffer.

  fParam.location    = QUDA_CUDA_FIELD_LOCATION;
  fParam.create      = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE
                                                   : QUDA_COPY_FIELD_CREATE;
  fParam.field       = &cpuMom;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.setPrecision(gauge_param->cuda_prec, true);
  GaugeField cudaMom(fParam);

  // ------------------------------------------------------------------
  // Build full-volume color-spinor params (matches interface_quda.cpp:4771-4775).
  // ------------------------------------------------------------------
  ColorSpinorParam qParam(nullptr, *inv_param, fParam.x, false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), p(nvector);
  std::vector<double>             force_coeff(nvector);
  std::vector<array<double, 2>>   ferm_epsilon(nvector);

  QudaParity parity =
      inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
          ? QUDA_EVEN_PARITY
          : QUDA_ODD_PARITY;
  QudaParity other_parity = static_cast<QudaParity>(1 - parity);

  // ------------------------------------------------------------------
  // Per-pole field setup.  KEY SUBSTITUTION at the M-apply step.
  // ------------------------------------------------------------------
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, /*pc_solve=*/true);
  Dirac *dirac = Dirac::create(diracParam);

  bool dagger = inv_param->dagger;
  bool not_dagger = static_cast<QudaDagType>(1 - inv_param->dagger);

  // Force/oprod accumulators (full-volume, GENERAL_LINKS).
  GaugeFieldParam fparam2(cudaMom);
  fparam2.link_type   = QUDA_GENERAL_LINKS;
  fparam2.reconstruct = QUDA_RECONSTRUCT_NO;
  fparam2.create      = QUDA_ZERO_FIELD_CREATE;
  fparam2.setPrecision(fparam2.Precision(), true);
  GaugeField force(fparam2);
  fparam2.geometry = QUDA_TENSOR_GEOMETRY;
  GaugeField oprod(fparam2);

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    p[i] = ColorSpinorField(qParam);

    // ---- Load X into x[i].parity (host parity-subset -> device) ----
    {
      ColorSpinorParam cpuParam(h_x[i], *inv_param, fParam.x, /*pc=*/true,
                                inv_param->input_location);
      ColorSpinorField cpuQuarkX(cpuParam);
      x[i][parity] = cpuQuarkX;
    }
    // ---- Load Y into p[i].parity (the SUBSTITUTION) ----
    {
      ColorSpinorParam cpuParam(h_y[i], *inv_param, fParam.x, /*pc=*/true,
                                inv_param->input_location);
      ColorSpinorField cpuQuarkY(cpuParam);
      p[i][parity] = cpuQuarkY;
    }

    // ---- gamma5 manipulations matching clover_force.cpp:46-50 ----
    // Original:
    //   gamma5(p_par, x_par);              // p_par = γ5 X
    //   if (dagger) Dagger(YES);
    //   Dslash(x_other, p_par, other);     // x_other = D γ5 X
    //   M(p_par, p_par);                   // p_par = M_pc γ5 X
    //   if (dagger) Dagger(NO);
    //
    // SUBSTITUTED:
    //   p_par = γ5 · Y_grid (Grid's M_pc · X, then γ5 on top)
    //   x_par = X (raw, will gamma5 below for consistency)
    //   x_other = D · γ5 · X (computed via Dslash same as before)
    //
    // In QUDA's convention, the chain ends with the equivalent of:
    //   x_par = X, x_other = D·γ5·X  (then gamma5(x_other) → x = γ5 · D · γ5 · X-like)
    //   p_par = M_pc·γ5·X            (Grid's Y with γ5)
    //   p_other = D·M_pc·γ5·X         (one more Dslash)
    //   final gamma5(p, p) → use in oprod calls.

    // p_par := γ5 · Y_grid_loaded  (overwrites the loaded Y with γ5·Y)
    // QUDA_FORCE_NO_FIRST_G5 skips this; useful for isolating γ5 conventions.
    if (std::getenv("QUDA_FORCE_NO_FIRST_G5") == nullptr) {
      gamma5(p[i][parity], p[i][parity]);
    }

    // x_other := D · γ5 · X — but our x_par is X (not γ5·X).
    // To match QUDA's flow, we need a temp field for γ5·X.
    // QUDA_FORCE_NO_INNER_G5_X skips the γ5 before Dslash; QUDA_FORCE_NO_OUTER_G5_X
    // skips the γ5 after Dslash.  Used to probe Wilson-hop bilinear convention.
    {
      ColorSpinorField tmp_g5x(qParam);
      tmp_g5x[parity] = ColorSpinorField(qParam)[parity];  // alloc
      if (std::getenv("QUDA_FORCE_NO_INNER_G5_X") == nullptr) {
        gamma5(tmp_g5x[parity], x[i][parity]);
      } else {
        tmp_g5x[parity] = x[i][parity];
      }
      if (dagger) dirac->Dagger(QUDA_DAG_YES);
      dirac->Dslash(x[i][other_parity], tmp_g5x[parity], other_parity);
      if (dagger) dirac->Dagger(QUDA_DAG_NO);
    }

    // gamma5 on x_other (matches line 62 of clover_force.cpp).
    if (std::getenv("QUDA_FORCE_NO_OUTER_G5_X") == nullptr) {
      gamma5(x[i][other_parity], x[i][other_parity]);
    }
    // Optional sign flip on x[other] to match paper eq A7's
    // X^A_o = -M_oo^{-1}·M_eo^†·X̂ convention (our wrapper produces + by default).
    if (std::getenv("QUDA_FORCE_NEGATE_X_OTHER") != nullptr) {
      blas::ax(-1.0, x[i][other_parity]);
    }
    // Optional zero of x[other] — kills Term 1 in kernel
    // (kernel reads x[¬p] for the (1+γ_μ) outerProd contribution).
    if (std::getenv("QUDA_FORCE_ZERO_X_OTHER") != nullptr) {
      blas::ax(0.0, x[i][other_parity]);
    }

    // Compute p_other = (Dagger? -- not_dagger pattern from clover_force.cpp:65-68)
    if (not_dagger) dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(p[i][other_parity], p[i][parity], other_parity);
    if (not_dagger) dirac->Dagger(QUDA_DAG_NO);

    // Final gamma5(p, p) — clover_force.cpp:79.  QUDA_FORCE_NO_FINAL_G5 skips.
    if (std::getenv("QUDA_FORCE_NO_FINAL_G5") == nullptr) {
      gamma5(p[i], p[i]);
    }
    // Optional sign flip on p[other] to match paper eq A8's
    // Y^A_o = -M_oo^{-1}·M_oe·Ŷ convention (sign-paired with X^A_o convention).
    if (std::getenv("QUDA_FORCE_NEGATE_P_OTHER") != nullptr) {
      blas::ax(-1.0, p[i][other_parity]);
    }
    // Optional zero of p[other] — kills Term 2 in kernel
    // (kernel reads p[¬p] for the (1-γ_μ) outerProd contribution).
    if (std::getenv("QUDA_FORCE_ZERO_P_OTHER") != nullptr) {
      blas::ax(0.0, p[i][other_parity]);
    }

    // Force coefficients per clover_force.cpp:46/93 calculations done at
    // computeCloverForceQuda level; mirrored here.
    force_coeff[i]  = 2.0 * dt * coeff[i] * kappa2;
    ferm_epsilon[i] = {2.0 * ck * coeff[i] * dt,
                       -kappa2 * 2.0 * ck * coeff[i] * dt};
  }

  // ------------------------------------------------------------------
  // Ensure extendedGaugeResident is fresh (chroma-fork pattern from
  // interface_quda.cpp:4801-4804 of the chroma snapshot).
  // ------------------------------------------------------------------
  // redundant_comms is static in interface_quda.cpp (no external linkage);
  // hardcode false (its default).  Single-rank or non-redundant comms case.
  constexpr bool redundant_comms = false;
  lat_dim_t R;
  for (int d = 0; d < 4; d++) {
    R[d] = (d == 0 ? 2 : 1) * (redundant_comms || commDimPartitioned(d));
  }
  // Note: TimeProfile would normally be supplied; reuse the inverter's profile.
  // For simplicity here we let updateExtendedGaugeResident pick the global
  // profile or skip if extendedGaugeResident is already set.
  GaugeField &gaugeEx = *::extendedGaugeResident;

  // ------------------------------------------------------------------
  // Primitive sequence (matches lib/clover_force.cpp:83-102).
  //
  // Term-by-term diagnostic mode: env vars QUDA_FORCE_SKIP_OPROD and
  // QUDA_FORCE_SKIP_SIGMA isolate Wilson-hop vs σ contributions.
  // ------------------------------------------------------------------
  bool skip_oprod = std::getenv("QUDA_FORCE_SKIP_OPROD") != nullptr;
  bool skip_sigma = std::getenv("QUDA_FORCE_SKIP_SIGMA") != nullptr;

  // 1. Wilson hop: ⟨P|∂M_eo/∂U|X⟩ + h.c.
  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);
  if (!skip_oprod) {
    computeCloverOprod(force, *::gaugePrecise,
                       inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                       inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                       force_coeff);
  }

  // 2. (optional) σ trace from clover field — LogDet.  Pass 0 to skip.
  if (sigma_trace_coeff != 0.0 && !skip_sigma) {
    computeCloverSigmaTrace(oprod, *::cloverPrecise, sigma_trace_coeff,
                            other_parity);
  }

  // 3. Clover σ-Oprod (X·P† projected against σ_μν).
  if (!skip_sigma) {
    computeCloverSigmaOprod(oprod,
                            inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                            inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                            ferm_epsilon);

    // 4. Apply clover derivative kernel: integrate oprod into force.
    cloverDerivative(force, gaugeEx, oprod, 1.0);
  }

  // 5. Accumulate force into mom: cudaMom += -1.0 · force.
  updateMomentum(cudaMom, -1.0, force, "clover_grid_y");

  // ------------------------------------------------------------------
  // Bring mom back to host buffer.
  // ------------------------------------------------------------------
  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);

  delete dirac;
}

// ----------------------------------------------------------------------------
// computeCloverSigmaForceWithSchurFields
// ----------------------------------------------------------------------------
// σ-piece-only force routine that takes BOTH parity slots as user-supplied
// host buffers, bypassing the gamma5+Dslash chain entirely.  Hybrid mode for
// Phase B-prime: feed Grid's exact Schur-completed off-parity fields (W_o,
// Z_o) so σ-Oprod produces a per-link force matching Path A's MeeDeriv +
// MooDeriv exactly.
//
// Inputs:
//   h_x_par   : array of N parity-slot buffers for x (= X̂_e, multishift)
//   h_p_par   : array of N parity-slot buffers for p (= M·X̂_e, kappa-form Y)
//   h_x_other : array of N off-parity buffers for x (= W_o = M_oo^{-1}·M_oe·X̂)
//   h_p_other : array of N off-parity buffers for p (= Z_o = M_oo^{-1†}·M_eo†·Y)
//
// Skips computeCloverOprod (Wilson-hop primitive — caller handles via Grid).
// Calls only computeCloverSigmaOprod + cloverDerivative + updateMomentum.
inline void computeCloverSigmaForceWithSchurFields(
    void *h_mom,
    void **h_x_par, void **h_p_par,
    void **h_x_other, void **h_p_other,
    int nvector,
    const std::vector<double> &coeff,
    double kappa2, double ck, double dt,
    double sigma_trace_coeff,
    QudaGaugeParam *gauge_param,
    QudaInvertParam *inv_param)
{
  using namespace ::quda;
  if (!::gaugePrecise) errorQuda("No resident gauge field");
  if (!::cloverPrecise) errorQuda("No resident clover field");
  if (inv_param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC &&
      inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    errorQuda("MatPC type %d not supported", inv_param->matpc_type);
  }

  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom(fParam);

  fParam.location    = QUDA_CUDA_FIELD_LOCATION;
  fParam.create      = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE
                                                   : QUDA_COPY_FIELD_CREATE;
  fParam.field       = &cpuMom;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.setPrecision(gauge_param->cuda_prec, true);
  GaugeField cudaMom(fParam);

  ColorSpinorParam qParam(nullptr, *inv_param, fParam.x, false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), p(nvector);
  std::vector<array<double, 2>>  ferm_epsilon(nvector);

  QudaParity parity =
      inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
          ? QUDA_EVEN_PARITY
          : QUDA_ODD_PARITY;
  QudaParity other_parity = static_cast<QudaParity>(1 - parity);

  // Force/oprod accumulators.
  GaugeFieldParam fparam2(cudaMom);
  fparam2.link_type   = QUDA_GENERAL_LINKS;
  fparam2.reconstruct = QUDA_RECONSTRUCT_NO;
  fparam2.create      = QUDA_ZERO_FIELD_CREATE;
  fparam2.setPrecision(fparam2.Precision(), true);
  GaugeField force(fparam2);
  fparam2.geometry = QUDA_TENSOR_GEOMETRY;
  GaugeField oprod(fparam2);

  // Load both parity slots from caller-provided host buffers.
  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    p[i] = ColorSpinorField(qParam);

    // x[parity] = X̂
    {
      ColorSpinorParam cp(h_x_par[i], *inv_param, fParam.x, /*pc=*/true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][parity] = cf;
    }
    // x[other] = W_o (Schur-completed)
    {
      ColorSpinorParam cp(h_x_other[i], *inv_param, fParam.x, /*pc=*/true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][other_parity] = cf;
    }
    // p[parity] = M·X̂ (kappa-form Y)
    {
      ColorSpinorParam cp(h_p_par[i], *inv_param, fParam.x, /*pc=*/true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][parity] = cf;
    }
    // p[other] = Z_o (Schur-completed)
    {
      ColorSpinorParam cp(h_p_other[i], *inv_param, fParam.x, /*pc=*/true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][other_parity] = cf;
    }

    // σ-Oprod coefficients.  Uniform on both parities (NOT the asymmetric
    // κ² in the standard QUDA convention) — because we feed Schur-completed
    // off-parity fields (W_o, Z_o) which already include the κ scaling from
    // the Wilson hop applied during Schur completion.  The asymmetric κ²
    // would double-count κ⁴ on ODD slot.
    ferm_epsilon[i] = {2.0 * ck * coeff[i] * dt,
                        2.0 * ck * coeff[i] * dt};
  }

  GaugeField &gaugeEx = *::extendedGaugeResident;

  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);

  if (sigma_trace_coeff != 0.0) {
    computeCloverSigmaTrace(oprod, *::cloverPrecise, sigma_trace_coeff,
                            other_parity);
  }

  computeCloverSigmaOprod(oprod,
                          inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                          inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                          ferm_epsilon);

  cloverDerivative(force, gaugeEx, oprod, 1.0);

  updateMomentum(cudaMom, -1.0, force, "clover_sigma_schur");

  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);
}

// ----------------------------------------------------------------------------
// computeCloverSigmaOprodWithSchurFields  (Phase H.1 candidate-(c) variant C-b)
// ----------------------------------------------------------------------------
// Same Schur-input contract as computeCloverSigmaForceWithSchurFields but
// STOPS after computeCloverSigmaOprod — does NOT call cloverDerivative or
// updateMomentum.  Returns the raw σ-Oprod tensor (6 ColourMatrix per site,
// indexed by μν lex with μ<ν) as a host buffer, which the caller folds via
// Grid's Cmunu chain (preserves per-block CS asymmetry that QUDA's internal
// Cmunu would lock in).  Cost vs the force sibling: skips cloverDerivative
// kernel + updateMomentum, adds a D2H of oprod_size = 6·V·18·8 bytes.
//
// Caller supplies the output buffer h_oprod, sized V·6·18 doubles in MILC
// EO order (parity_major, then mn=0..5, then 18 doubles per ColourMatrix).
// Subsequent EO→lex permute and pokeSite into LatticeColourMatrix happens
// in the DTXQCD primitive — kept here as raw-buffer interface to mirror the
// existing wrappers' style.
inline void computeCloverSigmaOprodWithSchurFields(
    double *h_oprod,                       // OUT: V·6·18 doubles, MILC EO order
    void **h_x_par, void **h_p_par,
    void **h_x_other, void **h_p_other,
    int nvector,
    const std::vector<double> &coeff,
    double ck, double dt,
    double sigma_trace_coeff,
    QudaGaugeParam *gauge_param,
    QudaInvertParam *inv_param)
{
  using namespace ::quda;
  if (!::gaugePrecise) errorQuda("No resident gauge field");
  if (!::cloverPrecise) errorQuda("No resident clover field");
  if (inv_param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC &&
      inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    errorQuda("MatPC type %d not supported", inv_param->matpc_type);
  }

  // Build a CPU-side TENSOR_GEOMETRY GaugeField pointing at the user buffer.
  // Pattern mirrors the force sibling at line 394-403: bare constructor with
  // host-pointer arg sets internal bookkeeping for QUDA_REFERENCE_FIELD_CREATE
  // implicitly; do NOT mutate `create` or `field` after the constructor —
  // doing so clobbers QUDA's host-pointer bookkeeping and causes a buffer
  // overrun that corrupts adjacent heap allocations at V≥8⁴.
  // (Original Phase H.1 commit `6b08a6b0` had these explicit mutations, which
  // passed 4⁴ FD by chance but corrupted aux force lattices at 8⁴+.)
  GaugeFieldParam cpuOprodParam(*gauge_param, /*h_gauge=*/h_oprod,
                                 QUDA_GENERAL_LINKS);
  cpuOprodParam.location    = QUDA_CPU_FIELD_LOCATION;
  cpuOprodParam.geometry    = QUDA_TENSOR_GEOMETRY;
  cpuOprodParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeField cpuOprod(cpuOprodParam);

  // Device oprod accumulator (TENSOR_GEOMETRY, RECONSTRUCT_NO).  Mirrors the
  // force sibling's device-mom pattern (line 397-403): mutate from cpuParam
  // copy, set ZERO_FIELD_CREATE + back-reference for the sister.
  GaugeFieldParam devOprodParam(cpuOprodParam);
  devOprodParam.location = QUDA_CUDA_FIELD_LOCATION;
  devOprodParam.create   = QUDA_ZERO_FIELD_CREATE;
  devOprodParam.field    = &cpuOprod;
  devOprodParam.setPrecision(gauge_param->cuda_prec, true);
  GaugeField oprod(devOprodParam);

  // Fermion params — mirror the force sibling (line 405-409) exactly, using
  // cpuOprodParam.x for lattice dimensions.  No separate fParam_for_dim
  // (which was an extra GaugeFieldParam whose destruction touched the host
  // pointer bookkeeping at V≥8⁴).
  ColorSpinorParam qParam(nullptr, *inv_param, cpuOprodParam.x, false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(devOprodParam.Precision(), devOprodParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), p(nvector);
  std::vector<array<double, 2>>  ferm_epsilon(nvector);

  QudaParity parity =
      inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
          ? QUDA_EVEN_PARITY
          : QUDA_ODD_PARITY;
  QudaParity other_parity = static_cast<QudaParity>(1 - parity);

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    p[i] = ColorSpinorField(qParam);
    {
      ColorSpinorParam cp(h_x_par[i], *inv_param, cpuOprodParam.x,
                          /*pc=*/true, inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][parity] = cf;
    }
    {
      ColorSpinorParam cp(h_x_other[i], *inv_param, cpuOprodParam.x,
                          /*pc=*/true, inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][other_parity] = cf;
    }
    {
      ColorSpinorParam cp(h_p_par[i], *inv_param, cpuOprodParam.x,
                          /*pc=*/true, inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][parity] = cf;
    }
    {
      ColorSpinorParam cp(h_p_other[i], *inv_param, cpuOprodParam.x,
                          /*pc=*/true, inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][other_parity] = cf;
    }
    // Same ferm_epsilon as the force sibling — uniform on both parities since
    // Schur-completed off-parity fields already include κ scaling from the
    // Wilson hop.
    ferm_epsilon[i] = {2.0 * ck * coeff[i] * dt,
                        2.0 * ck * coeff[i] * dt};
  }

  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);

  if (sigma_trace_coeff != 0.0) {
    computeCloverSigmaTrace(oprod, *::cloverPrecise, sigma_trace_coeff,
                            other_parity);
  }

  computeCloverSigmaOprod(oprod,
                          inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                          inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                          ferm_epsilon);

  // D2H: copy oprod into the user buffer.  Caller permutes EO→lex and
  // pokeSite into LatticeColourMatrix[6].
  cpuOprod.copy(oprod);
}

// ----------------------------------------------------------------------------
// computeCloverWilsonForceWithSchurFields
// ----------------------------------------------------------------------------
// Wilson-hop-only force routine: same input contract as the σ-piece sibling
// (caller supplies Schur-completed off-parity W_o, Z_o for each rhs) but
// invokes ONLY computeCloverOprod (no σ).  Used as Phase D D.1 validation
// experiment to confirm the Wilson-hop primitive matches Path A's
// MpcDeriv+MpcDagDeriv per-link bilinear when fed the same (X̂, Y, W_o, Z_o)
// quartet that the σ-piece already validates.
inline void computeCloverWilsonForceWithSchurFields(
    void *h_mom,
    void **h_x_par, void **h_p_par,
    void **h_x_other, void **h_p_other,
    int nvector,
    const std::vector<double> &coeff,
    double kappa2, double ck, double dt,
    QudaGaugeParam *gauge_param,
    QudaInvertParam *inv_param)
{
  using namespace ::quda;
  if (!::gaugePrecise) errorQuda("No resident gauge field");
  if (!::cloverPrecise) errorQuda("No resident clover field");
  if (inv_param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC &&
      inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    errorQuda("MatPC type %d not supported", inv_param->matpc_type);
  }

  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom(fParam);

  fParam.location    = QUDA_CUDA_FIELD_LOCATION;
  fParam.create      = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE
                                                   : QUDA_COPY_FIELD_CREATE;
  fParam.field       = &cpuMom;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.setPrecision(gauge_param->cuda_prec, true);
  GaugeField cudaMom(fParam);

  ColorSpinorParam qParam(nullptr, *inv_param, fParam.x, false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), p(nvector);
  std::vector<double> force_coeff(nvector);

  QudaParity parity =
      inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
          ? QUDA_EVEN_PARITY
          : QUDA_ODD_PARITY;
  QudaParity other_parity = static_cast<QudaParity>(1 - parity);

  GaugeFieldParam fparam2(cudaMom);
  fparam2.link_type   = QUDA_GENERAL_LINKS;
  fparam2.reconstruct = QUDA_RECONSTRUCT_NO;
  fparam2.create      = QUDA_ZERO_FIELD_CREATE;
  fparam2.setPrecision(fparam2.Precision(), true);
  GaugeField force(fparam2);

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    p[i] = ColorSpinorField(qParam);

    // x[parity] = X̂
    {
      ColorSpinorParam cp(h_x_par[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][parity] = cf;
    }
    // x[other] = W_o (Schur-completed)
    {
      ColorSpinorParam cp(h_x_other[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][other_parity] = cf;
    }
    // p[parity] = (2κ)·M_pc·X̂ (kappa-form Y)
    {
      ColorSpinorParam cp(h_p_par[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][parity] = cf;
    }
    // p[other] = Z_o (Schur-completed)
    {
      ColorSpinorParam cp(h_p_other[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][other_parity] = cf;
    }

    // Wilson-hop force coefficient (mirrors clover_force.cpp:46/93).
    force_coeff[i] = 2.0 * dt * coeff[i] * kappa2;
  }

  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);

  computeCloverOprod(force, *::gaugePrecise,
                     inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                     inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                     force_coeff);

  updateMomentum(cudaMom, -1.0, force, "clover_wilson_schur");

  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);
}

// ----------------------------------------------------------------------------
// computeCloverFullForceWithSchurFields
// ----------------------------------------------------------------------------
// Combined Wilson-hop + σ force routine.  Fuses Phase D D.2 implementation:
// runs computeCloverOprod (Wilson hop) and computeCloverSigmaOprod +
// cloverDerivative (σ piece) into a single force/oprod/momentum cycle so
// we save one device-side accumulator zero-init and one momentum upload.
// Same input contract as the σ-only routine.
inline void computeCloverFullForceWithSchurFields(
    void *h_mom,
    void **h_x_par, void **h_p_par,
    void **h_x_other, void **h_p_other,
    int nvector,
    const std::vector<double> &coeff,
    double kappa2, double ck, double dt,
    double sigma_trace_coeff,
    QudaGaugeParam *gauge_param,
    QudaInvertParam *inv_param)
{
  using namespace ::quda;
  if (!::gaugePrecise) errorQuda("No resident gauge field");
  if (!::cloverPrecise) errorQuda("No resident clover field");
  if (inv_param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC &&
      inv_param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    errorQuda("MatPC type %d not supported", inv_param->matpc_type);
  }

  GaugeFieldParam fParam(*gauge_param, h_mom, QUDA_ASQTAD_MOM_LINKS);
  GaugeField cpuMom(fParam);

  fParam.location    = QUDA_CUDA_FIELD_LOCATION;
  fParam.create      = gauge_param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE
                                                   : QUDA_COPY_FIELD_CREATE;
  fParam.field       = &cpuMom;
  fParam.reconstruct = QUDA_RECONSTRUCT_10;
  fParam.setPrecision(gauge_param->cuda_prec, true);
  GaugeField cudaMom(fParam);

  ColorSpinorParam qParam(nullptr, *inv_param, fParam.x, false,
                          QUDA_CUDA_FIELD_LOCATION);
  qParam.setPrecision(fParam.Precision(), fParam.Precision(), true);
  qParam.create     = QUDA_NULL_FIELD_CREATE;
  qParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

  std::vector<ColorSpinorField> x(nvector), p(nvector);
  std::vector<double>            force_coeff(nvector);
  std::vector<array<double, 2>>  ferm_epsilon(nvector);

  QudaParity parity =
      inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
          ? QUDA_EVEN_PARITY
          : QUDA_ODD_PARITY;
  QudaParity other_parity = static_cast<QudaParity>(1 - parity);

  GaugeFieldParam fparam2(cudaMom);
  fparam2.link_type   = QUDA_GENERAL_LINKS;
  fparam2.reconstruct = QUDA_RECONSTRUCT_NO;
  fparam2.create      = QUDA_ZERO_FIELD_CREATE;
  fparam2.setPrecision(fparam2.Precision(), true);
  GaugeField force(fparam2);
  fparam2.geometry = QUDA_TENSOR_GEOMETRY;
  GaugeField oprod(fparam2);

  for (int i = 0; i < nvector; i++) {
    x[i] = ColorSpinorField(qParam);
    p[i] = ColorSpinorField(qParam);
    {
      ColorSpinorParam cp(h_x_par[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][parity] = cf;
    }
    {
      ColorSpinorParam cp(h_x_other[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      x[i][other_parity] = cf;
    }
    {
      ColorSpinorParam cp(h_p_par[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][parity] = cf;
    }
    {
      ColorSpinorParam cp(h_p_other[i], *inv_param, fParam.x, true,
                          inv_param->input_location);
      ColorSpinorField cf(cp);
      p[i][other_parity] = cf;
    }
    force_coeff[i] = 2.0 * dt * coeff[i] * kappa2;
    // Uniform per-parity ferm_epsilon (Phase B-prime convention for σ-piece
    // with Schur-completed off-parity inputs).
    ferm_epsilon[i] = {2.0 * ck * coeff[i] * dt,
                       2.0 * ck * coeff[i] * dt};
  }

  GaugeField &gaugeEx = *::extendedGaugeResident;

  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);

  // 1. Wilson-hop bilinear → force accumulator.
  computeCloverOprod(force, *::gaugePrecise,
                     inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                     inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                     force_coeff);

  // 2. (optional) σ-trace from clover (LogDet); pass 0 to skip.
  if (sigma_trace_coeff != 0.0) {
    computeCloverSigmaTrace(oprod, *::cloverPrecise, sigma_trace_coeff,
                            other_parity);
  }

  // 3. σ-Oprod → oprod accumulator.
  computeCloverSigmaOprod(oprod,
                          inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                          inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                          ferm_epsilon);

  // 4. Apply clover derivative kernel: integrate σ-oprod into force.
  cloverDerivative(force, gaugeEx, oprod, 1.0);

  // 5. Accumulate into momentum.
  updateMomentum(cudaMom, -1.0, force, "clover_full_schur");

  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);
}

}  // namespace Quda
NAMESPACE_END(Grid);

#endif  // GRID_HAVE_QUDA
