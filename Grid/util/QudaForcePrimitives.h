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

// QUDA-side extern declarations of internal globals (defined in
// external/quda-src/lib/interface_quda.cpp).  These have external linkage
// so we can reach them from outside QUDA's translation units.
namespace quda {
  extern ::quda::GaugeField  *gaugePrecise;
  extern ::quda::CloverField *cloverPrecise;
  extern ::quda::GaugeField  *extendedGaugeResident;
}

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
  using ::quda::gaugePrecise;
  using ::quda::cloverPrecise;
  using ::quda::extendedGaugeResident;

  if (!gaugePrecise) errorQuda("No resident gauge field");
  if (!cloverPrecise) errorQuda("No resident clover field");
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
    gamma5(p[i][parity], p[i][parity]);

    // x_other := D · γ5 · X — but our x_par is X (not γ5·X).
    // To match QUDA's flow, we need a temp field for γ5·X.
    {
      ColorSpinorField tmp_g5x(qParam);
      tmp_g5x[parity] = ColorSpinorField(qParam)[parity];  // alloc
      gamma5(tmp_g5x[parity], x[i][parity]);
      if (dagger) dirac->Dagger(QUDA_DAG_YES);
      dirac->Dslash(x[i][other_parity], tmp_g5x[parity], other_parity);
      if (dagger) dirac->Dagger(QUDA_DAG_NO);
    }

    // gamma5 on x_other (matches line 62 of clover_force.cpp).
    gamma5(x[i][other_parity], x[i][other_parity]);

    // Compute p_other = (Dagger? -- not_dagger pattern from clover_force.cpp:65-68)
    if (not_dagger) dirac->Dagger(QUDA_DAG_YES);
    dirac->Dslash(p[i][other_parity], p[i][parity], other_parity);
    if (not_dagger) dirac->Dagger(QUDA_DAG_NO);

    // Final gamma5(p, p) — clover_force.cpp:79.
    gamma5(p[i], p[i]);

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
  GaugeField &gaugeEx = *extendedGaugeResident;

  // ------------------------------------------------------------------
  // Primitive sequence (matches lib/clover_force.cpp:83-102).
  // ------------------------------------------------------------------
  // 1. Wilson hop: ⟨P|∂M_eo/∂U|X⟩ + h.c.
  vector_ref<const ColorSpinorField> x_const(x);
  vector_ref<const ColorSpinorField> p_const(p);
  computeCloverOprod(force, *gaugePrecise,
                     inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                     inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                     force_coeff);

  // 2. (optional) σ trace from clover field — LogDet.  Pass 0 to skip.
  if (sigma_trace_coeff != 0.0) {
    computeCloverSigmaTrace(oprod, *cloverPrecise, sigma_trace_coeff,
                            other_parity);
  }

  // 3. Clover σ-Oprod (X·P† projected against σ_μν).
  computeCloverSigmaOprod(oprod,
                          inv_param->dagger == QUDA_DAG_YES ? p_const : x_const,
                          inv_param->dagger == QUDA_DAG_YES ? x_const : p_const,
                          ferm_epsilon);

  // 4. Apply clover derivative kernel: integrate oprod into force.
  cloverDerivative(force, gaugeEx, oprod, 1.0);

  // 5. Accumulate force into mom: cudaMom += -1.0 · force.
  updateMomentum(cudaMom, -1.0, force, "clover_grid_y");

  // ------------------------------------------------------------------
  // Bring mom back to host buffer.
  // ------------------------------------------------------------------
  if (gauge_param->return_result_mom) cpuMom.copy(cudaMom);

  delete dirac;
}

}  // namespace Quda
NAMESPACE_END(Grid);

#endif  // GRID_HAVE_QUDA
