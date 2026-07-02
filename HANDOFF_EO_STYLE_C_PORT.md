# DTXQCD EO fast-GPU-path port — handoff to fresh fork

## ═══ PROGRESS 2026-07-02 (same day) ═══ — ALL GATES GREEN, IN PRODUCTION
- **Gate 1 PASSED** (machine precision 4e-16, piecewise: hop / MooeeInv /
  Mooee / full Mpc / Mpc†, 10 random vectors — `production/Test_dtxqcd_mpc_op_quda_eo`).
- **Gate 2 PASSED** (10-shift Style C EO vs Grid-DP: |dpsi| 1e-11..1e-15,
  true residuals < tol — `production/Test_dtxqcd_multishift_quda_eo`).
- **Gate 3 PASSED** (4⁴ λ=4 A/B: dH agrees to 10-15 digits, =0 vs =1).
- **Gate 4 PASSED** (16³×48 4-GPU multishift 2.0× vs pure-DP, ~1.2× vs the
  MP-CG incumbent per solve).
- **Gate 5 PASSED** (λ=6 MDS=10 stream forked at ckpt 10, trajs 11-20 full
  Metropolis: dH envelope statistically identical to the MP-CG incumbent's
  own trajs 11-18 — both ~+2 with the same sticky-patch config, Style C even
  escaped it (accepts at 17-18); plaq continuity exact; rc=134 = benign QUDA
  teardown.  **Production per-traj: 573 s (Style C, job 1296217) vs 1040 s
  (MP-CG incumbent) at λ=6 16³×48 1.1.1.4 — 1.8×, clears the 1.5× bar.**)
- Production cycled onto the fast path 2026-07-02: jobs 1296215/16/17
  (λ=4/5/6 `_eo_fromchroma_cw_mds10` continuations, `DTXQCD_MULTISHIFT_QUDA=1`
  via the patched launcher).  λ=5/6 healthy; λ=4 unstable at MDS=10 (LogDet
  Fdt max ~59, all trajs rejected — pre-cycle incumbent identically sick ⇒
  small-λ EO-Schur pathology, NOT a Style C regression; non-EO λ=3-6
  comparison streams launched, jobs 1296276-79).
- **EO measurements**: `dtxqcd_meas_helper.h` gained `DTXQCD_MEAS_EO=1`
  (Schur solve on odd CB via `DTXQCDMpcOpQUDA_EO` + Style C single-zero-shift
  CG, even block back-substituted through the site caches).  λ=10 thermal-aux
  A/B: EO ~8.5 s/solve (380-400 iters) vs full-volume ~82 s/solve — ~10×.
  h5 cross-validation + λ=6/4/3 A/Bs in flight.
- Implementation deltas vs the plan below:
  * `DTXQCDMpcOpQUDA_EO` SUBCLASSES `DTXQCDMpcOpQUDA` (non-virtual shadowing;
    solvers templated on the concrete type → static binding, full reuse of
    gauge-load/persistent-Dirac).  Base `private:` → `protected:`.
  * Site matrices via the fermion class's EXISTING batched device caches
    (`InvDevCache(Even)`/`FwdDevCache(Odd)` accessors added) + `csf.copy`
    layout conversion + `cudaMemcpy2D` interleave + `gemmBatched` — **zero
    new device kernels**.  Hop = persistent-Dirac parity `Dslash` (+ a second
    persistent dagger-Dirac) with the existing Pre/PostMatLower natives
    (elementwise ⇒ parity-safe); hop scale = −1/2 exactly.
  * `DoubledStateCSF::copy_{from,to}_grid` grew a checkerboarded branch
    (RB order == cb_site ⇒ unvectorize+memcpy, pc=true params).
  * StyleC solver Grid scratches need CB stamping on RB grids (done).
  * Action wiring: env-dispatch INSIDE `DTXQCDWilsonCloverRationalEOAction::
    RunMultiShiftCG` (opt-in `DTXQCD_MULTISHIFT_QUDA=1`) — no driver changes,
    no new action subclass needed.  `ApplyRational` (S()) dispatch: TODO —
    currently only refresh/deriv route through RunMultiShiftCG; S() still
    Grid-DP (correct, just not accelerated) — wire after Gate 4 if S() is a
    measurable fraction of traj time.

**Date:** 2026-07-02
**Status (original):** scoped, ready to implement.  No code changes yet.
**Task:** #239
**Branch:** `dtxqcd-v2` (Grid-DTXQCD repo)
**Related memory:** `reference_dtxqcd_style_c_default_on`, `reference_dtxqcd_w3_wilson_off_scale`, `feedback_mp_cg_refinement_family_ruled_out`, `feedback_microbench_evaporates_under_concurrency`

## Why this port now

**Physics:** λ=4-6 is the emerging sweet spot for DTXQCD paper 1.
- Chroma-consistent plaq at all λ ≥ 1 (no sign-catastrophe basin like λ ≤ 0.5).
- Best per-config iso-π⁰ aux SNR peaks near λ=1-2 and stays usable through λ~6 (memory / SNR-vs-λ table earlier this session).
- λ ≥ 4 the aux VEV is small enough that the operator's σ-cliff is well-behaved, so Schur-EO factorization works — none of the low-λ Mpc singularity problems that forced the non-EO detour.
- Nucleon mass gap and pion mass make λ ≈ M_N − (3/2)m_π ≈ 6.6 the operational target (`project_txqcd_lambda_sweet_spot`).

**Perf:** the EO rational action is the ~stale codepath and misses every fast-GPU win of the last month.  Non-EO gets ~2× wallclock over EO simply from Style C being default ON there (`reference_dtxqcd_style_c_default_on`).  If λ=4-6 production runs on EO (which it should — Schur factorization halves the volume the multishift lives on, and MG becomes possible later), we need Style C in EO too.

## State of the code (as of this session)

`DTXQCDWilsonCloverRationalEOAction.h` — the Grid-only EO rational action, `#include`s only `DTXQCDMultiShiftCG.h` (Grid pure DP) + `DTXQCDMultiShiftCGMixedPrec.h` (Grid DP outer, Grid SP inner via twin op).  **Zero references to `DTXQCD_MULTISHIFT_QUDA`, Style C, StageB, `DTXQCDMpcOpQUDA`.**

`DTXQCDWilsonCloverRationalEOActionQudaPrimitive.h` — subclass of the EO action that overrides the *gauge* force path to route through `Quda::computeCloverWilsonForceWithSchurFields` (Phase H/W.3 wilson-hop force).  Base solver is still the Grid CG inherited from EOAction.  Gated by `DTXQCD_QUDA_HYBRID=1`.

`DTXQCDWilsonCloverRationalFullAction.h` — non-EO, has the full fast-path stack:
  - `#include DTXQCDMpcOpQUDA.h` (device-resident QUDA Mop wrapper)
  - `#include DTXQCDMultiShiftCGQUDA.h` (Stage A, host-mode)
  - `#include DTXQCDMultiShiftCGQUDA_StageB.h` (Stage B, device-resident)
  - `#include DTXQCDMultiShiftCGQUDA_StyleC.h` (native-CSF FloatN + blas_quda fusion)
  - `#include DTXQCDMultiShiftCGQUDA_StyleC_MpMultishift.h` (SP multishift → DP polish)
  - `#include DTXQCDMultiShiftCGQUDA_StyleC_MpMultishiftRefine.h` (Session H, MP refinement)
  - Env dispatch: `DTXQCD_MULTISHIFT_QUDA=1` (default), `DTXQCD_MULTISHIFT_QUDA_STAGE=B` (default), `DTXQCD_STAGEB_STYLE=C` (default), optional `DTXQCD_MPMULTISHIFT_REFINE`.

Production driver (`gen_dtxqcd_cfgs.cc` L438-488) dispatches:
1. `USE_FULL_PF=1` + `USE_FULL_PF_QUDA_WILSON=1` (default) → non-EO + QUDA Wilson hop (`FullActionQudaPrimitive`, gets Style C).
2. `DTXQCD_QUDA_HYBRID=1` → EO + QUDA Wilson hop (`EOActionQudaPrimitive`, Grid CG still).
3. Otherwise → plain `EOAction` (Grid CG, Grid Wilson hop).

**All current λ=0.1/0.25/0.5 low-λ chains run path 1** (non-EO, forced by USE_FULL_PF=1 because Schur-EO breaks in the σ-cliff regime).  **λ=4-6 production would happily run on the EO variant** if we brought its perf up.

## What to port

**Concretely, three deltas:**

### 1. Write `DTXQCDMpcOpQUDA_EO.h` — the EO analog of `DTXQCDMpcOpQUDA`

Structure to mirror: the non-EO one at `Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA.h`.

Difference: the EO wrapper's `M(v)` is `Mpc†·Mpc·v` on odd-CB fields (half-volume) instead of `M†·M·v` on full-vol fields.  Under the hood the device-side CSF geometry is halved (odd-CB only), the aux kernel operates on the odd sublattice, and the boundary pack/unpack talk to `LatticeFermion` on `RBGrid` odd checkerboard.

Key APIs to expose (mirror the non-EO wrapper):
- `M_device_csf(...)` / `Mdag_device_csf(...)` on device-resident CSF (odd-CB half-vol)
- `M(...)` (Grid interface) that wraps device_csf with host↔device pack
- `MakeNativeCsfParam()` / `MakeNativeCsfParamSp()` (QUDA CSFParam for odd-CB)
- `GaugeParam()`, `InvertParam()` accessors (used by the StyleC solver's `copy_from_grid` helpers)

The QUDA CSF geometry is: `X[0..3] = odd-CB volume dims`.  Halve the last (non-full) dim per QUDA convention.  See existing non-EO Mop for the pack ordering and `pc_type = QUDA_5D_PC` or `QUDA_4D_PC` (should be 4D for us).

### 2. Instantiate `DTXQCDMultiShiftCGQUDA_StyleC` on the EO Mop type

The StyleC solver at `DTXQCDMultiShiftCGQUDA_StyleC.h:69` is already templated:
```cpp
template <class DTXQCDMopD>
inline void DTXQCDMultiShiftCGQUDA_StyleC(
    DTXQCDMopD &Mop_d,                 // Grid-DP HermOp for reliable update
    DTXQCDMpcOpQUDA &Mop_quda,         // QUDA-DP HermOp for inner mat-vec
    ...)
```

`Mop_d` template parameter accepts any Grid Hermop that exposes `M(in, out)`, so `DTXQCDMpcOpEO<...>` (the existing Grid-DP Schur Mpc) plugs in directly.

**But** `Mop_quda` is hardcoded to `DTXQCDMpcOpQUDA` (full-vol) in the solver's function signature.  Either:
- (a) template the second parameter too, then instantiate on `DTXQCDMpcOpQUDA_EO`, OR
- (b) make `DTXQCDMpcOpQUDA_EO` a subclass of `DTXQCDMpcOpQUDA` that overrides all the CSF-dim / geometry methods, then no template change needed.

**(a) is cleaner** — the two Mop wrappers don't naturally share a v-table, and templating avoids a virtual-dispatch cost on the inner loop.

Same call at StyleC_MpMultishift.h and StyleC_MpMultishiftRefine.h — template all three consistently.

### 3. Add a `DTXQCDWilsonCloverRationalEOActionStyleC` subclass (or extend `EOActionQudaPrimitive`)

Follow the non-EO pattern in `RationalFullAction.h:323-425`: subclass takes the same ctor args, in `refresh`/`S`/`deriv` dispatches to the QUDA multishift path based on env `DTXQCD_MULTISHIFT_QUDA`, `_STAGE`, `_STAGEB_STYLE` (same env names as non-EO — user changes zero flags per chain).

Wire it into `gen_dtxqcd_cfgs.cc` selection logic:
- If `USE_FULL_PF=0` AND `DTXQCD_QUDA_HYBRID=1` AND `DTXQCD_MULTISHIFT_QUDA=1` (default) → new EO Style C variant.
- If `DTXQCD_QUDA_HYBRID=1` but `DTXQCD_MULTISHIFT_QUDA=0` → existing `EOActionQudaPrimitive` (Wilson hop QUDA, Grid CG).
- Etc.

Cleanest: same subclass hierarchy as non-EO — `EOActionQudaPrimitive` (Wilson hop QUDA) is the base, `EOActionQudaPrimitiveStyleC` extends it with the CG override.  Non-EO does it this way in `FullActionQudaPrimitive.h` — study that layout before diverging.

## What NOT to port

- **MpMultishiftRefine / Session H** — ruled out as net-loss vs pure-DP Style C at production λ (`feedback_mp_cg_refinement_family_ruled_out`).  Include the header for completeness but leave `DTXQCD_MPMULTISHIFT_REFINE` defaulting to 0 (opposite of Style C, which is default ON).  Match non-EO defaults.
- **The old Stage A path** — subsumed by Stage B / Style C.  Leave the code accessible via `DTXQCD_MULTISHIFT_QUDA_STAGE=A` for forensics but never as default.
- **MG (multigrid) preconditioning** — DTXQCD's doubled operator + aux blocks need bespoke MG that doesn't exist yet.  Explicit non-goal here.

## Validation gates (in order)

Each must pass before moving to the next.  All are the same gates the non-EO M-wrap phases used — no new physics.

### Gate 1: 4⁴ bit-exact operator equivalence
Test `Test_dtxqcd_mpc_op_quda_eo.cc` (new; mirror `Test_dtxqcd_mpc_op_quda.cc`): assert `M_QUDA_EO(v) ≡ M_Grid_EO(v)` to ≤ 1e-12 over 10 random odd-CB fermions at csw=1.24930970916466, mass=-0.245, λ=4.  If this fails, the CSF pack/unpack geometry for odd-CB is wrong; check `X[]` dim halving and `pc_type`.

### Gate 2: 4⁴ multishift bit-exact
Test `Test_dtxqcd_multishift_quda_eo.cc` (new; mirror the existing non-EO version): 10-shift solve on random RHS, both Grid EO CG and QUDA StyleC EO backends from the same seed.  Assert per-shift residuals within tol and per-shift `‖ψ_grid − ψ_quda‖ / ‖ψ_grid‖ < 1e-10`.  If this fails and Gate 1 passed, the failure is in the blas_quda inner loop wiring — walk the RationalFullAction blas dispatch and confirm same primitives used.

### Gate 3: 4⁴ FD force bit-exact (deriv match Grid ref)
1-traj `gen_dtxqcd_cfgs` at 4⁴ λ=4 with `DTXQCD_MULTISHIFT_QUDA=0` (Grid ref) vs `=1` (Style C).  Compare `dSdU` slot-wise rel diff.  Gate: ≤ 1e-7 (Remez floor).  This exercises the whole action including refresh() Φ-draw.

### Gate 4: 16³×48 4-GPU wallclock at λ=4
Full 3-traj `gen_dtxqcd_cfgs` at 16³×48 with existing EO settings vs new Style C EO settings.  Baseline: use the current EO wallclock (need to measure — run one traj first).  Success = ≥ 1.5× substep-1 [0][0] wallclock speedup.  Per `feedback_microbench_evaporates_under_concurrency`, always validate at 4-GPU production scale — 1-GPU numbers can lie.

### Gate 5: Full-length 20-traj dH stability at λ=6
Same run, extended.  Watch dH histogram is chroma-comparable (mean O(1), no long tails).  Style C reproduces production dH to ~7 decimals per non-EO experience (`reference_dtxqcd_style_c_default_on`); EO should be the same.

## Concrete file list

**New:**
- `Grid/qcd/action/dtxqcd/DTXQCDMpcOpQUDA_EO.h`
- `Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalEOActionQudaPrimitiveStyleC.h` (or extend existing `EOActionQudaPrimitive.h` in place — depends on how much shared)
- `tests/dtxqcd/Test_dtxqcd_mpc_op_quda_eo.cc`
- `tests/dtxqcd/Test_dtxqcd_multishift_quda_eo.cc`

**Modified:**
- `Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCGQUDA_StyleC.h`, `_MpMultishift.h`, `_MpMultishiftRefine.h` — template the QUDA-side Mop parameter (currently hardcoded `DTXQCDMpcOpQUDA &`).
- `production/gen_dtxqcd_cfgs.cc` — add selection branch for EO+Style C.
- `tests/dtxqcd/Make.inc` — register two new tests.
- `production/slurm_dtxqcd_nonEO_lam4_*.sh` (or drop `_nonEO_` — the *point* is EO): write a `slurm_dtxqcd_EO_lam4_*.sh` and `_lam6_*.sh` for the 4-GPU perf validation and follow-up production.

**Reused as-is:**
- `Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h` — already GPU-optimized.
- `Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h` — precomputed inverses on GPU.
- `Grid/qcd/action/dtxqcd/dtxqcd_quda_aux_kernel*.h` — the CUDA aux kernels ported for non-EO in M-wrap.3 work on ANY CB (they take the aux fields + spinors as flat arrays); no rewrite needed.
- `Grid/qcd/action/dtxqcd/dtxqcd_quda_gamma2_conj.h` — same, CB-agnostic.

## Rollback

- All new env gates default to the existing behavior (EO variant enters Style C only when `DTXQCD_MULTISHIFT_QUDA=1` is *explicitly* set alongside `DTXQCD_QUDA_HYBRID=1`).  Existing production scripts unaffected.
- New headers and tests are standalone; `git revert` restores the pre-port state.

## Open risks

- **Aux kernel odd-CB indexing.**  Non-EO aux kernels iterate over the full lattice; the EO port needs to iterate on odd-CB and read aux fields that are still full-vol.  Check the aux-slot access pattern — a bare `ordered_dim`-based iterator won't be right; needs the odd-checkerboard rank2 or the QUDA `parity` param.  See `dtxqcd_quda_aux_kernel_native_v2_impl.cc` for the pattern that got merged; the new EO instantiation may only need a `parity=QUDA_ODD_PARITY` flag.
- **`γ_2·conj` on odd-CB.**  The 4⁴ non-EO γ_2·conj kernel operates on full-vol spinors; halving the volume changes launch geometry but not per-site work.  Should be a one-line kernel launch change.
- **StageB CG-state alloc.**  Half-vol reduces the buffer size by 2×; the CG params (poles, tolerances) are unchanged.  Straightforward but be sure to update `X_full_dims[]` in the StyleC solver from `Mop_quda.GaugeParam().X` — the EO Mop's `GaugeParam.X` should already be the halved geometry.

## Success criterion

At Gate 5, the λ=6 EO chain reproduces dH statistics of the current EO chain to floating-point precision, plaq drift matches, AND traj wallclock is ≥ 1.5× faster than the pre-port EO baseline at 16³×48 4-GPU.  Ship the flag defaults ON, add a memory reference for the new default, and re-enable λ=4-6 production forks with the new fast-EO defaults.

## Followups (out of scope for this port, worth flagging)

- **MG for the doubled operator.**  Real path to 32³×64 b6.5 EO production — requires bespoke MG null-space generation on the aux+σ blocks.  Blocker for paper 2.  See `reference_txqcd_mg_status` for the current landscape (nothing exists yet).
- **Fused hermop wiring in the EO CG.**  Non-EO Style C already uses the fused M†M path; EO version should follow.  This is likely a 10-line change once Style C-EO is in place.
- **`M-wrap.5b.2` for EO?**  Task #218 is the non-EO device-resident inner loop with blas_quda fusion, still `in_progress`.  Whatever comes out of that lands in the EO port for free if the port is built on the same StyleC scaffold.

Good luck.
