# DTXQCD cluster handoff — GPU port + 16³×48 production

**Date:** 2026-06-19
**Branch:** `dtxqcd-v2` (pushed to `origin/dtxqcd-v2`).  Cut from
`develop`@`794f69e6` (TXQCD: laptop-CPU build path + t-condensate
overnight harness).  DO NOT rebase onto `develop` without re-running
the full gate sweep — the v1 / sigmaHerm transition crosses several
commits.
**Starting commit:** `c13fec9a` (DTXQCD: lock in sigmaHerm convention;
refloor saddle VEV abs tol with 5·SE)

You are picking up a working laptop-CPU DTXQCD baseline.  Your job is
to (1) port the DTXQCD action chain to the GPU paths TXQCD already
uses, and (2) run 16³×48 dtxqcd production at λ ∈ {10, 5, 2, 1, 0.5,
0.2, 0.1}, m=0.3, csw=0 (pure Wilson — same scout the laptop is
currently doing at 4³×8 to fix per-λ integrator).

---

## 1. The locked-in convention (sigmaHerm)

DTXQCD now runs ONE convention.  Do not introduce alternatives.

- σ, π are **Hermitian** (36 real DOFs/site each).
- d, n are **truly complex-symmetric** under the joint (color+flavor)
  transpose: M(k1,k2) = M(k2,k1), no conjugate (42 real DOFs/site each).
- M48 has **M_LL = conj(M_UR)** at both site-LU assembly
  (`DtxqcdAssembleDoubled48`) and on-the-fly Apply
  (`DtxqcdApplyDnCross(..., apply_conj=true)`).
- M_lower's diagonal block reads **σ^T, π^T** (joint transpose) via
  `DtxqcdBuildDiagBlock24(..., transpose_aux=true)` /
  `DtxqcdApplyDeltaDiagLower(..., transpose_aux=true)`.

Three formal gates defended (all green):
1. γ5-Hermiticity of full M48 (`Test_dtxqcd_gamma5_herm_full`)
2. Pfaffian antisymmetry (K·M48)^T = -(K·M48) (`Test_dtxqcd_pfaffian_antisymmetry`)
3. Full complex DOFs in every aux field (lossy projection caught
   silently before and gave ~0.5% Fierz residual at light mass; freefield
   tests now cover this — `Test_dtxqcd_freefield_qbarq_{heavy,light}{,_eo}`).

`DTXQCD_DN_COMPLEX_SYMMETRIC`, `DTXQCD_SIGMA_PI_HERMITIAN_ONLY`, and
`DTXQCD_DN_REAL_SYMMETRIC` env knobs are **IGNORED**.  Setting them
prints a `GridLogWarning` once at first
`DtxqcdComplexSymmetricCFGaussian` call.  The helper functions are
`constexpr`-true / -false, so dead branches inside
`if (DtxqcdDnComplexSymmetric()) {...}` survive in source as an audit
trail of what the legacy alternative did, but compile out under
constant folding.  A future cleanup pass can rip them.

---

## 2. Checkpoint format — DTX3 (not DTX2!)

Magic `0x44545833` = `'DTX3'`.  Per-site payload = 158 doubles =
1264 bytes:
- σ Hermitian pack: 36 reals
- π Hermitian pack: 36 reals
- d complex-symm pack: 42 reals (6 diag complex + 15 off-diag complex)
- n complex-symm pack: 42 reals
- s, p singlet scalars: 1 real each

DTX2 configs are unreadable (different size, will FAIL at header
check).  The complex-symm pack stores the imag-diagonal DOFs that DTX2
silently dropped — that was the source of the Fierz residual in the
6/18 debug pass.  Code: `Grid/qcd/action/dtxqcd/DTXQCDCheckpointer.h`.

---

## 3. Action chain — what's on CPU, what's on GPU

### Currently CPU-only (your GPU port targets)

| File | Hot path | TXQCD analog already on GPU? |
|---|---|---|
| `DTXQCDWilsonCloverFermionEO.h` (`ImportFields` → `PrecomputeInverses`) | Per-site 24×24 LU (CPU `partialPivLu().inverse()`) | `TXQCDWilsonCloverFermionEO`: ✅ (`TXQCD_PRECOMPUTE_GPU=1`, cuBLAS getrfBatched + getriBatched) |
| `DTXQCDLogDetCloverEOAction.h` (`S_gpu`, `deriv_gpu`) | NOT IMPLEMENTED — currently always falls to S_cpu/deriv_cpu | `TXQCDLogDetCloverEOAction.h`: ✅ (Phases J.1/J.2/J.3/J.4 done — see `TXQCDLogDetGpuKernel.h`) |
| `DTXQCDSiteForceKernel.h` | Per-site Eigen inverse + 24×24 traces (CPU thread_for) | TXQCD force kernel has GPU fusion |
| `DTXQCDWilsonCloverRationalEOAction.h` | Multishift CG already on GPU via Grid CG; force assembly CPU | TXQCD rational EO: same pattern |

### Already on GPU (don't re-port)

- All Grid stock: gauge action, hopping (Dhop) via WilsonFermion, CG,
  multishift CG, smearing chain.
- TXQCD's LogDet/EO GPU pipeline (the model to copy for DTXQCD): see
  `TXQCDLogDetGpuKernel.h` + the GPU paths inside
  `TXQCDLogDetCloverEOAction.h`.

### Recently-fixed bug (DON'T re-break it)

`TXQCDLogDetCloverEOAction::S_gpu` had its `cublasZgetrfBatched`
LU step inside `#ifdef GRID_CUDA` BUT the subsequent
`accelerator_for` log-diagonal trace read the same buffer
unconditionally.  On non-CUDA builds this returned a numerically-
plausible but functionally wrong S, silently breaking force FD checks.
Fixed at `8c0f721f`: `S_gpu` dispatch is now gated on `GRID_CUDA` so
non-CUDA builds always fall to `S_cpu`.  When porting DTXQCD's
`S_gpu`/`deriv_gpu`, mirror this pattern: gate the **dispatch** on
`GRID_CUDA`, not just the cuBLAS call.

---

## 4. Production driver — `production/gen_dtxqcd_cfgs.cc`

Single binary, all knobs env-overridable.  Geometry/physics from
`params.h`.

```
LATT=L.L.L.T              lattice (default 16.16.16.48 from params.h)
MASS_LIGHT_DTXQCD=value   bare mass (default mass_light = -0.245; for
                          your scout use 0.3)
LAMBDA_DTXQCD=value       aux coupling
CSW=value                 clover; 0 = pure Wilson (your scout uses 0)
BETA=value                gauge coupling (production 6.1; for 4³×8
                          weak-coupling scout 5.6–6.0 fine)
INTEGRATOR=ForceGradient  (default; OmelyanForceGradient also fine)
MDSTEPS, TRAJL            integrator
NO_METROP=K               skip Metropolis for first K trajs only
                          (NOT all; see reference_no_metrop_semantics.md)
TRAJ=K                    override total traj count (smoke runs)
RHMC_LO, RHMC_HI, RHMC_DEG    rational solver bracket
CG_TOL, CG_MAX                CG params
GAUGE_MULT, AUX_MULT          outer-level step multipliers
TXQCD_LOGDET_S_GPU=1      enable GPU S path (once you port it; default
                          ON in cuBLAS builds, automatically OFF on
                          non-CUDA per fix above)
TXQCD_LOGDET_GPU=1        enable GPU deriv path (same)
```

`gen_dtxqcd_cfgs` is **EO-only** — it uses
`DTXQCDWilsonCloverRationalEOAction` + `DTXQCDLogDetCloverEOAction`.
The `Test_dtxqcd_2pt_gencfgs` test driver in `tests/dtxqcd/` supports
`USE_FULL_PF=1` (non-EO full pseudofermion) for ablation.  Memory
records `project_dtxqcd_nonEO_action.md` and
`project_dtxqcd_eo_saddle_landscape.md` describe when EO is cliff-y
(λ ∈ [1, 3] without saddle init) vs full PF.  The laptop scout in
progress is testing whether the sigmaHerm convention closes that cliff.

`AUX_INIT_AUTO=1` bisects to the self-consistent saddle; needed at
small λ.  See `reference_dtxqcd_gencfgs_env_knobs.md` and
`project_dtxqcd_aux_init_self_consistency.md`.

---

## 5. HMC convention reminder (Grid eps ≠ chroma dt)

`reference_hmc_convention_grid_vs_chroma.md`: Grid's
`eps = trajL / MDsteps` is ~1/4 of chroma's dt for the same physical
step size.  `MDSTEPS=20 TRAJL=√2` is comparable to chroma `n_steps=7
tau0=√2` at 4× finer eps — production-tuned for 50–70% acceptance.

---

## 6. Test coverage status (all green at `c13fec9a`)

**DTXQCD gate tests (20/20 PASS):**
`Test_dtxqcd_aux_gaussian aux_io clover_herm delta_diag_lower delta_herm
gamma5_herm_full logdet_aux_force meooe_free_field mooee_clover_consistency
mooee_consistency mooee_herm mooeeinv_n pfaff_logdet_blockdiag
pfaffian_antisymmetry rational_aux_force rational_full_force site_lu
trminv_zeroaux wilson_clover_fermion_eo zero_aux_doubled_qcd`

**TXQCD gate tests (14/14 PASS):**
`Test_aux_gaussian aux_kinetic qcd_eo_clover qcd_logdet_clover_force
txqcd_clover_vs_grid txqcd_delta_color txqcd_clover_mooee
txqcd_hasenbusch_force txqcd_logdet_clover_eo_force txqcd_logdet_eo_force
txqcd_pf_force txqcd_rational_eo_force txqcd_rational_force txqcd_vev`

**Freefield Fierz tests (4 + 4 = 8 PASS at default settings):**
`Test_{dtxqcd,txqcd}_freefield_qbarq_{heavy,light}{,_eo}` — ratio
Σ_DTX/Σ_W (and Σ_TX/Σ_W) match unity to ~1e-3 with both abs and 3σ
gates green.  Saddle VEV gates use
`max(5·SE, 5·pass_tol·|pred|)` for abs + 3σ stat.

Re-running the full sweep after any structural change:
```
cd build
make -C tests/dtxqcd tests -j
make -C tests/txqcd  tests -j
# then loop the test list above
```

---

## 7. Common pitfalls (laptop tripwires)

1. **BC defaults are APBC time.**  Every TXQCD/DTXQCD Wilson-family
   operator defaults to antiperiodic time (chroma convention).  Stock
   Grid `WilsonFermion`/`WilsonCloverFermion` default to PBC.  When
   comparing TXQCD↔stock-Grid, pass matching `WilsonImplR::ImplParams`
   with `boundary_phases[Nd-1] = -1.0`.  Already-fixed sites:
   `Test_dtxqcd_zero_aux_doubled_qcd`, `Test_dtxqcd_meooe_free_field`,
   `Test_txqcd_clover_vs_grid`, `Test_qcd_logdet_clover_force`.  See
   `reference_bc_fix_audit.md`.

2. **Clover field strength source.**  `TXQCDLogDetCloverEOAction::GetEvenClover`
   computes `F_μν` from RAW U via `WilsonLoops::FieldStrength` — BC
   phases are NOT applied.  Stock `WilsonCloverFermion::ImportGauge`
   applies BC phases first.  This means TXQCD's clover and stock Grid's
   clover disagree by O(BC effect) at the time boundary.  This is
   architectural, not a bug.

3. **NO_METROP semantics.**  `NO_METROP=K` skips Metropolis on the
   first K trajs only.  It's NOT "skip metropolis for the run".

4. **Don't kill long-running tests on conversational asides.**  The
   laptop's "looks like a great stress test" comment does NOT mean
   "stop the run".  Ask before killing trajs that have been going for
   hours.  See `feedback_dont_kill_long_running_tests.md`.

5. **Always `make` before running.**  Stale `.o` files do not get
   automatic rebuild after header changes to `DTXQCDCompositeImpl.h`
   etc.  Production scripts should `make` first.

---

## 8. Likely first moves for the cluster instance

1. **Rebuild against the cluster's CUDA + MPI toolchain.**
   Known-good systems config: `systems/sdcc-genoa/`,
   `systems/SDCC-ICE/`, etc.  TXQCD-specific
   `--disable-fermion-reps --disable-gparity` are already the genoa
   default.  GPU build: `--enable-accelerator=cuda --enable-simd=GPU`.

2. **Run the gate sweep on the cluster** to confirm bit-identity with
   the laptop CPU baseline (modulo Grid CG roundoff).  Any test
   FAILING that passes on laptop is a porting bug — DO NOT proceed to
   production until clean.

3. **Port `DTXQCDLogDetCloverEOAction`'s GPU path** by mirroring
   `TXQCDLogDetCloverEOAction.h:S_gpu` / `deriv_gpu`:
   - Persistent device buffers for M_fwd_dev_, M_inv_dev_, pivots,
     getrf/getri pointer arrays (mirror TXQCD's pattern).
   - One LU per even site (24×24 cuBLAS getrfBatched).
   - Reuse the 5-trace fusion kernel structure from
     `TXQCDLogDetGpuKernel.h` but adapted to the doubled 48×48 site
     matrix (Note: DTXQCD's per-site matrix is 48×48 for the doubled
     fermion, not 24×24 — this is the main structural difference).
   - **CRITICAL: gate dispatch on `GRID_CUDA`, not just the cuBLAS
     call.**  See section 3 above.
   - Verify against `Test_dtxqcd_logdet_aux_force` (FD vs analytic).

4. **`DTXQCDWilsonCloverFermionEO::PrecomputeInverses`** is the next
   GPU target.  TXQCD already has this via `TXQCD_PRECOMPUTE_GPU=1`
   path.  Same getrfBatched/getriBatched but on the 48×48 site matrix.

5. **Don't touch the `DtxqcdAssembleDoubled48` algebra.**  M_LL =
   conj(M_UR) is the convention.  M_lower uses σ^T.  Forces use the
   sigpi_T / dn_cs formulas in `DTXQCDSiteForceKernel.h`.  These are
   load-bearing for γ5-Hermiticity and Pfaffian antisymmetry — if you
   touch them, re-run gates 1+2 from section 1.

6. **Production scaling at 16³×48** uses the same `gen_dtxqcd_cfgs`
   driver.  Once the GPU path lands, expect ~3–5× per-trajectory
   speedup based on TXQCD's experience (`per-deriv=5.242 ms` is the
   CPU baseline for one of the timer lines).

---

## 9. Open items / TODO

- `production/gen_dtxqcd_cfgs.cc` has no observers besides the
  checkpointer.  Add aux-VEV observer, eigenvalue diagnostics, stout
  chain (mirror `gen_txqcd_cfgs_2plus1.cc`).
- `Test_dtxqcd_aux_io` currently uses `HotConfiguration` which
  internally calls `DtxqcdComplexSymmetricCFGaussian` for d/n — that
  call now emits the legacy-knob warning if the user sets env knobs.
  Cluster instance should make sure those knobs are NOT set in
  production sbatch scripts.
- The laptop is actively scouting per-λ MDsteps at 4³×8.  Check the
  most-recent commit on `dtxqcd-v2` for a follow-up note on per-λ
  settings before launching 16³×48.
- DTXQCD's measurement programs (`meas_conn_dtxqcd`,
  `meas_disco_dtxqcd`, `meas_aux_dtxqcd`, `meas_baryon_dtxqcd`)
  remain CPU.  They run one-shot per config, not per HMC trajectory,
  so GPU speed-up is lower priority than the HMC chain.

---

## 10. Memory pointers (laptop has these saved)

Look in `~/.claude/projects/-Users-mwagman-Lattice-QCD-TXQCD-Grid-TXQCD/memory/`
(laptop only) for narrative context.  Relevant for cluster work:

- `project_dtxqcd_correct_convention_gates.md` — the sigmaHerm
  convention, code touches, gates.
- `project_dtxqcd_complex_symmetric.md` — complex-symm vs Hermitian
  history.
- `project_dtxqcd_pfaffian_realsymmetric_fix.md` — early correction
  pass.
- `reference_dtxqcd_gencfgs_env_knobs.md` — `Test_dtxqcd_2pt_gencfgs`
  env-knob catalog.
- `reference_hmc_convention_grid_vs_chroma.md` — eps vs dt convention.
- `reference_bc_fix_audit.md` — APBC default per Wilson-family op.
- `reference_no_metrop_semantics.md` — what NO_METROP=K means.

Cluster instance: build a parallel memory bank in
`~/.claude/projects/.../memory/` on the cluster login node as you
work.  Save a memory the first time you trip on each of the pitfalls
in section 7.

— Laptop, 2026-06-19
