# HANDOFF: QUDA acceleration of the DTXQCD deformed-contour solves

## 0. OUTCOME SUMMARY (2026-07-16 — campaign complete, program ready for reweighting measurements)

The §3 baseline below is HISTORICAL.  Final measured state (16³×48, λ=5,
traj 300, per wall-source solve, CG_TOL=1e-7):

| operating point | then (Grid sandwich) | now | total speedup |
|---|---|---|---|
| η=0.2 | ~2000 iters, ~450 s | cos profile + EO device CG: 756 iters, ~16 s | **~28×** |
| η=0.4 | ~4200 iters, ~950 s | cos: 2,113 iters, ~44 s | **~22×** |
| η=0.6 (differential 14.4) | ~8100 full-vol iters, ~30 min | **cos + M̃-MG: 5,187 fine Ops, 74 s** | **~24× (MG itself 10.3× vs cos-CGNR, 1.7× vs the best plain-CG point)** |
| η=0.8 | UNCONVERGEABLE (20k cap) | rounded tent w=12: 2,435 iters, ~1 min | **∞ → routine** |
| η=0.9 (the original ideal) | unmeasurable | rounded w=12: 3,934 iters, ~2 min | opened |

Component chain: QUDA hop-corr port + native kernel (~7×/iter) × EO Schur
(~2×) × PROFILE physics (the dominant late win — the tent's gradient-sign
flips are μ-domain walls binding near-zero-mode PAIRS; rounding both
(tent_rounded2) or the cosine profile removes them) × two-level
Petrov-Galerkin M̃-MG on smooth profiles (dense-exact coarse on a second
GPU).  Design constraints measured: |η·p′(t)| ≤ μ_c ≈ 0.85 everywhere;
smooth gradient sign flips; MG window-volume law ≈ 6 supercritical slices
at nbasis=16 (nb-scaling test in flight).  Grading differential =
η·(T/2 − (w_apex+w_src)/2) for rounded tents; full η·T/2 for cos.
Correlator magnitude inherits the profile grading ANALYTICALLY (envelope
collapse across η to 1–2%).  Full definitions/diagnostics:
DTXQCD_DEFORM_MG_NOTE.tex, DTXQCD_DEFORM_MG_MTILDE_DESIGN.md, commits
9d39fc2c / 39ed7524 / 1b274eba and the dtxqcd-eta-contour-deformation
memory.  §5b (EO reweighting estimator) remains the next consumer and
adopts the EO deformed solve + profiles as-is.

**PRODUCTION CERTIFICATION (2026-07-16 evening, commits 6f5b32b8/e3f24bf7).**
The M̃-MG is production-certified inside `meas_baryon_deform_dtxqcd`
(`DTXQCD_DEFORM_MG`, default ON for η≥0.6 smooth profiles) and wired into
the ported reweighting estimator `Test_dtxqcd_eta_stoch_pf` (fast Qdef EO
quadrature solves verified to 2e-16/1e-9 vs host, 46 s → 6.5 s at 4⁴).
Gate job 1301895 (cos η=0.6, λ=5, traj 300): gaussian selftest **4,501
fine Ops** (standalone: 5,187), wall solve **8,695** vs 26,761 plain CGNR
(3.1× iters), h5 A/B vs banked reference WORST 7.9e-7 (thr 1e-4) **PASS**.
The porting saga produced the campaign's most important algorithmic
finding, the **masked-basis NV law**: CONVERGED inverse iterates are the
wrong transfer basis — at NVTOL=1e-3 they collapse onto the supercritical
band (Rayleigh |M̃v|/|v| = 2.86 vs bulk 4.38) and the V-cycle AMPLIFIES
(probe 3.9); shallow clean iterates (6 its, 3.63) contract on random
vectors (0.60) but under-resolve the band.  The winning basis is a
**checkerboard-MASKED deep iterate** — the standalone's RB-allocated
struct fields made SolveTilEO's internal pickCheckerboard iterate the
wrong grid, partially masking the solve back to the white-noise Rayleigh
value while keeping the band's spatial envelope — replicated verbatim as
`DTXQCD_MG_NV_RBQUIRK=1` (default).  Every Setup is now gated: S2
coarse-op consistency (7e-14), per-vector Rayleigh prints, A_c LU-diag
health, and a one-V-cycle contraction probe (0.13 healthy / >1 broken).
Localization methodology for the record: 3-arm single-flip discriminator
on the validated standalone + a clean-recipe depth ladder; and the
operational law learned twice in one day — **bit-identical results across
a supposed change mean the change never ran** (missing header dep, env
whitelist; every knob now has a proof-of-life print).

**Date:** 2026-07-13. **Audience:** the QUDA/MG fork (the one that built the
StyleC device CG, `DTXQCD_MULTISHIFT_QUDA`/`HYBRID`, and the Grid-MG
coarsening work). **Goal:** put QUDA under the η-deformed operator solves.
mwagman's framing: most of what makes η=0.8 intractable today is the O(1)
overhead of Grid GPU kernels vs QUDA for the clover dslash; unlock that and
the landscape changes.

## 1. What the deformed operator is (30 seconds of math)

The η contour deformation scales the charge-2 aux cross-blocks of M₄₈ with a
t-profile: UR block (d,n) → e^{−η·p(t)}·(d,n), LL → e^{+η·p(t)}·conj(d,n).
Production profile is the **tent** p(t) = min((t−t₀) mod T, T−(t−t₀) mod T),
**source-centered**: t₀ = t_src for each source (each offset is a different
contour; reweighting is per-offset).

We solve in the **chemical-potential frame**: with
G = diag(e^{−ηp/2}·𝟙_upper, e^{+ηp/2}·𝟙_lower), the operator
M̃ = G⁻¹·M_def·G has

- **undeformed** aux cross-blocks (the G-ends cancel the aux twist exactly),
- **untouched** site-diagonal blocks (mass, clover, σ/π — scalar-diagonal G
  commutes with them),
- and the entire deformation residue sitting on the **temporal hops of the
  block-diagonal Wilson-clover dslash** as scalar weights:

```
upper block:  fwd t-hop (x→x+t̂):  w_f(t) = e^{−η·Δp(t)/2},  Δp(t) = p(t+1 mod T) − p(t)
              bwd t-hop (x→x−t̂):  w_b(t) = e^{+η·Δp(t−1)/2} = 1/w_f(t−1)
lower block:  reciprocal weights (opposite charge)
```

This is a **time-dependent real chemical potential** μ(t) = η·Δp(t)/2 on the
upper block, −μ(t) on the lower. Key facts:

- w_f·w_b = 1 per link. Links stay **unitary**; the weights ride on the hop
  terms, not the gauge field.
- For the tent, Δp(t) ∈ {+1, −1}: the weights take only **two values**
  e^{∓η/2}, piecewise constant with kinks at t₀ and t₀+T/2. Zero total
  winding (Σ_t Δp = 0).
- APBC composes trivially (the −U_t at t=T−1 is inside the link; weights are
  an extra scalar).
- **γ5-hermiticity is BROKEN** by real μ: M̃† = γ5·M̃(w→1/w)·γ5. Never call
  the base operator's Mdag; the dagger needs the **reciprocal weight arrays**
  (and swapped UR/LL aux weights). This has bitten before — it is the #1
  correctness trap.

## 2. Where the current implementation lives (all Grid kernels — the thing to replace)

- `production/meas_baryon_deform_dtxqcd.cc` — the deformed aux baryon
  measurement. `Mdef` (`:249`) = `Mop.M` + `AddCross` with pre-weighted d,n
  copies; `GApply` (`:279`) = precomputed `LatticeComplex` diagonal
  multiplies; `Mtil`/`MtilDag` (`:287`/`:293`) = the G-sandwich;
  `SolveTil` (`:300`) = hand-rolled **unpreconditioned CGNR on M̃†M̃, pure
  Grid**, 2 matvecs + 2 sandwiches per iteration.
  `BuildDeformWeights(t0)` (`:261`) rebuilds every weight field per source
  (source-centering).
- `production/Test_dtxqcd_eta_stoch_pf.cc` — the stochastic ΔlogPf
  (reweighting-factor) estimator. Same sandwich stack (`:184–231`), plus the
  **hop-form derivative insertion** `dMhop` (`:241–273`): W3 = APBC-phased
  fwd t-link, W7 = Cshift(adj(W3), t, −1), weights
  af_u(t) = (−Δp/2)·e^{−ηΔp/2} etc. — note these are ∂/∂η of exactly the
  w_f/w_b above, so the same patched dslash structure serves the insertion.
- Undeformed reference (what QUDA composition should look like):
  `production/dtxqcd_meas_helper.h` StyleC device CG — QUDA clover dslash
  per block + Grid/cuBLAS aux cross work. The deformed solve should become
  *that*, plus weight arrays.

## 3. Measured baseline (16³×48, λ=4, 1 A100, CG_TOL=1e-7)

| η    | CG iters/solve | s/solve | s/iter |
|------|----------------|---------|--------|
| 0 (StyleC QUDA, undeformed meas) | ~1700 | — | (reference: several× faster/iter) |
| 0 (Grid sandwich path, ETA=0 gate) | ~1700 | ~390 | 0.23 |
| 0.2  | ~2000          | ~450    | 0.22   |
| 0.4  | ~4100–4300     | ~950    | 0.23   |
| 0.8  | **HIT the 20000-iter CG_MAX cap unconverged** (4418 s) | — | 0.22 |

Per-iteration cost is **flat in η** — all η-scaling is operator conditioning
(iteration count ≈ e^{3.7η} empirically, and possibly super-exponential:
η=0.8 needs >20k where the extrapolation predicted ~18.5k). QUDA compresses
the constant; 16 solves/config means η=0.8 goes from ~19 h/config to ~4 h at
a 5× kernel win — but η=0.8 ALSO needs the iteration count attacked
(η-continuation guesses, and eventually MG), not just the kernel. The
ΔlogPf estimator fleet (O(60) solves/config × 4 offsets planned) is the
biggest total consumer of the same matvec.

**CG_MAX gotcha (fixed 2026-07-13 in the source, binary rebuild pending):**
`SolveTil` used to exit the iteration loop silently at CG_MAX and write the
unconverged propagator into the h5. It now prints relres and _Exit(3)s on
non-convergence. Any η=0.8 output produced before the rebuild is tainted —
check logs for `iters=20000`.

## 4. The proposed QUDA change — per-timeslice hop weights, NOT independent links

We do **not** need general independent fwd/bwd link storage. The asymmetry
is exactly two length-T scalar arrays.

**Option A (the real fix): patch the Wilson-clover dslash core.**
Multiply the μ=3 fwd/bwd hop terms by `wf[t]`, `wb[t]` (real, or complex for
future generality) read from a small device array indexed by the site's t
coordinate. Precedents: the twisted-mass i_mu plumbing, staggered chemical
potential. Plumb at dslash-arg level (not gauge-load level) so swapping
arrays between the 16 per-source solves costs nothing — **source-centered
tents become free** (today each t_src rebuilds ~8 lattice-size weight
fields). Links stay unitary, RECONSTRUCT unaffected, clover term untouched.
The dagger call must take the reciprocal arrays (§1 trap); lower block =
reciprocal of upper (plus the existing conjugate-gauge handling StyleC
already does).

**Option B (no-patch fallback, ships faster):** pass scaled links
W_f(t)·U_t to QUDA under RECONSTRUCT_NO (non-unitary links: HISQ precedent).
QUDA computes bwd = adj(w_f·U) = w_f·U†, but we need w_b·U† = (1/w_f)·U†, so
add a Grid-side correction term on the bwd t-hop only:
(w_b − w_f)·(−½)(1+γ_t)·U†ψ(x−t̂) — one Cshift + one su3 multiply per
matvec, small vs the full dslash. Caveats: keep the clover built from the
*unscaled* field (StyleC passes clover explicitly, so fine — but check);
non-unitary links must not hit any unitarity-assuming reconstruct path.

**MG caveat:** each contour (each t₀!) is a different operator → MG setup
per offset = 4 setups/config under source-centering. Do NOT start with MG;
plain QUDA CG/BiCGStab first, and see how far the constant factor goes.
Orthogonal cheap win available regardless: **η-continuation initial guesses**
(reuse the η=0.4 solution as the η=0.8 guess — smooth in η, nearly free).

## 5. Acceptance gates (all machinery exists)

1. **η=0 gate:** `meas_baryon_deform_dtxqcd` with ETA=0 on λ=4 traj 300 must
   reproduce `meas_2pt/dtxqcd_lam4_baryon8/baryon_dtxqcd_300.h5` to ~1e-6
   rel (the existing gate; passed by the Grid path, job 1300482).
2. **Dense operator check:** `Test_dtxqcd_eta_stoch_pf` OPCHECK mode
   compares the sandwich matvec against the dense 4⁴ matrix
   (`Test_dtxqcd_eta_deform_pf` machinery). Run at η=0.2 and 0.8.
3. **Dagger check:** ⟨u, M̃v⟩ = ⟨M̃†u, v⟩ to machine precision with the
   reciprocal arrays (γ5-herm trap).
4. **η=0.2 end-to-end:** reproduce
   `meas_2pt/dtxqcd_lam4_deform_eta0p2/baryon_deform_dtxqcd_300.h5` to
   CG-tolerance level.
5. **GAUGE_TWIST=1** on the measurement (permanent gate after any
   contraction-adjacent change).
6. Report s/iter vs the 0.22 baseline. Target ≥3×; anything ≥5× makes the
   η=0.8 corner routine.

## 5b. NEXT CONSUMER of the EO operator: the EO reweighting-factor estimator (designed, not built — adopt once your EO solve lands)

The Pf convergence study says the ΔlogPf estimator is noise-limited (dlogdet
= −6.8 ± 3.0 at 20 solves on λ=4 traj 300, η=0.2 tent; brute force to σ=0.5
is ~700 solves/config/offset). The EO structure buys an exact factorization
win on top of the solver speedup, because in the chemical-potential frame
**M̃ee = Mee and M̃oo = Moo identically** (the deformation lives ONLY on the
t-hops, which are even↔odd; every site-local block including the d,n cross
terms is untouched):

  Δlog det M₄₈ = log det S̃ − log det S,   S̃ = Moo − M̃oe·Mee⁻¹·M̃eo

with the Mee logdets cancelling **exactly per config**. The estimator
becomes Hutchinson on the odd sublattice (half the DOF, none of the
Mee-block noise):

  d/dη log det S̃ = Tr[S̃⁻¹ dS̃],   dS̃ = −(dM̃oe)·Mee⁻¹·M̃eo − M̃oe·Mee⁻¹·(dM̃eo)

Implementation plan (all pieces exist in `DTXQCDWilsonCloverFermionEO` +
`DTXQCDMpcOp`):
- Twisted CB hops: M̃eo = G_e⁻¹·Meooe·G_o etc. — CB-picked wGu/wGl weight
  fields (`pickCheckerboard` of the full-grid builders in
  `Test_dtxqcd_eta_stoch_pf.cc:188`), GApply sandwiches around the native
  `Meooe`/`MeooeDag`. Dagger swaps G→G⁻¹ (γ5-herm broken, as always).
- S̃/S̃† = `DTXQCDMpcOp::M/Mdag` (`DTXQCDMpcOp.h:27`) with the G-sandwiched
  hop calls substituted; Mee⁻¹ = the cached `MooeeInv` (undeformed, exact).
- dS̃: the hop-form insertion `dMhop` (`Test_dtxqcd_eta_stoch_pf.cc:294`)
  applied via full-grid promote/pick (once per noise vector, not per CG
  iteration — cost negligible); M̃eo/M̃oe legs CB-native.
- Noise: full-grid gaussian → `pickCheckerboard(Odd, ...)`; keep the
  grading-antithetic pair (v → block-flip v).
- CG: CGNR on S̃†S̃ over odd-CB doubled fermions (per-component
  `.Checkerboard() = Odd`, pattern of `DTXQCDWilsonCloverRationalEOAction.h:148`).
- Gates: (i) DERIVCHECK: dS̃ vs FD[S̃(η±d)] on a random odd vector; (ii) the
  decisive dense gate on 4⁴: assemble dense S̃ (48·V/2 = 6144 columns via
  unit-vector applies), `DetLU` (cuSOLVER pattern in
  `Test_dtxqcd_eta_deform_pf.cc:57`) at η and 0 — Δlogdet must equal the
  full dense M₄₈ ETA_SCAN value for the same config/profile to ~1e-10
  (proves the Mee cancellation end-to-end); (iii) statistical match vs the
  full-grid estimator on 4⁴.
- First target: λ=5 (94 production cfgs in
  `cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw`, 4⁴ gates stream
  `dtxqcd_lam5.0000_4t4_signhop_lam5`) — above the Mee cliff, ⟨signPf⟩=1,
  clean decorrelated chain.

## 5d. REWEIGHTING FACTORS: SOLVED — the production battery (2026-07-18)

**Status: the 5c program is complete.  Reweighting factors went from
"showstopper" (σ ≈ 20–40/config at 24 solves) to a running 4-config
pilot at η=0.2 with σ_anchor ≈ 1 in 24 GPU-h/config.**  Every element
below is measured and gated; commits dcc64dbc / ea28a970 / 6528a8aa.

### The measured stack (all multiplicative)

| lever | win | evidence |
|---|---|---|
| Schur-form estimator | ×1.7 σ | 4⁴ gate 1302156 |
| comb-4 t-dilution (DEFAULT) | ×1.2–1.5 σ | F-map ladder: universal optimum 0.68–0.82 at ep=0.05–0.40 |
| CG_TOL=1e-4 sampling (DEFAULT) | ×1.8 time | ladder gate 1302547: bias 1.5e-4 vs ±1.25 stat |
| MG-lite L1 (single GPU, DEFAULT) | ~×9/solve above ep\*≈0.15 | crossover 1302548/65; marginal ~7 s/solve at ep=0.4 |
| REUSE_NV shared nullspace (battery env) | setup 110 s → 20 µs/node | A/B 1302589; probe-gated auto-fallback |
| Neyman allocation (DEFAULT) | ~13% σ | measured flat-ish σ(ep) table baked in |
| η-segments ETA_LO | ~2× across target family + lower variance | consistency gate 1302595 |
| TSRC-RATIO quadrature | 48 offsets ≈ 2–3 anchors | ∂ₛ insertion O(η p″): node σ 14–31 vs 540; DERIVCHECK 1.3e-8; closure sum rule |
| oblique-projector deflation | variance-neutral, exact-term capture, ~free cached | prod A/B 1302160; NO bias from loose pairs (6-digit dense match) |

Deflation verdict: the deep band carries large MEAN (exact terms −105 at
K=16, −269 at K=48) but NOT the variance (σ 540→507 at K=48) — the bulk
floor dominates (√V scaling from 4⁴ + the σ∝η-was-the-prefactor
finding: σ(ep) is flat-ish {0.05: 512, 0.16: 536, 0.32: 621, 0.40: 750}).

### How to run reweighting factors (per config)

```bash
# 1) ANCHOR per segment (nested across the eta-target family):
env LATT=16.16.16.48 TRAJ=<t> CFG_DIR=<lam5 dir> PROFILE=cos PROF_T0=0 \
    ETA=0.2  ETA_LO=0.0 N_GL=2 N_NOISE=625 DTXQCD_STOCHPF_ANTI=0 \
    DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_MG_REUSE_NV=1 \
    ./Test_dtxqcd_eta_stoch_pf ...        # sigma_anchor ~ 1.0, ~10 GPU-h
# repeat with ETA=0.4 ETA_LO=0.2 and ETA=0.6 ETA_LO=0.4 as budget allows;
# each target's DlogPf = sum of its segments (STOCHPF lines carry eta_lo).
# 2) TSRC ratios (ALL 48 offsets from one periodic quadrature):
env ... ETA=0.2 DTXQCD_STOCHPF_TSRC_RATIO=12 N_NOISE=500 \
    DTXQCD_MG_REUSE_NV=1 ./Test_dtxqcd_eta_stoch_pf ...
# offline: cumulative trapezoid of the TSRCNODE means, bridge-corrected
# by the TSRCCLOSURE sum rule; DlogPf(t0) = anchor + ratio(t0).
```
Defaults already ON: comb-4 dilution, Neyman, CG_TOL=1e-4, Schur form,
MG-lite (ETAMIN 0.15).  Reference launcher: `slurm_rwf_battery_smoke.sh`
(+ `_seg.sh`).  Pricing: η=0.2 full ensemble ≈ 2,300 GPU-h (VIABLE);
η=0.4 ×2; η=0.6 ~100+ GPU-h/config (subset campaign; its top node
ep=0.558 is marginally supercritical — verified only by extrapolation).
Accuracy contract: per-config anchor σ ≲ 1 (the e^{σ²} ESS argument:
exponentiation bias CANCELS in reweighting ratios for independent noise);
relative offsets tolerate σ_D ~ 2–3.

### PILOT RESULTS (2026-07-19: the first production reweighting factors)

4 configs (traj 300/400/500/600), full locked stack, measured 6.5 s/solve
at the [0,0.2] nodes (cost model confirmed):

| traj | ΔlogPf[0,0.2] | ΔlogPf[0.2,0.4] | cumulative η=0.4 |
|---|---|---|---|
| 300 | −0.047 ± 0.74 | 2.38 ± 1.12 | 2.33 ± 1.34 |
| 400 | +0.958 ± 0.77 | 2.44 ± 1.13 | 3.40 ± 1.36 |
| 500 | −0.125 ± 0.76 | 4.45 ± 1.16 | 4.33 ± 1.39 |
| 600 | +1.465 ± 0.79 | 1.27 ± 1.13 | 2.74 ± 1.38 |

σ_anchor 0.74–0.79 at 5,000 solves ([0,0.2]) and 1.12–1.16 at 4,000
([0.2,0.4]) — both BETTER than the model's 1.0/1.5.  O(1) per-config
spread (η=0.2 weights e^{ΔlogPf} ≈ 0.9–4.3; η=0.4 spanning ~e³) —
genuine, resolvable reweighting structure.  darg resolved as well.
[0.4,0.6] smoke + TSRC halves pending.  η=0.6 TSRC QUADRATURE: use
M=24 — the M=12 closure violated the sum rule at 2.4σ (−2193±898),
M=24 restores it (+355±647): supercritical integrands have finer
s-structure; η ≤ 0.4 is clean at M=12.

### Scaling to higher stats: multi-job averaging + packing

Averaging runs on the same config is valid IFF every run has a DISTINCT
NOISE_SEED (identical seeds = identical samples = fully correlated).
Convention: `NOISE_SEED = 1000*traj + 100*segment_id + job_index`.
Plain averages of STOCHPF (and of per-node TSRCNODE means, before the
cumulative integral) are then correct; deflation caches and MG bases
may be shared freely (operator data, not statistics).

One 24 h × 4-GPU job on a single config (4 seeds averaged) delivers:
anchors σ ≈ 0.23 / 0.40 / 0.55 for [0,.2]/[.2,.4]/[.4,.6] (or run one
GPU per segment: 0.46/0.81/1.1 in a single job — anchors saturate
fast); TSRC (all 48 sources, always) σ_D ≈ 1.0 (η=0.2, M=12) / 2.6
(η=0.4) / ~28 (η=0.6, M=24 — INFEASIBLE at this budget: accept
σ_D ~ 5–10, spend 2–4 GPU-days, or run η=0.6 anchor-only with a fixed
t0=0 contour).  The η=0.6 TSRC integrand is intrinsically expensive:
per-sample σ ≈ 190 with 24 nodes.

### TENT PIVOT (2026-07-20): the production contour is the SHARP TENT

The [0.4,0.6] same-config head-to-head ended the cos era for
production reweighting: cos's supercritical band (η > 0.54·(2/π)…, i.e.
nodes above ep ≈ 0.45) is GENUINE config-to-config ΔlogPf variance —
cos σ_config ≈ 5–6 at η=0.6 ⇒ ESS ≈ 0 regardless of estimator
precision — while the subcritical tent holds σ_config ≈ 0.5 with ×2.4
lower estimator noise on top.  Tent segment ledger (4 configs):
σ = 0.66 / 0.85 / 1.67 for [0,.2]/[.2,.4]/[.4,.6].

Tent solver reality (λ=5, wall sources, measured): plain CG
1,260 / 4,480 / 15,900 / 59,930 iters at η = 0.4/0.6/0.7/0.8 —
**production η ≤ 0.6 needs NO MG and NO deflation.**  Census: the
wall PAIR (exact 2-fold degeneracy) descends smoothly to 2.0e-5 at
η=0.6; towers (~24 levels) form only at 0.7+.  Deflation for reach
beyond 0.6 MUST use the certified eigensolve settings
`DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 _TOL=1e-5` — the meas defaults
(301/1e-3) build bases that actively HURT (measured 4.5k→33k).
Tent TSRC coverage: use the common-noise Δt0 telescope
(ρ = 0.98–0.99 measured) — the s-line-integral needs a smooth profile.
Mask engineering post-mortem: explicit split/parity masks stall; the
RB-quirk NV recipe is not a spatial mask (site-scrambled half-copy) and
remains the only working basis — moot for tent production (no MG).

### The tent ceiling program (launched 2026-07-20)

Solver feasibility now EXCEEDS ensemble feasibility: estimator solves
at tol=1e-4 cost only ~700 iters even at ep=0.558 (CG stops before the
deep modes matter), and deflation covers ep to ~0.8 — so the tent reach
limit is σ_config per segment, i.e. ESS(η) = e^{−Σσ²_config}.  Running
on the pilot's 4 configs: [0.6,0.7] (DEFL=24) and [0.7,0.8] (DEFL=32)
anchor smokes + a DEFL=8 variance arm on [0.4,0.6] (does the wall pair
carry trace variance at deep nodes?).  nev policy from the census:
pairs (8) at nodes ≤ 0.6; tower (24–32) at 0.68+.  ALWAYS run
deflation through the seg-script DEFL knob (it forces the certified
ord=1001/tol=1e-5; the bare meas defaults 301/1e-3 build bases that
HURT).  Eigensolves are cached per (config, node) and t0-translatable
— reusable by TSRC/telescope runs.

### Propagator-side MG update (MG-lite L1 = NEW DEFAULT)

`DTXQCD_MG_BLOCK=4.4.4.4` + `DTXQCD_MG_LUDEV=0` (both now header
defaults) put the dense coarse LU (N=12,288, 2.4 GB) on the SAME GPU:
**h5-certified at cos η=0.6 (job 1302640) and BETTER than the original
two-GPU config — selftest 3,891 vs 4,501 Ops, wall 5,664 vs 8,695,
coarsen+LU 36 vs 50 s.**  The GPU-pairs constraint (DEFGPUS) is
OBSOLETE — deform-MG runs single-GPU everywhere; fleets get 2×
throughput from the same allocation.  `DTXQCD_MG_REUSE_NV=1` extends to
source-centered prop measurements (per-t_src setup 640–840 s of NV → 36 s
coarsen+LU; probe-gated fallback), certified for s-shifts, probe-guarded
per t_src.  GMRES coarse (`DTXQCD_MG_COARSE=gmres`) converges at
subcritical nodes but is slower than the LU — the zero-memory fallback.

## 5c. DEFLATED TRACE ESTIMATOR — the reweighting critical path (2026-07-16)

**The measured problem.** The fast EO estimator works (56 s/solve at λ=5
η=0.4, guards + proper GL rules in, commits 65ea769d/a93707d3), but the
rwf battery prices per-config ΔlogPf noise at 24 solves (N_GL=2, N_NOISE=12,
cos, PROF_T0=0):

| point | ΔlogPf ± σ | darg ± σ |
|---|---|---|
| λ=4 η=0.2 traj300 | +0.1 ± 17.3 | +2.8 ± 12.8 |
| λ=4 η=0.4 traj300 | −48.4 ± 52.9 | +64.4 ± 40.5 |
| λ=4 η=0.6 traj300 | +49.6 ± 88.7 | +49.9 ± 75.7 |
| λ=5 η=0.2 traj500 | −16.5 ± 18.3 | −8.0 ± 18.9 |
| λ=5 η=0.4 traj500 | +8.8 ± 42.8 | +1.2 ± 46.9 |
| λ=5 η=0.6 traj500 | +31.4 ± 72.7 | −22.3 ± 68.9 |

(σ ≈ 18·(η/0.2), λ=4 ~25% noisier than λ=5 at fixed η; every value ≲1σ
from zero — noise-dominated across the whole map.  Per-node integrand σ ≈
120–215 at N12 — the same per-sample σ ≈ 400 as the pre-perf era; the fast
solver made samples cheap, not quiet.  η=0.6 numbers from the GPU-PAIRED
rerun (EO+MG); earlier "−6.8 ± 3.0" was the TRUNCATED-rule artifact.)  Reweighting needs σ ≲ O(1) per config/offset ⇒
brute force ~10⁵ solves/config/offset.  Cheap solves cannot buy this; the
Hutchinson VARIANCE must come down — this is now the single blocker for
production reweighting, and it is a job for the deflation stack.

**Proposed estimator (singular-triplet split).**  With the SVD
S̃ = Σ σ_k u_k v_k†, split at K smallest σ:

  Tr[S̃⁻¹ dS̃] = Σ_{k≤K} σ_k⁻¹ ⟨u_k| dS̃ |v_k⟩          (EXACT, no noise)
              + E_w ⟨w, S̃⁻¹ P⊥_u dS̃ w⟩                 (Hutchinson, deflated)

- v_k, σ_k² = lowest eigenpairs of M̃pc†M̃pc (the stack's bread and butter;
  ord~1001 window discipline per [[dtxqcd-deform-deflation]]);
  u_k = S̃ v_k / σ_k.
- Remainder sample: y = dS̃ w (hop-form insertion, exists); project y ⊥
  span{u_k}; deflated solve S̃x = P⊥y (Galerkin deflation accelerates this
  too); accumulate ⟨w, x⟩.  Keep the grading-antithetic pair.
- The variance is dominated by the σ_k⁻¹-enhanced directions — exactly the
  low band the census maps.  NOTE the known solve-side result does NOT
  carry over: "eigen-deflation returns ZERO at η=0.8" was about CGNR
  iteration counts (mid-band dominated); the TRACE variance ∝ σ_k⁻² is
  low-mode dominated, so deflation should pay here even where it failed
  for solves.  The open risk is the same dense mid-band: variance-vs-K may
  flatten once K exhausts the resolvable outliers.  MEASURE IT FIRST on
  4⁴ dense (exact SVD + exact trace via cuSOLVER — one afternoon, decides
  the production K before any big wiring).
- Eigenpair tolerance → trace BIAS (not just variance): the exact term
  needs refined triplets or a residual-correction term; gate at 4⁴ against
  the dense trace to 1e-8.

**The offset unlock (why this also rescues NTSRC=48).**  Source-centered
contours need one ΔlogPf per offset; the stack's **basis cache + frame
translation** (e98a3c9c) translates an eigenbasis between t0 frames — one
eigensolve per config, translated per offset, amortizes the dominant cost
across all 48 offsets.  Combined with per-offset deflated Hutchinson this
is the only visible path to reweighted max-stats datasets.

**ALSO FOR THE MG OWNER — λ=4 wall-source stall (2026-07-16, farm 1302010):**
cos η=0.6 λ=4, EO+MG forced, ALL setup gates green (S2 7e-14, LU-diag 52.8,
contraction probe 0.111, NV Rayleigh 4.589) yet the WALL solve hard-stalls
at relres 0.044 @ 60k while the same pairing's gaussian-source estimator
solves succeed.  The gates are gaussian probes; wall sources excite what
the λ=4 coarse span misses (the old λ-scan's "λ=4 needs nbasis>32").  λ=4
η=0.6 correlator farm is parked on this; λ=5 unaffected.  Suggested: add a
WALL-source contraction probe to the Setup gates so this class is caught
before a production run.

**Contract to preserve** (Test_dtxqcd_eta_stoch_pf): GLNODE/STOCHPF output
format; proper 1/2/4/8-pt GL rules; NOT-CONVERGED ⇒ _Exit(3) (unconverged
solves bias the trace); PROF_T0 / profile dispatch (cos default in scripts,
tent in code); MG needs GPU PAIRS (CoarseDirect dev=1 — the 2026-07-16
single-GPU lesson, commit 6b524e97).

## 6. Context pointers

- Program history + physics: memory `dtxqcd-eta-contour-deformation.md`;
  commit messages 278c1af3, e41dd0cb, 8e1b8201, d6fd37d8.
- First physics result (2026-07-13): per-config graded-magnitude m_eff shift
  = +η confirmed (4/5 configs at η=0.2; traj 300 gives +0.495 at η=0.4).
  Analysis tool: `production/analyze_baryon_deform.py`.
- Running fleets: η=0.2 full-ensemble (57 cfgs, jobs 1300601/03/06/07),
  η=0.4 (1300492), η=0.8 (1300493), Pf convergence study (1300491).
- Operational: srun --mpi=pmix (never mpirun), --shm-mpi 1, LATT env (not
  --grid) sets the volume, one-statement bash hook.
