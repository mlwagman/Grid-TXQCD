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
