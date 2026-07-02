# DTXQCD non-EO Tr[π] vs p asymmetry — handoff to next session

## ═══ RESOLVED 2026-07-01 (same day, follow-up session) — NO CODE BUG ═══

**The asymmetry is a frozen initial condition of the fermion-blind soft mode,
not a force or action defect.**  Full chain of evidence:

1. **π and p enter M only through Y = π + p·I₆** (identical γ5 coupling, both
   blocks).  The orthogonal per-site mode w ∝ p − Tr[π]/… (volume-averaged
   w̄ = ⟨p⟩ − ⟨Trπ⟩) couples to NOTHING: not the fermions, not the gauge — only
   the aux Gaussian with mass λ².  Exact equilibrium identity: ⟨Trπ⟩ = ⟨p⟩
   (constrained-Gaussian split of the 6π₀² + p² metric).  Same for (σ, s).
2. **The w force is algebraically zero in the code**: Σ_diag F_pi = F_p is an
   identity of the site kernel for ANY bilinear input (γ5 diagonal), preserved
   by the Wirtinger transpose, Hermitize, accumulation, and integrator.  So w̄
   can only diffuse (SD ≈ 0.003/traj at 16³×48) — verified against the h5
   traj-to-traj steps.
3. **The "traj-1 kick" was a reconstruction artifact.**  gen_dtxqcd_cfgs
   RE-SEEDS pRNG to {6,7,8,9,10} right before the final FillAuxFields
   (line 268), so the production aux draw starts at stream position 0.  The
   Test_dtxqcd_aux_drive_symmetry reconstruction drew the weak-field gauge
   first (consuming stream) → different Gaussian sample → w̄_init = −0.081
   instead of the chain's true −0.134.  Verified by
   production/Test_dtxqcd_init_repro: driver order gives ⟨Trπ⟩=+0.1334,
   w̄=−0.1342; chain h5 traj 1 = +0.1317, −0.1323 (one traj of diffusion away).
4. **The mode is frozen**: per-entry aux frequency ω = λ√2 → per-traj mean
   decay cos(ωτ) → relaxation ~800 trajs at λ=0.1 (τ=√2/4).  110–140 trajs
   ≪ that.  The λ=0.25 / λ=0.5 chains decay at the predicted faster rates ✓.
5. **λ-scaling + common sign across chains**: all chains share seed
   {6,7,8,9,10}; init width ∝ 1/λ ⇒ w̄_init ∝ 1/λ with the same normalized
   draw ⇒ ⟨Trπ⟩ ≈ −6w̄/7 ∝ 1/λ, same sign everywhere.  Matches 0.15 / 0.05 /
   0.02 at λ = 0.1 / 0.25 / 0.5.  The quantitative split
   ⟨Trπ⟩ = 6(ū−w̄)/7, ⟨p⟩ = (6ū+w̄)/7 reproduces the observed values with
   ū ≈ 0.03.  The "270σ" claim assumed independent samples; with τ_w ~ 800
   trajs the chain holds ~one sample of a Gaussian with SD ≈ 0.06 → ~2σ.
6. **The P-even sector shows the same frozen mode** (s − Trσ = −0.025 at init
   → −0.044 by traj 140, hidden by the large common condensate 44) — same
   mechanism, same explanation.
7. **Every code probe is green** (Test_dtxqcd_aux_force_aggregate +
   Test_dtxqcd_ratforce_gpu multi-rank): host M u-equivalence M(π+cI,p) ≡
   M(π,p+c) at 1e-16 (4⁴ + 16³×48); LogDet & Rational S u-equivalence exact,
   incl. QUDA Style C; deriv-vs-FD = 1.000000 for all four actions on all
   four singlet slots, CPU and production-GPU paths; GPU force kernel
   bit-comparable multi-rank.

**Physics impact: none on fermionic observables.**  w does not enter M, so
plaq / correlators / Fierz / signPf are exactly w-independent.  Only the raw
aux VEVs ⟨Trπ⟩ vs ⟨p⟩ (and ⟨Trσ⟩ vs ⟨s⟩) display the frozen split.  The
λ=0.1 plaq 0.454-vs-0.514 question is a SEPARATE (thermalization) issue —
the σ condensate is still visibly relaxing (44.3 → 42.3 over 140 trajs, no
plateau), so that chain simply is not equilibrated yet.

**Recommended fixes (not yet implemented):**
- Exact per-site Gibbs resample of the two fermion-blind directions each
  trajectory: given Y = π + pI, draw p' ~ N(Tr Y_ct/7·…, 1/(7λ²)) and set
  π = Y − p'I (and the σ/s analogue).  Site-local, analytic, zero cost —
  removes the frozen modes entirely.  Until then, either init with the
  conditional draw (w typical instead of 2σ) or simply don't interpret
  ⟨Trπ⟩−⟨p⟩ as physics.
- Reconstructions of the production init MUST mimic the driver RNG order:
  seed → weak-field gauge → RESEED {6,7,8,9,10} → FillAuxFields.

Diagnostics added: tests/dtxqcd/Test_dtxqcd_aux_force_aggregate.cc,
production/Test_dtxqcd_init_repro.cc, scratch analyzer of h5 aux VEV series.
The remainder of this document is the ORIGINAL (superseded) bug-hunt handoff,
kept for the record.

## ═══ original handoff below (superseded) ═══

**Date:** 2026-07-01
**Status:** localized to something between the aux force kernels (verified symmetric) and the field observables (evolve asymmetrically).  Ready for targeted deriv() diagnostic.
**Branch:** `dtxqcd-v2` (Grid-DTXQCD repo)
**Related memory:** `project_dtxqcd_low_lambda_isosinglet_condensate`, `reference_dtxqcd_parity_transform`

## The mystery in one paragraph

At λ=0.1 (16³×48, non-EO, USE_FULL_PF), the P-odd aux `<Tr[π]>` (isosinglet color-traced π) develops a large VEV `+0.154 ± 0.0006` (270σ) while its Q-numbers-identical partner `<p>` sits at `+0.005 ± 0.0003` (17σ, ~30× smaller).  P-even partners `<Tr[σ]>` and `<s>` track each other to 3-4 digits at every λ — that channel is clean.  The parity heatbath (Test #237, task #238) symmetrizes across P sectors but `<|Tr[π]|>` still shows the same SSB-like magnitude, so the asymmetry is NOT a P-sector trapping problem, it's a genuine force asymmetry.

From analytical work: if `∂M/∂π_singlet = ∂M/∂p` as operators (verified by inspection of `DtxqcdApplyX`), then equilibrium `<Tr[π]> = <p>` exactly.  Observation contradicts by a factor of ~30.

## What's been verified (this session)

`tests/dtxqcd/Test_dtxqcd_aux_drive_symmetry.cc` (committed 894ea80f) probes at the **FD linear-response level**:

1. **LogDet-EO action** — perturb σ→σ+ε·I₆, π→π+ε·I₆, s→s+ε, p→p+ε separately; measure `(S(+ε) − S(−ε))/(2ε)`.  Result on 4⁴ with hot gauge + Sigma_init=3.07:
   - `f_Trσ / f_s = 1.000000000000` (bit-exact)
   - `f_Trπ / f_p = 1.000000000000` (bit-exact)
2. **Pseudofermion single-shift `Φ†(M†M)⁻¹Φ`** — same perturbations, force via CG on `M†M X = Φ` (also 4⁴):
   - `f_pf_Trσ / f_pf_s = 1.000000000027`
   - `f_pf_Trπ / f_pf_p = 1.000000000000`
3. **Both actions with production weak-field init recipe** (Sigma at init shift, `GenerateWeakFieldGauge`) — same result, symmetry to 10-12 digits.

So the FD sum identity `Σ_{ai} pi_force(a,a)(i,i) = p_force` holds at every level we can currently probe.

## The critical observation the FD test doesn't catch

Ran the test at 16³×48 with the EXACT driver seed `pRNG.SeedFixedIntegers({6,7,8,9,10})` and `Sigma_init=0.221654383` (matches ref chain `[AUX_INIT_AUTO converged]` log).  Init aux averages (per-site):

```
<Trsig>_init = +44.317    <s>_init = +44.308     (P-even, expected match)
<Trpi>_init  = +0.070     <p>_init  = -0.011     (P-odd, both consistent with Gaussian init)
```

Init Gaussian statistics: `SD(<Trpi>_init) = √6/(λ·√V₄) = 0.032`, `SD(<p>_init) = 1/(λ·√V₄) = 0.013`.  So `<Trpi>_init = +0.070` is +2.2σ from zero (a mildly-positive Gaussian sample), `<p>_init = -0.011` is -0.85σ (noise).  Nothing weird at init.

**Ref chain after 1 trajectory (from h5):**
```
             init      traj 1     Δ per traj
<Trpi>      +0.070    +0.132     +0.062
<p>         -0.011    -0.001     +0.010
```

**Ratio Δ<Trpi> / Δ<p> ≈ 6.2 in one trajectory.**

If forces were symmetric per the FD test, Δ<Trpi> should equal Δ<p>.  Observed asymmetry is factor of 6 — suspiciously equal to `N_c · N_f = 6`, the aggregation factor in the FD identity `Σ_{ai} pi_force = p_force`.  Looks like **each pi diagonal entry is getting force ≈ p_force (not p_force/6)** in the integrator update — the exact bug the algebraic identity would hide if only the sum-over-entries were correct, or if the pi force is being applied 6× per site while p is applied once.

## Suspect list (ranked by likelihood)

The FD test proves the **kernel output** is correct.  So the bug is downstream of `DtxqcdSiteForceKernel::AuxForceAt`.  Candidates:

1. **AuxGaussianAction::deriv() on pi.**  Look at `Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h:78`: `dSdU.pi = c * U.pi;` for the Hermitian 6×6 matrix.  Compare to `dSdU.p = c * U.p;` on the scalar.  Are they consistent with the DOF counting for the singlet-mode Gaussian mass?  Explicitly: for π = π₀·I₆, `norm2(π) = 6·π₀²`, so the singlet-mode mass is 6λ² per site; each diagonal entry of `dSdU.pi` should be `λ²·π₀` (correct: `c*U.pi` gives this).  Off-diagonals of pi have DIFFERENT effective mass because `norm2` double-counts them (see comment on line 15).  This is the OBVIOUS place to check for a factor-6 issue.

2. **DTXQCDCompositeImpl::projectForce for pi.** `Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h:272-273` calls `DtxqcdHermitizeAndTracelessCFInPlace(out.pi)` (now Hermitize-only per commit note on line 26-36).  Verify this doesn't accidentally amplify the diagonal.  For a Hermitian input, `0.5·(X + adj(X))` is a no-op on diagonal entries.

3. **DTXQCDCompositeImpl::update_P vs update_field for pi.** Lines 310-311, 314-315: `U.pi = U.pi + P.pi * ep;` and `U.p = U.p + P.p * ep;`.  Look uniform.  BUT check that `P.pi` momentum was generated with the right variance (line 229-230): `DtxqcdHermitianCFGaussian(pRNG, P.pi); P.pi = scale * P.pi;` where `scale = √HMC_MOMENTUM_DENOMINATOR = √2`.  Per-entry momentum SD is √2 for both pi entries and p — consistent.

4. **Rational full-action deriv() sum over Remez poles.**  The rational action iterates over ~10 poles and accumulates aux forces via `AddFullLatticeToFull(F_pi, dSdU.pi, coef)` where `coef = 2·a_k`.  If somehow `coef` differs between pi and p accumulation, we'd see the asymmetry.  Same call for F_p on the next line, so this looks symmetric.

5. **The `Wirtinger transpose` write in the GPU kernel** (line 253 of `DTXQCDRationalForceGpuKernel.h`): `piv[s]()(a,b)(i,j) = pi_k[b][a][j][i]`.  On the diagonal (a=b, i=j) this is a no-op, but if there's an off-diagonal read that overflows into the diagonal slot in some SIMD lane, that could produce a 6× amplification.  Unlikely but worth checking with a print at a specific site.

## Concrete next test to write

Write `tests/dtxqcd/Test_dtxqcd_aux_force_aggregate.cc` — a targeted deriv() diagnostic that runs on the actual production setup and prints, for each action separately:

```
[GaugeAction]         F_pi_diagonal_sum = ...   F_p = ...   ratio = ...
[AuxGaussianAction]   F_pi_diagonal_sum = ...   F_p = ...   ratio = ...
[LogDetCloverEOAction] F_pi_diagonal_sum = ...  F_p = ...   ratio = ...
[RationalFullAction]  F_pi_diagonal_sum = ...   F_p = ...   ratio = ...
```

where `F_pi_diagonal_sum = Σ_x Σ_{ai} dSdU.pi(x)^{ii}_{aa}` and `F_p = Σ_x dSdU.p(x)`.

**Setup:** same as the existing Test_dtxqcd_aux_drive_symmetry — driver seed {6,7,8,9,10}, weak-field gauge, FillAuxFields with production Sigma_init.  Preferably 4⁴ first (fast), then 16³×48 for confirmation.

**Expected result if code were symmetric:** all four ratios = 1.  If the AuxGaussianAction ratio = 6, that's the bug — the Gaussian force on pi has a factor-6 miscount somewhere.  If the RationalFullAction ratio = 6 but AuxGaussian is 1, the fermion path has the issue.

**Implementation hint:** the existing Test_dtxqcd_aux_drive_symmetry already sets up the config correctly; just extend it to call each action's `deriv(U, dSdU)` and print the diagonal-sum / p aggregate.  For rational: needs `refresh()` first to draw Φ.  Take care to check the Remez auto-scale.

**Auxiliary check while you're there:** print the same aggregates for σ↔s.  The P-even sector observationally works, so ratios should be 1 (or something that gives `<Trsig> = <s>` on equilibration).  If they're NOT 1, we've learned that the empirical P-even matching is a coincidence, not a code symmetry.

## Parallel work still running

- **Slurm job 1295582** (parity-heatbath fork at λ=0.1) is still running.  It reproduces the same `<Tr[π]>` magnitude with sign-flips symmetric across P sectors, confirming the fix is *not* just parity trapping.  It's a useful control chain — let it continue.  Log at `slurm-logs/dtx_nEO_l0p1_parHB.1295582.out`.
- **Reference chain** at `production/cfgs/dtxqcd_lam0.1000_nonEO_mds10_fullL_deg8/` — 110+ trajs of data available for aux VEV analysis.

## Key files (map for the next session)

| Path | What it does |
|:---|:---|
| `tests/dtxqcd/Test_dtxqcd_aux_drive_symmetry.cc` | The FD probe (committed).  Its structure is the template for the aggregate-deriv test. |
| `Grid/qcd/action/dtxqcd/DTXQCDAuxGaussianAction.h` | `S = (λ²/2) norm2(π)` and `dSdU.pi = λ²·U.pi`.  First place to instrument. |
| `Grid/qcd/action/dtxqcd/DTXQCDSiteForceKernel.h` | AuxForceAt kernel — proven symmetric by FD. |
| `Grid/qcd/action/dtxqcd/DTXQCDCompositeImpl.h` lines 269-283, 308-316 | projectForce, update_field for aux.  Second suspect. |
| `Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverRationalFullAction.h` lines 550-660 | Rational deriv() — pole loop + AddFullLatticeToFull.  Third suspect. |
| `Grid/qcd/action/dtxqcd/DTXQCDRationalForceGpuKernel.h` lines 170-185 (pi), 232-245 (p), 249-257 (transpose write) | GPU kernel path; default OFF but worth checking for parity with CPU. |
| `production/gen_dtxqcd_cfgs.cc` lines 76-77, 137, 268-269 | The pRNG seed sequence.  Match this exactly. |

## Don't waste time on

- **Re-verifying the FD identity.** It holds to 12 digits at 4⁴ and 16³×48.  The bug is not there.
- **The parity transformation code (`DTXQCDParity.h`).** It's validated bit-exact (`reference_dtxqcd_parity_transform`).  Independent of this bug.
- **Alt hypothesis "long autocorrelation on (Trpi - p) mode".** At 110+ trajs with 91% accept, this doesn't fit — the SD across late-half trajs is small (2.7% of mean), i.e., well within equilibrium fluctuation.
- **Alt hypothesis "SSB direction picks Tr[π] over p".** Would need `∂²M/∂aux²` to be asymmetric, but `∂M/∂aux` is linear in both, so `∂²M = 0` and there's no higher-order term to distinguish them.

## Success criterion

You have found the bug when one of the four deriv() aggregate ratios (Gauge, AuxGaussian, LogDet, Rational) prints ≠ 1 for the (Tr[π]-diag-sum vs p) comparison.  Fix that ratio to 1 and rerun.  If a fresh λ=0.1 chain with the fix converges to `<Tr[π]> = <p> = 0` (both zero at equilibrium, since no true SSB drive), the physics interpretation is confirmed.

Good luck.
