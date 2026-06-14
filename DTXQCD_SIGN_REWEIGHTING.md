# DTXQCD Pfaffian sign reweighting

Design note for handling the Pfaffian sign problem identified in low-λ
DTXQCD HMC ensembles. Companion to `TXQCD_ARCHITECTURE.md`.

## What HMC samples vs. what we want

The DTXQCD measure is

    dμ ∝ |Pf(K·M₄₈)|² · e^{−S_aux} dU dσ dπ dd dn ds dp

HMC samples this correctly (action is a sum of squares; PSD verified).
But **observables that depend on sign(Pf)** — anything carrying an odd
number of fermion lines through M⁻¹ in a way that doesn't square the
Pfaffian — are sampled with the wrong weight. Physical expectation:

    ⟨O⟩_phys = ⟨s · O⟩_HMC / ⟨s⟩_HMC,    s = sign(Pf(K·M₄₈))

Statistical penalty: var(⟨O⟩_phys) ∝ 1/⟨s⟩² · var(O). When ⟨s⟩ → 0
this is the classic sign problem; feasibility hinges on **measuring
⟨s⟩** on each ensemble.

## Which observables need it

| Observable                             | Sign reweight? | Reason |
|---------------------------------------|----------------|--------|
| Plaq, ⟨σ⟩, ⟨π⟩, ⟨d⟩, ⟨n⟩, ⟨s⟩, ⟨p⟩   | No             | No fermion loop; sampled correctly. |
| Σ ≡ ⟨q̄q⟩ via Hutchinson Tr M⁻¹       | **Yes**        | Odd # of M⁻¹; sign-sensitive. |
| Connected 2pt ⟨q̄q · q̄q⟩ (correlator)  | **Yes**        | Each M⁻¹ propagator carries sign info; product squares it only when q̄q at both ends are exactly identical, which is not the case for separated points in a doubled Pf measure. Treat as sign-dependent. |
| Disconnected (loops × loops)           | **Yes**        | Same reason. |
| Baryon 3-quark via doubled operator    | **Yes**        | Odd fermion line through Pf-weight. |
| Aux 2pt (σ-σ, π-π, etc.)               | No             | Pure boson; sign-blind. |

## Sign from γ5·M₄₈ spectrum

`sign(Pf(K·M₄₈)) = (−1)^{n_neg(γ5·M₄₈)}`. The Lanczos diagnostic
already committed (`9fe39fbb`) gives the **lowest K signed eigenvalues**
per traj. Two regimes:

1. **|λ|_min stays >> 0 across all trajs**: no high modes can cross
   zero either (they're far from the boundary), so parity_lowK is the
   correct global parity. λ=10 scout (this session) confirmed this.
2. **|λ|_min approaches 0**: parity_lowK still captures the crossing,
   *provided* the eval that crosses lies in the lowest K. Empirically
   λ=0.1 sign flips have happened in the lowest 2 evals. With K=6 we
   have headroom but should verify by spot-checking with a wider K at
   a thermalized configuration on each ensemble.

**Gap-from-zero indicator**: `|λ|_min` per traj is the warning signal.
Anything < ~0.5 means a sign-flip is in the cards next traj or two.

**Verification protocol (one-shot per ensemble at thermalization)**:
run `Test_dtxqcd_g5M_evals` with `N_EV=20 N_KRYLOV=80` on a saved
checkpoint to confirm the (K+1)-th eigenvalue is well-separated from
the lowest K and never crosses sign in the time history. If not,
bump default `G5M_NEV` for that λ.

## Implementation: post-processing

`production/analyze_sign_reweighting.py` (this commit) ingests
`hmc_diagnostics.*.h5` files produced by `Test_dtxqcd_2pt_gencfgs` and:

- derives per-traj `s_n = (−1)^{n_neg(g5M_evals[n])}`
- computes `⟨s⟩ ± σ` and effective sample size N_eff = N·⟨s⟩²
- writes a `signs.<traj>.h5` sidecar with per-traj signs ready to be
  multiplied into any downstream observable
- prints a sign-reweighting summary: ⟨s⟩, σ(s), histogram of |λ|_min,
  flag if any |λ|_min < `MIN_LAM_WARN` (default 0.5)

For a measured observable `O_n` (in a separate h5/npz), the reweighted
mean is `mean(s·O) / mean(s)` with errors via jackknife or bootstrap
over the joint (s, O) ensemble.

## Diagnostic & decision plan

1. **Spot-check each existing ensemble**: run the verification protocol
   on the latest checkpoint of each λ stream. Confirm K=6 is enough.
2. **Measure ⟨s⟩**: rerun gencfgs *from the current checkpoint forward*
   on each existing ensemble with the new diagnostic on, accumulating
   ~50–100 trajs. Pure observation cost is the Lanczos overhead
   (~10–15%).
3. **Triage by ⟨s⟩**:
   - ⟨s⟩ ≥ 0.9 → reweight directly; stat penalty tiny.
   - 0.5 ≤ ⟨s⟩ < 0.9 → reweight, expect ~2–4× stat loss; usable.
   - 0.1 ≤ ⟨s⟩ < 0.5 → reweighting feasible but 10–100× stat loss;
     consider Hasenbusch-style sign-protection (frozen-sign window) or
     bigger ensemble.
   - ⟨s⟩ < 0.1 → effectively a hard sign problem at this λ. Options
     below.
4. **Reapply to Σ_DTXQCD vs Σ_bare**: this is what motivated the
   investigation. Reweighted Σ should restore Fierz at small λ if
   the sign problem is the whole story.

## Fallback strategies if ⟨s⟩ is too small

1. **Restrict comparison to "safe" λ**: report Fierz at λ ≥ 1 where
   sign problem is mild; document the small-λ limit as algorithmically
   inaccessible without a sign-circumventing method.
2. **Sign-quenched (biased) Σ for small-λ**: report `⟨s·O⟩_HMC` (no
   denominator) as the sign-quenched approximation. Useful as a
   plausibility check but not a physics result.
3. **Reflection-positivity argument**: investigate whether DTXQCD has
   an additional discrete symmetry that would force `s_n ≥ 0` in the
   thermodynamic limit. (Open question; out of scope for this note.)
4. **Operator-level sign reduction**: parameterize aux fields to push
   the spectrum further from zero (e.g., constrained-aux HMC). Adds
   bias unless carefully done.

## Concrete next steps (ordered)

- [ ] Run verification protocol on `configs_2pt_dtxqcd_v2_lam0.1_*`
      and `..._lam5_*` / `..._lam10_*` checkpoints — confirm K=6
      sufficient or bump.
- [ ] Resume each existing λ ensemble for ~50 trajs with γ5M tracking
      to measure ⟨s⟩.
- [ ] Apply `analyze_sign_reweighting.py` on collected h5; produce
      ⟨s⟩ vs λ table.
- [ ] Reweight Σ_DTXQCD measurements and compare against Σ_bare —
      test of the hypothesis that the sign problem is the whole story.
- [ ] If Fierz is restored at, e.g., λ ≥ 1: write up and call it done
      for the accessible λ window. If even reweighted Σ doesn't match
      Σ_bare at some λ, there's a second source of Fierz violation
      and we'll need a deeper diagnostic.
