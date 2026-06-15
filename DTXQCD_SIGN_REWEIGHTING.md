# DTXQCD Pfaffian sign reweighting

Design note for handling the Pfaffian sign problem identified in low-λ
DTXQCD HMC ensembles. Companion to `TXQCD_ARCHITECTURE.md`.

## The physical measure has signed Pf, HMC samples |Pf|

The DTXQCD rational pseudofermion uses the Remez approximation to
`x^(−1/4)` (see `DTXQCDWilsonCloverRationalEOAction.h:75` and
`...FullAction.h:75`). The resulting fermion weight is

    w_phys ∝ Pf(K·M₄₈) · e^{−S_aux}

— signed. The 1/4 root accounts for two things at once: the Nf=2 quark
flavor degeneracy (one √), and the doubled charge-conjugate
construction (the other √, taking det(M₄₈) → Pf(K·M₄₈)). HMC needs a
positive weight, so it samples

    w_HMC ∝ |Pf(K·M₄₈)| · e^{−S_aux}.

The mismatch is the *signed* Pfaffian. For **any** observable O:

    ⟨O⟩_phys = ⟨signPf · O⟩_HMC / ⟨signPf⟩_HMC,
        signPf ≡ sign(Pf(K·M₄₈)) = (−1)^{n_neg(γ5·M₄₈)}

Statistical penalty: var(⟨O⟩_phys) ∝ 1/⟨signPf⟩² · var(O). When
⟨signPf⟩ → 0 this is the classic sign problem; feasibility hinges on
**measuring ⟨signPf⟩** on each ensemble.

## Which observables need reweighting

**All of them.** This was not what the earlier draft of this note said,
but it is correct: HMC literally samples a different measure, and the
reweighting `⟨signPf·O⟩/⟨signPf⟩` is the unbiased estimator no matter
what O is — gauge, aux, or fermionic.

Observed empirical pattern at λ=0.1:

| Observable                            | Sensitivity to signPf flips                                    |
|---------------------------------------|----------------------------------------------------------------|
| `⟨π_ab⟩`, `⟨p⟩`, `⟨Trπ⟩` (parity-odd) | **HIGH** — symmetric across signPf sectors, so finite-N HMC bias is visible immediately. ⟨Trπ⟩ jumped 0.16 in one step at the λ=0.1 traj-130 flip. |
| `⟨σ_ab⟩`, `⟨s⟩`, `⟨Trσ⟩` (parity-even) | LOW — sectors give the same value (parity-blind), so even imbalanced sampling converges. Still wrong at finite N. |
| Plaq, gauge action observables        | LOW — gauge sector is parity-blind given symmetric aux. |
| Σ = ⟨q̄q⟩ via Tr M⁻¹                  | HIGH — fermion-line observable; biased by sign sampling. |
| Connected pion/nucleon 2pt            | HIGH — every propagator carries the sign.              |
| Disconnected (loop × loop)            | HIGH                                                   |
| Doubled-baryon 3-quark                | HIGH                                                   |

The right policy: **reweight everything** with the same `signPf` array.
Parity-even observables will look the same to within statistical
fluctuations (consistency check on the sign tracker); parity-odd and
fermion-line observables are where the correction actually matters.

## signPf from γ5·M₄₈ spectrum

`signPf = (−1)^{n_neg(γ5·M₄₈)}`. The Lanczos diagnostic committed in
`9fe39fbb` returns the **lowest K signed eigenvalues** per traj. Two
regimes:

1. **|λ|_min stays >> 0 across all trajs**: no high mode can cross
   zero either (they're far from the boundary), so `parity_lowK` is
   the correct global parity. λ=10 scout (this session) confirmed:
   30 trajs, |λ|_min ∈ [0.78, 1.18], no flips, ⟨signPf⟩ = −1 exactly.
2. **|λ|_min approaches 0**: parity_lowK still captures the crossing,
   *provided* the eval that crosses lies in the lowest K. Empirically
   λ=0.1 flips have happened in the lowest 2 evals. K=6 has headroom
   but should be spot-checked.

**Gap-from-zero indicator**: `|λ|_min` per traj is the warning signal.
< ~0.5 means a sign-flip is in the cards within a few trajs.

**Verification protocol (one-shot per ensemble at thermalization)**:
run `Test_dtxqcd_g5M_evals` with `N_EV=20 N_KRYLOV=80` on a saved
checkpoint to confirm the (K+1)-th eigenvalue is well-separated from
the lowest K and never crosses sign during the run. Bump default
`G5M_NEV` for that λ if not.

## Prerequisite: is Pf actually real?

`signPf = (−1)^{n_neg}` is only meaningful if Pf(K·M₄₈) is real-valued
(so it has a well-defined sign). That follows from:
1. `(K·M₄₈)^T = −(K·M₄₈)` — the Pfaffian antisymmetry. Validated by
   `Test_dtxqcd_pfaffian_antisymmetry` at the single-site level after
   the 2026-06-13 real-symmetric aux fix.
2. `γ5·M₄₈·γ5 = M₄₈†` — γ5-Hermiticity, implies γ5·M₄₈ is Hermitian,
   so det(γ5·M₄₈) is real, so det(M₄₈) is real, so Pf = ±√det is real.

`Test_dtxqcd_pf_realness` (added this commit) runs check (2) on a
loaded configuration — a stochastic γ5-Hermiticity probe
`‖(γ5·M − M†·γ5)·η‖ / ‖M·η‖` — and reports the violation. If this is
clean (< 1e-12) on the ensembles we care about, sign reweighting is
well-defined. If it's not clean (e.g. on the pre-fix hermFULL configs),
the reweighting setup itself is the wrong object — full complex
reweighting `Pf/|Pf|` would be needed.

## Implementation: post-processing

`production/analyze_sign_reweighting.py` ingests
`hmc_diagnostics.*.h5` files produced by `Test_dtxqcd_2pt_gencfgs` and:

- derives per-traj `signPf_n = (−1)^{n_neg(g5M_evals[n])}`
- computes `⟨signPf⟩ ± σ` and effective sample size N_eff = N·⟨signPf⟩²
- writes a `signs.<traj>.h5` sidecar with per-traj signPf values ready
  to be multiplied into any downstream observable
- prints a reweighting summary: ⟨signPf⟩, σ(signPf), histogram of
  |λ|_min, flags trajs near the sign boundary

For a measured observable `O_n` (stored in a separate h5/npz), the
reweighted mean is

    mean(signPf · O) / mean(signPf)

with errors via jackknife or bootstrap over the joint ensemble.

## Diagnostic & decision plan

1. **Verify Pf realness** on each existing ensemble's latest cfg
   (and on a "gnarly" λ=0.1 cfg near a sign flip): run
   `Test_dtxqcd_pf_realness`. If clean, proceed; if not, treat the
   hermFULL data as suspect and re-generate post-fix.
2. **Spot-check K=6 sufficiency** with `Test_dtxqcd_g5M_evals N_EV=20`
   on the same checkpoints.
3. **Measure ⟨signPf⟩**: resume each existing λ stream for ~50–100
   trajs with γ5M tracking on (no other run cost change).
4. **Triage by ⟨signPf⟩**:
   - ≥ 0.9 → reweight everything; stat penalty negligible.
   - 0.5 ≤ |⟨signPf⟩| < 0.9 → reweight; expect 2–4× stat loss.
   - 0.1 ≤ |⟨signPf⟩| < 0.5 → reweighting feasible but 10–100× stat
     loss; consider longer runs.
   - < 0.1 → effectively a hard sign problem at this λ.
5. **Reapply to Σ_DTXQCD vs Σ_bare** at every λ. This is the original
   motivating test: if Pf-real and sign-reweighting is the whole story,
   reweighted Σ should restore Fierz at small λ.

## Fallback strategies if ⟨signPf⟩ is too small

1. **Restrict comparison to "safe" λ**: report Fierz at λ ≥ 1 where
   the sign problem is mild; document the small-λ regime as
   algorithmically inaccessible without a sign-circumventing method.
2. **Sign-quenched (biased) report**: `⟨signPf·O⟩_HMC` (no
   denominator) is the sign-quenched approximation. Useful as a
   plausibility check, not a physics result.
3. **Reflection-positivity argument**: investigate whether DTXQCD has
   an additional discrete symmetry that would force `signPf_n ≥ 0` in
   the thermodynamic limit. (Open question; out of scope here.)
4. **Constrained-aux HMC**: parameterize aux fields to push the
   spectrum further from zero. Biased unless carefully constructed.

## Concrete next steps (ordered)

- [ ] Run `Test_dtxqcd_pf_realness` on `hermFULL/ckpoint_lat.{120,130}`
      to verify Pf is real on the existing λ=0.1 sign-flip data.
- [ ] If clean: run `Test_dtxqcd_g5M_evals N_EV=20` on the same trajs
      to confirm K=6 is enough.
- [ ] Resume each existing λ ensemble for ~50 trajs with γ5M tracking
      to measure ⟨signPf⟩.
- [ ] Apply `analyze_sign_reweighting.py`; produce ⟨signPf⟩ vs λ table.
- [ ] Reweight Σ_DTXQCD measurements and compare against Σ_bare —
      direct test of the hypothesis that the sign problem is the
      whole story.
