# HMC tuning lessons — vanilla Wilson-Clover Nf=3 (deg.) on cl3_16_48_b6p1_m0p2450

Notes from a multi-week tuning effort on the 16³×48 ensemble (`cl3_16_48_b6p1_m0p2450`,
clover Wilson `m=-0.2450`, `csw=1.249`, β=6.1, `u0=0.832605`, stout 1× ρ=0.125,
antiperiodic time BC) using both Grid (this fork) and chroma `hmc` binary, weak-field
and chroma-equilibrium starts.  The chroma equilibrium plaq on this ensemble is
**0.5138**.  Audience: someone tuning vanilla QCD HMC performance from scratch.

## 1. Action structure matters more than integrator choice

Two equivalent ways to write `det(M)^3` for 3 degenerate light flavors:

- **Nf=2+1**: 1× `TwoFlavour(M)` (= `det(M_pc M_pc†) = |det(M_pc)|²`) + 1× `OneFlavourRational(M)` (= `|det(M_pc)|^1`).  Total: `|det(M_pc)|³`.
- **Nf=1+1+1 / Nf=3**: 1× `LogDet(num_flavors=3)` + 1× `OneFlavourRational(M, num_pf=3)` (3 independent PFs, each contributing a `|det(M_pc)|¹` factor stochastically).  Total: `|det(M_pc)|³`.

These have the same Boltzmann measure but **dramatically different integrator stability** on this ensemble.  We confirmed in **both Grid and chroma** that:

- **Nf=2+1** chains thermalize 3-5 trajectories from weak-field then **catastrophically blow up** (dH ~ 10³–10⁴ at chroma MDs=10, dH ~ +60-300 at chroma MDs=7, sustained rejection).  Chains lock into a metastable basin near plaq ≈ 0.530–0.575 depending on eps.
- **Nf=3** chains (chroma's exact `.lime` action structure: `1× N_FLAVOR_LOGDET num_flavors=3 + 1× ONE_FLAVOR_RAT num_pf=3 + LW gauge`) thermalize **smoothly** through 0.530 and reach plaq ≈ 0.519 by trajectory 50.

**Why:** the Nf=2+1 light pair uses Schur EO preconditioning where the force is `M_pc⁻¹ × derivative`.  Near-zero modes of M_pc get amplified as `1/eigenvalue` in the force, producing huge spikes whenever the chain encounters a stiff configuration.  Rational HMC bounds the force inside `Σ_k α_k (M†M+β_k)⁻¹`, capping per-mode amplification at `1/√β_min`.  The bound matters at the basin boundary.

**Recommendation**: even if your action is two-flavor, never use `TwoFlavourSchurClover` (or chroma's `TWO_FLAVOR_EOPREC_CONSTDET`) for the light pair on a thermalization run from cold/weak-field.  Use Nf=2 written as 2× rational PFs instead, or equivalently `num_pf=2` in chroma.  The cost overhead is minimal and the integrator stability is qualitatively better.

This is the single biggest factor we identified.  See [project memory](memory/project_nf2p1_vs_nf3_basin.md) for more details.

## 2. Integrator step-size convention: Grid eps = chroma dt / 4

A factor of 4 enters from σ(P) and kinetic-mass conventions:

- Grid samples momenta with `Var(P_grid)=2`, has effective `m_grid=2`, so `dU = (P/m)·eps` evolves U with effective rate `√2/2` per unit eps.
- Chroma samples with `Var(P_chroma)=1/2`, has effective `m_chroma=1/2`, so dU evolves with rate `√2` per unit dt.
- Ratio: 4×.

**Empirical**: chroma MDs=7 trajL=√2 (dt=0.202) ≡ Grid MDs=28 trajL=√2 (eps=0.0505).  Same physical evolution.

Practical: when porting integrator settings between codes, divide chroma's dt by 4 to get the equivalent Grid eps.  Do NOT divide trajL — that's a physical quantity both codes use the same way.

## 3. Stable eps boundary is ~0.05 for this ensemble at MN2

For Nf=3 / Nf=1+1+1 from weak field (16³×48, MN2 with multi-rate gauge sub-integrator
n_steps=4, λ=0.1789):

| eps | Grid MDs | chroma MDs | thermalization status |
|-----|----------|------------|----------------------|
| 0.0707 | 5 | 10 (with trajL=√2/4)  | unstable in some streams |
| 0.0505 | 7 | 7 | stable for chroma; marginal for Grid |
| 0.024  | 15 | — | stable |
| 0.012  | 30 | — | stable + best wallclock per useful traj |
| 0.0088 | 40 | — | stable + slower |

**dH scales as O(eps²) cleanly across this range** — verified to ~1.5% on `dH/eps²` constant.  MN2 is doing what it should.

For PRODUCTION, MDs=30 trajL=√2/4 (Grid) or equivalently MDs=8 trajL=√2 (chroma) is the sweet spot: deep into the stable region, ~25 useful trajs/hour.  Below MDs=15 (Grid), the integrator hits an instability that locks chains in metastable basins after a few trajectories.

## 4. Stuck-basin trap near plaq ≈ 0.530 (Nf=2+1 only)

The **Nf=2+1 action specifically** has a metastable basin near plaq=0.530.  Once the chain enters it:
- dH starts oscillating ±1-2 around zero (rejection-heavy)
- A few trajectories later, integrator finds a near-zero mode and **dH explodes** to >10³.
- Chain freezes (plaq exact-bit-identical for many trajectories).

The wall **does not exist** for the Nf=3 / Nf=1+1+1 action structure — chains pass through 0.530 cleanly with dH ~ −0.1.

If you must use Nf=2+1 (e.g. for a comparison test), the workaround is **smaller eps**:
- MDs=15 (Grid) blows up at traj 30-40 with dH=+613 → +2700.
- MDs=20 (Grid) limps through with rejection-heavy dH=+1-4 oscillation.
- MDs=30 (Grid) integrates through cleanly.

## 5. Multi-rate integrator ratio: gauge sub-steps = 4

Both codes' default for this ensemble: gauge sub-integrator does 4× the steps of the fermion-force level (chroma `<n_steps>4</n_steps>` in the SubIntegrator block; Grid `GAUGE_INNER_MULT=4`).  This is consistent with the gauge force being ~4× cheaper per eval than the fermion force, so amortizing over more sub-steps pays off.

Don't try AUX/gauge multiplier > 8 without checking dH stability: the inner integrator can introduce its own instabilities at very small inner-eps.

## 6. Rational expansion bounds: cover the M†M spectrum

For chroma's `OneFlavourRational` and Grid's `OneFlavourSchurCloverRationalActionMP`,
chroma's reference uses `(lo=0.0001, hi=32, degree=15)` for the `cl3_16_48_b6p1_m0p2450` ensemble at m=−0.245.

We diagnosed that **hi=32 is too small for chroma-equilibrium configurations** (where λ_max(M†M) ≈ 60-80).  Multishift CG inside the rational PF refresh can fail to converge when modes outside the design band exist.  For weak-field starts the spectrum is narrower and hi=32 works initially; once the chain thermalizes and λ_max grows past 32, refresh starts hanging.

**Recommendation**: bump to `(1e-4, 100, deg=20)` or similar to cover λ_max comfortably.  Cost: ~5% per multishift solve (more poles, but iteration count is set by `√κ(M†M)` not by degree).  Big stability win.

To diagnose: write a small `eigspec_diag.cc` that runs a Lanczos-with-Chebyshev-filter on M†M for a handful of cfgs.  Look at λ_max and the few lowest λ.  If λ_max > rational_hi, raise the bound.

## 7. Mixed-precision multishift CG

Chroma's `MULTI_CG_QUDA_CLOVER_INVERTER` is the production solver — heavily-optimized
mixed precision with QUDA backends.  Grid's analogue is `OneFlavourSchurCloverRationalActionMP`
(see `gen_qcd_cfgs.cc`).

Speed: chroma+QUDA's full multishift is dramatically faster than Grid's MP CG.  We measured ~50× per-trajectory wallclock for the same problem (16³×48, Nf=2+1 MN2 MDs=7-15, weak-field).  For QCD-only production, prefer chroma.

For correctness verification: run with `RsdCG=1e-9` and `MaxCG=10000`.  If you see iterations approaching MaxCG with residual not converging, your rational bounds (point 6) probably miss part of the spectrum.

## 8. Stout smearing convention

Grid stout default in `Smear_Stout` ρ=0.125 1-smear matches chroma's `STOUT_FERM_STATE` ρ=0.125 1-smear with `orthog_dir=-1` (4D isotropic).  No conversion needed.

**One subtlety**: chroma's STOUT_FERM_STATE smears ONLY for the fermion action (not the gauge action).  The LW_TREE_GAUGEACT operates on thin (unsmeared) links.  Grid's `is_smeared=false` flag on gauge actions reproduces this.  We had a bug here for several weeks where Grid's gauge action used smeared links — cost us ~0.001 in equilibrium plaq.

## 9. Antiperiodic time BC

Chroma's `<boundary>1 1 1 -1</boundary>` for fermions == Grid's `WilsonImplParams::boundary_phases = {1, 1, 1, -1}`.  Always set this for QCD.  Grid's default is all-periodic; forgetting to override gives the wrong det(M) by O(volume) in time, shifting equilibrium plaq.

## 10. dH/eps² as a per-stream sanity check

Once your chain is thermalized (dH centered near zero), compute `dH / eps²` per trajectory.  This should be approximately constant per ensemble (~6800 on this lattice, both codes, all MDs).  Drift in this number signals either:
- Integrator instability (eps too large)
- Force-vs-action inconsistency (numerical issue in the rational/CG solvers)
- Action-structure issue (e.g. M_pc⁻¹ amplification firing)

It's a cheap, sensitive diagnostic.

## 11. Eigenvalue spectrum diagnostics

Routinely measuring `λ_min(M†M)`, `λ_max(M†M)`, `|γ5M|_min` on a representative cfg
catches problems early:

- If `λ_max > rat_hi` → rational bounds are too narrow (point 6).
- If `|γ5M|_min` is small (~10⁻³ on chroma equilibrium for this ensemble vs ~0.1 typical) → near-zero modes that will bite the integrator.
- If `κ(M†M) > 10⁵` → CG iteration counts will be excessive; consider Hasenbusch preconditioning.

We use Chebyshev-filtered Lanczos via `eigspec_diag.cc`.  Order ~31 with Cheby band `(0.05, 120)` resolves the lowest 5 eigenvalues to ~1e-4 absolute on 16³×48 in ~10 minutes wallclock per cfg.

## 12. RNG seed → multiple independent streams

Both codes seed from `<Seed>` block.  Chroma's seed is 4 ints, default `{11, 0, 0, 0}`;
Grid's is a 5-int vector.  Different seeds → different weak-field initial gauges →
truly independent chains.  Useful for ensemble averaging or thermalization race testing.

For directly comparing the two codes' dynamics from the same starting point, seeding
matters: same RNG seed but different action structures (Nf=2+1 vs Nf=3) lets you watch
the action-structure-induced divergence cleanly.

## 13. Per-trajectory wallclock (16³×48, this fork on lq2_gpu A100):

- chroma Nf=3 MDs=7 (CG_INVERTER, no QUDA): ~63 s/traj
- chroma Nf=2+1 MDs=7 (CG_INVERTER): ~23 s/traj
- chroma Nf=3 MDs=10: ~89 s/traj
- chroma Nf=2+1 MDs=10: ~37 s/traj
- Grid QCD Nf=2+1 MDs=15 trajL=√2/4 (TwoFlavourSchurMP+rational+LogDets): ~22 min/traj
- Grid QCD Nf=2+1 MDs=20 trajL=√2/4: ~28 min/traj
- Grid QCD Nf=2+1 MDs=30: ~48 min/traj
- Grid QCD Nf=2+1 MDs=40: ~63 min/traj

After accounting for the 4× factor between Grid trajL=√2/4 and chroma trajL=√2, **chroma is ~50–230× faster per unit physical evolution** than this Grid fork on this problem.  Most of the gap is QUDA-optimized clover apply/inv kernels even when chroma is configured to use `CG_INVERTER`.

For pure QCD-Nf=3 ensemble generation: use chroma.  Grid's value is for action structures chroma doesn't have (TXQCD, custom integrators).

## 14. What to ignore

We spent a lot of time on these and they were not the issue:

- **Hasenbusch preconditioning** at HASEN_DM=0.05–0.10 on the cl3_16_48 ensemble didn't qualitatively change thermalization.  The bigger lever is the action structure choice (point 1).
- **`AUX_INIT_AUTO`** for TXQCD aux fields (only relevant if you're doing TXQCD; vanilla QCD doesn't have this).
- **Force-FD test** at `CG_TOL=1e-13` shows force matches `(S(U+εp) − S(U−εp))/2ε` to 8 digits at all action structures — so action gradients are right.  The instability is genuine, not a gradient bug.

## 15. Tooling worth building first

If you're writing your own HMC infrastructure or auditing one:

1. **Plaq-vs-traj plot from slurm log** — first thing to check; lets you spot stuck-basin / blowup at a glance.
2. **dH per traj** — should oscillate near zero in equilibrium.  Trend visible at a glance.
3. **`eigspec_diag` on a representative cfg** — ~10 min of compute, gives you κ, validates rational bounds.
4. **Force breakdown per action term** — Grid prints this via `[Level Force Log]`; chroma prints `Force average`/`Force max` per monomial.  Useful for debugging which action term is generating spikes.

---

Specific to this fork (Grid-TXQCD): the Nf=3 conversion (commits `b04af352`, `76ca2a75`)
is the production action structure.  Don't run with `gen_qcd_cfgs_2plus1` /
`gen_txqcd_cfgs_2plus1` unless you're doing a comparison test.
