# HANDOFF (engineering fork): per-offset reweighting factors — the direct TSRC circuit is refuted, telescope/kink instruments needed

2026-09-03.  From the scout/production session running the layer-1 reweighting
farm.  Everything below is measured at 16^3x48, lambda=5, tent profile,
CG@1e-5, MG off (see the tolerance/MG verdicts in `dtxqcd-deform-mg` memory
and commit b7adcc29's message for why those are locked).

## 1. The problem

The graded-down reweighted estimators (baryon + dibaryon) want one
reweighting factor per (config, source offset t0): translation averaging
over the 48 sources is worth up to sqrt(48) in the graded channels.  Anchors
at t0=0 are production-grade (layer-1 farm `rwf_farmA10/`, ~5-10 configs and
growing).  The per-offset RATIOS were designed to come from the TSRC
derivative circuit: d(dlogPf)/ds sampled on a full-period 2-pt-GL grid
(~96 nodes), integrated in s with a closure/bridge correction
(`DTXQCD_STOCHPF_TSRC_RATIO=12` in `Test_dtxqcd_eta_stoch_pf`).

## 2. What is now measured (the refutation)

Production-scale circuits at NT=32 noise/node exist for traj 1000 (both
arms) and partially for 1100-1400 (`rwf_farmA10/rwf_tsrc_tr*_e{1,2}_j1.log`;
e1 = eta 0.2 circuit, e2 = eta 0.4).  The NEW offline reconstruction
`production/analyze_rwf_tsrc.py` (bridge-corrected cumulative + error
propagation) gives, for traj 1000:

- closure sum rules PASS (z = 0.5 / 1.3 re, 0.2 / 0.0 im) — the machinery
  (circuit, conventions, bridge) is CORRECT end to end;
- per-offset errors: **±4-6 (eta=0.2), ±9-13 (eta=0.4)** against a target
  of ~0.5 (the anchor-scale spread);
- the bridge noise is strongly correlated across offsets (shared cumulative
  path), so offset-averaging in the final estimator does not rescue it.

Brute-force fix needs NT ~ 3,000/node => ~700 GPU-h per config-segment.
**The "48 offsets ≈ 2-3 anchors" design estimate is dead at production
volume/eta.**  Root cause: the direct circuit integrates noise over the
FULL period; the per-node derivative noise (already ~100x larger at
eta=0.6 than 0.2 — see the smoke logs) accumulates over ~48 slices.

Smoke-era data for noise-scaling diagnosis: `gridmg_setups/logs/
rwf_smoke_seg{00_02,02_04,04_06}_tr*_tsrc.log` (NT=8), `rwf_tsrck_tr*.log`
(kink-window variant), `rwf_ts07mg.log` (eta=0.7 full circuit).

## 3. What to build (your instruments, now production-critical)

The killer is the long integration path.  Both of your designed-but-untested
instruments attack exactly that:

1. **Kink-localized quadrature** (`BuildHopWeightsS` hop-form d/ds insertion:
   for the sharp tent, profp differences are square pulses supported ONLY on
   the links containing the two kinks — coefficients O(eta*p'') instead of
   O(p)).  Per-offset differences between NEIGHBORING t0 need only the local
   kink response: noise should not accumulate with |t0 - t0'|.
2. **RHOT0 difference-telescope** (`DTXQCD_STOCHPF_RHOT0`): measure
   correlated differences of nearby offsets directly (shared noise vectors
   => the same-seed cancellation that made the solver-bias triangle sharp).
   Chain short telescoped steps 0->4->8->...; variance grows linearly in the
   number of STEPS, not in the integrated path length, if each step is
   measured at fixed noise.

Fallback if both underperform: **sparse anchors** — full anchors at
t0 = 0, 12, 24, 36 (4x anchor cost, sqrt(4) of the sqrt(48) averaging) plus
interpolation only within ±6 slices.  Price before building: layer-1 anchor
cost is ~50-110 GPU-h/config (hard configs up to 2x — see traj 1300,
~2,200 iters/solve at [0.4,0.6]; size seed-jobs from each config's measured
solve rate, the layer-1 logs have them).

## 4. Acceptance gates

- Per-offset sigma <= 0.5 at <= 2x the anchor cost per segment.
- **Ground-truth gate (mandatory, cheap): run ONE full anchor at t0=24 on
  traj 1000 and compare the instrument's reconstructed shift
  w(24)-w(0) against the direct difference.**  That is the decisive
  correctness+precision test; ~1 anchor-equivalent of GPU time.
- Same-seed A/B against CG-1e-8 truth for any new solver path (the MG
  lesson: a tolerance-INDEPENDENT bias means a perturbed operator; MG
  remains excluded until its operator mismatch is found — that is a second,
  separate item on your queue).

## 5. CORRECTION (2026-09-04) — read before building anything

The premise of §1 ("t0=0-only = one source per config") is WRONG (caught by
mwagman).  The graded factor depends only on p(d3 end) − p(dbar3 end), and
the conjugated end of each correlator pair is a free choice: with ONE fixed
t0=0 contour and ONE anchor per config, EVERY same-slope pair (a,b) with
|p(a)−p(b)| = dt is graded at the full rate — ~2·(T/2−dt) pairs, i.e.
near-full translation averaging from the anchors we already have.  Measured
(`analyze_dib_graded_prod.py`, 5 configs, layer-1 anchors): dt=1 at 3.9σ;
4 configs at eta=0.6 beat the 175-config unreweighted ensemble at dt=2–3.

Consequently the per-offset instruments of §3 buy only ~2× (second
direction per pair + straddling pairs) — **DEPRIORITIZED**.  They matter
again only for eta ≥ 0.7 programs (deformed-HMC phase-only reweighting) or
if that last ~2× ever becomes the binding constraint.  The acceptance
gates in §4 stand if you do build them.

## 6. Conventions that must not drift

- CG_TOL=1e-5 (bias ladder 0.12/0.56/3.2 at 1e-4 per segment; +0.04 at 1e-5).
- Tent profile, PROF_T0 the offset, segments [0,0.2]+[0.2,0.4]+[0.4,0.6] add.
- Seeds: NOISE_SEED = 3000 + traj + 100000*seg + 10000000*seedjob (anchors);
  5000+... (tsrc).  Multi-job layers combine by inverse variance
  (`analyze_rwf_farm.py`).
- Farm layout/resume: `rwf_farmA10/` done-markers; targeted retries via
  `sbatch --array=<ids>` (never blanket resubmits).
