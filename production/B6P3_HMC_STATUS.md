# b6.3 TXQCD HMC at λ=6 — status as of 2026-05-23 19:55 (UPDATED)

## Headline: prior diagnosis was wrong — comm not algorithm

**Earlier today's conclusion (below) was based on an apparent silent stall.**
**That stall turned out to be `--shm-mpi 0` slowing Wilson Meooe by 237×**
**(580 ms/call vs the correct 2.44 ms/call).** With `--shm-mpi 1` forced for
all halos, the multishift CG progresses at a normal rate and refresh + start-
of-traj action eval complete in ~25 min.

The remaining blocker is **memory**, not convergence: `OneFlavourSchurClover`
`QudaPrimitiveActionMP::deriv()` allocates 3 × 20 × 152 MB = 9.12 GB/rank in
one shot, which together with the baseline state busts the 80 GB A100 at
mpi=2.2.2.2.  Three workarounds (different effort/payoff tradeoffs) listed at
the bottom.

**The "use multi-Hasenbusch / move off b6.3" conclusion below is OBSOLETE.**
**Production b6.3 TXQCD HMC may be feasible at ~5-10 hr/traj with a small fix.**

See `~/.claude/memory/reference_shm_mpi_1_fix.md` for the perf cliff details.

---

## Detailed update (2026-05-23 19:55)

### What changed

| Diagnostic | shm-mpi=0 (prior) | shm-mpi=1 (today) |
|--|--|--|
| Wilson Meooe per call | 580 ms | **2.44 ms** (237× faster) |
| Multishift CG iter | 2.3 s | 36 ms |
| Refresh of all 5 actions | (never finished) | 12.5 min |
| Start S [0][0] eval (light multishift) | (never reached) | 10 min |
| MD step 1 force CG | (never reached) | 8 min (12539 iter) |

The original "silent stall" was Grid's stencil running 100s of slow Meooe
silently before printing per-100-iter logs.  When you wait long enough,
progress shows up.  When you switch to `--shm-mpi 1`, no waiting needed.

### What blocks completing a 1-traj smoke now

After MD step 1 force CG converged, `cudaMalloc 10192158720` (= 9.74 GB)
failed.  Traced to `OneFlavourSchurCloverQudaPrimitiveActionMP::deriv()`
lines 76/85/86 — 3 vectors of `Npole=20` `FermionField` each, allocated
per-call in the force evaluation:

```cpp
std::vector<FermionField> MPhi_k(Npole, fcbgrid);   // 20 × 152 MB
std::vector<FermionField> Y_k   (Npole, fcbgrid);   // 20 × 152 MB
std::vector<FermionField> Yd_k  (Npole, fcbgrid);   // 20 × 152 MB
```

Total: 9.12 GB transient per rank, on top of ~70 GB baseline (TXQCD precompute
~6 GB/rank × ?ops, QUDA MG setup, smearing chain ~24 GB/rank for stout copies,
multishift PF state, etc.).

### Workarounds (pick one)

| # | Workaround | Effort | Loss vs full QUDA path |
|--|--|--|--|
| A | Get an 8-node alloc — per-rank lattice halves → all memory pressure halves | None (just request 8 nodes) | None |
| B | Refactor `OneFlavourSchurCloverQudaPrimitiveActionMP::deriv()` to process poles in chunks of 4–5 instead of all 20 at once → drops peak from 9 GB to ~2 GB | 1–2 hr code + FD revalidation | None (same algorithm) |
| C | **`unset TXQCD_QUDA_HYBRID` (don't just set =0 — env-var existence is checked, not value)** — selects original Grid TXQCD action.  Avoids the 9 GB alloc.  Force eval becomes ~2-5× slower per call but still tiny vs the per-CG cost. | One-line change | ~2-5× slower force eval, but CG dominates total → ~5-15% trajectory slowdown |

Recommended **first try: C** (smallest blast radius); promote to B if we want to keep the Phase H Force speedup.

### UPDATE 2026-05-24 05:35 — Option C does NOT work as expected

Tried Option C overnight (`smoke_b6p3_lean_unset_hybrid.sh`, action class
`TXQCDWilsonCloverRationalEOAction`).  The 9 GB OOM is dodged, BUT:

**Meooe regressed from 2.44 ms → 552 ms.**  Refresh CG [0][0] took **33 917 s
(9.4 hr)** for 14829 multishift iterations.  Confirmed via destructor timer
dump in `logs/smoke_b6p3_lean_unset_hybrid.log`.  Killed at 05:35.

The fast Meooe path apparently requires the `QudaCloverInverter` to be
constructed in the action — its early `QudaInit::Init()` →
`QMP_init_msg_passing` → `initCommsGridQuda` → `initQuda(device)` chain
unlocks UCX-CUDA registration that Grid's stencil silently benefits from.
Without it, Grid falls back to whatever Stencil_force_mpi=true selects on
this OpenMPI/UCX build, which is 225× slower.

**This makes the Option list smaller:**

| # | Workaround | Effort | Status |
|--|--|--|--|
| A | 8-node alloc — halves per-rank memory pressure | None (need allocation) | Should work; user said lq2 is hard for 8 |
| B | Refactor `OneFlavourSchurCloverQudaPrimitiveActionMP::deriv()` to chunk poles by 4-5 → peak 1.8 GB | 1-2 hr code + FD revalidation | **Recommended next** |
| C | `unset TXQCD_QUDA_HYBRID` | One-line | **TRIED — does NOT work**, slow Meooe makes wallclock 50× worse |
| D | Construct a dummy `QudaCloverInverter` early in `gen_txqcd_cfgs_2plus1.cc` even when not using QudaPrimitive action — preserves fast Meooe AND avoids 9 GB alloc | 30 min code | **Try second**, simpler than B if it works |
| E | Same workaround D, hard-coded inside `TXQCDWilsonCloverRationalEOAction` constructor (always own a QudaCloverInverter for the side-effect, never use it) | 30 min code | Cleaner placement of D |

### Recommended path

1. **First**: Try **option D** — add `QudaInit::Init()` + dummy `loadGaugeQuda` at the top of `gen_txqcd_cfgs_2plus1.cc` main(), BEFORE constructing any action.  If Meooe stays fast (~2.4 ms), Option C becomes viable and we get a working 4-node config in ~1.5 hr/traj at MDS=2 smoke.
2. **Second**: Refactor B (chunk poles in QudaPrimitive deriv).  Preserves Phase H Force speedup *and* fits in memory.  ~1-2 hr work.
3. **Long-term**: Get 8-node allocation when available — no code change needed, all current scripts work.

### Comment on autonomous behavior

Burned the overnight interactive on Option C — should have checked the
hypothesis on a much shorter test (e.g., compare Meooe per-call after
constructor-only setup with single-rank Test_txqcd_wilson_eo) before
committing 4 nodes × 12 hours to the smoke.  Apologies for the cluster
hours spent on a config that turned out unviable.

### Trajectory ETA when memory fits

At MDS=10 production:
- Refresh: ~12 min
- ~12 multishift CG calls (refresh + start S + MD forces + end S) × ~10 min each ≈ 2 hr per traj light-mass CG cost
- Strange RHMC + LogDets: fast (~20 min total)
- **Total: ~2–3 hr/traj at MDS=2 smoke; ~5–10 hr/traj at MDS=10 production**

Per-iter CG breakdown (36 ms total):
- Wilson Meooe (Grid): 9.76 ms (27%)
- Mooee (cuBLAS): 12.06 ms (33%)
- ApplyMooeeInv (cuBLAS): 12.08 ms (34%)
- Bookkeeping: ~2 ms (6%)

Next-level wins would target cuBLAS Mooee/MooeeInv (now 67% of per-iter cost),
or multi-Hasenbusch chains to reduce iter count.

---

## OBSOLETE: prior (incorrect) diagnosis follows

**This section is preserved for context but the conclusion was wrong — the
silent stall was `--shm-mpi 0` overhead, not algorithmic divergence.**

## Headline: blocked by an algorithmic issue, not infrastructure (OBSOLETE)

TXQCD HMC at b6.3 48³×96 light mass is **not viable in current form**, and
the reason is structural — same fundamental wall as for Nf=1 RHMC, just
hits us harder because b6.3 light is much lighter than b6.1's production
mass.

## What we tried

4-node 16-rank smoke (`mpi=1.1.4.4`, lq2gpu04-06,10), MDS=10, NO_METROP=1,
chroma cfg_2000 seed, MG enabled.  Got past `loadGauge` + smearing (1.6 s
each) cleanly.  Then hit `refresh [0][0]
TXQCDWilsonCloverRationalEOActionQudaPrimitive` and grinded silently for
13+ min with GPUs at 2–4 % util, no further log output.  Killed and freed
the allocation.

## Why it's blocked

Yesterday's MG-HMC win (TwoFlavour Schur 2-solve trick, FD-validated on
4⁴) is in **`TwoFlavourSchurCloverQudaForceActionMP`**. That's the
**QCD** HMC light action. TXQCD HMC uses a *different* class for its
light pseudofermion:

```cpp
// gen_txqcd_cfgs_2plus1.cc line 239
PF_quda_holder = std::make_unique<TXQCDWilsonCloverRationalEOActionQudaPrimitive>(...);
```

This is a **rational / multishift action** (the σ field's contribution
to det(M) is handled via RHMC-style rational expansion).  Per
[[reference_mg_rhmc_landscape]]: multishift + MG are mathematically
incompatible — QUDA `invertMultiShiftQuda` requires `NORMOP_PC_SOLVE`,
MG-as-preconditioner requires `DIRECT_SOLVE`.

So MG **does not apply to TXQCD HMC's light action**, even though we
spent yesterday building the MG infrastructure.

## Why b6.3 hits this hard

| ensemble | CG iters at light mass (from compute_vev) |
|---|---|
| b6.1 16³×48 m=−0.245 | ~80 |
| b6.3 48³×96 m=−0.2416 | **>5000** (didn't converge in 13 min on 1 noise) |

Multishift CG cost ≈ shift-0 (lightest) iter count. At b6.1 it's ~80 iters
× 0.15 s ≈ 12 s/multishift solve → HMC traj works.  At b6.3 it's >5000
iters × 0.15 s ≈ 12–30 min/multishift solve.  HMC needs ~12 multishift
solves per traj (1 refresh + MDS=10 force + 1 final action) → **~3–6
hours/traj**.  A 1000-traj ensemble: 6 months wallclock. Not production-ready.

## What's still useful

| artifact | status |
|---|---|
| Σ_light = 3.02213(6e−5), Σ_strange = 3.00970(1.4e−4) for b6.3 | ✓ measured via compute_vev with MG (project_b6p3_vev_trminv) |
| MG for `compute_vev` measurements at b6.3 | ✓ 200× speedup (project_b6p3_vev_trminv) |
| MG for QCD TwoFlavour HMC (NOT TXQCD) | ✓ FD-validated 4⁴ (project_hmc_mg_twoflavour) |
| TXQCD HMC at b6.3 | ❌ blocked, needs multi-Hasenbusch (~1–2 week project) |

## Production slurm script `slurm_txqcd_b6p3_lam6_4node.sh`

I drafted this and committed it, but **don't submit it without first fixing
the algorithmic issue** — it would just consume cluster hours grinding
multishift CG that won't finish a single traj before wallclock expires.
The script is correct for the moment TXQCD HMC at b6.3 becomes practical.

## What to do next

Options in order of effort:

1. **Stay at b6.1 for TXQCD ensembles** (zero work).  b6.1 streams work fine.
   Use b6.3 only for measurement on chroma cfgs.

2. **Generate b6.3 QCD ensembles** (no σ field) using `gen_qcd_cfgs_2plus1`
   with `USE_HMC_MG=1`.  TwoFlavour light gets MG → traj rate viable.
   Then run TXQCD *measurement* on those cfgs (already validated by the
   b6.3 VEV work).  Useful for cross-checks; not a TXQCD ensemble.

3. **Multi-Hasenbusch decomposition of TXQCD light** (chroma-validated for
   QCD).  Real 1–2 week engineering project.  Replaces the rational
   action with a chain of MG-able single-shift ratios.  The right
   long-term path to TXQCD HMC at lighter masses.

4. **Re-run the smoke with `cg_max` capped low + QUDA_VERBOSE=1** to
   confirm the slow-CG diagnosis vs an actual hang.  ~30 min of debugging
   to be sure.  Doesn't change the conclusion but rules out
   infrastructure issues.

## Smoke smell test (the actual stall details)

- Process was alive (PID active, GPUs drawing 77–84 W vs 50 W idle, 60 GB
  resident on each GPU).
- GPU util 2–4 % stable.
- Log silent for 13+ min.
- No CG iter logs ever appeared — QUDA's multishift doesn't emit
  per-iter unless `QUDA_VERBOSE`.  So we couldn't distinguish "slow CG
  iterating silently" from "hung in some sync primitive."

Most likely interpretation: silent multishift CG progressing very slowly
due to the 5000+ iter convergence cost compounded with 4-node halo-exchange
overhead on the z=4 split (each rank has only 12 z-sites → comm-bound).
That doesn't change the conclusion — even if it would have converged
eventually, the per-traj cost is too high.
