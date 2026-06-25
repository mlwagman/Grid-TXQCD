# Easy perf wins shipped — DTXQCD

Session date: 2026-06-24
Scope: DTXQCD HMC production scripts on 16³×48 b6.1 (lq2 / lq2gpu13).
Out of scope: TXQCD-aware multishift CG in QUDA (real research project).

## TL;DR

- DTXQCD's canonical runner `production/run_dtxqcd_gencfgs.sh` already
  defaults-on the full Phase H + GPU + cuBLAS + MP-CG stack. **No new
  flag flips required.**
- `DTXQCD_QUDA_FULL=1` (Phase H.1 σ-clover via batched QUDA
  computeCloverSigmaOprod, shipped 2026-06-24 in commit 6b08a6b0) is wired
  but stays off pending production validation per the commit message
  (4⁴ FD bit-exact, but no 16³×48 dH smoke gate yet). Not flipped this
  session.
- **DTXQCD has no Hasenbusch wiring**. The TXQCD ladder pattern in
  `gen_txqcd_cfgs_2plus1.cc` lines 461-604 plus class
  `TXQCDWilsonCloverHasenbuschAction` could in principle be ported to
  `DTXQCDWilsonCloverHasenbuschAction` + analogous integrator-level wiring
  in `gen_dtxqcd_cfgs.cc`. **NOT done this session** — see Section 4.
- cudaMemcpy storm hunt: same conclusion as TXQCD (BatchedBlas already
  patched, nothing else obvious without nsys ground truth).
- Phase I revival: NOT recommended per profile data — see Section 6.

## 1. Env vars audit

`production/run_dtxqcd_gencfgs.sh` is well-tuned. All these are default-on
in the runner (override-able via env):

| Flag | Default | Wired? |
|---|---|---|
| `DTXQCD_PRECOMPUTE_GPU=1` | yes | yes |
| `DTXQCD_LOGDET_S_GPU=1`   | yes | yes |
| `DTXQCD_LOGDET_GPU=1`     | yes | yes |
| `DTXQCD_MOOEEINV_CUBLAS=1`| yes | yes |
| `DTXQCD_MOOEE_CUBLAS=1`   | yes | yes |
| `DTXQCD_MOOEE_FWDCACHE=1` | yes | yes |
| `DTXQCD_RATFORCE_GPU=1`   | yes | yes |
| `DTXQCD_MP_CG=1`          | yes | yes |
| `QUDA_FORCE=1`            | yes | yes |
| `QUDA_FORCE_KERNEL=1`     | yes | yes |
| `DTXQCD_QUDA_HYBRID=1`    | yes | yes |
| `AUX_INIT_AUTO=1`         | yes | yes |
| `RAT_AUTO_HI=1`           | yes | yes |

These are conditional / opt-in:

- `DTXQCD_QUDA_FULL`: empty default (opt-in). Shipped in commit 6b08a6b0
  (Phase H.1). FD-validated on 4⁴, awaiting production smoke gate. Leave
  off this session.
- `ADD_STRANGE`: 0 default. User toggles when running Nf=2+1.
- `USE_HMC_MG`: not in the runner. The DTXQCD light Nf=2 rational PF
  doesn't benefit from MG (multishift CG mathematically blocks MG
  preconditioning — `reference_mg_rhmc_landscape`).

**Result: no script changes shipped for DTXQCD env vars** — the canonical
runner is already correct.

### Source-level defaults

Source defaults (no script change required for these):

- `DTXQCD_MOOEE_FWDCACHE=1` (commit 491d5e4b, PR 5')
- `DTXQCD_MP_CG=1` (commit 2a45d8c4, mixed-precision multishift CG,
  ~17% per-traj faster)
- `QUDA_FORCE_KERNEL=1` (commit 5a76aa37, strange Path B)
- `DTXQCD_MOOEE_CUBLAS=1` (commit 12cb51d4)
- `DTXQCD_RATFORCE_GPU=1` (commit 6692a930)
- `DTXQCD_MOOEEINV_CUBLAS=1` (commit f47a985c)

## 2. cudaMemcpy storm hunt findings

Same as TXQCD-side: `Grid/algorithms/blas/BatchedBlas.h` alpha/beta storm
already eliminated via CUBLAS_POINTER_MODE_HOST in both repos.

Next-tier candidates surveyed in DTXQCD code (parallel to TXQCD survey):

- `Grid/qcd/action/dtxqcd/DTXQCDMultiShiftCG*.h`: no per-iter small HtoD
  copies; uses pure Lattice expressions inside CG.
- `Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverFermionEO.h`: `acceleratorCopy*`
  only at setup/import time, not in CG inner loop.
- `Grid/qcd/action/dtxqcd/DTXQCDLogDetCloverEOAction.h`: setup-time
  copies (per S/deriv call), large payload — not a storm.

No clean, easy alpha/beta-style win visible without nsys ground truth.

The DTXQCD profile (DTXQCD_NONEO_PROFILE.md) shows 54k cudaMemcpy/run at
8⁴ (avg 32 µs each), mostly **precision-change buffers** (SP↔DP for MP-CG
reliable update). This is a structural feature of MP-CG, not a storm
amenable to single-line caching.

## 3. Hasenbusch wallclock — NOT measured for DTXQCD

DTXQCD doesn't have Hasenbusch wiring (see Section 4). Therefore no
wallclock comparison possible this session.

For DTXQCD specifically, the multishift CG (light Nf=2 rational PF) is
**43% of trajectory** per `DTXQCD_NONEO_PROFILE.md`. Hasenbusch can't
compose with multishift CG (memory `reference_mg_rhmc_landscape`),
so Hasenbusch would not directly accelerate the dominant cost.

The TXQCD light rational PF has the same multishift-CG structure — the
TXQCD Hasenbusch class
(`Grid/qcd/action/txqcd/TXQCDWilsonCloverHasenbuschAction.h`, used by
`gen_txqcd_cfgs_2plus1.cc`) reduces the FORCE magnitude by mass-ratio
preconditioning (not the CG cost). Whether the analogous reduction
materializes for DTXQCD is open until the wiring is done.

## 4. DTXQCD Hasenbusch porting — work plan (NOT shipped this session)

### What would need to happen

1. **New class** `Grid/qcd/action/dtxqcd/DTXQCDWilsonCloverHasenbuschAction.h`
   modeled after `Grid/qcd/action/txqcd/TXQCDWilsonCloverHasenbuschAction.h`.
   The doubled-block operator structure makes this more than a textual
   port — the Hasenbusch ratio `det(M_light)/det(M_heavy)` acts on the
   doubled fermion (Nflavor × 2 blocks), and the force chain rule needs
   to include the per-block delta dependence.
2. **Driver wiring** in `production/gen_dtxqcd_cfgs.cc` lines 595-610:
   add `HASEN_LADDER` env parser, build N-1 ratio actions, push onto L1.
   Mirror TXQCD's lines 461-604 + 1044-1058.
3. **4⁴ FD validation gate**: N=2 ladder at HASEN_DM=0 must reduce
   bit-exact to current single-rational path.

### Why not this session

- The doubled-block force chain rule is the load-bearing complexity — it's
  not a textual port. Real C++ engineering, not flag-flipping. Two-three
  day project at minimum.
- Even after porting, the gain at DTXQCD's current setting is unclear:
  TXQCD's bench numbers (Section 5 below) are the first data point for
  whether Hasenbusch in this regime is worth deploying at all.

**Recommended for next session** after the TXQCD Hasenbusch numbers land.

## 5. Profiler cross-pollination

`DTXQCD_NONEO_PROFILE.md` already enumerates 5 optimization candidates
with effort + expected gain. None are "easy stuff" by the session criterion:

- Cand. 1 (HP MP-CG SP-matrix): ~10-15% trajectory, ~5 days effort, medium
  risk (half-precision can break rational poles).
- Cand. 2 (SP-only cleanup): ~10% trajectory, ~3 days effort, low risk.
- Cand. 3 (Light [0][0] gap): ~6-8% trajectory, ~5 days effort.
- Cand. 4 (siteforce kernel batching): ~3% trajectory, ~3 days.
- Cand. 5 (Stout fwd smear fused): ~3-4% trajectory, ~7 days (touches
  upstream Grid).

All require new C++ + bit-exact validation. **Recommend Cand. 2 (SP-only
cleanup) as the first follow-up project** — lowest risk + highest
gain-per-day-of-effort.

## 6. Phase I revival assessment (per session addendum)

**Gate**: profile shows Wilson Meooe ≥50% of CG inner-loop time.

Per `DTXQCD_NONEO_PROFILE.md` rank table:

- Rank 1: `DTXQCDMpcOpF::ApplySimd` (SP MdagM in MP-CG): 48% of GPU time.
  This IS the doubled-block Wilson Meooe at SP precision **but** with
  the per-site delta-clover folded in via the 48×48 SIMD cache — QUDA's
  plain Wilson dslash has no Δ awareness.
- Rank 6: `WilsonKernels SP Dslash` (upstream Grid Wilson kernel): 2% of
  GPU time. This is what plain QUDA Wilson dslash would replace.

So porting QUDA Wilson dslash into the CG would touch ~2% of GPU time, not
~48%. **Below the 50% gate. Phase I shelved correctly.**

The DTXQCD-analog would be a QUDA-aware doubled-block Wilson + Δ kernel —
that's the same months-long research project explicitly out of session
scope.

**Recommendation**: No quick-probe Phase I work for DTXQCD. The
profile-identified Cand. 1 (HP MP-CG SP-matrix) directly accelerates the
48% kernel and is the productive next step instead.

## 7. Commit SHAs

None this session — DTXQCD canonical runner already correct; no script
changes shipped.

## 8. Files touched

- `production/EASY_PERF_WINS_SHIPPED.md` (new — this doc)

No code or runner changes shipped to DTXQCD this session.
