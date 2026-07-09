# Aux/Dirac fusion for the full-volume M_device_csf — scoping (2026-07-04)

## Why

The full-volume doubled operator is the MG/measurement substrate at λ≲4
(the `M_ee` cliff poisons the EO Schur complement there — see the λ-scan in
the gridmg memory / commit `93043526`).  Its device matvec runs 21.8 ms per
M†M at 16³×48 (2× the volume of the EO chain's 16.0 ms, so already efficient
per site), but the **SP-smoother experiment proved the cost is not dslash
precision**: SP matvecs came back 21.3 vs 23.3 ms.  Something
precision-independent dominates.  Candidates, per apply of `M_device_csf`
(`DTXQCDMpcOpQUDA.h`):

- 3 `cudaDeviceSynchronize` (plus γ5 wrap traffic on Mdag: 12 extra
  elementwise launches per M†M),
- the elementwise chain: PreMatLower ×2, dirac ×4 + blas::ax ×4,
  PostMatLower ×2, aux kernel ×1 ≈ 20+ kernel launches,
- the aux kernel's DP cache traffic (~2.3 KB/site — precision-shared between
  DP and SP paths by design),
- the 4 clover-Wilson dslashes themselves.

## Step 0 — MEASURED (job 1297599 j0, λ=4 cached setup, 800 applies)

`DTXQCD_MDEVICE_PROFILE=1` (default off) per-phase wallclock inside
`M_device_csf`, steady state after QUDA autotune:

    total 11.5 ms/apply = entry-sync 0.11 + pre 0.19 + dirac 2.39
                        + post 0.23 + AUX 8.60

**The aux kernel is 75% of the matvec** — ~10× off its bandwidth bound
(~2.3 KB/site aux cache + 8 spinors ≈ 0.6-0.8 ms of unavoidable traffic).
This fully explains the SP-smoother wash (the aux kernel + DP cache are
precision-shared).  The launch/sync/γ5 overheads I expected to matter are
0.5 ms combined.

## REVISED priority: optimize `ApplyAuxKernelNative`, everything else is
## secondary

Target: `DtxqcdQudaAuxKernelNativeV2` (dtxqcd_quda_aux_kernel_native_v2_
impl.cc; SP twin).  Suspects, in the order to check: (a) perm-table
indirection producing uncoalesced FloatN accesses; (b) `transpose_aux`
reading σ/π at swapped (b,a)(j,i) strides — a transposed copy in the cache
would make both orientations coalesced for +72 doubles/site of storage;
(c) register pressure / occupancy from the 6×6 complex flavor⊗color
accumulation over 4 spinor pairs in one thread; (d) one-thread-per-site
layout where one-thread-per-(site,spin) or warp-per-site would parallelize
the inner sums.  The TXQCD fused-aux kernel (1.68× on the Grid side,
DTXQCD_FUSED_HERMOP_RESULTS.md) and the EO SiteMatDevice gemmBatched
pattern are the in-house references — note gemmBatched on a per-site 48×48
assembled aux matrix is a viable fallback implementation that reuses the
DTXQCDBatchedInverse48 machinery.

Expected: aux 8.6 → 2-3 ms ⇒ apply 11.5 → 5-6 ms, M†M 23 → 10-12 ms.
Downstream: λ=4 full CG 77 → ~40 s, full MG 82 → ~45 s (both well under the
69 s EO CG); λ=3 CG 1711 → ~900 s; every non-EO HMC trajectory (the aux
kernel runs per multishift iteration there too — post-hoist profile said
multishift is still ~50% of trajectory).

## Kernel-optimization verdict: NEGATIVE — V2 stays (2026-07-08, clean
## same-job pairs on the lambda=4 cached setup, 200 applies each)

| variant | decomposition / layout                          | aux ms/apply |
|---------|--------------------------------------------------|-------------|
| V2      | thread/site, AoS cache, alpha-redundant loads    | **8.60** (8.82 in the V4-pair job) |
| V3a     | thread/site, SoA cache, row-hoist LOCAL ARRAYS   | 21.62       |
| V3b     | thread/site, SoA cache, scalar ops + full unroll | 12.24       |
| V4      | thread/(site,flavor), AoS cache (V2 loads)       | 9.89 (+12% vs same-job V2) |

The ordering is monotonic in PER-THREAD STATE, not memory-access quality:
the kernel is OCCUPANCY/REGISTER-PRESSURE limited, not coalescing limited.
The "10x off bandwidth bound" framing in the original scoping was the wrong
model — at ~190k sites/1 GPU with ~150+ live complex values per thread,
V2's redundant-but-streamed loads ride L2 while its (comparatively) small
live state keeps more warps resident.  Fixing the "obvious" load redundancy
(V3) or layout (SoA) INCREASES live state and loses.  V4's flavor split
halves output state but doubles input-rotation work per site and still
loses ~12%, closing the decomposition axis too.  V3a's specific lesson:
runtime-indexed per-thread arrays compile to local memory — never do that
in these kernels.

All variants remain in-tree, env-gated OFF (DTXQCD_AUX_KERNEL_V3/_V4), with
V2 the default; correctness of V3a/V3b/V4 is machine-precision verified
(gate B at 4^3x8 + 16^3x48; V3a additionally dH-bit-identical end-to-end in
the auxk gates A/B).  **Verdict: V2 sits near its practical bound for this
per-site arithmetic intensity.  The full-substrate matvec is capped at
~10.2-11.5 ms/apply as-is; further gains must come from mrhs batching
(amortize the aux-cache reads across right-hand sides, which genuinely
raises arithmetic intensity) or the fusion options below — not from
rewriting this kernel.**

## Fusion options (ordered by expected value / effort — pre-verdict list,
## superseded above)

1. **Fuse the lower-block elementwise chain + aux + rescale** (2-4 days).
   Today: Pre(γ2·conj) → [dirac] → Post(γ2·conj) → blas::ax → aux-kernel,
   each a separate pass over full spinor fields.  Fusable into TWO custom
   kernels (one pre-dirac, one post-dirac) in the house native-impl style
   (`dtxqcd_quda_*_native_impl.cc`, trove accessors): saves ~3 full
   read+write passes over 4 spinor fields (~1.2 GB traffic per apply) and
   ~12 launches.  The Mdag γ5 wraps fold into the same kernels (flag arg),
   killing 12 more launches per M†M.  Validation: gate B machine-precision
   + the multishift gate test, exactly like the M-wrap ports.
2. **M†M fusion** (1 day).  `Mdag_device_csf` re-applies γ5 to inputs and
   restores them (4 passes over inputs per M†M) purely to preserve caller
   state; a fused HermOp entry point that owns its intermediate avoids the
   restore pair and one sync.
3. **Stream-event ordering instead of device-wide syncs** (1-2 days,
   riskier).  Replace the 3 `cudaDeviceSynchronize`/apply with
   `cudaStreamWaitEvent` between Grid's computeStream and QUDA's stream.
   Value depends entirely on the measured `entry-sync` share; the fork
   estimated ~1% but that predates the aux hoist (fixed costs are now a
   larger fraction).
4. **SP aux cache for the SP path** (half day).  The aux cache is shared DP;
   an SP copy halves aux traffic for smoother matvecs.  Only worth it if
   the profile shows aux ≥ ~20% AND the SP smoother is in production use.
5. **True aux-into-dslash fusion: NOT viable externally.**  `dirac->M` is a
   closed QUDA kernel; injecting the aux term into its epilogue means
   patching QUDA (the DiracCoarse route from the native-MG assessment is
   the honest version of that project).

## Projected effect (to be replaced by measured numbers)

If the profile splits roughly as dirac ~8 / elementwise+aux ~8 / syncs+launch
~6 (ms per M†M): options 1+2 target ~10 of the 21.8 → full-volume HermOp
~14 ms.  Downstream at λ=4 16³×48: full CG 77 → ~50 s, full MG 82 → ~55 s —
both under the 69 s EO CG, flipping the λ≲4 wallclock verdict before any
volume scaling.

## Validation gates (house style)

1. Gate B machine precision (packed adapter vs Grid struct path) at 4³×8 and
   16³×48, both substrates.
2. `Test_dtxqcd_multishift_quda` bit-exactness (the fused path feeds the
   production non-EO HMC).
3. dH bit-match on a 1-traj A/B (the aux-static pattern, sbatch driver
   already exists: `slurm_dtxqcd_auxstatic_ab.sh` — clone).
4. BENCH_HERMOP before/after + a λ=4 full-MG solve on the cached setup.
