# Grid ↔ QUDA linkage and vanilla-QCD optimizations

A technical summary of the Grid↔QUDA glue and the pure-QCD HMC / measurement
optimizations we've built in `Grid-TXQCD`. Written to scope what's
upstreamable to Grid — the TXQCD-specific pieces (composite gauge+auxiliary
field, multishift CG, force decomposition, etc.) are kept out of this
document; they live in `Grid/qcd/action/txqcd/` and are tightly coupled to
the TXQCD design.

Repo: `https://github.com/mlwagman/Grid-TXQCD` (branch `develop`).

## TL;DR

We bolted QUDA onto Grid as a set of clean, env-gated wrappers (`QudaInit.h`,
`QudaCloverInverter.h`, `QudaCloverMultiShiftInverter.h`, `QudaForcePrimitives.h`,
plus `QudaPropSolver<WCF>` at the user level), with a GPU-resident pack/unpack
that eliminates host roundtrips. Measured on **one A100 at 16³×48,
β=6.1, m=-0.245, csw=1.249**:

| operation | Grid-only | QUDA-accelerated | speedup |
|---|---|---|---|
| 1 propagator inversion (Wilson-clover, target 1e-8) | **2.25 s** (Grid CG, full-vol M†M, ~1380 iters) | **0.58 s** single-src / **0.37 s** in 12-rhs batch (QUDA Schur-EO, ~315 iters) | **3.9× / 6.1×** |
| 1 HMC force eval (strange Nf=1 RHMC level) | ~3 s (est.) | **~1.8 s** measured in production | ~1.6× |
| 1 trajectory of Nf=2+1 HMC (chroma-equilib cfg) | baseline | -36% wallclock | 1.6× |

The wrappers are off-by-default (`QUDA_SOLVER=1` / `QUDA_FORCE=1` gates the
QUDA path; same binary still runs pure-Grid). The TXQCD-specific code is
kept separate; the Grid↔QUDA infrastructure described below is what we'd
propose upstreaming.

## Headline performance (full table)

All on 16³×48, β=6.1, m=-0.245, csw=1.249, single A100-80GB.

| Component | Path | Wallclock | Δ vs Grid-only |
|---|---|---|---|
| Propagator inversion (Grid CG, full-vol M†M) | `ConjugateGradient<LatticeFermion>` | **2.25 s/inv** (1380 iters median) | baseline |
| Propagator inversion (QUDA Schur-EO, single-src) | `QudaCloverInverter::operator()` | **0.58 s/inv** (315 iters) | 3.9× |
| Propagator inversion (QUDA, 12 RHS batched) | `QudaPropSolver::solve_multi` → `invertMultiSrcQuda` | **0.37 s/inv** (4.47 s for 12) | 6.1× |
| Strange Nf=1 RHMC force eval (level [0][2]) | `OneFlavourSchurCloverQudaRationalActionMP::deriv` | **~1.8 s/eval** w/ QUDA on | ~1.6× |
| Light Nf=2 RHMC force eval (level [0][1]) | `TwoFlavourSchurCloverQudaForceActionMP::deriv` | **~3.8 s/eval** w/ QUDA on | ~1.5× |
| End-to-end QCD HMC traj (chroma-equilib cfg) | composed | — | -36% wallclock |
| Disconnected QCD measurement (128 stoch sources, light+strange) | `meas_disco_qcd` | **174 s → 21.6 s** w/ QUDA Schur-EO | **8.0×** |
| GPU-resident pack (no host roundtrip) | `Grid/util/QudaPackGpu.h` | saves ~30 ms/call | dominant win for fast meas kernels |

**Re-bench command** (single source, 12 spin/color inversions, A100):

```bash
# Grid CG baseline:
CUDA_VISIBLE_DEVICES=0 QCD_SUFFIX=_chroma_bench \
  IMPORT_CFG=cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime \
  LATT=16.16.16.48 MEAS_SPACE_SRC=1 MEAS_TIME_SRC=1 MEAS_CG_TOL=1e-8 \
  mpirun -np 1 --bind-to none ./meas_conn_qcd 11100 --mpi 1.1.1.1
# ~27 s wall, 12 × 2.25 s = ~2.25 s/inv (1380 CG iters)

# QUDA multi-src (12 RHS batched):
... QUDA_SOLVER=1 QCD_MULTISRC=1 ... ./meas_conn_qcd 11100 --mpi 1.1.1.1
# ~4.5 s wall, 0.37 s/inv (315 CG iters via Schur-EO)
```

## Layer 1 — buffer plumbing

The fundamental piece is converting between Grid's SIMD-vectorized
checkerboard layout and QUDA's lexicographic EO buffers.

### `Grid/util/QudaInit.h`
- Singleton-style `Quda::initialize()` that hands QUDA the right device,
  MPI communicator, and lattice topology (read from a Grid `GridBase`).
  Idempotent; safe to call repeatedly.
- `local_volume(grid)`: trivial helper; QUDA needs the per-rank EO volume.

### `Grid/util/QudaPack.h` (host pack/unpack)
Reference implementation:
- `Quda::fermion_to_eo_buffer(LatticeFermion, double *eo24V)`
  — pulls a Grid fermion off device, unvectorizes via Grid's
  `unvectorizeToLexOrdArray`, then walks `lex_index → eo_index` mapping
  the way QUDA expects (parity-first, then site-major).
- `Quda::eo_buffer_to_fermion` is the inverse.

Pack format: 24 doubles/site (2 parities × 4 spin × 3 color × 2 reim).
QUDA's `QudaInvertParam::input_location = QUDA_CPU_FIELD_LOCATION` with
`gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS`. Empirically the right
combination after a multi-week convention hunt (see Layer 4 below).

### `Grid/util/QudaPackGpu.h` (GPU-resident pack)
For paths where the source/result lives on the GPU already (most of HMC
and measurement), the host-roundtrip pack is ~30-100ms per call which
dominates for fast kernels. The GPU-resident version uses:

- `BuildLexTable(grid)` — precomputes the `lex_index → eo_index` map as
  a Grid `Lattice<vInteger>` on device. One-time setup.
- `GpuPackFermionRbLex(LatticeFermion, void *device_eo_ptr)` — a single
  `accelerator_for` over sites doing the gather. Writes directly into a
  device pointer that QUDA can consume via `input_location =
  QUDA_CUDA_FIELD_LOCATION`.

This single optimization removes the host buffer roundtrip from every
Grid↔QUDA crossing.

## Layer 2 — inverter wrappers

### `Grid/algorithms/iterative/QudaCloverInverter.h`

Drop-in `LinearOperator`-style wrapper around `invertQuda`:

```cpp
struct QudaCloverParams {
    RealD mass, csw, tol;
    int max_iter;
    bool anti_periodic_t = true;
    QudaGammaBasis gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
};

class QudaCloverInverter {
    QudaCloverInverter(GridBase *grid, const QudaCloverParams &);
    void SetGauge(const LatticeGaugeField &U);     // 1× per cfg
    void operator()(HermOp &, LatticeFermion &src,
                    LatticeFermion &x);             // M^{-1}·src
    QudaInvertParam &InvertParam();                 // exposed for tweaks
};
```

The `operator()` calls `dslashQuda` or `invertQuda` depending on what's
needed. Mostly used through the higher-level wrappers below.

### `Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h`

Wraps `invertMultiShiftQuda` for RHMC pseudofermion solves. Takes a
vector of shifts σᵢ, solves `(M†M + σᵢ)·xᵢ = b` for all i in one call.
Replaces Grid's `ConjugateGradientMultiShift` in HMC paths where the
rational approximation is built externally.

Both inverters are mostly transparent: they hold a QUDA gauge-and-clover
state, refresh it on `SetGauge`, and accept Grid lattice inputs.

### `production/quda_helper.h::QudaPropSolver<WCF>` (high level)

The cleanest user-facing wrapper. Drop-in replacement for `(Mdag·src;
ConjugateGradient)` Grid CG:

```cpp
WCF Dw(U_inv, ...);
MdagMLinearOperator<WCF, LatticeFermion> HermOp(Dw);
Grid::QudaPropSolver<WCF> solver(Dw, HermOp, U_inv,
                                 mass, csw, tol, max_iter);
solver.solve(src, x);                 // QUDA path if QUDA_SOLVER=1
solver.solve_multi(srcs, xs);         // invertMultiSrcQuda for Nrhs batch
```

Falls back to Grid CG when QUDA isn't built/enabled. Multi-source path
uses `invertMultiSrcQuda` (the underexposed batched API) — ~2.7× over
per-source `invertQuda` calls because it shares operator setup across
RHS. Measurements that loop over Ns·Nc=12 spin/color sources or N space-
time sources are the prime beneficiaries.

## Layer 3 — force wrapper

### `Grid/util/QudaForcePrimitives.h`

This is the most subtle of the wrappers — it routes the fermion force
through `computeCloverForceQuda` (which evaluates
`p·F_clover[Q]·p† − dQ/dU · (Mψ)(Mψ)†` internally) and packages the
result as a Grid `LatticeGaugeField` momentum increment.

Key calling convention (after a long discovery process):
- `gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS`
- `dagger = QUDA_DAG_YES`  ← essential, missing it gives a 7° structural mismatch (cos=0.93 vs 1.0)
- `matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC`
- `multiplicity` controls flavor count (Nf=1 strange, Nf=3 rational etc.)
- final scale = `-1/(8·κ²)` per pseudofermion contribution
- output is Ta-projected before adding to `mom`

Validation: `production/bench_clover_oprod.cc` constructs a Grid
reference `f_Grid` (Mob from `WilsonCloverFermionImplementation`) and
compares cos(f_Grid, f_QUDA) on a fixed gauge + random pseudofermion.
Should report cos = 1.000000, scale_ratio = 1.000000 at the matching
convention. We use this as the gate for any wrapper change.

`OneFlavourSchurCloverQudaRationalActionMP` (in `Grid/qcd/action/
pseudofermion/`) is the Nf=1 RHMC action that uses this wrapper for
both the rational solves (multishift) and the force eval. Plugs into
Grid's `Integrator` like any other `Action<LatticeGaugeField>`. Gated
by `QUDA_FORCE=1` and `QUDA_FORCE_KERNEL=1`.

## Layer 4 — convention dictionary (the painful part)

QUDA exposes Wilson-clover with a few orthogonal conventions that all
need to agree. We documented the final set in
`production/HMC_CONVENTIONS.md`. The factor we keep tripping over:

**Grid eps = chroma_dt / 4** for the same physical step. This comes from:
1. Momentum variance: Grid `Var(P)=2` vs chroma `Var(P)=1/2` (factor 2 in σ)
2. Implicit kinetic mass: Grid `m=2` vs chroma `m=1/2` (factor 2)

So to reproduce chroma's `n_steps=7 tau0=√2` (dt=0.202) in Grid:
- option A: same trajL=√2 with MDsteps=28 (eps=0.0505)
- option B: trajL=√2/4 with MDsteps=7 (eps=0.0505)

Both give the same physical trajectory. Empirically Grid `MDs=14
trajL=√2` (eps=0.10) is unstable with dH=22k; MDs=28 trajL=√2 is stable
with dH=O(1). The dictionary is verified against a chroma reference cfg
import (`IMPORT_CFG=path.lime`) with `NO_METROP=1` and single-trajectory
plaq held at chroma's 0.5135.

## Layer 5 — measurement-side integration

### `production/meas_conn_qcd.cc`
Connected-2pt measurement using the `QudaPropSolver` infrastructure.
Multi-source path (`QCD_MULTISRC=1`) batches 12 spin/color sources per
location through `invertMultiSrcQuda`. Output: `conn_qcd_<traj>.h5`
with pion, proton/neutron (pos/neg/FB), per-source variants, all in
source-relative time with antiperiodic-T sign applied where needed.

### `production/meas_disco_qcd.cc`
Stochastic disconnected loops (`Tr[γ5 M^{-1}]` and `Tr[M^{-1}]`) on
both light and strange. Lightweight; just calls `solve` per stochastic
source. Defaults to Grid CG; QUDA path optional via `QUDA_SOLVER=1`.

### `production/compute_plaq.cc`
Diagnostic. Reads NERSC OR LIME/ILDG cfg (auto-detected from the magic
bytes) and reports plaquette + link trace. Handy for spot-checking a
chroma chain's thermalization (we used this on a chroma `cl3_16_48`
chain of 4000 cfgs to verify the thermalization curve matches our HMC).

## Patterns we found useful

1. **`IMPORT_CFG` env var.** Every binary that needs a starting gauge
   field reads `IMPORT_CFG` first; if set, it imports the LIME/ILDG cfg
   directly. We use this for chroma-validation runs and for forking
   new HMC streams from an external ensemble cfg.

2. **`QUDA_SOLVER=1` gating.** All QUDA paths are off-by-default, gated
   by env var. Same binary works pure-Grid or with QUDA. Crucial during
   convention-hunt phase (could compare per-call).

3. **`bench_*` binaries** as conformance gates. We have benches for
   Dslash convention, clover oprod (force), multishift CG, etc. Each
   reports `cos(grid, quda)` and `||grid||/||quda||` against a Grid
   reference. The build pipeline runs them on a small lattice;
   regressions caught immediately.

4. **Force eval profiling timer block.** In `OneFlavourSchurCloverQudaRationalActionMP::deriv`
   we wrap each sub-component (multishift CG, force kernel, pack/unpack)
   in a counter; `[ID timer]` lines in the log give per-component time
   per HMC step. Made the GPU-pack optimization easy to measure.

## Suggested upstreaming order

1. **`QudaInit.h` / `QudaPackGpu.h`** — pure plumbing, no externally
   visible API change. Mergeable as Grid utility headers.
2. **`QudaCloverInverter.h` / `QudaCloverMultiShiftInverter.h`** —
   single-shift + multishift, slot in alongside Grid's existing CG
   classes. Needs `--with-quda` configure flag (already exists).
3. **`QudaPropSolver` (production helper)** — example wrapper showing
   how to drive `invertMultiSrcQuda`. Could go in `tests/` or
   `examples/` rather than core Grid.
4. **`QudaForcePrimitives` + `OneFlavourSchurCloverQudaRationalActionMP`**
   — the harder upstream because convention conformance is sensitive
   to the QUDA build. Suggest landing with `bench_clover_oprod.cc` as
   a regression test so it stays correct across QUDA versions.

The TXQCD-specific pieces (`TXQCDField`, `TXQCDCloverSchurOp`,
`TXQCDWilsonCloverFermionEO`, the auxiliary action and force adapters)
should be kept out of upstream Grid — they're the experimental physics
we're still validating.

## File map (Grid-TXQCD repo paths)

Pure infrastructure (upstreamable):
- `Grid/util/QudaInit.h`
- `Grid/util/QudaPack.h`, `Grid/util/QudaPackGpu.h`
- `Grid/algorithms/iterative/QudaCloverInverter.h`
- `Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h`
- `Grid/util/QudaForcePrimitives.h`
- `Grid/qcd/action/pseudofermion/OneFlavourSchurCloverQudaRationalActionMP.h`
- `Grid/qcd/action/pseudofermion/TwoFlavourSchurCloverQudaForceActionMP.h`

Production helpers:
- `production/quda_helper.h` — `QudaPropSolver<WCF>` template
- `production/HMC_CONVENTIONS.md` — chroma↔Grid dictionary
- `production/compute_plaq.cc`
- `production/bench_clover_oprod.cc` — force convention regression
- `production/bench_dslash.cc` — Dslash convention regression

TXQCD-only (keep separate):
- All of `Grid/qcd/action/txqcd/`
- `production/quda_txqcd_helper.h` — `QudaTxqcdPropSolver` for composite
- `production/{gen,meas}_txqcd_*.cc`
