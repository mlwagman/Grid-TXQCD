# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [Grid](https://github.com/paboyle/Grid) (data-parallel C++ lattice QCD library) extended with two related auxiliary-field constructions that couple tensor/scalar fields to quarks so the fermion determinant has a controlled phase, enabling correlated-sampling signal-to-noise improvements:

- **TXQCD** (Tensor eXtended QCD) — the original construction; `det(M)` stays real. Code under `Grid/qcd/action/txqcd/` (hub `Txqcd.h`). Design in `TXQCD_ARCHITECTURE.md` (read first for non-trivial TXQCD work) and `txqcd_notes.tex`.
- **DTXQCD** (Doubled TXQCD) — the current active line of work. The charge-conjugate-doubled construction turns `det(M)` into a Pfaffian `Pf(K·M₄₈)`; HMC samples `|Pf|` and the sign is handled by reweighting. Code under `Grid/qcd/action/dtxqcd/` (hub `Dtxqcd.h`). Design in `DTXQCD_SIGN_REWEIGHTING.md`.

The GPU/QUDA/cuBLAS acceleration layer (the production path) is documented in `GRID_QUDA_ARCHITECTURE.md`, `production/GPU_BUILD_NOTES.md`, and `production/MULTI_GPU_SRUN_NOTES.md`.

**Read the handoff notes before starting cluster work.** `HANDOFF_CLUSTER_DTXQCD_2026-06-19.md` (DTXQCD convention/checkpoint/GPU-port status) and `HANDOFF_AUX_GPU_PORT.md` (cluster setup, srun/QUDA perf gotchas, CPU↔GPU porting patterns) capture load-bearing operational facts that this file only summarizes.

## Build

Grid uses GNU autotools. There are typically **two build trees** side by side: `build-cpu/` (small tests / laptop) and `build-gpu/` (production on the lq2 A100 cluster).

```bash
./bootstrap.sh                         # only needed once (or after m4/configure.ac changes)
mkdir build-gpu && cd build-gpu
../configure <flags>                   # see systems/*/config-command for known-good flags
make -j                                # builds libGrid
make -C tests/txqcd  tests             # TXQCD correctness/force tests (not built by default)
make -C tests/dtxqcd tests             # DTXQCD gate tests (not built by default)
```

- Default `make` only builds tests at the root of `tests/` — **subdirectory tests (all `tests/txqcd/` and `tests/dtxqcd/`) must be built explicitly.**
- Known-good configs live under `systems/<machine>/config-command` (sdcc-genoa, SDCC-ICE, SDCC-A100, Frontier, Perlmutter, Summit, …). `sourceme.sh` in each sets environment. On lq2, `source env_lq2_grid.sh` before invoking the harness (sets the https proxy + modules).
- CPU SIMD flag is `--enable-simd=AVX512|AVX2|…`; **GPU builds use `--enable-accelerator=cuda --enable-simd=GPU`** (HIP on Frontier/Lumi).
- TXQCD/DTXQCD compile-time deps: `--disable-fermion-reps --disable-gparity` (the genoa default, a reasonable baseline everywhere).
- After editing a header (e.g. `*CompositeImpl.h`), stale `.o` files do **not** auto-rebuild dependents — always `make` before running.

### Production programs (`production/`)

`production/Makefile` builds out-of-tree against an existing Grid build via `grid-config`. It now **defaults to `GRID_BUILD_DIR=../build-gpu`**; override on the command line for a laptop/CPU build:

```bash
make GRID_BUILD_DIR=../build gen_dtxqcd_cfgs    # or just `make` (=> all PROGRAMS)
```

Produces the `gen_{qcd,txqcd,dtxqcd}_cfgs*` generators, `meas_{conn,disco,aux,baryon}_{qcd,txqcd,dtxqcd}` measurements, `compare`, `compute_{plaq,vev}`, plus the `test_quda_*` and `bench_*` programs.

All physics parameters for production live in `production/params.h` (lattice size, masses, csw, beta, stout smearing, lambda, trajectory counts). `lambda` (the auxiliary coupling) is encoded in output paths: `cfgs/txqcd_lam0.5000/`, `meas_2pt/txqcd_lam0.5000/`. Production generators are also heavily **env-knob overridable** (e.g. `LATT`, `LAMBDA_DTXQCD`, `MASS_LIGHT_DTXQCD`, `CSW`, `BETA`, `INTEGRATOR`, `MDSTEPS`, `TRAJL`, `NO_METROP`) so the same binary runs parallel streams — see the handoff notes and `reference_dtxqcd_gencfgs_env_knobs` for the catalog.

## Running

**On the lq2 cluster use `srun --mpi=pmix`, never `mpirun`** — `mpirun` falls back to a slow transport and is ~8× slower on the same hardware/binary. And **always pass `--shm-mpi 1`** (the default `--shm-mpi 0` makes Wilson Meooe up to 237× slower). These are not optional perf tweaks; they have load-bearing UCX-CUDA/IPC side effects.

```bash
# Single test (after `make -C tests/dtxqcd tests`), interactive GPU node:
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
     ./tests/dtxqcd/Test_dtxqcd_wilson_clover_fermion_eo --grid 4.4.4.8 --mpi 1.1.1.1

# Production generation (resumes from latest checkpoint):
srun --overlap --mpi=pmix -N 1 -n 4 --cpu-bind=none --gres=gpu:4 \
     env <QUDA/TXQCD flags> ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.4 --shm 512 --shm-mpi 1

# Measurements (sequential drivers, skip existing output):
./run_all_qcd_measurements.sh   --grid ... --mpi ...
LAMBDA=0.5000 ./run_all_txqcd_measurements.sh --grid ... --mpi ...
./compare --grid ...                                                   # Fierz-identity cross-check
```

- On a laptop/CPU build, `mpirun -np N ./gen_… ` is fine (no srun there).
- **`--mpi 1.1.1.4` (one 4-GPU job per node) is the production config at 16³×48 — one λ stream per node.** A single 1.1.1.4 runs ~1.7× the per-trajectory throughput of an *uncontended* 1.1.1.1 (~42% parallel efficiency — MPI sync dwarfs the tiny per-rank compute). The tempting alternative, 4 independent 1.1.1.1 jobs on one node, does **not** deliver 4× aggregate: the four single-GPU jobs contend on shared host bandwidth / NVLink / PCIe and slow each other enough to end up *worse* than one 1.1.1.4 (confirmed empirically in TXQCD; the DTXQCD per-deriv profile is hopping+CG-dominated and contends the same way). So run **one 1.1.1.4 per node**, not 4×1.1.1.1. (Clean-node single-GPU numbers — measured with idle neighbours — overstate 1.1.1.1 throughput for exactly this reason.) Watch for the per-rank-volume QUDA cliff (~256k→524k sites/rank) when scaling up.
- `params.h` also governs measurement trajectory range via `n_therm`, `n_prod`, `meas_skip`.
- `N_TRAJ` in slurm scripts is a **target, not an increment**: if the latest checkpoint is `.150` and `N_TRAJ=100`, the driver exits immediately.

## TXQCD architecture — what to know before editing

Almost all TXQCD code lives under `Grid/qcd/action/txqcd/` (hub: `Txqcd.h`). Two existing Grid files were modified: `WilsonCloverFermionImplementation.h` (added `MooeeDeriv`/`MooeeInvDeriv` clover force helpers) and `PseudoFermion.h` (new `#include`s). Read `TXQCD_ARCHITECTURE.md` for the full picture; key structural facts:

- **Composite field.** `TXQCDField` bundles `LatticeGaugeField U` with five auxiliary lattice fields (`sigma`, `pi` flavor-space Hermitian; `s`, `p`, `t` color-space Hermitian). `TXQCDCompositeImpl` implements Grid's `FieldImplementation` static interface (momenta, update, kinetic) by delegating gauge → `PeriodicGimplR` and aux → additive/Hermitian. This is how `HybridMonteCarlo`, `LeapFrog`/`ForceGradient`, smearing, checkpointing all work unchanged.
- **Kinetic energy conventions.** Gauge momentum is antihermitian; aux momentum is Hermitian (conjugate to Hermitian fields) — the `1/2` factor on aux traces in `TXQCDCompositeImpl::KineticEnergy` is not a bug. Aux momentum generation scales by `sqrt(HMC_MOMENTUM_DENOMINATOR)` to match Grid's integrator convention.
- **Action adapters.** Mixing TXQCD and plain-QCD pieces in one HMC requires two wrappers: `GaugeActionAdapter<T>` (any `GaugeAction<PeriodicGimplR>` on the composite field) and `QCDActionAdapter` (any `Action<LatticeGaugeField>`, e.g. the strange quark). They extract `U`, delegate, and repack force with aux components zeroed.
- **EO preconditioning.** `TXQCDWilsonCloverFermionEO` stores precomputed LU-factored 24×24 site matrices (Nf·Ns·Nc = 2·4·3) built from the aux fields + clover. LogDet of `M_ee` is exact (no stochastic estimation); the odd-sublattice piece uses RHMC with multishift CG. Per-site scalar ops live in `TXQCDSiteMatrix.h` — historically the main CPU bottleneck and the GPU-optimisation target (batched LU via cuBLAS).
- **Clover force convention.** Grid's `WilsonCloverFermion` stores "Convention B" internally; the force needs "Convention A" — conversion factor `-1/2`. See `MooeeDeriv`/`MooeeInvDeriv` in `WilsonCloverFermionImplementation.h`.
- **Smearing.** `TXQCDSmearedConfiguration` wraps Grid's `SmearedConfiguration<PeriodicGimplR>`: gauge component goes through the stout chain; aux components pass through unchanged. All fermion/gauge actions in `production/` run with `is_smeared = true`.
- **Checkpoint format.** Gauge is standard NERSC (interoperable). TXQCD aux fields are a packed-Hermitian binary sidecar `ckpoint_lat_aux.<traj>` (magic `TXQA`, IEEE64BIG). Multi-rank writes use MPI-IO collective at rank-major offsets (format v3); single-rank v2 is still readable. Don't invent a new format — extend the existing one.
- **HMC driver pattern.** TXQCD/DTXQCD drivers bypass `HMCResourceManager` (which assumes a gauge-only field) and wire `HybridMonteCarlo + Integrator + RNGs` by hand. See `HMC/TXQCD_Wilson_small.cc` or `production/gen_txqcd_cfgs.cc` as templates.
- **Measurement inversions** (`meas_conn_*`, `meas_disco_*`): use full-volume `MdagM` CG, *not* EO. This is intentional — measurement inversions are one-shot; HMC forces are what benefit from EO.

## DTXQCD architecture — the active construction

DTXQCD code lives under `Grid/qcd/action/dtxqcd/` (hub: `Dtxqcd.h`); tests in `tests/dtxqcd/`; production generator `production/gen_dtxqcd_cfgs.cc`; measurements `meas_{conn,aux,baryon}_dtxqcd`. The current branch is `dtxqcd-v2`. Read `DTXQCD_SIGN_REWEIGHTING.md` and `HANDOFF_CLUSTER_DTXQCD_2026-06-19.md` before touching the algebra.

- **Composite field.** `DTXQCDField` = gauge `U` + auxiliary `{σ, π, d, n, s, p}`. σ, π are Hermitian (36 reals/site each); d, n are **truly complex-symmetric** under the joint color+flavor transpose `M(k1,k2)=M(k2,k1)` (42 reals/site each: 6 complex diag + 15 complex off-diag); s, p are singlet scalars (1 real each).
- **Doubled site matrix.** The per-site fermion matrix is **48×48** (the charge-conjugate doubling of TXQCD's 24×24), assembled by `DtxqcdAssembleDoubled48`. The locked-in **sigmaHerm convention**: `M_LL = conj(M_UR)`, and `M_lower`'s diagonal block reads `σ^T, π^T`. This is load-bearing for γ5-Hermiticity and Pfaffian antisymmetry — do not change it. Legacy env knobs (`DTXQCD_DN_COMPLEX_SYMMETRIC`, `DTXQCD_SIGMA_PI_HERMITIAN_ONLY`, `DTXQCD_DN_REAL_SYMMETRIC`) are **ignored** (constant-folded out); they only print a one-time warning. Do not set them in production scripts.
- **Pfaffian / sign problem.** The rational PF approximates `x^(−1/4)`: one √ for Nf=2 degeneracy, one √ for `det(M₄₈) → Pf(K·M₄₈)`. HMC samples `|Pf|·e^{−S_aux}`; the signed Pfaffian `signPf = (−1)^{n_neg(γ5·M₄₈)}` is recovered by reweighting `⟨O⟩_phys = ⟨signPf·O⟩_HMC / ⟨signPf⟩_HMC`. Feasibility hinges on measuring `⟨signPf⟩` per ensemble — see `DTXQCD_SIGN_REWEIGHTING.md` and `production/analyze_sign_reweighting.py`.
- **Checkpoint format DTX3.** Magic `'DTX3'` (`0x44545833`), per-site payload 158 doubles (1264 bytes): σ 36 + π 36 + d 42 + n 42 + s 1 + p 1. **DTX2 configs are unreadable** (different size; the complex-symm pack restores imag-diagonal DOFs DTX2 dropped, which was the source of a ~0.5% Fierz residual). Code: `DTXQCDCheckpointer.h`.
- **EO-only generator.** `gen_dtxqcd_cfgs` uses `DTXQCDWilsonCloverRationalEOAction` + `DTXQCDLogDetCloverEOAction`. The `tests/dtxqcd/Test_dtxqcd_2pt_gencfgs` driver supports `USE_FULL_PF=1` (non-EO) for ablation.
- **Production generator capabilities** (all in `gen_dtxqcd_cfgs.cc`): checkpoint **resume** (scans `ckpoint_lat{,_daux,_rng}.<t>`, NERSC-checksum-verified), **runtime integrator selection** (`INTEGRATOR=MinimumNorm2|ForceGradient`; both templates compiled — default `MinimumNorm2`), and the **signed-Pfaffian diagnostics observer** `production/dtxqcd_diag.h` (per-traj γ5·M₄₈ low eigenvalues → `signPf`, aux VEVs/Tr M⁻¹/correlators/per-action Fdt → `hmc_diagnostics.<traj>.h5`, consumed by `analyze_sign_reweighting.py`).
- **Fresh-start aux init.** `AUX_INIT_AUTO=1` finds the self-consistent saddle by **Picard-first** iteration (`Σ_{n+1}=Σ_DTXQCD(Σ_n)` from aux=0; Jacobian `~N_f/λ²` so it contracts in ~2 evals at large λ — at λ=10 the saddle ≈ mean-field to <0.1%) with a **bisection fallback** when not contracting (small λ). Tol is `AUX_INIT_TOL` (default **0.1** — 10% is ample for an HMC init; the trajectory equilibrates the rest). Other knobs: `AUX_INIT=<Σ>` (explicit, skip the solve), `AUX_FLUCT_LAMBDA` (aux draw width, default 10), `AUX_INIT_PICARD[_MAX]`, `AUX_INIT_MAX_ITER`, `ZERO_DN_INIT`/`ZERO_ALL_AUX`, `VEV_NOISE`, `G5M_NEV`/`G5M_NKRYLOV`/`G5M_EVALS_OFF`.
- **RHMC rational autoscaling** (`DTXQCDRemezAutoScale.h`, ported from TXQCD's `TXQCDRemezAutoScale.h`). The rational `x^(−1/4)`/`x^(+1/8)` bounds are hardcoded (`RHMC_LO=0.1`/`RHMC_HI=64`); when the doubled-M48 spectrum drifts out of that window the per-shift multishift-CG tolerances go absurdly tight (CG stalls) and the Remez over-extrapolates (huge PF forces → dH blowup). `RAT_AUTO_HI=1` (default **OFF**) runs a plain Lanczos (`RAT_LANCZOS_NM=30`) on the action's own operator at each `refresh()` — EO: `Mpc†Mpc` RB-Odd; Full: `M†M` full grid — and rebuilds the Remez if `[RAT_LO_SAFETY·λ_min, RAT_HI_SAFETY·λ_max]` (defaults 0.5/1.5) exceeds the current bounds. Default-off is byte-identical to the fixed-bound path. Wired into both `DTXQCDWilsonCloverRationalEOAction::refresh` and the Full action's `refresh`.
- **Stout smearing.** `DTXQCDSmearedConfiguration.h` wraps Grid's `SmearedConfiguration<PeriodicGimplR>` exactly like the TXQCD analogue: gauge `.U` through the stout chain, the six aux fields pass through unchanged, `smeared_force` chain-rules only the gauge force (aux forces untouched — aux don't enter the stout map). Gated by `STOUT_NSMEAR` (params.h default **1**; `=0` is the unsmeared csw=0 scout path) and `STOUT_RHO` (default 0.125); the generator sets `is_smeared=true` on the gauge/fermion/LogDet actions when smearing. Required for the chroma-matched **csw=1.249** point (clover on rough links needs smearing). Validated end-to-end at csw=1.249 (finite dH, signPf observer live); run a reversibility/dH-scaling check at the production volume before a long stream.
- **Generator gotcha:** the production generators take the lattice from `lattice_size()` (env `LATT` / `params.h` default), **not** from Grid's `--grid` flag — `--grid` is parsed by Grid but ignored for the field size. Always set `LATT=…` to change volume.
- **Formal gates (must stay green).** `Test_dtxqcd_gamma5_herm_full`, `Test_dtxqcd_pfaffian_antisymmetry`, and the freefield Fierz checks (`Test_dtxqcd_freefield_qbarq_{heavy,light}{,_eo}`) defend the convention. Re-run the full `tests/dtxqcd` + `tests/txqcd` sweep after any structural change.
- **GPU acceleration.** The per-site 48×48 dense work (the dominant cost) is offloaded to batched cuBLAS (`getrfBatched`/`getriBatched`) via the shared `DTXQCDBatchedInverse48.h` helper. Env knobs (default ON under CUDA, OFF/ignored on non-CUDA builds), each with the Eigen CPU path retained as a bit-comparable reference: `DTXQCD_PRECOMPUTE_GPU` (EO `BuildInverseCacheCB` Mooee⁻¹ cache), `DTXQCD_LOGDET_S_GPU` (LogDet `S` = −½Σ log|det| via device LU diagonal), `DTXQCD_LOGDET_GPU` (LogDet `deriv` — batched inverse + the existing CPU force kernel). Dispatch is `#ifdef GRID_CUDA`-gated (not just the cuBLAS call) so non-CUDA builds always take the CPU path. **nvcc gotcha:** under `GRID_CUDA`, Grid's `ComplexD` is `thrust::complex` — keep Eigen matrices in `std::complex<double>` and use the portable `DtxqcdToStd/DtxqcdConj/DtxqcdAbs/DtxqcdArg` helpers (in `DTXQCDAuxFieldTypes.h`); never `peekSite` inside a `thread_for` (corrupts the GPU view lock — build per-site matrices serially or unvectorize first). **Also never hold an `autoView` open across a collective on the same lattice** (e.g. `autoView(v, X, CpuWrite); … sliceSum(X, …)` while `v` is alive) — `sliceSum`/`innerProduct` open their own view and the overlap trips `GRID_ASSERT AccCache.cpuLock==0` on the non-unified GPU manager (CPU-lenient, so it hides until a GPU run); scope the `autoView` in its own `{ }` block so it closes first (was the `DTXQCDAuxCorrelator.h` slice helpers' bug, surfaced by the production sign observer).

## GPU / QUDA / cuBLAS acceleration

Production runs on A100s. Read `GRID_QUDA_ARCHITECTURE.md` and `HANDOFF_AUX_GPU_PORT.md` for the full picture.

- **QUDA wrappers** (`Grid/.../Quda*.h`, `production/quda_*helper.h`): env-gated, off by default, same binary still runs pure-Grid. QUDA accelerates the **Wilson dslash + clover matvec** and the σ-force kernel, and multigrid CG for non-multishift solves (large speedup at light mass). Headline: ~3.9×/6.1× propagator inversion, −36% per-trajectory on Nf=2+1.
- **What QUDA cannot do:** anything involving the aux Δ operator (σ/π/s/p/t live in spin×flavor space; QUDA's clover is spin-only), and multishift CG on the full operator. **MG and multishift do not compose** — Nf=1 RHMC (strange) is forced multishift → no MG → use multi-Hasenbusch mass splitting. Nf=2 light has no multishift → MG works (`USE_HMC_MG=1`).
- **Per-site dense work** (24×24 / 48×48 LU and traces) uses **cuBLAS batched** (`cublasZgetrf/getri/gemmBatched`), not QUDA. Cache cuBLAS handles at class scope and `cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)` — both have bitten before.
- **Env flags** gate the GPU paths (e.g. `TXQCD_QUDA_HYBRID/FULL`, `TXQCD_PRECOMPUTE_GPU`, `TXQCD_MOOEE(INV)_CUBLAS`, `TXQCD_LOGDET_S_GPU`/`_GPU`, `USE_QUDA_MG`/`USE_HMC_MG`, `QUDA_ENABLE_MPS`, `QUDA_ENABLE_MANAGED_MEMORY`). The full production block is in `HANDOFF_AUX_GPU_PORT.md` §4. **Do not disable perf flags as a workaround** — they have load-bearing side effects (UCX-CUDA registration, IPC) and turning them off has caused 100×+ slowdowns.
- **CPU↔GPU porting:** dispatch must be gated on `GRID_CUDA` (not just the cuBLAS call) or non-CUDA builds silently compute wrong answers; `Eigen` and `if constexpr` don't work inside `accelerator_for`; view-lifetime bugs are the #1 "works on CPU, breaks on GPU" cause. The TXQCD LogDet GPU pipeline (`TXQCDLogDetGpuKernel.h` + Phase J paths in `TXQCDLogDetCloverEOAction.h`) is the template to copy when porting DTXQCD's `S_gpu`/`deriv_gpu` (still CPU-only on `dtxqcd-v2`).

## HMC integrator conventions vs chroma

**Grid's `eps = trajL/MDSTEPS` is ~1/4 of chroma's `dt = tau0/n_steps` for the same physical step size.** So chroma's `n_steps=7 tau0=√2` (dt=0.202) matches Grid's `MDSTEPS=28 trajL=√2` (eps=0.0505) — *not* Grid's MDSTEPS=7. The factor of 4 comes from two compounding conventions:

- **Var(P) ratio = 4×.** Grid samples `Var(P)=2` (CPS_MD_TIME, `HMC_MOMENTUM_DENOMINATOR=2`); chroma samples `Var(P)=1/2`.
- **Implicit kinetic mass.** Grid `H=(1/2)|P|²` with Var(P)=2 ⇒ `m_grid=2`; chroma `H=|P|²` with Var(P)=1/2 ⇒ `m_chroma=1/2`. Combined, `dU=P·dt` relates the two by 4×.

**Empirical (chroma `cl3_16_48_b6p1_m0p2450`, NO_METROP, trajL=√2):**

| Grid MDs | eps | dH | plaq |
|---|---|---|---|
| 7 (chroma nominal, 4× too large) | 0.20 | 1.5M | 0.42 |
| 14 | 0.10 | 22k | 0.51 |
| 28 (≡ chroma's stable setting) | 0.05 | -0.18 | 0.5135 ✓ |
| 56 | 0.025 | 0.029 | 0.5135 ✓ |

The instability at chroma's nominal MDs=7 is a true symplectic-stability boundary, not a force bug: `TEST_FORCE_FD=1` in the generators verifies every action gradient against `(S(U+εp)−S(U−εp))/2ε` to 8 digits at `CG_TOL=1e-13`.

**Production settings.** Canonical for 16³×48 (matches chroma gauge-gen 1:1): `INTEGRATOR=MinimumNorm2 MDSTEPS=10 TRAJL=√2/4` — the ¼ rescale gives traj-to-traj correspondence with chroma `n_steps=10`. The older `INTEGRATOR=ForceGradient MDSTEPS=20-28 TRAJL=√2` (~50-70% acceptance) is equivalent at the same physical step size. Multi-rate: gauge sub-integrator ×4 finer than fermion, aux ×1 (slowest). See `production/HMC_CONVENTIONS.md` and `production/HMC_TUNING_LESSONS.md`. `NO_METROP=K` force-accepts the **first K trajectories** (not just traj 0, and not the whole run). Watch `Fdt max` in the log: <0.5 comfortable, >1 unstable, >10 catastrophic.

## Repo layout pointers

- `Grid/` — the library; new code under `Grid/qcd/action/txqcd/` and `Grid/qcd/action/dtxqcd/`, plus the QUDA glue (`Grid/.../Quda*.h`).
- `tests/txqcd/`, `tests/dtxqcd/` — correctness (operator identities, finite-difference force checks, Fierz/freefield end-to-end comparisons) and benchmarks. The reference for any new feature; gate tests must stay green.
- `HMC/` — short HMC driver prototypes (stock Grid + TXQCD small-lattice examples).
- `production/` — out-of-tree generators + measurements for the real ensembles (16³×48 b6.1, 32³×64 b6.5, 48³×96 b6.3, Wilson-Clover m=-0.245 csw=1.24930970916466 β=6.1). Many `slurm_*.sh`/`run_*.sh` launch scripts, plus status notes (`B6P3_HMC_STATUS.md`, `DTXQCD_GPU_HANDOFF.md`, `PERLMUTTER_HANDOFF.md`) and Python analysis (`analyze_*.py`).
- `systems/<machine>/` — known-good `configure` invocations and `sourceme.sh` per target machine.
- `benchmarks/`, `BLAS_benchmark/`, `MPI_benchmark/` — performance tests (separate from correctness).

## Conventions and pitfalls

- **APBC time default.** Every TXQCD/DTXQCD Wilson-family operator defaults to antiperiodic time (chroma convention); stock Grid `WilsonFermion`/`WilsonCloverFermion` default to PBC. When comparing TXQCD/DTXQCD ↔ stock Grid, pass `boundary_phases[Nd-1] = -1.0`. Also: TXQCD clover computes `F_μν` from raw U (no BC phases), while stock Grid applies BC phases first — they disagree by O(BC effect) at the time boundary by design.
- **Branches.** Work on `develop` for TXQCD-core changes; `master` is for releases (don't PR there). The DTXQCD effort lives on `dtxqcd-v2` (cut from `develop`) — **do not rebase it onto `develop` without re-running the full gate sweep**, as the sigmaHerm/v1→v2 transition spans several commits. Keep TXQCD/DTXQCD as additive diffs to upstream Grid (no merge commits without an explicit ask) so rebasing onto new Grid releases stays manageable.
- **Bash hook.** A repo hook blocks compound bash (`;`, `&&`, `||`, newlines) after a past `cd`-then-`rm` data-loss incident — one statement per call; use absolute paths instead of `cd`. Pipes/redirects are allowed.
