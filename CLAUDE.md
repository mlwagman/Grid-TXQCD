# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [Grid](https://github.com/paboyle/Grid) (data-parallel C++ lattice QCD library) extended with **TXQCD** (Tensor eXtended QCD): auxiliary tensor fields coupled to quarks so `det(M)` stays real, enabling correlated-sampling signal-to-noise improvements. Physics and design are written up in `TXQCD_ARCHITECTURE.md` (read first for non-trivial TXQCD work) and `txqcd_notes.tex`.

## Build

Grid uses GNU autotools. The typical flow:

```bash
./bootstrap.sh                         # only needed once (or after m4/configure.ac changes)
mkdir build && cd build
../configure <flags>                   # see systems/*/config-command for known-good flags
make -j                                # builds libGrid
make -C tests/txqcd tests               # builds TXQCD correctness/force tests (not built by default)
```

- Default `make` only builds tests at the root of `tests/` — subdirectory tests (including all `tests/txqcd/`) must be built explicitly.
- Known-good configs live under `systems/<machine>/config-command` (sdcc-genoa, SDCC-ICE, Frontier, Summit, Perlmutter, …). `sourceme.sh` in each sets environment. CPU SIMD flag is `--enable-simd=AVX512|AVX2|…`; GPU builds use `--enable-accelerator={cuda,hip} --enable-simd=GPU`.
- TXQCD-specific compile-time dependencies: `--disable-fermion-reps --disable-gparity` are used in the Grid-TXQCD genoa config and are a reasonable default.

### Production programs (`production/`)

`production/Makefile` builds out-of-tree against an existing Grid build (uses `grid-config`). Edit line 1 (`GRID_CONFIG = ../build/grid-config`) to point at the right build dir (e.g. `../build-gpu/grid-config` for GPU). Then `make` from `production/`. Produces: `gen_{qcd,txqcd}_cfgs`, `meas_{conn,disco}_{qcd,txqcd}`, `meas_aux_txqcd`, `compare`.

All physics parameters for production live in `production/params.h` (lattice size, masses, csw, beta, stout smearing, lambda, trajectory counts). `lambda` (the TXQCD auxiliary coupling) is encoded in output paths: `cfgs/txqcd_lam0.5000/`, `meas_2pt/txqcd_lam0.5000/` — change it in `params.h` and rebuild to run parallel streams.

## Running

```bash
# Single test executable (after `make -C tests/txqcd tests`):
./tests/txqcd/Test_txqcd_wilson_eo --grid 4.4.4.8 --mpi 1.1.1.1

# HMC driver:
./HMC/TXQCD_Wilson_small --grid 4.4.4.4

# Production:
cd production
mpirun -np N ./gen_qcd_cfgs   --grid Lx.Ly.Lz.Lt --mpi mx.my.mz.mt   # resumes from latest checkpoint
mpirun -np N ./gen_txqcd_cfgs --grid Lx.Ly.Lz.Lt --mpi mx.my.mz.mt
./run_all_qcd_measurements.sh   --grid ... --mpi ...                  # sequential driver, skips existing
LAMBDA=0.5000 ./run_all_txqcd_measurements.sh --grid ... --mpi ...
./compare --grid ...                                                    # Fierz-identity cross-check
```

`params.h` also governs measurement behaviour — trajectory range comes from `n_therm`, `n_prod`, `meas_skip`.

## TXQCD architecture — what to know before editing

All new code lives under `Grid/qcd/action/txqcd/` (hub: `Txqcd.h`). Two existing Grid files were modified: `WilsonCloverFermionImplementation.h` (added `MooeeDeriv`/`MooeeInvDeriv` clover force helpers) and `PseudoFermion.h` (new `#include`s). Read `TXQCD_ARCHITECTURE.md` for the full picture; key structural facts:

- **Composite field.** `TXQCDField` bundles `LatticeGaugeField U` with five auxiliary lattice fields (`sigma`, `pi` flavor-space Hermitian; `s`, `p`, `t` color-space Hermitian). `TXQCDCompositeImpl` implements Grid's `FieldImplementation` static interface (momenta, update, kinetic) by delegating gauge → `PeriodicGimplR` and aux → additive/Hermitian. This is how `HybridMonteCarlo`, `LeapFrog`/`ForceGradient`, smearing, checkpointing all work unchanged.
- **Kinetic energy conventions.** Gauge momentum is antihermitian; aux momentum is Hermitian (conjugate to Hermitian fields) — the `1/2` factor on aux traces in `TXQCDCompositeImpl::KineticEnergy` is not a bug. Aux momentum generation scales by `sqrt(HMC_MOMENTUM_DENOMINATOR)` to match Grid's integrator convention.
- **Action adapters.** Mixing TXQCD and plain-QCD pieces in one HMC requires two wrappers: `GaugeActionAdapter<T>` (any `GaugeAction<PeriodicGimplR>` on the composite field) and `QCDActionAdapter` (any `Action<LatticeGaugeField>`, e.g. the strange quark). They extract `U`, delegate, and repack force with aux components zeroed.
- **EO preconditioning.** `TXQCDWilsonCloverFermionEO` stores precomputed LU-factored 24×24 site matrices (Nf·Ns·Nc = 2·4·3) built from the aux fields + clover. LogDet of `M_ee` is exact (no stochastic estimation); the odd-sublattice piece uses RHMC with multishift CG. Per-site scalar ops live in `TXQCDSiteMatrix.h` — this is the main CPU bottleneck and the obvious GPU-optimisation target (batched 24×24 LU via cuBLAS/rocBLAS).
- **Clover force convention.** Grid's `WilsonCloverFermion` stores "Convention B" internally; the force needs "Convention A" — conversion factor `-1/2`. See `MooeeDeriv`/`MooeeInvDeriv` in `WilsonCloverFermionImplementation.h`.
- **Smearing.** `TXQCDSmearedConfiguration` wraps Grid's `SmearedConfiguration<PeriodicGimplR>`: gauge component goes through the stout chain; aux components pass through unchanged. All fermion/gauge actions in `production/` run with `is_smeared = true`.
- **Checkpoint format.** Gauge is standard NERSC (interoperable). Aux fields are a packed-Hermitian binary sidecar `ckpoint_lat_aux.<traj>` (magic `TXQA`, IEEE64BIG). Don't invent a new format — extend the existing one.
- **HMC driver pattern.** TXQCD drivers bypass `HMCResourceManager` (which assumes a gauge-only field) and wire `HybridMonteCarlo + Integrator + RNGs` by hand. See `HMC/TXQCD_Wilson_small.cc` or `production/gen_txqcd_cfgs.cc` as the templates.
- **Measurement inversions** (`meas_conn_*`, `meas_disco_*`): use full-volume `MdagM` CG, *not* EO. This is intentional — measurement inversions are one-shot, HMC forces are what benefit from EO.

## Repo layout pointers

- `Grid/` — the library itself; almost all new code lives under `Grid/qcd/action/txqcd/`.
- `tests/txqcd/` — TXQCD correctness (operator identities, finite-difference force checks, Fierz-identity end-to-end comparisons) and benchmark tests. Use these as the reference for any new TXQCD feature.
- `HMC/` — short HMC driver prototypes (stock Grid + TXQCD small-lattice examples).
- `production/` — out-of-tree Nf=2+1 gauge-generation and measurement programs for the real ensemble (16³×48, Wilson-Clover, m=-0.245, csw=1.24930970916466, β=6.1).
- `systems/<machine>/` — known-good `configure` invocations and `sourceme.sh` for each target machine.
- `benchmarks/`, `BLAS_benchmark/`, `MPI_benchmark/` — performance tests (separate from correctness tests).

## Upstream / branch conventions

- Work on `develop`. `master` is reserved for releases — do not PR to master.
- Don't create merge commits against upstream Grid without an explicit ask; TXQCD extensions are kept as additive diffs to make rebasing onto new Grid releases manageable.
