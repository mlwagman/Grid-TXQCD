# Multi-GPU / multi-node srun notes (lq2 cluster, Grid-TXQCD)

How to run any Grid-built binary (`gen_txqcd_cfgs_2plus1`, `compute_vev`,
`meas_*`, etc.) on more than one node, and the gotchas we hit on 2026-05-23
that aren't obvious.

## TL;DR

Inside an `salloc`-allocated interactive bash session on N nodes × 4 GPUs:

```bash
source ../env_lq2_grid.sh

srun --overlap --mpi=pmix \
     -N $SLURM_NNODES -n $(( SLURM_NNODES * 4 )) \
     --cpu-bind=none \
     ./srun_gpu_wrapper.sh \
     ./<binary> --grid Lx.Ly.Lz.Lt --mpi mx.my.mz.mt <args...>
```

`srun_gpu_wrapper.sh` is a 4-line shim (already in `production/`):

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
```

That's the whole recipe. The notes below explain *why* each piece is needed.

## Things that fail and why

### 1. Don't use `mpirun` for multi-node

```bash
mpirun -np 8 --bind-to none ... ./binary    # ❌ multi-node fails
```

Error you'll see:

```
An ORTE daemon has unexpectedly failed after launch and before
communicating back to mpirun.
```

OpenMPI 4.1.5's `mpirun` can't spawn daemons on remote nodes through our
slurm setup (no keyless ssh between nodes, no IP route OpenMPI can find on
its own). Use `srun` — slurm-native, talks to its own PMIx server.

### 2. You need `--overlap` if you're already inside an interactive shell

When you `salloc -N 2 ...`, slurm creates step `<jobid>.0 = bash` that
*occupies the full allocation*. A naive `srun` inside that bash will hang
indefinitely waiting for resources that already belong to the bash step.

`--overlap` tells slurm "let this new step share the resources of the
existing step." Without it: silent hang. With it: works.

If you instead launch from `sbatch` (so the script *is* the first step),
you don't need `--overlap` — the reference NPLQCD `lq2_production_*.sh`
scripts run via sbatch and use plain `srun --mpi=pmix`.

### 3. `--mpi=pmix` (NOT `pmi2`)

Slurm offers several MPI plugins:

```
$ srun --mpi=list
   none
   pmi2
   pmix
   cray_shasta
   specific pmix plugin versions available: pmix_v5,pmix_v4
```

Our OpenMPI 4.1.5 build (via `gompi/2023a`) wants `pmix`. `pmi2` will give
mismatched-protocol errors. The reference NPLQCD scripts all use `pmix`.

### 4. Grid is built with `--enable-setdevice=no` — you MUST set CUDA_VISIBLE_DEVICES per rank

This is the most insidious one. Without a wrapper, all 4 ranks on a node
bind to GPU 0 (same bus id), Grid reports:

```
local rank 0 device 0 bus id: 0000:2F:00.0
local rank 1 device 0 bus id: 0000:2F:00.0   ← same GPU!
local rank 2 device 0 bus id: 0000:2F:00.0
local rank 3 device 0 bus id: 0000:2F:00.0
```

→ effectively one GPU per node, instant OOM on anything larger than a toy
lattice. The Grid binary even prints this warning at startup:

> `AcceleratorCudaInit: assume user either uses (a) IBM jsrun, or
> (b) invokes through a wrapping script to set CUDA_VISIBLE_DEVICES`

The fix is a 4-line bash wrapper that sets `CUDA_VISIBLE_DEVICES` from
`SLURM_LOCALID` and exec's the binary. We keep one at
`production/srun_gpu_wrapper.sh`. **Always go through it for Grid
multi-rank-per-node runs.**

Sanity check after launch — the log should show distinct bus IDs:

```
local rank 0 device 0 bus id: 0000:2F:00.0
local rank 1 device 0 bus id: 0000:30:00.0
local rank 2 device 0 bus id: 0000:AF:00.0
local rank 3 device 0 bus id: 0000:B0:00.0
```

If you see the same bus id repeated, the wrapper isn't taking effect.

### 5. salloc shape that pairs with this recipe

```bash
salloc -N 2                       \
       --ntasks-per-node 4        \
       --gres=gpu:a100:4          \
       --cpus-per-task=16         \
       --time=14:00:00            \
       --partition=lq2_gpu --account=nplqcd.lq2_gpu --qos=normal
```

`--ntasks-per-node=4` matches the 4 GPUs/node. `--cpus-per-task=16`
gives 16 cores per rank (lq2gpu nodes have 64 cores / 4 ranks).

In sbatch scripts, mirror the same `#SBATCH` lines and just drop
`--overlap` from the `srun` invocation (the script itself is step .0).

### 6. MPI grid (`--mpi mx.my.mz.mt`) must divide the lattice and equal `ntasks`

Two independent constraints:
- `mx * my * mz * mt == N_RANKS` (total MPI ranks)
- Each `m_i` must divide `L_i` (so each rank holds an integer slab)

For 48³×96 on 8 ranks: `--mpi 1.1.2.4` gives per-rank `48×48×24×24`
(half-memory split in z and t). Could also use `1.1.4.2`, `2.2.2.1`, etc. —
the choice affects communication pattern, not correctness.

For 16³×48 on 8 ranks: `--mpi 1.1.2.4` gives `16×16×8×12`, fine. (Don't
oversplit a dimension below ~4 sites — halo overhead kills throughput.)

### 7. Module env — `env_lq2_grid.sh` is the canonical loader

Sourcing `../env_lq2_grid.sh` from `production/` (or `Grid-TXQCD/`)
loads: `cmake gompi ucx_cuda ucc_cuda gcc/12.3.0 hdf5/1.14.2_gompi_2023a`.
That gives us OpenMPI 4.1.5 + PMIx 4.2.6 + UCX-CUDA + NCCL — everything
the srun/pmix path needs. **Source it before every multi-node run.**

## Concrete working example (from 2026-05-23 b6.3 48³×96 smoke)

```bash
salloc -N 2 --ntasks-per-node 4 --gres=gpu:a100:4 --cpus-per-task=16 \
       --time=14:00:00 --partition=lq2_gpu --account=nplqcd.lq2_gpu \
       --qos=normal

# now inside the interactive bash on the head node...
cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh

CFG=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/...lime

srun --overlap --mpi=pmix -N 2 -n 8 --cpu-bind=none \
     ./srun_gpu_wrapper.sh ./compute_vev \
       --grid 48.48.48.96 --mpi 1.1.2.4 \
       --mass -0.2416 --csw 1.20536588031793 \
       --n-noise 4 --cg-tol 1e-8 --seed 1234567 \
       "$CFG"
```

This works. Took our 48³ runs from "single-node OOMs in smearing" (4-GPU
split into one node) to a fully-loaded 2-node CG that grinds steadily.

## Quick troubleshoot table

| Symptom | Likely cause |
|---|---|
| `ORTE daemon failed` | Used `mpirun` instead of `srun` |
| `srun` hangs with no output for >60s | Forgot `--overlap` (you're inside interactive) |
| All 4 ranks/node share GPU 0, OOMs immediately | Forgot `srun_gpu_wrapper.sh` |
| `mpi/pmi*: handshake error` | Try `--mpi=pmix` (not pmi2) |
| `Cannot allocate ... bytes` mid-run despite split | MPI grid too narrow on one node; try a wider `mt` split |
| Grid prints same bus id for multiple ranks | wrapper not exec-ing; check `chmod +x srun_gpu_wrapper.sh` |

## Where this came from

Worked out 2026-05-23 trying to fix b6.3 48³×96 VEV OOM (overnight\_orchestrator
hit OOM in smearing because mpi=1.1.2.2 split was per-rank 48²×24×48, too
big for one 80 GB A100). Moving to 2 nodes (mpi=1.1.2.4) halves per-rank
memory and unblocks the run. The friction was all infrastructure — getting
srun to actually launch 8 distinct GPU-bound ranks across 2 nodes.

For the reference scripts that taught us:
- `/lustre2/nplqcd/nplqcd_production_quda_cl3_48_64_b6p1_m0p2450/invert_test_wrapper-48*` (uses `srun -N 1 -n 4 --mpi=pmix` for single-node)
- `/lustre1/nplqcd/nplqcd_production-cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/lq2_run_*` (uses `srun -N 4 -n 16 --mpi=pmix` for multi-node sbatch jobs)
