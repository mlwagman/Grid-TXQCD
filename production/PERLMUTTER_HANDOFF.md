# Perlmutter handoff checklist — b6.3 TXQCD HMC at λ=6 (8-node)

## What's ready in this repo

| File | Purpose |
|--|--|
| `slurm_txqcd_b6p3_lam6_perlmutter_8node.sh` | 8-node sbatch (mpi=2.2.2.4, `--shm-mpi 1`, full QUDA stack, HASEN_LADDER) |
| `slurm_hasen_nscan_16x48.sh` | N-level wallclock scan on lq2 to tune the ladder masses |
| `gen_txqcd_cfgs_2plus1.cc` | New `HASEN_LADDER` env (light→heavy comma list) + `RAT_LO/HI/DEGREE/TOL` envs |
| `systems/Perlmutter/{sourceme.sh,config-command}` | Stock Grid Perlmutter build files (unchanged) |
| `B6P3_HMC_STATUS.md` | Full diagnosis history (shm-mpi cliff, 9 GB OOM, ladder solution) |

## One-time Perlmutter setup

```bash
# 1. Clone + build on Perlmutter
ssh perlmutter
cd $HOME
git clone <git-remote>/Grid-TXQCD.git    # or rsync from lq2
cd Grid-TXQCD
source systems/Perlmutter/sourceme.sh
./bootstrap.sh
mkdir build && cd build
../systems/Perlmutter/config-command     # check QUDA path inside!
make -j 32                                # ~30 min
cd ../production && make gen_txqcd_cfgs_2plus1

# 2. Stage chroma cfg2000 (~2 GB)
scp lq2:/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime \
    $PSCRATCH/cfgs/b6p3/

# 3. Validate build: 4⁴ NO_METROP smoke at 1 GPU (~12 min)
cd ../production
RAT_LO=0.01 RAT_HI=10.0 RAT_DEGREE=8 RAT_TOL=1e-5 \
HASEN_LADDER="0.3,0.5,0.7" MASS_LIGHT=0.3 MASS_STRANGE=1.5 CSW=0 \
LATT=4.4.4.4 NO_METROP=1 MDSTEPS=2 N_TRAJ=1 LAMBDA=2.0 AUX_INIT=0.1 \
TRAJL=0.353553390593274 \
srun -N 1 -n 1 --gpus-per-task=1 ./gen_txqcd_cfgs_2plus1 \
    --grid 4.4.4.4 --mpi 1.1.1.1 --shm 256 --shm-mpi 1
# Expect: finite dH printout, exit 0.
```

## HASEN_LADDER tuning (do this BEFORE 8-node sbatch)

**Default (matches chroma's b6.3 ladder for cfg2000 — extracted from LIME header):**

```
HASEN_LADDER="-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
```

This is a **5-level** ladder.  4 ratio actions + 1 rational at the heaviest anchor.
Mass steps mirror chroma's `has3/has2/has1/has0` monomial sequence exactly.
Chroma anchors with `TWO_FLAVOR_EOPREC_CONSTDET_FERM_MONOMIAL` at m=−0.1870;
TXQCD substitutes a rational action there (σ field requires rational treatment).

If running an N-scan to compare wallclock with shorter ladders, the
chroma-matched 5-level is the gold-standard reference — anything cheaper is a
tradeoff between trajectory time and Hasenbusch force suppression.

## Sbatch directives — check before submit

```
#SBATCH -A m3886_g            # your Perlmutter alloc charge code
#SBATCH -q regular            # `debug` for first smoke (≤30 min, ≤2 nodes), `regular` for prod
#SBATCH -t 12:00:00           # wallclock — at ~3-4 hr/traj, 12 hr gives ~3 trajectories
#SBATCH -N 8                  # 8 nodes × 4 GPU = 32 ranks
```

## Required env (already in the sbatch)

- `MPICH_GPU_SUPPORT_ENABLED=1`, `MPICH_RDMA_ENABLED_CUDA=1` (Perlmutter MPICH-CUDA)
- `--shm-mpi 1` (CRITICAL — see `reference_shm_mpi_1_fix` memory; without it, Meooe is 237× slower)
- `TXQCD_QUDA_HYBRID=1 TXQCD_QUDA_FULL=1` (Phase H force kernel)
- `USE_HMC_MG=1` (MG for QCD strange RHMC)
- `TXQCD_PRECOMPUTE_GPU=1 TXQCD_MOOEE{,_INV}_CUBLAS=1` (cuBLAS per-iter ops)
- `IMPORT_CFG=$PSCRATCH/cfgs/b6p3/cl21_..._cfg_2000.lime`

## Expected wallclock estimate (8-node, mpi=2.2.2.4)

- Per-rank lattice: 24×24×24×24 (cube — best halo balance)
- Per-rank fermion: ~330k sites
- Per-iter CG (post-shm-mpi-1): ~20-25 ms (extrapolated from lq2 4-node mpi=2.2.2.2 36ms scaling by compute/2 + comm)
- TXQCD light multishift CG iters at b6.3 m=−0.2416: ~14000-16000 per CG (per lq2 measurement)
- **MDS=10 trajectory: ~3-4 hr**
- 12 hr alloc → ~3 trajectories per submit; long ensembles need many resubmits

## Memory check

- 80 GB A100 per rank
- Phase H force kernel: ~9 GB transient at mpi=2.2.2.2 (lq2 finding) → ~4.5 GB at mpi=2.2.2.4 (Perlmutter halves per-rank lattice)
- TXQCD precompute + MG + smearing + multishift PF: ~50 GB base
- Comfortable headroom on Perlmutter at 8 nodes; OOM resolved vs lq2 4-node.

## What to watch in the first traj

1. `Meooe (Wilson):` per-call time in destructor dump — expect **2-3 ms** not 500+
2. CG iter counts per refresh/action eval — should match the lq2 numbers
3. `dH` per trajectory — first traj at MDS=10 expected dH ~1-10
4. No `cudaMalloc failed` lines

## Backout / debugging knobs

- If OOM: `unset TXQCD_QUDA_FULL` to fall back to Grid TXQCD force (slower but ~9 GB less memory)
- If multishift slow: add `QUDA_VERBOSE_HMC=1` to see per-iter solver detail
- If integrator unstable (large dH): drop MDSTEPS to 6 or 8, check basin via `EIG_DIAG=1` output
