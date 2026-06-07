#!/bin/bash
#SBATCH --job-name=b6p5_lam8_2n
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/b6p5_lam8_2n.%j.out

# Production 2-node 8-GPU HMC chain for b6.5 32³×64 m=-0.1788 TXQCD at λ=8.
# Chroma-matched parameters (cl3_32_64_b6p5 cfg4200 XML reference):
#   MDS=16, GAUGE_MULT=5, u0=0.85703554213273, stout ρ=0.125 n=1.
# Performance stack: full QUDA + cuBLAS + PRECOMPUTE_GPU + --shm-mpi 1.
# AUX_MULT=1 (4/2/1 outer×inner×aux) keeps redundant smear-chain calls at
# <0.1% of trajectory cost — no smear-skip optimization needed.
#
# Expected wallclock at MDS=16: ~65 min/traj → ~22 trajs/day per 24h job.
# N_SKIP=10 saves every 10th cfg (production cadence) → 2-3 saved ckpts/day.
#
# Chain via: sbatch --dependency=afterany:<prev> slurm_b6p5_lam8_2node.sh
# Resume: gen_txqcd_cfgs_2plus1 picks up latest ckpoint_lat in cfg_dir; first
# launch uses IMPORT_CFG (chroma cfg_4200), subsequent launches resume from
# last saved ckpt + aux sidecar + RNG state.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs cfgs

# ===== b6.5 ensemble parameters =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788                                  # Nf=2+1 degenerate
export LAMBDA=8
export AUX_INIT=2.9849                                       # Σ = vev_trminv

# ===== HMC parameters =====
export N_TRAJ=2000                                           # production target
# N_SKIP unset → use params.h default (10)
export NO_METROP=0                                           # WITH metropolis
export INTEGRATOR=MinimumNorm2
# Outer MN2 lambda: leave UNSET so Grid uses its default 0.1931833275037836,
# which matches chroma cl3_32_64_b6p5 cfg4200's outer fermion-level lambda
# (Omelyan minimum-error optimum).  Chroma uses a separate 0.1789 at the
# gauge sub-integrator level, but Grid's MN2 has a single class-level lambda
# applied at all levels — pick the outer optimum where stability matters most.
# (b6.1 scripts keep LAMBDA_MN2=0.1789 to match chroma b6.1's single value.)
unset LAMBDA_MN2
export MDSTEPS=16
export TRAJL=0.353553390593274
export GAUGE_MULT=5
export GAUGE_INNER_MULT=2
export AUX_MULT=1
export HASEN_DM=0

# ===== full QUDA + cuBLAS performance stack =====
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
export EIG_DIAG=1
export QUDA_ENABLE_MPS=0
export OMP_NUM_THREADS=4                                     # 4 GPUs/node × 4 threads = 16 (cgroup)

# ===== chroma source cfg (used only if no checkpoints in cfg_dir) =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

# ===== 2-node 8-GPU split =====
export SUFFIX="_b6p5_lam8_mds16_2node"                       # chroma-matched MDS=16, GAUGE_MULT=5
unset CUDA_VISIBLE_DEVICES                                   # Grid handles per-rank binding internally

LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"
echo "=== b6.5 32³×64 λ=8 MDS=16 MULT=5/2/1 chroma-matched — 2-node 8-GPU ==="
echo "cfg_dir=$CFG_DIR"
echo "N_TRAJ=$N_TRAJ (resume from latest ckpt if present)"
date

# sbatch script IS slurm step .0 → no --overlap needed
srun --mpi=pmix -N 2 -n 8 --cpu-bind=none \
    ./gen_txqcd_cfgs_2plus1 \
        --mpi 1.1.2.4 --shm 1024 --shm-mpi 1

echo "=== exit code $? — end $(date) ==="
echo "latest ckpts:"
ls "$CFG_DIR"/ckpoint_lat.* 2>/dev/null | sort -V | tail -5
