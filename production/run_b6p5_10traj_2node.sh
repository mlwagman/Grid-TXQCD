#!/bin/bash
# 2-node 8-GPU production run for b6.5 32³×64 m=-0.1788 TXQCD at λ=8.
# Saves cfgs every traj (N_SKIP=1).  Per-rank 32×32×16×16 = 262K sites
# (half the 4-GPU single-node case → full cuBLAS Mooee + PRECOMPUTE_GPU
# stack fits comfortably ~40-50 GB per GPU vs OOM at 78 GB on 4-GPU).
#
# Usage:   bash run_b6p5_10traj_2node.sh [<MDS>]
#          MDS default 10 (b6.1 production value).
# Output:  cfgs/txqcd_lam8.0000_b6p5_lam8_mds<MDS>_2node_first10/
#          logs/b6p5_10traj_2node_mds<MDS>.log
#
# Multi-node srun pattern uses srun_gpu_wrapper.sh for per-rank
# CUDA_VISIBLE_DEVICES binding (see MULTI_GPU_SRUN_NOTES.md).  Caller
# must already be inside the 2-node interactive allocation.

set -u
MDS="${1:-10}"

cd /lustre2/nplqcd/Grid-TXQCD/production
mkdir -p logs cfgs

source ../env_lq2_grid.sh

# ===== b6.5 ensemble parameters =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788                                  # Nf=2+1 degenerate
export LAMBDA=8
export AUX_INIT=2.9849                                       # Σ = vev_trminv

# ===== production HMC parameters =====
export N_TRAJ=10
export N_SKIP=1                                              # save every traj
export NO_METROP=0                                           # WITH metropolis
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=$MDS
export TRAJL=0.353553390593274
export GAUGE_MULT=5
export GAUGE_INNER_MULT=4
export AUX_MULT=4                                            # b6.1 production ratios
export HASEN_DM=0

# ===== full QUDA + cuBLAS performance stack (memory fits at 8-GPU per-rank 262K) =====
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
export EIG_DIAG=1                                            # signed γ5·M eigs in h5
export QUDA_ENABLE_MPS=0
export OMP_NUM_THREADS=4                                     # 4 GPUs per node × 4 threads = 16 (cgroup limit)

# ===== chroma source cfg =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

# ===== 2-node 8-GPU split =====
NGPU=8
MPI=1.1.2.4
export SUFFIX="_b6p5_lam8_mds${MDS}_2node_first10"
# Grid binary was rebuilt 2026-05-23 with --enable-setdevice=yes (see
# build-gpu-nvtx/Grid/Config.h note), so each rank calls cudaSetDevice
# itself based on its local rank BEFORE allocating.  Leave CUDA_VISIBLE_DEVICES
# unfiltered (all GPUs visible) so QUDA's per-node GPU count check passes —
# srun_gpu_wrapper.sh's single-GPU filter conflicts with QUDA's
# comm_init_common requiring n_visible >= n_local_ranks.
unset CUDA_VISIBLE_DEVICES

LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"

LOG="logs/b6p5_10traj_2node_mds${MDS}.log"
echo "=== b6.5 32³×64 m=-0.1788 λ=8 MDS=$MDS — 2-node 8-GPU N_TRAJ=10 ===" | tee -a "$LOG"
echo "cfg_dir=$CFG_DIR" | tee -a "$LOG"
echo "start=$(date)" | tee -a "$LOG"

# 2-node srun with --overlap (inside an interactive bash step).  No wrapper:
# Grid sets device from local rank itself; QUDA assigns from its view of all
# visible GPUs (rank N → device N%n_visible).
srun --overlap --mpi=pmix -N 2 -n 8 --cpu-bind=none \
    ./gen_txqcd_cfgs_2plus1 \
        --mpi "$MPI" --shm 1024 --shm-mpi 1 \
    >> "$LOG" 2>&1

echo "=== exit code $? — end=$(date) ===" | tee -a "$LOG"
echo "cfgs saved:" | tee -a "$LOG"
ls "$CFG_DIR"/ckpoint_lat.* 2>/dev/null | sort -V | tee -a "$LOG"
