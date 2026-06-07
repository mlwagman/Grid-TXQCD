#!/bin/bash
# 2-node 8-GPU production-ish run for b6.5 32³×64 with TWO optimizations vs
# the conservative baseline:
#   1. MULTs reduced 4/4/4 → 4/2/1 (Fdt diagnostic on v8 showed aux Fdt 13×
#      smaller than TXQCD light → big headroom to coarsen aux sub-stepping).
#   2. SMEAR-SKIP patch (TXQCDSmearedConfiguration.h, 2026-05-23): guards
#      gauge_smearing_.set_Field() with a norm2 check on U, so aux-only Q
#      steps no longer trigger re-smearing.  v8 was doing 910 smearings (≈
#      every aux Q step at ~360 ms each = 5.5 min wasted/traj).
#
# Combined target: ~50% wallclock reduction vs v8 (~54 min → ~27 min/traj).
# Stability check: dH should stay close to v8 baseline (-3.0).
#
# Usage:   bash run_b6p5_10traj_2node_mults.sh [<MDS>]
#          MDS default 10.
# Output:  cfgs/txqcd_lam8.0000_b6p5_lam8_mds<MDS>_2node_g4i2a1_first10/
#          logs/b6p5_10traj_2node_mds<MDS>_g4i2a1.log

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
export MASS_STRANGE=-0.1788
export LAMBDA=8
export AUX_INIT=2.9849

# ===== production HMC parameters =====
export N_TRAJ=10
export N_SKIP=1
export NO_METROP=0
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=$MDS
export TRAJL=0.353553390593274

# ===== REDUCED MULTs (the experiment) =====
export GAUGE_MULT=5
export GAUGE_INNER_MULT=2                                    # was 4 — Fdt 28× smaller than TXQCD light, headroom to coarsen
export AUX_MULT=1                                            # was 4 — Fdt 13× smaller; also halves smearing count
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
export OMP_NUM_THREADS=4

# ===== chroma source cfg =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

# ===== 2-node 8-GPU split =====
NGPU=8
MPI=1.1.2.4
export SUFFIX="_b6p5_lam8_mds${MDS}_2node_g4i2a1_first10"
unset CUDA_VISIBLE_DEVICES

LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"

LOG="logs/b6p5_10traj_2node_mds${MDS}_g4i2a1.log"
echo "=== b6.5 32³×64 m=-0.1788 λ=8 MDS=$MDS MULT=4/2/1 — 2-node 8-GPU N_TRAJ=10 ===" | tee -a "$LOG"
echo "cfg_dir=$CFG_DIR" | tee -a "$LOG"
echo "start=$(date)" | tee -a "$LOG"

srun --overlap --mpi=pmix -N 2 -n 8 --cpu-bind=none \
    ./gen_txqcd_cfgs_2plus1 \
        --mpi "$MPI" --shm 1024 --shm-mpi 1 \
    >> "$LOG" 2>&1

echo "=== exit code $? — end=$(date) ===" | tee -a "$LOG"
echo "cfgs saved:" | tee -a "$LOG"
ls "$CFG_DIR"/ckpoint_lat.* 2>/dev/null | sort -V | tee -a "$LOG"
