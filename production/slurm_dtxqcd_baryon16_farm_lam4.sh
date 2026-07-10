#!/bin/bash
#SBATCH --job-name=dtx_bar16_farm_l4
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/%x.%j.out
# ============================================================================
# Farm the 16-component aux baryon over ALL thermalized lambda=4 configs, one
# config per GPU, 4 concurrent, churning the backlog.  The lambda=4 tla stream
# is a thermalized chroma fork (aux VEV <s>l2~7.15 + plaq~0.5135 FLAT from
# traj 1), so thermalization cut = drop <=20; measure trajs 30..320 (N_SKIP 10).
# Backend: EO DEFAULT (lambda=4 EO CG ~38s/solve, the fastest single-rhs path;
# do NOT force DTXQCD_MEAS_EO=0).  NTSRC=48 = EVERY source timeslice (T=48):
# 4*48=192 solves/config ~2 h; 30 configs / 4 GPU ~15 h.  Max translation
# averaging for the noise-limited baryon (sqrt(48/8)=2.4x better/config).
# In-allocation 4-GPU parallelism: --gres=gpu:4 + per-launch
# CUDA_VISIBLE_DEVICES=k (NEVER --gres=gpu:1 per step -> same-GPU regrant).
# RESUMABLE: skips configs whose output h5 already exists -> chain to extend.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
cd /lustre2/nplqcd/Grid-DTXQCD/production

# LAM (default 4, via --export=ALL,LAM=5): EO default is the fastest wired
# single-rhs path at lambda>=4 (M_ee cliff is lambda<=3 only).
LAM=${LAM:-4}
CFG_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam${LAM}.0000_tla_fromchroma_cw
OUT_DIR=/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam${LAM}_baryon16
mkdir -p "$OUT_DIR"
NTSRC=${NTSRC:-48}   # every source timeslice (T); binary caps to T anyway
NGPU=4

# Config list: thermalized trajs (drop the <=20 thermalization head).
TRAJS=""
for t in $(ls -v "$CFG_DIR"/ckpoint_lat.* | grep -v daux | sed 's/.*\.//'); do
  [ "$t" -gt 20 ] 2>/dev/null && TRAJS="$TRAJS $t"
done
echo "=== baryon16 farm lambda=$LAM  trajs:$TRAJS  ntsrc=$NTSRC ==="

declare -a SLOT_PID
launch() {  # $1=traj $2=gpu
  local traj=$1 gpu=$2
  echo "  [launch] traj=$traj -> GPU $gpu"
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES="$gpu" \
        LATT=16.16.16.48 LAMBDA_DTXQCD="$LAM" MASS_LIGHT_DTXQCD=-0.245 \
        CFG_DIR="$CFG_DIR" DATA_DIR="$OUT_DIR" \
        MEAS_BARYON_NTSRC="$NTSRC" CG_TOL=1e-6 \
        DTXQCD_MEAS_QUDA=1 OMP_NUM_THREADS=16 \
        QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2 \
    ./meas_baryon_dtxqcd "$traj" --grid 16.16.16.48 --mpi 1.1.1.1 \
        --shm 512 --shm-mpi 1 \
    > "$OUT_DIR/farm.$traj.log" 2>&1 &
  SLOT_PID[$gpu]=$!
}

for traj in $TRAJS; do
  if [ -f "$OUT_DIR/baryon_dtxqcd_${traj}.h5" ]; then
    echo "  [skip] traj=$traj (done)"; continue
  fi
  placed=0
  while [ $placed -eq 0 ]; do
    for g in $(seq 0 $((NGPU-1))); do
      if [ -z "${SLOT_PID[$g]}" ] || ! kill -0 "${SLOT_PID[$g]}" 2>/dev/null; then
        launch "$traj" "$g"; placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 10
  done
done
wait
echo "=== baryon16 farm lambda=4 DONE $(date) ==="
