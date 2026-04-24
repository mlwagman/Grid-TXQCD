#!/bin/bash
#SBATCH --job-name=tx_mscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/tx_mscan.%j.out

# TXQCD MDsteps scan at fixed trajL=sqrt(2) (chroma's reference length).
# Post gauge-action is_smeared=false fix.  Vanilla RHMC + aux-init at λ=6.6.
#
#   GPU 0: MDs=5               cfgs/txqcd_lam6.6000_mscan_5
#   GPU 1: MDs=7 (chroma)      cfgs/txqcd_lam6.6000_mscan_7
#   GPU 2: MDs=10              cfgs/txqcd_lam6.6000_mscan_10
#   GPU 3: MDs=15              cfgs/txqcd_lam6.6000_mscan_15

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 suffix=$2 mds=$3 log=$4
  CUDA_VISIBLE_DEVICES=$gpu LAMBDA=6.6 SUFFIX="$suffix" \
    HASEN_DM=0 AUX_SIGMA_L=-5.9 \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=thermal \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 _mscan_5  5  "slurm-logs/tx_mscan_5.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=5"
sleep 2
launch_stream 1 _mscan_7  7  "slurm-logs/tx_mscan_7.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=7"
sleep 2
launch_stream 2 _mscan_10 10 "slurm-logs/tx_mscan_10.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=10"
sleep 2
launch_stream 3 _mscan_15 15 "slurm-logs/tx_mscan_15.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=15"
sleep 2

wait
echo "=== all streams exited ==="
date
