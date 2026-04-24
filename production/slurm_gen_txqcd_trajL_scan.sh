#!/bin/bash
#SBATCH --job-name=tx_tscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/tx_tscan.%j.out

# TXQCD trajL scan at production 16³×48 at fixed MDs=7.  Post gauge-action
# is_smeared=false fix.  Vanilla RHMC (no Hasenbusch) with aux-mean init
# at Σ_l=-5.9 (sign-flipped ThermalAuxConfiguration).
#
#   GPU 0: trajL=0.2               cfgs/txqcd_lam6.6000_tscan_0p2
#   GPU 1: trajL=0.5               cfgs/txqcd_lam6.6000_tscan_0p5
#   GPU 2: trajL=1.0               cfgs/txqcd_lam6.6000_tscan_1p0
#   GPU 3: trajL=1.4142 (chroma)   cfgs/txqcd_lam6.6000_tscan_1p4

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 suffix=$2 trajL=$3 log=$4
  CUDA_VISIBLE_DEVICES=$gpu LAMBDA=6.6 SUFFIX="$suffix" \
    HASEN_DM=0 AUX_SIGMA_L=-5.9 \
    MDSTEPS=7 TRAJL="$trajL" INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=thermal \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 _tscan_0p2 0.2               "slurm-logs/tx_tscan_0p2.${SLURM_JOB_ID}.out"
echo "[GPU 0] trajL=0.2"
sleep 2
launch_stream 1 _tscan_0p5 0.5               "slurm-logs/tx_tscan_0p5.${SLURM_JOB_ID}.out"
echo "[GPU 1] trajL=0.5"
sleep 2
launch_stream 2 _tscan_1p0 1.0               "slurm-logs/tx_tscan_1p0.${SLURM_JOB_ID}.out"
echo "[GPU 2] trajL=1.0"
sleep 2
launch_stream 3 _tscan_1p4 1.4142135623730951 "slurm-logs/tx_tscan_1p4.${SLURM_JOB_ID}.out"
echo "[GPU 3] trajL=sqrt(2)"
sleep 2

wait
echo "=== all streams exited ==="
date
