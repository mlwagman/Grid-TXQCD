#!/bin/bash
#SBATCH --job-name=tx_mscan2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/tx_mscan2.%j.out

# TXQCD MDsteps scan at fixed trajL=sqrt(2), λ=6.6, in the stable region
# established by the QCD dt-scan (MDs ≥ 28 for ForceGradient).  Aux-init at
# Σ_l = -5.9.
#
#   GPU 0: MDs=20            cfgs/txqcd_lam6.6000_mscan2_20
#   GPU 1: MDs=28            cfgs/txqcd_lam6.6000_mscan2_28
#   GPU 2: MDs=40            cfgs/txqcd_lam6.6000_mscan2_40
#   GPU 3: MDs=56            cfgs/txqcd_lam6.6000_mscan2_56

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

launch_stream 0 _mscan2_20 20 "slurm-logs/tx_mscan2_20.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=20"
sleep 2
launch_stream 1 _mscan2_28 28 "slurm-logs/tx_mscan2_28.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=28"
sleep 2
launch_stream 2 _mscan2_40 40 "slurm-logs/tx_mscan2_40.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=40"
sleep 2
launch_stream 3 _mscan2_56 56 "slurm-logs/tx_mscan2_56.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=56"
sleep 2

wait
echo "=== all streams exited ==="
date
