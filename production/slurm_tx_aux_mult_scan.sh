#!/bin/bash
#SBATCH --job-name=tx_aux_mult
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/tx_aux_mult.%j.out

# TXQCD aux_mult scan with chroma's exact MD settings (translated via the
# Grid eps = chroma_dt/4 convention; see production/HMC_CONVENTIONS.md).
#
#   INTEGRATOR=MinimumNorm2  LAMBDA_MN2=0.1789  GAUGE_MULT=4
#   TRAJL=sqrt(2)/4 = 0.353553   MDSTEPS=7      → physical match to chroma
#                                                  cl3_16_48_b6p1's
#                                                  n_steps=7 tau0=sqrt(2).
#
# Sweep aux_mult ∈ {1,2,4,8} to characterize aux-level stability vs cost.
# Aux action is quadratic (gaussian); its force is linear in field —
# possibly stiff enough that aux_mult>1 is needed for stability.
#
#   GPU 0: aux_mult=1   cfgs/txqcd_lam6.6000_aux1
#   GPU 1: aux_mult=2   cfgs/txqcd_lam6.6000_aux2
#   GPU 2: aux_mult=4   cfgs/txqcd_lam6.6000_aux4
#   GPU 3: aux_mult=8   cfgs/txqcd_lam6.6000_aux8

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 suffix=$2 aux=$3 log=$4
  CUDA_VISIBLE_DEVICES=$gpu LAMBDA=6.6 SUFFIX="$suffix" \
    HASEN_DM=0 AUX_SIGMA_L=-5.9 \
    MDSTEPS=7 TRAJL=0.353553390593274 \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    GAUGE_MULT=4 AUX_MULT="$aux" \
    NO_METROP=0 N_TRAJ=10 \
    WEAK_FIELD_SCALE=0.0 START_TYPE=thermal \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 _aux1 1 "slurm-logs/tx_aux1.${SLURM_JOB_ID}.out"
echo "[GPU 0] aux_mult=1"
sleep 2
launch_stream 1 _aux2 2 "slurm-logs/tx_aux2.${SLURM_JOB_ID}.out"
echo "[GPU 1] aux_mult=2"
sleep 2
launch_stream 2 _aux4 4 "slurm-logs/tx_aux4.${SLURM_JOB_ID}.out"
echo "[GPU 2] aux_mult=4"
sleep 2
launch_stream 3 _aux8 8 "slurm-logs/tx_aux8.${SLURM_JOB_ID}.out"
echo "[GPU 3] aux_mult=8"
sleep 2

wait
echo "=== all streams exited ==="
date
