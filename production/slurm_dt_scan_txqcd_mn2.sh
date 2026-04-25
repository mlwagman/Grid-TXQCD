#!/bin/bash
#SBATCH --job-name=tx_dtmn2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/tx_dtmn2.%j.out

# TXQCD MN2 dt-scan, λ=6.6.  No chroma reference cfg available for TXQCD,
# so we start from the same tepid + thermal-aux config across streams (RNG
# seed 870 fixed) and compare dH(dt) for ForceGradient vs MinimumNorm2 on
# the TXQCD action structure (composite gauge + 5 aux fields).  N_TRAJ=1
# NO_METROP=1 forces one accept-as-is trajectory to expose pure integrator
# error.
#
#   GPU 0: MN2 MDs=14  (dt=0.10)
#   GPU 1: MN2 MDs=28  (dt=0.05)
#   GPU 2: MN2 MDs=56  (dt=0.025)
#   GPU 3: MN2 MDs=112 (dt=0.0125)

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_dt() {
  local gpu=$1 mds=$2 log=$3
  CUDA_VISIBLE_DEVICES=$gpu LAMBDA=6.6 SUFFIX="_dtmn2_${mds}" \
    HASEN_DM=0 AUX_SIGMA_L=-5.9 \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=MinimumNorm2 NO_METROP=1 N_TRAJ=1 \
    START_TYPE=thermal WEAK_FIELD_SCALE=0.0 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_dt 0 14  "slurm-logs/tx_dtmn2_14.${SLURM_JOB_ID}.out"
echo "[GPU 0] MN2 MDs=14  (dt=0.10)"
sleep 2
launch_dt 1 28  "slurm-logs/tx_dtmn2_28.${SLURM_JOB_ID}.out"
echo "[GPU 1] MN2 MDs=28  (dt=0.05)"
sleep 2
launch_dt 2 56  "slurm-logs/tx_dtmn2_56.${SLURM_JOB_ID}.out"
echo "[GPU 2] MN2 MDs=56  (dt=0.025)"
sleep 2
launch_dt 3 112 "slurm-logs/tx_dtmn2_112.${SLURM_JOB_ID}.out"
echo "[GPU 3] MN2 MDs=112 (dt=0.0125)"
sleep 2

wait
echo "=== all streams exited ==="
date
