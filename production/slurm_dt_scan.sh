#!/bin/bash
#SBATCH --job-name=qcd_dtscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/qcd_dtscan.%j.out

# dt-convergence diagnostic.  Same chroma cfg, same RNG (STREAM_ID=850),
# N_TRAJ=1, NO_METROP=1, fixed trajL=sqrt(2).  Vary MDs ∈ {7,14,28,56},
# i.e. dt ∈ {0.20, 0.10, 0.05, 0.025}.  ForceGradient nominal order is 4 →
# dH should fall as ~16x per halving.  If it doesn't, force ≠ ∂S.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CHROMA_CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_dt() {
  local gpu=$1 mds=$2 log=$3
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=850 SUFFIX="_dt${mds}" \
    IMPORT_CFG="$CHROMA_CFG" \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=ForceGradient NO_METROP=1 N_TRAJ=1 \
    START_TYPE=tepid WEAK_FIELD_SCALE=0.0 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_dt 0 7   "slurm-logs/qcd_dt7.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=7  (dt=0.20)"
sleep 2
launch_dt 1 14  "slurm-logs/qcd_dt14.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=14 (dt=0.10)"
sleep 2
launch_dt 2 28  "slurm-logs/qcd_dt28.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=28 (dt=0.05)"
sleep 2
launch_dt 3 56  "slurm-logs/qcd_dt56.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=56 (dt=0.025)"
sleep 2

wait
echo "=== all streams exited ==="
date
