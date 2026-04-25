#!/bin/bash
#SBATCH --job-name=qcd_dtmn2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/qcd_dtmn2.%j.out

# MN2 dt-convergence diagnostic — mirror of slurm_dt_scan.sh but with
# MinimumNorm2 instead of ForceGradient.  Same chroma cfg, same RNG,
# N_TRAJ=1, NO_METROP=1, fixed trajL=sqrt(2).  MN2 is 2nd order so
# dH ~ dt^2 (vs dt^4 for FG); 2 force evals/step (vs ~4 for FG) — net
# cost-efficiency at our production dt depends on prefactor.
#
#   GPU 0: MDs=14 (dt=0.10) — was rejection-killer w/ FG
#   GPU 1: MDs=28 (dt=0.05) — production-viable w/ FG
#   GPU 2: MDs=56 (dt=0.025) — overkill w/ FG
#   GPU 3: MDs=112 (dt=0.0125)

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CHROMA_CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_dt() {
  local gpu=$1 mds=$2 log=$3
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=860 SUFFIX="_dtmn2_${mds}" \
    IMPORT_CFG="$CHROMA_CFG" \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=MinimumNorm2 NO_METROP=1 N_TRAJ=1 \
    START_TYPE=tepid WEAK_FIELD_SCALE=0.0 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_dt 0 14  "slurm-logs/qcd_dtmn2_14.${SLURM_JOB_ID}.out"
echo "[GPU 0] MN2 MDs=14  (dt=0.10)"
sleep 2
launch_dt 1 28  "slurm-logs/qcd_dtmn2_28.${SLURM_JOB_ID}.out"
echo "[GPU 1] MN2 MDs=28  (dt=0.05)"
sleep 2
launch_dt 2 56  "slurm-logs/qcd_dtmn2_56.${SLURM_JOB_ID}.out"
echo "[GPU 2] MN2 MDs=56  (dt=0.025)"
sleep 2
launch_dt 3 112 "slurm-logs/qcd_dtmn2_112.${SLURM_JOB_ID}.out"
echo "[GPU 3] MN2 MDs=112 (dt=0.0125)"
sleep 2

wait
echo "=== all streams exited ==="
date
