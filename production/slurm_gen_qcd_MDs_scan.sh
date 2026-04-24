#!/bin/bash
#SBATCH --job-name=qcd_mscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_mscan.%j.out

# MDsteps scan at fixed trajL=0.2 (confirmed stable in the earlier trajL scan
# on job 1271826).  Nf=2+1 Wilson-Clover on LWv2 gauge action, tepid start,
# NO_METROP=0.  Finds the coarsest dt that still gives acceptable dH — the
# computational optimum.
#
#   GPU 0: MDs=3   (dt=0.067 — aggressive)   cfgs/qcd_s810_MDs3
#   GPU 1: MDs=5   (dt=0.04)                 cfgs/qcd_s811_MDs5
#   GPU 2: MDs=8   (dt=0.025)                cfgs/qcd_s812_MDs8
#   GPU 3: MDs=15  (dt=0.013 — conservative) cfgs/qcd_s813_MDs15

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 stream=$2 suffix=$3 mds=$4 log=$5
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$stream SUFFIX="$suffix" \
    MDSTEPS="$mds" TRAJL=0.2 INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=tepid \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 810 _MDs3  3  "slurm-logs/qcd_s810_MDs3.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=3"
sleep 2
launch_stream 1 811 _MDs5  5  "slurm-logs/qcd_s811_MDs5.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=5"
sleep 2
launch_stream 2 812 _MDs8  8  "slurm-logs/qcd_s812_MDs8.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=8"
sleep 2
launch_stream 3 813 _MDs15 15 "slurm-logs/qcd_s813_MDs15.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=15"
sleep 2

wait
echo "=== all streams exited ==="
date
