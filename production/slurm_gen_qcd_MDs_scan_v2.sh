#!/bin/bash
#SBATCH --job-name=qcd_mscan2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_mscan2.%j.out

# Post-dt-scan MDs sweep at trajL=sqrt(2).
# dt-scan from chroma cfg (24-Apr) showed Grid's ForceGradient is unstable at
# chroma's MDs=7 (dH=1.5M from chroma cfg, plaq drift to 0.42); stable at
# MDs=28 (dH≈0.18) and finer.  Stability boundary is between MDs=14 and
# MDs=28 — pick {20, 28, 40, 56} to bracket the production regime.
#
#   GPU 0: MDs=20            cfgs/qcd_s840_MDs20
#   GPU 1: MDs=28            cfgs/qcd_s841_MDs28
#   GPU 2: MDs=40            cfgs/qcd_s842_MDs40
#   GPU 3: MDs=56            cfgs/qcd_s843_MDs56

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 stream=$2 suffix=$3 mds=$4 log=$5
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$stream SUFFIX="$suffix" \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=tepid \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 840 _MDs20 20 "slurm-logs/qcd_s840_MDs20.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=20"
sleep 2
launch_stream 1 841 _MDs28 28 "slurm-logs/qcd_s841_MDs28.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=28"
sleep 2
launch_stream 2 842 _MDs40 40 "slurm-logs/qcd_s842_MDs40.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=40"
sleep 2
launch_stream 3 843 _MDs56 56 "slurm-logs/qcd_s843_MDs56.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=56"
sleep 2

wait
echo "=== all streams exited ==="
date
