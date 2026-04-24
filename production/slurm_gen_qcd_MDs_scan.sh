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

# MDsteps scan at fixed trajL=sqrt(2) (chroma's reference trajectory length
# for cl3_16_48_b6p1_m0p2450).  Post is_smeared=false fix, the gauge action
# operates on thin links — same convention chroma uses.
#
#   GPU 0: MDs=5 (aggressive)   cfgs/qcd_s830_MDs5
#   GPU 1: MDs=7 (chroma)       cfgs/qcd_s831_MDs7
#   GPU 2: MDs=10               cfgs/qcd_s832_MDs10
#   GPU 3: MDs=15 (conservative) cfgs/qcd_s833_MDs15

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

launch_stream 0 830 _MDs5  5  "slurm-logs/qcd_s830_MDs5.${SLURM_JOB_ID}.out"
echo "[GPU 0] MDs=5"
sleep 2
launch_stream 1 831 _MDs7  7  "slurm-logs/qcd_s831_MDs7.${SLURM_JOB_ID}.out"
echo "[GPU 1] MDs=7"
sleep 2
launch_stream 2 832 _MDs10 10 "slurm-logs/qcd_s832_MDs10.${SLURM_JOB_ID}.out"
echo "[GPU 2] MDs=10"
sleep 2
launch_stream 3 833 _MDs15 15 "slurm-logs/qcd_s833_MDs15.${SLURM_JOB_ID}.out"
echo "[GPU 3] MDs=15"
sleep 2

wait
echo "=== all streams exited ==="
date
