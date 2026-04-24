#!/bin/bash
#SBATCH --job-name=qcd_tscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_tscan.%j.out

# trajL scan at fixed MDs=7 (chroma's baseline for cl3_16_48_b6p1_m0p2450).
# Post is_smeared=false fix: LW gauge action acts on thin links, so large
# trajL should be tractable again.  Four 1-GPU QCD streams per node.
#
#   GPU 0: trajL=0.5               cfgs/qcd_s820_trajL0p5
#   GPU 1: trajL=1.0               cfgs/qcd_s821_trajL1p0
#   GPU 2: trajL=1.4142 (√2, chroma)  cfgs/qcd_s822_trajL1p4
#   GPU 3: trajL=2.0 (aggressive)  cfgs/qcd_s823_trajL2p0

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

launch_stream() {
  local gpu=$1 stream=$2 suffix=$3 trajL=$4 log=$5
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$stream SUFFIX="$suffix" \
    MDSTEPS=7 TRAJL="$trajL" INTEGRATOR=ForceGradient NO_METROP=0 \
    WEAK_FIELD_SCALE=0.1 START_TYPE=tepid \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

launch_stream 0 820 _trajL0p5  0.5               "slurm-logs/qcd_s820_trajL0p5.${SLURM_JOB_ID}.out"
echo "[GPU 0] trajL=0.5"
sleep 2
launch_stream 1 821 _trajL1p0  1.0               "slurm-logs/qcd_s821_trajL1p0.${SLURM_JOB_ID}.out"
echo "[GPU 1] trajL=1.0"
sleep 2
launch_stream 2 822 _trajL1p4  1.4142135623730951 "slurm-logs/qcd_s822_trajL1p4.${SLURM_JOB_ID}.out"
echo "[GPU 2] trajL=sqrt(2)"
sleep 2
launch_stream 3 823 _trajL2p0  2.0               "slurm-logs/qcd_s823_trajL2p0.${SLURM_JOB_ID}.out"
echo "[GPU 3] trajL=2.0"
sleep 2

wait
echo "=== all streams exited ==="
date
