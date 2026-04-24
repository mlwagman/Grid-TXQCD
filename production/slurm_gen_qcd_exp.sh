#!/bin/bash
#SBATCH --job-name=qcd_exp
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_exp.%j.out

# Thermalization experiment: hot start + chroma lambda (MN2 lambda=0.1789
# compiled in via Integrator_algorithm.h).  4 streams on one node:
#   streams 100, 101  -- HOT start, MDsteps=7        (baseline)
#   streams 102, 103  -- HOT start, MDsteps=14       (2x MD steps)
#
# For streams 102/103, user plan is to drop MDsteps to 7 after the first
# 100 trajectories by resubmitting with MDSTEPS=7 (the binary picks the env
# var up at startup from the STREAM_ID's latest checkpoint).
#
# STREAM_IDs 100-103 deliberately separate these runs from the in-flight
# tepid-start streams in cfgs/qcd_s0..s3.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16

echo "=== Launching QCD thermalization experiment (lambda=0.1789, hot start) ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Per-stream (GPU_ID, STREAM_ID, MDSTEPS)
STREAM_SPECS=(
  "0 100 7"
  "1 101 7"
  "2 102 14"
  "3 103 14"
)

for spec in "${STREAM_SPECS[@]}"; do
  read -r gpu sid mdsteps <<< "$spec"
  logfile="slurm-logs/qcd_s${sid}_mds${mdsteps}.${SLURM_JOB_ID}.out"
  echo "[stream] GPU=$gpu  STREAM_ID=$sid  MDSTEPS=$mdsteps  log=$logfile"
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=$sid START_TYPE=hot MDSTEPS=$mdsteps \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All streams exited ==="
date
