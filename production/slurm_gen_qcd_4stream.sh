#!/bin/bash
#SBATCH --job-name=qcd_4s
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_4s.%j.out

# Four parallel NP=1 HMC streams, one per GPU on an exclusive node.
# Each stream writes to cfgs/qcd_s<ID>/ with a stream-specific RNG seed.
# Restart-safe: each stream resumes from its own latest checkpoint.
# No MPI: HMC is serial per GPU (no inter-stream communication).
#
# This pattern mirrors the lq2 chroma NPLQCD production convention
# (/lustre2/nplqcd/nplqcd_production_quda_cl3_32_48_b6p1_m0p2450) — launch
# N parallel single-GPU processes with CUDA_VISIBLE_DEVICES bound per rank.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

# Each process is single-rank; let OpenMP use 16 cores per GPU (64/4).
export OMP_NUM_THREADS=16

N_STREAMS=${N_STREAMS:-4}

echo "=== Launching $N_STREAMS parallel NP=1 HMC streams ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in $(seq 0 $((N_STREAMS-1))); do
  logfile="slurm-logs/qcd_s${i}.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i STREAM_ID=$i \
      ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2   # stagger startup so CUDA init messages don't interleave
done

# Wait for any to exit (they'll normally run to walltime).
wait

echo "=== All streams exited ==="
date
