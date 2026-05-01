#!/bin/bash
#SBATCH --job-name=eigspec
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/eigspec.%j.out

# Eigenvalue spectrum diagnostic: lowest eigenvalues of M†M for QCD vs TXQCD
# at multiple λ values on a fixed gauge config.
#
# Required env vars:
#   EIG_BIN       which eigspec binary (eigspec_diag for Nf=3, eigspec_diag_nf2 for Nf=2+1)
#   IMPORT_CFG    starting cfg path
#   LAMBDAS       space-separated λ values
#   OUT_FILE      results file (saved alongside log for cross-job comparison)
# Optional:
#   CHEBY_LO, CHEBY_HI, CHEBY_ORD, NEV, NM

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

EIG_BIN=${EIG_BIN:-./eigspec_diag}
OUT_FILE=${OUT_FILE:-slurm-logs/eigspec.${SLURM_JOB_ID}.out}

echo "=== eigspec on $IMPORT_CFG ==="
echo "binary: $EIG_BIN  LAMBDAS: $LAMBDAS"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDAS="$LAMBDAS" \
  CHEBY_LO="${CHEBY_LO-}" CHEBY_HI="${CHEBY_HI-}" \
  CHEBY_ORD="${CHEBY_ORD-}" NEV="${NEV-}" NM="${NM-}" \
  IMPORT_CFG="$IMPORT_CFG" \
  mpirun -np 1 --map-by ppr:1:socket:PE=16 \
      $EIG_BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0
date
