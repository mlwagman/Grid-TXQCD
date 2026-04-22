#!/bin/bash
# Per-rank GPU binding helper. Invoked as:
#   mpirun -np N ./select_gpu.sh ./binary args...
# Binds each MPI rank to its local GPU by the OpenMPI local rank.
GPU="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
unset ROCR_VISIBLE_DEVICES
echo "[select_gpu] rank=${OMPI_COMM_WORLD_RANK:-?}  local=${OMPI_COMM_WORLD_LOCAL_RANK:-?}  GPU=$GPU  host=$(hostname)"
exec "$@"
