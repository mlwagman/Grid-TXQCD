#!/bin/bash
# Per-rank GPU binding wrapper for multi-rank Grid runs.
# Grid is built with --enable-setdevice=no, so the user must export
# CUDA_VISIBLE_DEVICES before exec'ing the binary, distinct per rank.
#
# Works under both srun (SLURM_LOCALID) and mpirun (OMPI_COMM_WORLD_LOCAL_RANK).
# With --ntasks-per-node=4 (or -np 4 single-node mpirun) and 4 GPUs per node,
# mapping local rank -> CUDA_VISIBLE_DEVICES gives each rank a distinct GPU.
LOCAL_RANK=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
exec "$@"
