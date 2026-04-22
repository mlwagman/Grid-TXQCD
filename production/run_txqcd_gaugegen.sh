#!/bin/bash
# TXQCD gauge generation driver. Resumes from latest checkpoint in
# cfgs/txqcd_lam<X.XXXX>/.  Lambda comes from params.h.
#
# Set GRID_LAUNCH to control how the binary is invoked:
#   GRID_LAUNCH="srun"              (when running inside a SLURM job)
#   GRID_LAUNCH="mpirun -np 4"      (outside SLURM)
#   GRID_LAUNCH=""                  (single rank, default)
#
# CLI args are forwarded as Grid args (typically --mpi mx.my.mz.mt
# [--shm 2048 --shm-mpi 0]). Lattice size comes from params.h, not --grid.
set -e
cd "$(dirname "$0")"

GRID_LAUNCH="${GRID_LAUNCH:-}"
GRID_ARGS="${@:---mpi 1.1.1.1}"

mkdir -p cfgs/txqcd logs

echo "Starting TXQCD gauge generation..."
echo "Launcher : ${GRID_LAUNCH:-<direct>}"
echo "Grid args: $GRID_ARGS"

$GRID_LAUNCH ./gen_txqcd_cfgs $GRID_ARGS 2>&1 | tee logs/gen_txqcd_cfgs.log
