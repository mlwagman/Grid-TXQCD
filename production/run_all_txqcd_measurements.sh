#!/bin/bash
# TXQCD measurements driver (conn + disc + aux for each available config).
# Lambda is passed via LAMBDA env var and must match params.h.
#
# Set GRID_LAUNCH to control how each measurement binary is invoked:
#   GRID_LAUNCH="srun"              (inside a SLURM job)
#   GRID_LAUNCH="mpirun -np 4"      (outside SLURM)
#   GRID_LAUNCH=""                  (single rank, default)
#
# CLI args are forwarded as Grid args (typically --mpi mx.my.mz.mt
# [--shm 2048 --shm-mpi 0]). Lattice size comes from params.h, not --grid.
set -e
cd "$(dirname "$0")"

GRID_LAUNCH="${GRID_LAUNCH:-}"
LAMBDA="${LAMBDA:-0.5000}"
DATA_DIR="meas_2pt/txqcd_lam${LAMBDA}"
CFG_DIR="cfgs/txqcd_lam${LAMBDA}"
MIN_SIZE=1000  # minimum valid output file size in bytes
GRID_ARGS="${@:---mpi 1.1.1.1}"

mkdir -p "$DATA_DIR" logs

# Remove broken (incomplete) output files not currently being written
for f in "$DATA_DIR"/*_txqcd_*.h5; do
  [ -f "$f" ] || continue
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
  if [ "$sz" -lt "$MIN_SIZE" ]; then
    if ! lsof "$f" >/dev/null 2>&1; then
      echo "Removing incomplete file: $f ($sz bytes)"
      rm -f "$f"
    fi
  fi
done

# Measurement trajectories: n_therm to n_therm+n_prod step meas_skip
N_THERM=300
N_PROD=500
MEAS_SKIP=10

for ((t=N_THERM; t<N_THERM+N_PROD; t+=MEAS_SKIP)); do
  # Check config exists
  if [ ! -f "$CFG_DIR/ckpoint_lat.$t" ]; then
    continue
  fi

  # Connected
  if [ ! -f "$DATA_DIR/conn_txqcd_$t.h5" ]; then
    echo "=== meas_conn_txqcd traj=$t ==="
    $GRID_LAUNCH ./meas_conn_txqcd $t $GRID_ARGS 2>&1 | tee "logs/conn_txqcd_lam${LAMBDA}_$t.log"
  fi

  # Disconnected
  if [ ! -f "$DATA_DIR/disco_txqcd_$t.h5" ]; then
    echo "=== meas_disco_txqcd traj=$t ==="
    $GRID_LAUNCH ./meas_disco_txqcd $t $GRID_ARGS 2>&1 | tee "logs/disco_txqcd_lam${LAMBDA}_$t.log"
  fi

  # Auxiliary fields
  if [ ! -f "$DATA_DIR/aux_txqcd_$t.h5" ]; then
    echo "=== meas_aux_txqcd traj=$t ==="
    $GRID_LAUNCH ./meas_aux_txqcd $t $GRID_ARGS 2>&1 | tee "logs/aux_txqcd_lam${LAMBDA}_$t.log"
  fi
done

echo "All available TXQCD measurements (lambda=$LAMBDA) complete."
