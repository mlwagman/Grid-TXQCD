#!/bin/bash
# Scan H_init components vs AUX_SIGMA_L on weak-field gauge.
#
# Each scan point launches gen_txqcd_cfgs, watches for the "Total H before
# trajectory" log line (which is printed after action setup + initial action
# evaluation), then kills the process before MD evolution starts.  Total per
# scan point: ~100 sec wall.
#
# Usage:
#   ./scan_init_action.sh "0 -2 -4 -6 -8 -10 -12"
#
# All other settings fixed: LAMBDA=6.6, MDSTEPS=10, chroma physical match.
# Output: slurm-logs/scan_init_aux/aux_<value>.log + scan_summary.tsv

set -e
cd "$(dirname "$0")"

OUT_DIR="slurm-logs/scan_init_aux"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/scan_summary.tsv"

GPU=${SCAN_GPU:-0}
LIST=${1:-"0 -2 -4 -6 -8 -10 -12"}
SUFFIX_TAG=${SCAN_TAG:-init}

echo -e "AUX_SIGMA_L\tS_TXQCDWilson\tS_TXQCDLogDet\tS_QCDLogDet\tS_StrangeSchur\tS_Gauge\tS_AuxGauss\tH_total" > "$SUMMARY"

run_one () {
  local asl=$1
  local tag=$2
  local cfg_dir="cfgs/txqcd_lam6.6000_${tag}"
  local log="${OUT_DIR}/aux_${tag}.log"

  rm -rf "$cfg_dir" 2>/dev/null || true

  source ../env_lq2_grid.sh
  export OMP_NUM_THREADS=16

  CUDA_VISIBLE_DEVICES=$GPU LAMBDA=6.6 SUFFIX="_${tag}" \
    HASEN_DM=0 AUX_SIGMA_L=$asl \
    MDSTEPS=10 TRAJL=0.353553390593274 \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    GAUGE_MULT=4 AUX_MULT=2 \
    NO_METROP=0 N_TRAJ=1 \
    WEAK_FIELD_SCALE=0.05 START_TYPE=thermal \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
      ./gen_txqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > "$log" 2>&1 &
  local pid=$!

  # Wait for the "Total H before trajectory" line (or timeout).
  local elapsed=0
  while ! grep -q "Total H before trajectory" "$log" 2>/dev/null; do
    sleep 5
    elapsed=$((elapsed+5))
    if [ "$elapsed" -gt 240 ]; then
      echo "  TIMEOUT waiting for H_init at AUX_SIGMA_L=$asl (tag=$tag)"
      kill -9 $pid 2>/dev/null
      return 1
    fi
    if ! kill -0 $pid 2>/dev/null; then
      echo "  Process exited unexpectedly at AUX_SIGMA_L=$asl"
      return 1
    fi
  done
  # Give it 2 more seconds to finish writing the line.
  sleep 2
  # Kill the run + any sibling mpirun/bin.
  pkill -9 -P $pid 2>/dev/null || true
  kill -9 $pid 2>/dev/null || true
  pkill -9 -f "gen_txqcd_cfgs.*${tag}" 2>/dev/null || true
  sleep 2

  # Extract S components.
  local s00=$(grep "S \[0\]\[0\] H =" "$log" | head -1 | awk '{print $NF}')
  local s01=$(grep "S \[0\]\[1\] H =" "$log" | head -1 | awk '{print $NF}')
  local s02=$(grep "S \[0\]\[2\] H =" "$log" | head -1 | awk '{print $NF}')
  local s03=$(grep "S \[0\]\[3\] H =" "$log" | head -1 | awk '{print $NF}')
  local s10=$(grep "S \[1\]\[0\] H =" "$log" | head -1 | awk '{print $NF}')
  local s20=$(grep "S \[2\]\[0\] H =" "$log" | head -1 | awk '{print $NF}')
  local h=$(grep "Total H before trajectory" "$log" | head -1 | awk '{print $NF}')
  echo -e "${asl}\t${s00}\t${s01}\t${s02}\t${s03}\t${s10}\t${s20}\t${h}" >> "$SUMMARY"
  echo "  AUX_SIGMA_L=$asl tag=$tag: H_init=$h"
}

echo "=== Scanning AUX_SIGMA_L on GPU $GPU: $LIST ==="
for asl in $LIST; do
  tag="${SUFFIX_TAG}_${asl//-/m}"
  run_one "$asl" "$tag"
done

echo
echo "=== Scan complete; summary at $SUMMARY ==="
column -t "$SUMMARY"
