#!/bin/bash
# Determinism test: run gen_txqcd_cfgs at fixed AUX_SIGMA_L=-6.142 four times
# (2 baseline + 2 with CUDA_LAUNCH_BLOCKING=1).  Per-momentum-component norms
# are printed via [detbug] line in generate_momenta — compare across runs.

set -e
cd "$(dirname "$0")"
mkdir -p slurm-logs/det_test

run_one () {
  local tag=$1; shift
  local cfg_dir="cfgs/txqcd_lam6.6000_${tag}"
  local log="slurm-logs/det_test/${tag}.log"
  rm -rf "$cfg_dir" 2>/dev/null || true

  source ../env_lq2_grid.sh
  export OMP_NUM_THREADS=16

  "$@" CUDA_VISIBLE_DEVICES=0 LAMBDA=6.6 SUFFIX="_${tag}" \
    HASEN_DM=0 AUX_SIGMA_L=-6.142 \
    MDSTEPS=10 TRAJL=0.353553390593274 \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    GAUGE_MULT=4 AUX_MULT=2 \
    NO_METROP=0 N_TRAJ=1 \
    WEAK_FIELD_SCALE=0.05 START_TYPE=thermal \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
      ./gen_txqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > "$log" 2>&1 &
  local pid=$!
  local elapsed=0
  while ! grep -q "Total H before trajectory" "$log" 2>/dev/null; do
    sleep 5
    elapsed=$((elapsed+5))
    if [ "$elapsed" -gt 240 ]; then
      echo "  TIMEOUT $tag" >&2
      kill -9 $pid 2>/dev/null
      return 1
    fi
    if ! kill -0 $pid 2>/dev/null; then
      echo "  Process exited unexpectedly $tag" >&2
      return 1
    fi
  done
  sleep 2
  pkill -9 -P $pid 2>/dev/null || true
  kill -9 $pid 2>/dev/null || true
  pkill -9 -f "gen_txqcd_cfgs.*${tag}" 2>/dev/null || true
  sleep 2
  echo "  [$tag] done"
}

echo "=== det_test: 4 runs at AUX_SIGMA_L=-6.142 ==="
run_one det_a env
run_one det_b env
run_one det_c env CUDA_LAUNCH_BLOCKING=1
run_one det_d env CUDA_LAUNCH_BLOCKING=1
echo
echo "=== Comparison: detbug + S [N][N] H + Total H ==="
for tag in det_a det_b det_c det_d; do
  log="slurm-logs/det_test/${tag}.log"
  echo "--- $tag ---"
  grep -E "\[detbug\]|Total H before trajectory|S \[2\]\[0\] H =|S \[1\]\[0\] H =|S \[0\]\[0\] H =|S \[0\]\[1\] H =" "$log" | head -10
done
