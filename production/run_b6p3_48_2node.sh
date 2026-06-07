#!/bin/bash
# 2-node 8-GPU compute_vev runs on the b6.3 48^3 x 96 ensemble.
# Plan B: 4 cfgs x n_noise=4 at light (m=-0.2416) and strange (m=-0.2050).
# Run from inside the 2-node interactive allocation (1279852).
#
# Each compute_vev invocation processes all N cfgs as positional args
# (matches overnight_orchestrator pattern). Single seed per pass.

cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh

LOGDIR=logs
mkdir -p "$LOGDIR"

ENSEMBLE_DIR=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3
CSW=1.20536588031793
N_NOISE=4
SEED=1234567

# 4 widely-spaced cfgs from the ensemble
CFGS=(
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2030.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2060.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2090.lime"
)

run_vev_2node () {
  local tag=$1 mass=$2; shift 2
  local log="$LOGDIR/compute_vev_${tag}.log"
  echo "=== $tag (mass=$mass) starting $(date) ===" | tee -a "$log"
  srun --overlap --mpi=pmix -N 2 -n 8 --cpu-bind=none \
    ./srun_gpu_wrapper.sh ./compute_vev \
      --grid 48.48.48.96 --mpi 1.1.2.4 \
      --mass "$mass" --csw "$CSW" \
      --n-noise "$N_NOISE" --cg-tol 1e-8 --seed "$SEED" \
      "$@" >> "$log" 2>&1
  echo "=== $tag DONE $(date) ===" | tee -a "$log"
}

# Wait for the smoke test (cfg_2000 light) to finish; pick up cfg_2030/60/90 for light.
SMOKE_LOG=logs/compute_vev_b6p3_m0p2416_2node_smoketest.log
echo "[runner] waiting for smoke to print vev_trminv..."
while ! grep -q 'vev_trminv' "$SMOKE_LOG" 2>/dev/null; do
  sleep 30
done
echo "[runner] smoke complete, proceeding to plan B passes $(date)"

# Light pass: cfg_2030, 2060, 2090 (smoke already did 2000)
run_vev_2node "b6p3_m0p2416_2node_light_plus3" "-0.2416" "${CFGS[1]}" "${CFGS[2]}" "${CFGS[3]}"

# Strange pass: all 4 cfgs
run_vev_2node "b6p3_m0p2050_2node_strange_4cfg" "-0.2050" "${CFGS[@]}"

echo "[runner] ALL PASSES DONE $(date)"
