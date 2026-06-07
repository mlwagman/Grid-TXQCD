#!/bin/bash
# Plan B with MG: 4 cfgs x 2 masses x n_noise=16 on the b6.3 48^3 x 96 ensemble.
# Now that QudaCloverInverter has MG support (USE_QUDA_MG=1), each cfg takes
# ~6 min (vs ~30 min/noise without MG = unreachable in interactive).
#
# Run from inside the 2-node interactive (1279852, lq2gpu05+06).

cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh

LOGDIR=logs
mkdir -p "$LOGDIR"

ENSEMBLE_DIR=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3
CSW=1.20536588031793
N_NOISE=16
SEED=1234567

# 4 widely-spaced cfgs (matches the 32^3 b6.5 spacing — 30 traj between)
CFGS=(
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2030.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2060.lime"
  "$ENSEMBLE_DIR/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2090.lime"
)

run_pass () {
  local tag=$1 mass=$2; shift 2
  local log="$LOGDIR/compute_vev_${tag}.log"
  echo "=== $tag (mass=$mass) starting $(date) ===" | tee -a "$log"
  USE_QUDA_MG=1 \
  srun --overlap --mpi=pmix --export=ALL -N 2 -n 8 --cpu-bind=none \
    ./compute_vev \
      --grid 48.48.48.96 --mpi 1.1.2.4 \
      --mass "$mass" --csw "$CSW" \
      --n-noise "$N_NOISE" --cg-tol 1e-8 --seed "$SEED" \
      "$@" >> "$log" 2>&1
  echo "=== $tag DONE $(date) ===" | tee -a "$log"
}

run_pass "b6p3_m0p2416_4cfg_mg_n16" "-0.2416" "${CFGS[@]}"
run_pass "b6p3_m0p2050_4cfg_mg_n16" "-0.2050" "${CFGS[@]}"

echo "ALL PASSES DONE $(date)"
