#!/usr/bin/env bash
# Overnight: probe the tensor-condensate phase the cluster sees at λ=0.1.
# Same Symanzik+stout+Wilson-clover Nf=2+1 action as cluster, USE_FULL_PF=1
# (non-EO, no cliff), $4^3 \times 4$ lattice.
#
# Three runs to disentangle universality vs ergodicity:
#   (A) λ=0.1, cold start — primary condensate probe
#   (B) λ=0.1, hot start  — if (A) and (B) converge to same <t> structure
#                            → universality; if they don't → ergodicity issue
#   (C) λ=3.0, cold start — control; should NOT show <t> condensate
#
# After generation, loop meas_aux_txqcd on each saved cfg to record per-traj
# <Tr σ>, <Tr π>, <Tr s>, <Tr p>, <Tr t> (we compute <t> structure offline).
#
# Each run: 100 therm + 500 prod, save every 10 traj.  At ~17 s/traj on 4³×4,
# ~3 h per run, ~9 h total overnight.  Run from production/.

set -euo pipefail
cd "$(dirname "$0")"

GRID="--grid 4.4.4.4 --mpi 1.1.1.1"
LATT=4.4.4.4
NT=100
NP=500
NS=10
TS=$(date +%Y%m%d_%H%M%S)

CG="USE_FULL_PF=1 LATT=$LATT N_THERM=$NT N_PROD=$NP N_SKIP=$NS NO_METROP=$NT"

run_one() {
  local TAG=$1
  local LAMBDA=$2
  local START=$3
  local SUFFIX=$4              # disambiguates cfg dir for cold/hot at same λ
  local LOG=run_tcond_${TAG}_${TS}.log
  echo "[$(date)] ($TAG)  λ=$LAMBDA  start=$START  suffix=$SUFFIX  ->  $LOG"
  printf -v LAM_TAG "%.4f" "$LAMBDA"
  rm -rf "cfgs/txqcd_lam${LAM_TAG}${SUFFIX}"
  env $CG LAMBDA=$LAMBDA START_TYPE=$START SUFFIX=$SUFFIX \
    ./gen_txqcd_cfgs_2plus1 $GRID > "$LOG" 2>&1
  echo "[$(date)] ($TAG) gen done.  Loop meas_aux_txqcd on $((NP/NS)) cfgs..."
  for ((t = NT + NS; t <= NT + NP; t += NS)); do
    env $CG LAMBDA=$LAMBDA SUFFIX=$SUFFIX \
      ./meas_aux_txqcd $t $GRID >> "$LOG" 2>&1
  done
  echo "[$(date)] ($TAG) meas_aux done."
}

echo "============================================================="
echo "  TXQCD tensor-condensate overnight scan @ $(date)"
echo "  3 runs, 4^3 x 4, Symanzik+stout+clover Nf=2+1, USE_FULL_PF=1"
echo "============================================================="

run_one A_lam0p1_cold 0.1 cold _cold
run_one B_lam0p1_hot  0.1 hot  _hot
run_one C_lam3p0_cold 3.0 cold _cold

echo "[$(date)] all done.  Aux h5 files live in meas_2pt/txqcd_lam*"
echo "Per-traj traces ready for offline <t_munu> analysis."
