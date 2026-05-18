#!/bin/bash
# 4-GPU multi-cfg TXQCD measurements driver.  Splits the traj list into 4
# round-robin chunks; one worker per GPU runs its chunk sequentially.  Each
# measurement is single-GPU (--mpi 1.1.1.1) to avoid the 16³×48 multi-GPU
# comm-overhead penalty (memory: project_quda_phase_b_status / Phase H notes).
#
# Throughput vs the spatially-decomposed run_all_txqcd_measurements.sh:
# ~4× on cfg averaging when the queue has ≥ 4 cfgs to process.
#
# Required env:
#   LAMBDA=<x.xxxx>      lambda for cfg/data path
# Optional env:
#   N_THERM, N_PROD, MEAS_SKIP    traj range (defaults below)
#   SUFFIX                       cfg/data dir suffix (passes through to params.h)
#   TXQCD_MULTIRHS_CG=1 etc.     fast-path env for meas_conn_txqcd
set -e
cd "$(dirname "$0")"

LAMBDA="${LAMBDA:-0.5000}"
# Normalize LAMBDA to 4 decimals (matches params.h's lambda_tag()).  Accepts
# either "6" or "6.0000" as input.
LAMBDA=$(printf "%.4f" "$LAMBDA")
SUFFIX_TAG="${SUFFIX:-}"
DATA_DIR="meas_2pt/txqcd_lam${LAMBDA}${SUFFIX_TAG}"
CFG_DIR="cfgs/txqcd_lam${LAMBDA}${SUFFIX_TAG}"
N_THERM="${N_THERM:-300}"
N_PROD="${N_PROD:-500}"
MEAS_SKIP="${MEAS_SKIP:-10}"
NGPU="${NGPU:-4}"
MIN_SIZE=1000

# Auto-tune OMP threads to avoid CPU oversubscription with NGPU concurrent
# workers.  16 CPU/task on lq2_gpu split across NGPU = 4 OMP threads each.
# Caller can override by exporting OMP_NUM_THREADS before invoking.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=$((16 / NGPU))
fi
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS per worker"

mkdir -p "$DATA_DIR" logs

# Remove broken outputs
for f in "$DATA_DIR"/*_txqcd_*.h5; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null)
  if [ "${sz:-0}" -lt "$MIN_SIZE" ] && ! lsof "$f" >/dev/null 2>&1; then
    echo "Removing incomplete file: $f ($sz bytes)"
    rm -f "$f"
  fi
done

# Build the traj list
trajs=()
for ((t=N_THERM; t<N_THERM+N_PROD; t+=MEAS_SKIP)); do
  if [ -f "$CFG_DIR/ckpoint_lat.$t" ]; then
    trajs+=($t)
  fi
done

if [ ${#trajs[@]} -eq 0 ]; then
  echo "No cfgs to measure under $CFG_DIR"
  exit 0
fi

echo "=== TXQCD measurements (4-GPU multi-cfg, lambda=$LAMBDA) ==="
echo "    cfgs found: ${#trajs[@]}    GPUs: $NGPU"

# Worker: process its slice of the traj list on its own GPU.  Each meas is
# --mpi 1.1.1.1 single-rank, avoiding the multi-GPU comm penalty on this
# lattice volume.
worker() {
  local gpu=$1; shift
  local mytrajs=("$@")
  local count=0
  for t in "${mytrajs[@]}"; do
    if [ ! -f "$DATA_DIR/conn_txqcd_$t.h5" ]; then
      echo "[gpu $gpu] meas_conn_txqcd traj=$t"
      CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none \
        ./meas_conn_txqcd $t --mpi 1.1.1.1 \
        > "logs/conn_txqcd_lam${LAMBDA}${SUFFIX_TAG}_t${t}_gpu${gpu}.log" 2>&1
      count=$((count+1))
    fi
    if [ -x ./meas_disco_txqcd ] && [ ! -f "$DATA_DIR/disco_txqcd_$t.h5" ]; then
      echo "[gpu $gpu] meas_disco_txqcd traj=$t"
      CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none \
        ./meas_disco_txqcd $t --mpi 1.1.1.1 \
        > "logs/disco_txqcd_lam${LAMBDA}${SUFFIX_TAG}_t${t}_gpu${gpu}.log" 2>&1
    fi
    if [ -x ./meas_aux_txqcd ] && [ ! -f "$DATA_DIR/aux_txqcd_$t.h5" ]; then
      echo "[gpu $gpu] meas_aux_txqcd traj=$t"
      CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none \
        ./meas_aux_txqcd $t --mpi 1.1.1.1 \
        > "logs/aux_txqcd_lam${LAMBDA}${SUFFIX_TAG}_t${t}_gpu${gpu}.log" 2>&1
    fi
  done
  echo "[gpu $gpu] DONE ($count conn measurements ran)"
}

# Stripe trajs across NGPU workers.  Round-robin gives a more balanced load
# than a contiguous chunk split when later cfgs are heavier.
for ((gpu=0; gpu<NGPU; gpu++)); do
  mine=()
  for ((i=gpu; i<${#trajs[@]}; i+=NGPU)); do
    mine+=(${trajs[i]})
  done
  if [ ${#mine[@]} -gt 0 ]; then
    worker $gpu "${mine[@]}" &
  fi
done

wait
echo "=== All TXQCD measurements (lambda=$LAMBDA) complete ==="
