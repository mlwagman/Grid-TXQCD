#!/bin/bash
# 4-GPU multi-cfg QCD measurements driver.  Same pattern as the TXQCD version.
set -e
cd "$(dirname "$0")"

QCD_SUFFIX_TAG="${QCD_SUFFIX:-}"
DATA_DIR="meas_2pt/qcd${QCD_SUFFIX_TAG}"
CFG_DIR="cfgs/qcd${QCD_SUFFIX_TAG}"
N_THERM="${N_THERM:-300}"
N_PROD="${N_PROD:-500}"
MEAS_SKIP="${MEAS_SKIP:-10}"
NGPU="${NGPU:-4}"
MIN_SIZE=1000

if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=$((16 / NGPU))
fi
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS per worker"

mkdir -p "$DATA_DIR" logs

for f in "$DATA_DIR"/*_qcd_*.h5; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null)
  if [ "${sz:-0}" -lt "$MIN_SIZE" ] && ! lsof "$f" >/dev/null 2>&1; then
    echo "Removing incomplete file: $f ($sz bytes)"
    rm -f "$f"
  fi
done

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

echo "=== QCD measurements (4-GPU multi-cfg) ==="
echo "    cfgs found: ${#trajs[@]}    GPUs: $NGPU"

worker() {
  local gpu=$1; shift
  local mytrajs=("$@")
  local count=0
  for t in "${mytrajs[@]}"; do
    if [ ! -f "$DATA_DIR/conn_qcd_$t.h5" ]; then
      echo "[gpu $gpu] meas_conn_qcd traj=$t"
      CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none \
        ./meas_conn_qcd $t --mpi 1.1.1.1 \
        > "logs/conn_qcd${QCD_SUFFIX_TAG}_t${t}_gpu${gpu}.log" 2>&1
      count=$((count+1))
    fi
    if [ -x ./meas_disco_qcd ] && [ ! -f "$DATA_DIR/disco_qcd_$t.h5" ]; then
      echo "[gpu $gpu] meas_disco_qcd traj=$t"
      CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none \
        ./meas_disco_qcd $t --mpi 1.1.1.1 \
        > "logs/disco_qcd${QCD_SUFFIX_TAG}_t${t}_gpu${gpu}.log" 2>&1
    fi
  done
  echo "[gpu $gpu] DONE ($count conn measurements ran)"
}

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
echo "=== All QCD measurements complete ==="
