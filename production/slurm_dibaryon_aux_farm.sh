#!/bin/bash
#SBATCH --job-name=dtx_dibaryon
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_dibaryon.%j.out
# PURE-AUX dibaryon (triple-diquark) farm: no inversions -- seconds/config.
# Runs the GAUGE_TWIST gate on the first config FIRST; aborts the job if the
# gate fails (mandatory after any new contraction).  LAM via --export.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LAM=${LAM:-5}
CFG=$P/cfgs/dtxqcd_lam${LAM}.0000_tla_fromchroma_cw
OUT=$P/meas_2pt/dtxqcd_lam${LAM}_dibaryon_aux
mkdir -p "$OUT"

TRAJS=""
for t in $(ls -v "$CFG"/ckpoint_lat.* 2>/dev/null | grep -v daux | grep -v rng | sed 's/.*\.//'); do
  [ "$t" -gt 20 ] 2>/dev/null && TRAJS="$TRAJS $t"
done
FIRST=$(echo $TRAJS | awk '{print $1}')
echo "=== dibaryon farm lam=$LAM: $(echo $TRAJS | wc -w) configs ==="

# ---- gate ----
srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 LATT=16.16.16.48 GAUGE_TWIST=1 \
      CFG_DIR=$CFG DATA_DIR=$OUT LAMBDA_DTXQCD=$LAM OMP_NUM_THREADS=16 \
  $P/meas_dibaryon_aux_dtxqcd "$FIRST" --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > "$OUT/gate.$FIRST.log" 2>&1
if ! grep -q "PASSED" "$OUT/gate.$FIRST.log"; then
  echo "FATAL: GAUGE_TWIST gate FAILED -- see $OUT/gate.$FIRST.log"
  exit 1
fi
echo "=== GAUGE_TWIST gate PASSED ==="
grep "GAUGE_TWIST" "$OUT/gate.$FIRST.log"

# ---- farm ----
declare -a SLOT_PID
launch() {
  local traj=$1 gpu=$2
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$gpu LATT=16.16.16.48 \
        CFG_DIR=$CFG DATA_DIR=$OUT LAMBDA_DTXQCD=$LAM OMP_NUM_THREADS=16 \
    $P/meas_dibaryon_aux_dtxqcd "$traj" --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$OUT/dib.$traj.log" 2>&1 &
  SLOT_PID[$gpu]=$!
}
for traj in $TRAJS; do
  [ -f "$OUT/dibaryon_aux_dtxqcd_${traj}.h5" ] && continue
  placed=0
  while [ $placed -eq 0 ]; do
    for g in 0 1 2 3; do
      if [ -z "${SLOT_PID[$g]}" ] || ! kill -0 "${SLOT_PID[$g]}" 2>/dev/null; then
        launch "$traj" "$g"; placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 5
  done
done
wait
echo "=== dibaryon farm done $(date): $(ls $OUT/*.h5 2>/dev/null | wc -l) h5 ==="
