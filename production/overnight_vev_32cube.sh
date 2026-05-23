#!/bin/bash
cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=8

CFGDIR=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788
CFGS=( "$CFGDIR"/cl3_32_64_b6p5_m0p1788_cfg_{4070,4080,4100,4120,4130,4160,4170,4200}.lime )
LOGDIR=logs
mkdir -p "$LOGDIR"

MASS=-0.1788
CSW=1.170082389372972

# Wait for the current cfg_4200 n_noise=8 job to finish (it's running on GPUs 0,1)
# by waiting for the marker line in its log.
INIT_LOG=$LOGDIR/compute_vev_b6p5_cfg4200.log
echo "[wait] for init job to finish — watching $INIT_LOG"
while true; do
  if grep -q 'done at\|=== done ===' "$INIT_LOG" 2>/dev/null; then break; fi
  if ! pgrep -fu "$USER" 'compute_vev .* 32.32.32.64' >/dev/null; then break; fi
  sleep 30
done
echo "[wait] init job appears finished; starting overnight 32³ pass"

# Pass 1: all 8 cfgs, n_noise=16
LOG=$LOGDIR/compute_vev_b6p5_overnight_pass1.log
echo "=== 32³×64 m=$MASS pass 1 — 8 cfgs n_noise=16 ===" | tee -a "$LOG"
date | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun -np 2 --bind-to none \
    ./compute_vev "${CFGS[@]}" \
    --grid 32.32.32.64 --mpi 1.1.1.2 \
    --mass "$MASS" --csw "$CSW" \
    --n-noise 16 --cg-tol 1e-8 \
  >> "$LOG" 2>&1
echo "=== pass 1 done $(date) ===" | tee -a "$LOG"

# Pass 2: more stats — different seed, also n_noise=16
LOG=$LOGDIR/compute_vev_b6p5_overnight_pass2.log
echo "=== 32³×64 m=$MASS pass 2 — 8 cfgs n_noise=16 seed=7654321 ===" | tee -a "$LOG"
date | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun -np 2 --bind-to none \
    ./compute_vev "${CFGS[@]}" \
    --grid 32.32.32.64 --mpi 1.1.1.2 \
    --mass "$MASS" --csw "$CSW" \
    --n-noise 16 --cg-tol 1e-8 --seed 7654321 \
  >> "$LOG" 2>&1
echo "=== pass 2 done $(date) ===" | tee -a "$LOG"

# Pass 3 (if time): n_noise=32 with another seed
LOG=$LOGDIR/compute_vev_b6p5_overnight_pass3.log
echo "=== 32³×64 m=$MASS pass 3 — 8 cfgs n_noise=32 seed=11111 ===" | tee -a "$LOG"
date | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
  mpirun -np 2 --bind-to none \
    ./compute_vev "${CFGS[@]}" \
    --grid 32.32.32.64 --mpi 1.1.1.2 \
    --mass "$MASS" --csw "$CSW" \
    --n-noise 32 --cg-tol 1e-8 --seed 11111 \
  >> "$LOG" 2>&1
echo "=== pass 3 done $(date) ===" | tee -a "$LOG"
