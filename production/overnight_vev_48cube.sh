#!/bin/bash
cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=8

CFGDIR=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3
CFGS=( "$CFGDIR"/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_{2000,2010,2020,2030,2040,2050,2060,2070,2080,2090}.lime )
LOGDIR=logs
mkdir -p "$LOGDIR"

CSW=1.20536588031793

for MASS_LABEL in "m0p2416_light:-0.2416" "m0p2050_strange:-0.2050"; do
  LABEL="${MASS_LABEL%:*}"
  MASS="${MASS_LABEL#*:}"
  LOG="$LOGDIR/compute_vev_b6p3_${LABEL}_overnight.log"
  echo "=== 48³×96 $LABEL  mass=$MASS  csw=$CSW  10 cfgs n_noise=16 ===" | tee -a "$LOG"
  date | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=2,3 \
    mpirun -np 2 --bind-to none \
      ./compute_vev "${CFGS[@]}" \
      --grid 48.48.48.96 --mpi 1.1.1.2 \
      --mass "$MASS" --csw "$CSW" \
      --n-noise 16 --cg-tol 1e-8 \
    >> "$LOG" 2>&1
  echo "=== $LABEL done $(date) ===" | tee -a "$LOG"
done

# Optional: third pass with different seed for more stats (will happen if time allows)
for MASS_LABEL in "m0p2416_light:-0.2416" "m0p2050_strange:-0.2050"; do
  LABEL="${MASS_LABEL%:*}"
  MASS="${MASS_LABEL#*:}"
  LOG="$LOGDIR/compute_vev_b6p3_${LABEL}_overnight2.log"
  echo "=== 48³×96 $LABEL  PASS 2  seed=7654321 ===" | tee -a "$LOG"
  date | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=2,3 \
    mpirun -np 2 --bind-to none \
      ./compute_vev "${CFGS[@]}" \
      --grid 48.48.48.96 --mpi 1.1.1.2 \
      --mass "$MASS" --csw "$CSW" \
      --n-noise 16 --cg-tol 1e-8 --seed 7654321 \
    >> "$LOG" 2>&1
  echo "=== $LABEL pass2 done $(date) ===" | tee -a "$LOG"
done
