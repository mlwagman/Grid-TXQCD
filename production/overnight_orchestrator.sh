#!/bin/bash
cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=8

LOGDIR=logs
mkdir -p "$LOGDIR"

CFGDIR32=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788
CFGS32=( "$CFGDIR32"/cl3_32_64_b6p5_m0p1788_cfg_{4070,4080,4100,4120,4130,4160,4170,4200}.lime )
MASS32=-0.1788
CSW32=1.170082389372972

CFGDIR48=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3
CFGS48=( "$CFGDIR48"/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_{2000,2010,2020,2030,2040,2050,2060,2070,2080,2090}.lime )
CSW48=1.20536588031793

run_vev () {
  local tag=$1; shift
  local grid_=$1; shift
  local mpi_=$1; shift
  local mass=$1; shift
  local csw=$1; shift
  local noise=$1; shift
  local seed=$1; shift
  local cuda_dev=$1; shift
  local log="$LOGDIR/compute_vev_${tag}.log"
  local np
  np=$(echo "$mpi_" | awk -F. '{print $1*$2*$3*$4}')
  echo "=== $tag  grid=$grid_  mpi=$mpi_  np=$np  mass=$mass  csw=$csw  n_noise=$noise  seed=$seed  $(date) ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=$cuda_dev \
    mpirun -np "$np" --bind-to none \
      ./compute_vev "$@" \
      --grid "$grid_" --mpi "$mpi_" \
      --mass "$mass" --csw "$csw" \
      --n-noise "$noise" --cg-tol 1e-8 --seed "$seed" \
    >> "$log" 2>&1
  echo "=== $tag DONE $(date) ===" | tee -a "$log"
}

# 1. Wait for in-progress 32³ pass1 (mpirun PID 44566) to finish
echo "[wait] for current 32³ pass1 mpirun PID 44566 to finish... $(date)"
while ps -p 44566 > /dev/null 2>&1; do sleep 60; done
echo "[wait] 32³ pass1 finished $(date)"

# 2. 48³×96 light mass on all 4 GPUs (mpi=1.1.2.2 → per-rank 48²×24×48)
run_vev "b6p3_m0p2416_light_4gpu_n16" "48.48.48.96" "1.1.2.2" "-0.2416" "$CSW48" 16 1234567 "0,1,2,3" "${CFGS48[@]}"

# 3. 48³×96 strange mass
run_vev "b6p3_m0p2050_strange_4gpu_n16" "48.48.48.96" "1.1.2.2" "-0.2050" "$CSW48" 16 1234567 "0,1,2,3" "${CFGS48[@]}"

# 4. 32³ pass2 — different seed (more stats) on 4 GPUs (mpi=1.1.1.4 → per-rank 32³×16)
run_vev "b6p5_pass2_4gpu_n16" "32.32.32.64" "1.1.1.4" "$MASS32" "$CSW32" 16 7654321 "0,1,2,3" "${CFGS32[@]}"

# 5. 48³ light pass 2 — different seed (if time)
run_vev "b6p3_m0p2416_light_4gpu_pass2_n16" "48.48.48.96" "1.1.2.2" "-0.2416" "$CSW48" 16 7654321 "0,1,2,3" "${CFGS48[@]}"

# 6. 48³ strange pass 2
run_vev "b6p3_m0p2050_strange_4gpu_pass2_n16" "48.48.48.96" "1.1.2.2" "-0.2050" "$CSW48" 16 7654321 "0,1,2,3" "${CFGS48[@]}"

echo "=== ORCHESTRATOR COMPLETE $(date) ==="
