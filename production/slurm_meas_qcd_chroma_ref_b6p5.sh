#!/bin/bash
#SBATCH --job-name=meas_qcd_b6p5_ref
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_qcd_b6p5_ref.%j.out

# Pure QCD measurements on the 15 chroma reference cfgs at b6.5 32^3×64
# (cl3_32_64_b6p5_m0p1788), cfg numbers 4070-4200 at Δ=10 (14 consecutive)
# plus the standalone cfg_410.  Establishes the vanilla-QCD spectrum at
# b6.5 against which the TXQCD b6.5 λ=8 chain will be compared.
#
# Each measurement uses ALL 4 GPUs (mpi=1.1.1.4) — same topology as the
# b6.5 HMC.  No parallelism across cfgs at this lattice size.
#
# Cfg numbers >> 1000 so the new per-cfg deterministic src_shift is active
# (see params.h src_grid_origin).  Shift written to h5 as 'src_shift'.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs logs

# ===== b6.5 ensemble parameters =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788

export OMP_NUM_THREADS=4
export MEAS_CG_TOL="${MEAS_CG_TOL:-1e-8}"

export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1
export QCD_MULTISRC=1
export QCD_TIME_REVERSED=1
export SHIFT_ALL_CFGS=1          # translation averaging from cfg 0 (cfg 410 etc are < 1000; legacy gate skips them)

SX="${MEAS_SPACE_SRC:-2}"
ST="${MEAS_TIME_SRC:-8}"
export MEAS_SPACE_SRC=$SX
export MEAS_TIME_SRC=$ST

CHROMA_DIR=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788
SUFFIX="_chroma_ref_b6p5"
ALL_DIR="meas_2pt/qcd${SUFFIX}"
mkdir -p "$ALL_DIR"

CFG_LIST=(410 4070 4080 4090 4100 4110 4120 4130 4140 4150 4160 4170 4180 4190 4200)

run_one () {
  local cfg=$1
  local lime="${CHROMA_DIR}/cl3_32_64_b6p5_m0p1788_cfg_${cfg}.lime"
  local outfile="${ALL_DIR}/conn_qcd_${cfg}.h5"
  if [ ! -f "$lime" ]; then
    echo "[$(date +%H:%M:%S)] SKIP cfg.$cfg — lime missing: $lime"
    return
  fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then
    echo "[$(date +%H:%M:%S)] SKIP cfg.$cfg — already done"
    return
  fi
  local logfile="logs/conn_qcd_chroma_ref_b6p5_${cfg}.log"
  echo "[$(date +%H:%M:%S)] cfg.$cfg → $outfile"
  QCD_SUFFIX="$SUFFIX" \
    IMPORT_CFG="$lime" \
    srun --mpi=pmix --cpu-bind=none -N1 -n4 \
    ./meas_conn_qcd $cfg --grid 32.32.32.64 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1 \
    > "$logfile" 2>&1
  if [ ! -f "$outfile" ] || [ "$(stat -c%s "$outfile")" -lt 50000 ]; then
    echo "[$(date +%H:%M:%S)] FAILED cfg.$cfg — see $logfile"
  fi
}

echo "=== QCD chroma-ref b6.5 measurements ==="
date
echo "  ${#CFG_LIST[@]} cfgs queued"
echo "  cfgs: ${CFG_LIST[*]}"

for cfg in "${CFG_LIST[@]}"; do
  run_one "$cfg"
done

echo "=== b6.5 QCD chroma-ref measurements complete ==="
date
ls -la "$ALL_DIR" | tail -20
