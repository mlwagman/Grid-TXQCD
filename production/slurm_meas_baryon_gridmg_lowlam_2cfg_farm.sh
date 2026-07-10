#!/bin/bash
#SBATCH --job-name=bar_mg_2cfg_farm
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/bar_mg_2cfg_farm.%j.out
# ============================================================================
# NTSRC=48 (every timeslice) lambda<=3 aux_baryon Grid-MG farm, 2cfg/node.
# Each job measures the NEXT 2 UNMEASURED configs -- one per GPU pair,
# 1.1.1.2 each (config A GPUs 0-1 / GPU_OFFSET=0, config B GPUs 2-3 /
# GPU_OFFSET=2), concurrently.  At ~360 s/solve, 192 solves = ~19 h/config;
# the two run in parallel so a pair finishes in ~20 h (< 24 h walltime).
# RESUMABLE (skips existing h5); CHAIN copies with --dependency=afterany to
# churn the whole ensemble 2 configs / ~24 h / node.
# Params (--export): LAM(3), NTSRC(48), CFG_DIR, DATA_DIR.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
cd $P

LAM=${LAM:-3}
NTSRC=${NTSRC:-48}
CFG_DIR=${CFG_DIR:-$P/cfgs/dtxqcd_lam${LAM}.0000_tla_fromchroma_cw}
DATA_DIR=${DATA_DIR:-$P/meas_2pt/dtxqcd_lam${LAM}_baryon16}
SETUPS=$P/gridmg_setups
WRAPO=$P/srun_gpu_wrapper_offset.sh
mkdir -p "$DATA_DIR" "$SETUPS/logs"

if [ "$LAM" -ge 4 ] 2>/dev/null; then
  echo "REFUSING: lambda>=4 uses the EO-default farm, not Grid-MG." ; exit 2
fi

# All UNMEASURED thermalized trajs (drop <=20; tla fork flat from traj 1).
TRAJS=""
for t in $(ls -v "$CFG_DIR"/ckpoint_lat.* | grep -v daux | sed 's/.*\.//'); do
  [ "$t" -gt 20 ] 2>/dev/null || continue
  [ -f "$DATA_DIR/baryon_dtxqcd_${t}.h5" ] && continue
  TRAJS="$TRAJS $t"
done
if [ -z "$TRAJS" ]; then
  echo "=== all lambda=$LAM configs measured -- farm idle, exiting ==="; exit 0
fi
echo "=== 2cfg LOOP farm lambda=$LAM NTSRC=$NTSRC  trajs:$TRAJS  $(date) ==="

COMMON="LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM MASS_LIGHT_DTXQCD=-0.245 CFG_DIR=$CFG_DIR DATA_DIR=$DATA_DIR MEAS_BARYON_NTSRC=$NTSRC CG_TOL=1e-8 OMP_NUM_THREADS=16 QUDA_ENABLE_MPS=1 DTXQCD_MEAS_GRIDMG=1 DTXQCD_MEAS_MG_NBASIS=64 DTXQCD_MEAS_MG_GEOM=nn DTXQCD_MEAS_MG_ORD=1500 DTXQCD_MEAS_MG_REFINE_MAXIT=400 DTXQCD_MEAS_MG_NEV=96 COARSE_IRL_MAXIT=30 QUDA_RESOURCE_PATH=/lustre2/nplqcd/cache/quda_resource_cuda12p2"
GRID="--grid 16.16.16.48 --mpi 1.1.1.2 --shm 512 --shm-mpi 1"

# 2 GPU-PAIR slots: slot 0 = GPUs 0-1 (offset 0), slot 1 = GPUs 2-3 (offset 2).
# Setup-cache cleanup: the per-config nb64 basis (~9.4 GB) is only needed while
# that config solves.  clean_cache <traj> removes it once the h5 is on disk.
# The trailing "." in the glob prevents t30 matching t300.
clean_cache() {  # $1=traj
  local t=$1
  [ -n "$t" ] || return
  [ -f "$DATA_DIR/baryon_dtxqcd_${t}.h5" ] || return
  if ls "$SETUPS/measmg_lam${LAM}_t${t}".* >/dev/null 2>&1; then
    rm -f "$SETUPS/measmg_lam${LAM}_t${t}".*
    echo "  [cleanup] removed setup cache measmg_lam${LAM}_t${t}.*"
  fi
}
# Start-of-job sweep: reclaim caches for every ALREADY-measured config (covers
# accumulation from earlier chain links / concurrent farms).
for h5 in "$DATA_DIR"/baryon_dtxqcd_*.h5; do
  [ -e "$h5" ] || continue
  bt=$(basename "$h5"); bt=${bt#baryon_dtxqcd_}; bt=${bt%.h5}
  clean_cache "$bt"
done

# Keep both busy, refilling with the next config as each finishes -- one 24 h
# job churns ~2-3 pairs (at ~6-8.5 h/config, measured ~150 s/solve).
declare -a SLOT_PID SLOT_TRAJ
launch() {  # $1=traj  $2=slot(0|1)
  local traj=$1 slot=$2 off=$((slot * 2))
  echo "  [launch] traj=$traj -> slot $slot (GPUs $off,$((off+1)))"
  srun --overlap --mpi=pmix -N1 -n2 -c16 --cpu-bind=none \
    env GPU_OFFSET=$off $COMMON \
        DTXQCD_MEAS_MG_SETUP_PREFIX=$SETUPS/measmg_lam${LAM}_t${traj} \
    $WRAPO ./meas_baryon_dtxqcd $traj $GRID \
    > "$DATA_DIR/gridmg.$traj.log" 2>&1 &
  SLOT_PID[$slot]=$!
  SLOT_TRAJ[$slot]=$traj
}

for traj in $TRAJS; do
  # (re-skip in case a chained sibling grabbed it)
  [ -f "$DATA_DIR/baryon_dtxqcd_${traj}.h5" ] && continue
  placed=0
  while [ $placed -eq 0 ]; do
    for slot in 0 1; do
      if [ -z "${SLOT_PID[$slot]}" ] || ! kill -0 "${SLOT_PID[$slot]}" 2>/dev/null; then
        clean_cache "${SLOT_TRAJ[$slot]}"   # reclaim the just-finished config
        launch "$traj" "$slot"; placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 30
  done
done
wait
clean_cache "${SLOT_TRAJ[0]}"; clean_cache "${SLOT_TRAJ[1]}"   # final two
echo "=== 2cfg LOOP farm lambda=$LAM DONE $(date) ==="
