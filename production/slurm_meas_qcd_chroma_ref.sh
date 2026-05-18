#!/bin/bash
#SBATCH --job-name=meas_qcd_ref
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_qcd_ref.%j.out

# QCD reference measurements on 100 chroma cfgs (every 10 chroma trajectories)
# from cfg_11610 to cfg_12600.  Apples-to-apples cadence with TXQCD streams:
# both use cfg_11100 as the initialization point, skip 500 trajs of
# thermalization (= cfg_11100 → cfg_11600), then measure the next 1000 trajs
# with 100 cfgs at 10-traj spacing.  Source: dense chroma chain at
# /lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a (cfg.10 → cfg.40100 step 10).
#
# Uses QUDA Schur-EO inverter for fair comparison with TXQCD's Schur-EO solver.
# Output dir: meas_2pt/qcd_chroma_ref/conn_qcd_<NNNNN>.h5

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=4

# QUDA Schur-EO inverter (matches TXQCD measurement convention)
export QUDA_SOLVER=1
# Multi-source batching — invertMultiSrcQuda groups 12 RHS per source location
export QCD_MULTISRC=1
# Chroma-style time-reversed FB averaging
export QCD_TIME_REVERSED=1

LATT="${LATT:-16.16.16.48}"
SX=2; ST=6
MEAS_CG_TOL="${MEAS_CG_TOL:-1e-8}"

CHROMA_DIR=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a
SUFFIX="_chroma_ref"
ALL_DIR="meas_2pt/qcd${SUFFIX}"
mkdir -p "$ALL_DIR"

# 100 cfgs at 10-traj spacing: cfg_11610, 11620, ..., cfg_12600.
# Init point = cfg_11100 (matches our TXQCD/QCD HMC chroma initialization);
# skip 500 chroma trajs of equivalent-thermalization (cfg_11100 → cfg_11600);
# measure next 1000 trajs.
CFG_LIST=()
for c in $(seq 11610 10 12600); do
  CFG_LIST+=("$c")
done

run_one () {
  local cfg=$1 gpu=$2
  local lime="${CHROMA_DIR}/cl3_16_48_b6p1_m0p2450_a_cfg_${cfg}.lime"
  local outfile="${ALL_DIR}/conn_qcd_${cfg}.h5"
  if [ ! -f "$lime" ]; then
    echo "[gpu $gpu] SKIP cfg.$cfg — lime missing: $lime"
    return
  fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then
    return  # already done
  fi
  local logfile="logs/conn_qcd_chroma_ref_${cfg}_gpu${gpu}.log"
  echo "[gpu $gpu] cfg.$cfg → $outfile"
  CUDA_VISIBLE_DEVICES=$gpu \
    QCD_SUFFIX="$SUFFIX" \
    QUDA_SOLVER=1 QCD_MULTISRC=1 QCD_TIME_REVERSED=1 \
    IMPORT_CFG="$lime" \
    LATT=$LATT MEAS_SPACE_SRC=$SX MEAS_TIME_SRC=$ST MEAS_CG_TOL=$MEAS_CG_TOL \
    mpirun -np 1 --bind-to none ./meas_conn_qcd $cfg --mpi 1.1.1.1 \
    > "$logfile" 2>&1
}

# Distribute CFG_LIST across 4 GPUs in parallel
echo "=== QCD chroma-ref measurements ==="
date
echo "  ${#CFG_LIST[@]} cfgs queued"

i=0
while [ $i -lt ${#CFG_LIST[@]} ]; do
  pids=()
  for gpu in 0 1 2 3; do
    idx=$(( i + gpu ))
    [ $idx -lt ${#CFG_LIST[@]} ] || break
    run_one "${CFG_LIST[$idx]}" "$gpu" &
    pids+=($!)
  done
  wait "${pids[@]}"
  i=$(( i + 4 ))
  done_count=$(ls "$ALL_DIR"/conn_qcd_*.h5 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] progress: $done_count / ${#CFG_LIST[@]} cfgs done"
done

echo "=== QCD chroma-ref measurements complete ==="
date
ls -la "$ALL_DIR" | tail -10
