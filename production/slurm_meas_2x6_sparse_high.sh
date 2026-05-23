#!/bin/bash
#SBATCH --job-name=meas_hiL
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_hiL.%j.out

# Sparse hadron-thermalization monitor — HIGH-λ half (λ ≥ 8 plus QCD).
# Cadence: every 5th saved cfg (50, 100, 150, ...).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs logs
source ../env_lq2_grid.sh

export OMP_NUM_THREADS=4
export TXQCD_MULTIRHS_CG=1
export TXQCD_MOOEE_CUBLAS=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_TIME_REVERSED=1
export QCD_MULTISRC=1
export QCD_TIME_REVERSED=1
# QUDA Schur-EO inverter for QCD — matches TXQCD measurement convention
# (TXQCD uses TXQCD_EO_SOLVER via Schur EO).  Apples-to-apples comparison.
export QUDA_SOLVER=1
# Enable kaon correlator (TXQCD light × vanilla-QCD strange) in TXQCD meas
export TXQCD_KAON=1

LATT="${LATT:-16.16.16.48}"
SX=2; ST=6
MEAS_CG_TOL="${MEAS_CG_TOL:-1e-8}"

STREAMS=(
  "txqcd 8    _nf2p1_mds10_fork"
  "txqcd 8    _fromchroma_md20_fork_t50_mds10"
  "txqcd 9    _fromchroma_md10"
  "txqcd 10   _nf2p1_mds10_fork"
  "txqcd 10   _fromchroma_md10"
  "txqcd 12   _fromchroma_md10"
  "txqcd 14   _fromchroma_md10"
  "txqcd 16   _fromchroma_md10"
  "qcd   -    _s702_nf2p1_mdscan_mds30"
)

ALL_DIR="meas_2pt/all_2x6_sparse"
mkdir -p "$ALL_DIR"

run_one () {
  local kind=$1 lam=$2 suffix=$3 traj=$4 gpu=$5
  local cfg_dir outfile desc_name lam_tag
  if [ "$kind" = "txqcd" ]; then
    if [[ "$lam" == *.* ]]; then
      lam_tag=$(printf "lam%.4f" "$lam")
    else
      lam_tag="lam${lam}.0000"
    fi
    cfg_dir="cfgs/txqcd_${lam_tag}${suffix}"
    outfile="meas_2pt/txqcd_${lam_tag}${suffix}/conn_txqcd_${traj}.h5"
    desc_name="conn_txqcd_${lam_tag}${suffix}_traj${traj}.h5"
  else
    cfg_dir="cfgs/qcd${suffix}"
    outfile="meas_2pt/qcd${suffix}/conn_qcd_${traj}.h5"
    desc_name="conn_qcd${suffix}_traj${traj}.h5"
  fi
  local desc_path="${ALL_DIR}/${desc_name}"
  if [ ! -f "$cfg_dir/ckpoint_lat.$traj" ]; then return; fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then
    [ -f "$desc_path" ] || ln "$outfile" "$desc_path" 2>/dev/null
    return
  fi
  mkdir -p "$(dirname "$outfile")"
  local logfile="logs/conn_sparseHI_${kind}_${lam}${suffix}_t${traj}_gpu${gpu}.log"
  echo "[gpu $gpu] $kind λ=$lam$suffix cfg.$traj"
  if [ "$kind" = "txqcd" ]; then
    CUDA_VISIBLE_DEVICES=$gpu \
      LAMBDA=$lam SUFFIX="$suffix" \
      LATT=$LATT MEAS_SPACE_SRC=$SX MEAS_TIME_SRC=$ST MEAS_CG_TOL=$MEAS_CG_TOL \
      mpirun -np 1 --bind-to none ./meas_conn_txqcd $traj --mpi 1.1.1.1 \
      > "$logfile" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu \
      QCD_SUFFIX="$suffix" \
      LATT=$LATT MEAS_SPACE_SRC=$SX MEAS_TIME_SRC=$ST MEAS_CG_TOL=$MEAS_CG_TOL \
      mpirun -np 1 --bind-to none ./meas_conn_qcd $traj --mpi 1.1.1.1 \
      > "$logfile" 2>&1
  fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then
    [ -f "$desc_path" ] || ln "$outfile" "$desc_path" 2>/dev/null
  fi
}

CFG_FROM="${CFG_FROM:-510}"; CFG_STEP="${CFG_STEP:-10}"; CFG_TO="${CFG_TO:-2000}"
CFG_NUMBERS=($(seq "$CFG_FROM" "$CFG_STEP" "$CFG_TO"))   # default=production 510..2000; override CFG_FROM/STEP/TO for backlog

PASS=0
while true; do
  PASS=$(( PASS + 1 ))
  echo ""
  echo "###########################################################"
  echo "#  HIGH-λ PASS $PASS  starting $(date +'%H:%M:%S')"
  echo "###########################################################"
  ANYTHING_RUN=0
  for cfg in "${CFG_NUMBERS[@]}"; do
    AVAIL=()
    for s in "${STREAMS[@]}"; do
      read kind lam suffix <<< "$s"
      if [ "$kind" = "txqcd" ]; then
        if [[ "$lam" == *.* ]]; then
          lam_tag=$(printf "lam%.4f" "$lam")
        else
          lam_tag="lam${lam}.0000"
        fi
        cfg_path="cfgs/txqcd_${lam_tag}${suffix}/ckpoint_lat.$cfg"
        outfile="meas_2pt/txqcd_${lam_tag}${suffix}/conn_txqcd_${cfg}.h5"
      else
        cfg_path="cfgs/qcd${suffix}/ckpoint_lat.$cfg"
        outfile="meas_2pt/qcd${suffix}/conn_qcd_${cfg}.h5"
      fi
      if [ -f "$cfg_path" ]; then
        if [ ! -f "$outfile" ] || [ "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -lt 50000 ]; then
          AVAIL+=("$s")
        else
          # Already measured — ensure hardlink in ALL_DIR exists
          if [ "$kind" = "txqcd" ]; then
            desc="conn_txqcd_${lam_tag}${suffix}_traj${cfg}.h5"
          else
            desc="conn_qcd${suffix}_traj${cfg}.h5"
          fi
          [ -f "$ALL_DIR/$desc" ] || ln "$outfile" "$ALL_DIR/$desc" 2>/dev/null
        fi
      fi
    done
    [ ${#AVAIL[@]} -eq 0 ] && continue
    echo ""
    echo "=== cfg.$cfg  ${#AVAIL[@]} streams TODO ==="
    ANYTHING_RUN=1
    i=0
    while [ $i -lt ${#AVAIL[@]} ]; do
      pids=()
      for gpu in 0 1 2 3; do
        idx=$(( i + gpu ))
        [ $idx -lt ${#AVAIL[@]} ] || break
        read kind lam suffix <<< "${AVAIL[$idx]}"
        run_one "$kind" "$lam" "$suffix" "$cfg" "$gpu" &
        pids+=($!)
      done
      wait "${pids[@]}"
      i=$(( i + 4 ))
    done
    echo "cfg.$cfg done at $(date +'%H:%M:%S')"
  done
  if [ "$ANYTHING_RUN" = "0" ]; then
    echo "  nothing new → exit (will resubmit via dependency chain)"
    break
  fi
done
