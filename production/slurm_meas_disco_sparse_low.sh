#!/bin/bash
#SBATCH --job-name=disco_lowL
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/disco_lowL.%j.out

# Disconnected (single-flavor stochastic loop + Tr[M^-1]) measurements at
# every-5th-cfg cadence on the low-λ half.  Companion to slurm_meas_2x6_sparse_low.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=4
export QUDA_SOLVER=1

LATT="${LATT:-16.16.16.48}"

# Low-λ members of the 12 streams actively being extended (2026-05-18 audit;
# λ=6/6.5 RENAMED to _fromchroma_md30; λ=5 fork_t50_mds10 added).
STREAMS=(
  "txqcd 5    _nf2p1_mds10_fork"
  "txqcd 5    _fromchroma_md20_fork_t50_mds10"
  "txqcd 6    _fromchroma_md30"
  "txqcd 6.5  _fromchroma_md30"
  "txqcd 7    _nf2p1_mds10_fork"
  "txqcd 7    _fromchroma_md20_fork_t50_mds10"
)

ALL_DIR="meas_2pt/all_disco_sparse"
mkdir -p "$ALL_DIR"

run_one () {
  local kind=$1 lam=$2 suffix=$3 traj=$4 gpu=$5
  local cfg_dir outfile desc_name lam_tag binary
  if [ "$kind" = "txqcd" ]; then
    if [[ "$lam" == *.* ]]; then
      lam_tag=$(printf "lam%.4f" "$lam")
    else
      lam_tag="lam${lam}.0000"
    fi
    cfg_dir="cfgs/txqcd_${lam_tag}${suffix}"
    outfile="meas_2pt/txqcd_${lam_tag}${suffix}/disco_txqcd_${traj}.h5"
    desc_name="disco_txqcd_${lam_tag}${suffix}_traj${traj}.h5"
    binary=./meas_disco_txqcd
  else
    cfg_dir="cfgs/qcd${suffix}"
    outfile="meas_2pt/qcd${suffix}/disco_qcd_${traj}.h5"
    desc_name="disco_qcd${suffix}_traj${traj}.h5"
    binary=./meas_disco_qcd
  fi
  if [ ! -f "$cfg_dir/ckpoint_lat.$traj" ]; then return; fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 500 ]; then
    [ -f "$ALL_DIR/$desc_name" ] || ln "$outfile" "$ALL_DIR/$desc_name" 2>/dev/null
    return
  fi
  mkdir -p "$(dirname "$outfile")"
  local logfile="logs/disco_sparseLO_${kind}_${lam}${suffix}_t${traj}_gpu${gpu}.log"
  echo "[gpu $gpu] $kind disco λ=$lam$suffix cfg.$traj"
  if [ "$kind" = "txqcd" ]; then
    CUDA_VISIBLE_DEVICES=$gpu LAMBDA=$lam SUFFIX="$suffix" LATT=$LATT \
      mpirun -np 1 --bind-to none $binary $traj --mpi 1.1.1.1 > "$logfile" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu QCD_SUFFIX="$suffix" LATT=$LATT \
      mpirun -np 1 --bind-to none $binary $traj --mpi 1.1.1.1 > "$logfile" 2>&1
  fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 500 ]; then
    [ -f "$ALL_DIR/$desc_name" ] || ln "$outfile" "$ALL_DIR/$desc_name" 2>/dev/null
  fi
}

CFG_FROM="${CFG_FROM:-510}"; CFG_STEP="${CFG_STEP:-10}"; CFG_TO="${CFG_TO:-2000}"
CFG_NUMBERS=($(seq "$CFG_FROM" "$CFG_STEP" "$CFG_TO"))   # default=production 510..2000; override CFG_FROM/STEP/TO for backlog

PASS=0
while true; do
  PASS=$(( PASS + 1 ))
  echo ""
  echo "###########################################################"
  echo "#  DISCO LOW-λ PASS $PASS  $(date +'%H:%M:%S')"
  echo "###########################################################"
  ANYTHING_RUN=0
  for cfg in "${CFG_NUMBERS[@]}"; do
    AVAIL=()
    for s in "${STREAMS[@]}"; do
      read kind lam suffix <<< "$s"
      if [ "$kind" = "txqcd" ]; then
        if [[ "$lam" == *.* ]]; then lam_tag=$(printf "lam%.4f" "$lam"); else lam_tag="lam${lam}.0000"; fi
        cfg_path="cfgs/txqcd_${lam_tag}${suffix}/ckpoint_lat.$cfg"
        outfile="meas_2pt/txqcd_${lam_tag}${suffix}/disco_txqcd_${cfg}.h5"
      else
        cfg_path="cfgs/qcd${suffix}/ckpoint_lat.$cfg"
        outfile="meas_2pt/qcd${suffix}/disco_qcd_${cfg}.h5"
      fi
      if [ -f "$cfg_path" ]; then
        if [ ! -f "$outfile" ] || [ "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -lt 500 ]; then
          AVAIL+=("$s")
        else
          # already done → hardlink
          if [ "$kind" = "txqcd" ]; then desc="disco_txqcd_${lam_tag}${suffix}_traj${cfg}.h5"
          else desc="disco_qcd${suffix}_traj${cfg}.h5"; fi
          [ -f "$ALL_DIR/$desc" ] || ln "$outfile" "$ALL_DIR/$desc" 2>/dev/null
        fi
      fi
    done
    [ ${#AVAIL[@]} -eq 0 ] && continue
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
    echo "cfg.$cfg done $(date +'%H:%M:%S')"
  done
  if [ "$ANYTHING_RUN" = "0" ]; then
    echo "  nothing new → exit (will resubmit via dependency chain)"
    break
  fi
done
