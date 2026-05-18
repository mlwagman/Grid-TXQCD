#!/bin/bash
#SBATCH --job-name=meas_sparse
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_sparse.%j.out

# Sparse hadron-level thermalization monitor.
# Measures EVERY 5th saved cfg (cfg.50, 100, 150, 200, 250, ...) across the
# CURRENT batch of streams, prefering MDS=10 versions where they exist
# (the new MDS=10 forks make several MDS=30 streams redundant).
#
# Sparse enough that one job (4 GPUs, ~5 s/src ⇒ ~30 s/cfg amortized) keeps
# pace with HMC production (~1 measureable cfg/hr across the 16 streams).

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

LATT="${LATT:-16.16.16.48}"
SX=2; ST=6
MEAS_CG_TOL="${MEAS_CG_TOL:-1e-8}"

# (kind, lam, suffix).  Streams to monitor — preferring MDS=10 where it
# exists. λ=6, 6.5 keep chroma MDS=30 since MDS=10 forks are broken there.
# λ=4 keeps chroma MDS=30 (no MDS=10 source).
# For λ=7, 8, 10 we measure TWO streams (the weak-field MDS=10 fork AND the
# chroma path) to compare paths at fixed λ — this is the new redundancy.
STREAMS=(
  # Weak-field MDS=10 forks (preferred source where available)
  "txqcd 5    _nf2p1_mds10_fork"
  "txqcd 7    _nf2p1_mds10_fork"
  "txqcd 8    _nf2p1_mds10_fork"
  "txqcd 10   _nf2p1_mds10_fork"
  # Chroma-fork-t50 MDS=10 (new from nodeC)
  "txqcd 7    _fromchroma_md20_fork_t50_mds10"
  "txqcd 7.5  _fromchroma_md20_fork_t50_mds10"
  "txqcd 8    _fromchroma_md20_fork_t50_mds10"
  # Fresh chroma MDS=10 streams
  "txqcd 9    _fromchroma_md10"
  "txqcd 10   _fromchroma_md10"
  "txqcd 12   _fromchroma_md10"
  "txqcd 14   _fromchroma_md10"
  "txqcd 16   _fromchroma_md10"
  # Chroma MDS=30 — both for λ where MDS=10 unavailable AND for λ=5
  # to compare weak-field-fork vs chroma-source paths at fixed λ.
  "txqcd 4    _fromchroma_md20"
  "txqcd 5    _fromchroma_md20"
  "txqcd 6    _fromchroma_md20"
  "txqcd 6.5  _fromchroma_md20"
  # QCD weak-field MDS=30 (newly resumed)
  "qcd   -    _s702_nf2p1_mdscan_mds30"
)

# Self-describing output dir
ALL_DIR="meas_2pt/all_2x6_sparse"
mkdir -p "$ALL_DIR"

run_one () {
  local kind=$1 lam=$2 suffix=$3 traj=$4 gpu=$5
  local cfg_dir outfile desc_name
  if [ "$kind" = "txqcd" ]; then
    # Build cfg_dir using lambda_tag convention (lam5 → lam5.0000, lam7.5 → lam7.5000)
    local lam_tag
    if [[ "$lam" == *.* ]]; then
      # has decimal: format with sprintf %.4f
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
  local logfile="logs/conn_sparse_${kind}_${lam}${suffix}_t${traj}_gpu${gpu}.log"
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

# Every-5th cadence: cfg.50, 100, 150, 200, 250, 300, 350, 400.
CFG_NUMBERS=(50 100 150 200 250 300 350 400)

PASS=0
while true; do
  PASS=$(( PASS + 1 ))
  echo ""
  echo "###########################################################"
  echo "#  SPARSE PASS $PASS  starting $(date +'%H:%M:%S')"
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
      if [ -f "$cfg_path" ] && \
         { [ ! -f "$outfile" ] || [ "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -lt 50000 ]; }; then
        AVAIL+=("$s")
      fi
    done
    [ ${#AVAIL[@]} -eq 0 ] && continue
    echo ""
    echo "==================================================="
    echo "  pass $PASS — CFG = $cfg  ${#AVAIL[@]} streams TODO"
    echo "==================================================="
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
    echo "  pass $PASS — cfg.$cfg done at $(date +'%H:%M:%S')"
  done
  if [ "$ANYTHING_RUN" = "0" ]; then
    # No new cfgs — sleep 30 min and re-scan, instead of exiting.
    # This keeps the job alive as a true continuous monitor.
    echo "===  pass $PASS found nothing new → sleep 30 min then re-scan"
    sleep 1800
  else
    echo "===  pass $PASS done → re-scan for new cfgs"
  fi
done

# (loop never exits cleanly; killed by SLURM time limit)
