#!/bin/bash
#SBATCH --job-name=meas_b6p5_l8
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_b6p5_l8.%j.out

# Continuous TXQCD measurement monitor for the b6.5 λ=8 32^3×64 chain.
#
# Watches cfgs/txqcd_lam8.0000_b6p5_lam8_mds10_1node and runs connected +
# disconnected measurements on every saved checkpoint that hasn't been
# measured yet.  Each measurement uses ALL 4 GPUs (mpi=1.1.1.4 = same
# topology as the HMC production) — distinct from the 16³ sparse monitor
# which fans 4 cfgs across 4 GPUs in parallel.
#
# Output:
#   meas_2pt/txqcd_lam8.0000_b6p5_lam8_mds10_1node/conn_txqcd_<traj>.h5
#   meas_2pt/txqcd_lam8.0000_b6p5_lam8_mds10_1node/disco_txqcd_<traj>.h5
#
# Source-grid shift: cfg numbers in the chain are >> 1000, so the new
# per-cfg deterministic origin shift is automatically active in the conn
# measurement (see params.h src_grid_origin).  Same shift is reproducible
# from the cfg number; written to h5 as 'src_shift'.
#
# Cadence: measure every saved cfg (production writes every 10 trajs by
# default).  Sleeps 30 min between rescans when nothing new is available.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs logs

# ===== b6.5 ensemble parameters (must match HMC slurm_b6p5_lam8_mds10_1node.sh) =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788
export LAMBDA=8
export AUX_INIT=2.9849

# ===== Inversion settings =====
# 32³ at b6.5 m_π is heavier than the b6.1 ensemble; 1e-7 is sufficient for
# the connected and a touch loose for disconnected stochastic estimates.
export MEAS_CG_TOL="${MEAS_CG_TOL:-1e-7}"
export OMP_NUM_THREADS=4

# ===== GPU performance / memory fences (match the HMC env at this scale) =====
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_MANAGED_MEMORY=1
export QCD_MULTISRC=1            # batched QUDA multi-src in meas_conn_qcd path
export QCD_TIME_REVERSED=1       # chroma-style FB baryon avg
export TXQCD_TIME_REVERSED=1     # TXQCD-side counterpart
export SHIFT_ALL_CFGS=1          # translation averaging from cfg 0 (b6.5 cfgs start low; default gates >=1000)
export TXQCD_MULTIRHS_CG=1       # Schur-EO multi-RHS CG in meas_conn_txqcd
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_DISCO_MULTIRHS=1    # Schur-EO multi-RHS disco
export QCD_DISCO_MULTISRC=1      # QUDA multi-src for strange-quark VEV

# ===== Source-grid resolution =====
# Tuneable.  At 32^3×64 with b6.5 (m_π~0.4), 2³×8 = 64 sources gives a
# reasonable SNR per cfg; bump to 2³×16 = 128 if more is needed.
SX="${MEAS_SPACE_SRC:-2}"
ST="${MEAS_TIME_SRC:-8}"
export MEAS_SPACE_SRC=$SX
export MEAS_TIME_SRC=$ST

# Streams to monitor (kind, lambda, suffix).  Single TXQCD stream right now.
STREAMS=(
  "txqcd 8 _b6p5_lam8_mds10_1node"
)

run_conn () {
  local kind=$1 lam=$2 suffix=$3 traj=$4
  local lam_tag cfg_dir outfile logfile
  if [ "$kind" = "txqcd" ]; then
    if [[ "$lam" == *.* ]]; then
      lam_tag=$(printf "lam%.4f" "$lam")
    else
      lam_tag="lam${lam}.0000"
    fi
    cfg_dir="cfgs/txqcd_${lam_tag}${suffix}"
    outfile="meas_2pt/txqcd_${lam_tag}${suffix}/conn_txqcd_${traj}.h5"
  else
    cfg_dir="cfgs/qcd${suffix}"
    outfile="meas_2pt/qcd${suffix}/conn_qcd_${traj}.h5"
  fi
  [ ! -f "$cfg_dir/ckpoint_lat.$traj" ] && return
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then return; fi
  mkdir -p "$(dirname "$outfile")"
  logfile="logs/conn_b6p5_l8_${kind}_${lam}${suffix}_t${traj}.log"
  echo "[$(date +%H:%M:%S)] conn  $kind λ=$lam$suffix cfg.$traj"
  if [ "$kind" = "txqcd" ]; then
    LAMBDA=$lam SUFFIX="$suffix" \
      srun --mpi=pmix --cpu-bind=none -N1 -n4 \
      ./meas_conn_txqcd $traj --grid 32.32.32.64 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1 \
      > "$logfile" 2>&1
  else
    QCD_SUFFIX="$suffix" \
      srun --mpi=pmix --cpu-bind=none -N1 -n4 \
      ./meas_conn_qcd $traj --grid 32.32.32.64 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1 \
      > "$logfile" 2>&1
  fi
}

run_disco () {
  local kind=$1 lam=$2 suffix=$3 traj=$4
  local lam_tag cfg_dir outfile logfile
  if [ "$kind" = "txqcd" ]; then
    if [[ "$lam" == *.* ]]; then
      lam_tag=$(printf "lam%.4f" "$lam")
    else
      lam_tag="lam${lam}.0000"
    fi
    cfg_dir="cfgs/txqcd_${lam_tag}${suffix}"
    outfile="meas_2pt/txqcd_${lam_tag}${suffix}/disco_txqcd_${traj}.h5"
  else
    cfg_dir="cfgs/qcd${suffix}"
    outfile="meas_2pt/qcd${suffix}/disco_qcd_${traj}.h5"
  fi
  [ ! -f "$cfg_dir/ckpoint_lat.$traj" ] && return
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 5000 ]; then return; fi
  mkdir -p "$(dirname "$outfile")"
  logfile="logs/disco_b6p5_l8_${kind}_${lam}${suffix}_t${traj}.log"
  echo "[$(date +%H:%M:%S)] disco $kind λ=$lam$suffix cfg.$traj"
  if [ "$kind" = "txqcd" ]; then
    LAMBDA=$lam SUFFIX="$suffix" \
      srun --mpi=pmix --cpu-bind=none -N1 -n4 \
      ./meas_disco_txqcd $traj --grid 32.32.32.64 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1 \
      > "$logfile" 2>&1
  else
    QCD_SUFFIX="$suffix" \
      srun --mpi=pmix --cpu-bind=none -N1 -n4 \
      ./meas_disco_qcd $traj --grid 32.32.32.64 --mpi 1.1.1.4 --shm 1024 --shm-mpi 1 \
      > "$logfile" 2>&1
  fi
}

PASS=0
while true; do
  PASS=$(( PASS + 1 ))
  echo ""
  echo "###########################################################"
  echo "#  b6.5 λ=8 meas pass $PASS  starting $(date +'%H:%M:%S')"
  echo "###########################################################"

  ANYTHING_RUN=0
  for s in "${STREAMS[@]}"; do
    read kind lam suffix <<< "$s"
    if [ "$kind" = "txqcd" ]; then
      if [[ "$lam" == *.* ]]; then
        lam_tag=$(printf "lam%.4f" "$lam")
      else
        lam_tag="lam${lam}.0000"
      fi
      cfg_dir="cfgs/txqcd_${lam_tag}${suffix}"
    else
      cfg_dir="cfgs/qcd${suffix}"
    fi
    if [ ! -d "$cfg_dir" ]; then
      echo "  $kind $lam$suffix : cfg dir $cfg_dir does not exist yet — skipping"
      continue
    fi
    # Enumerate cfgs by numeric suffix on ckpoint_lat.<traj>
    cfg_list=$(ls "$cfg_dir"/ckpoint_lat.* 2>/dev/null \
               | grep -oE 'ckpoint_lat\.[0-9]+$' \
               | sed 's/ckpoint_lat\.//' \
               | sort -n)
    if [ -z "$cfg_list" ]; then
      echo "  $kind $lam$suffix : no saved cfgs yet — skipping"
      continue
    fi
    for traj in $cfg_list; do
      # conn first, then disco — both use the full node.
      pre_conn=0
      pre_disco=0
      if [ "$kind" = "txqcd" ]; then
        cf="meas_2pt/txqcd_${lam_tag}${suffix}/conn_txqcd_${traj}.h5"
        df="meas_2pt/txqcd_${lam_tag}${suffix}/disco_txqcd_${traj}.h5"
      else
        cf="meas_2pt/qcd${suffix}/conn_qcd_${traj}.h5"
        df="meas_2pt/qcd${suffix}/disco_qcd_${traj}.h5"
      fi
      [ -f "$cf" ] && [ "$(stat -c%s "$cf" 2>/dev/null || echo 0)" -gt 50000 ] && pre_conn=1
      [ -f "$df" ] && [ "$(stat -c%s "$df" 2>/dev/null || echo 0)" -gt 5000  ] && pre_disco=1
      if [ "$pre_conn" = "0" ]; then
        run_conn "$kind" "$lam" "$suffix" "$traj"
        ANYTHING_RUN=1
      fi
      if [ "$pre_disco" = "0" ]; then
        run_disco "$kind" "$lam" "$suffix" "$traj"
        ANYTHING_RUN=1
      fi
    done
  done

  if [ "$ANYTHING_RUN" = "0" ]; then
    echo "===  pass $PASS found nothing new → sleep 30 min then re-scan"
    sleep 1800
  else
    echo "===  pass $PASS done at $(date +'%H:%M:%S')"
  fi
done
