#!/bin/bash
#SBATCH --job-name=meas_txqcd_on_qcdref
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_txqcd_on_qcdref.%j.out

# TXQCD connected measurements on VANILLA QCD configs (mean-field aux test).
#
# Runs meas_conn_txqcd at LAMBDA = 5, 6, 7, 8 on the EXACT SAME 100 chroma
# vanilla-QCD configs that qcd_chroma_ref measured (cfg_11610 .. cfg_12600,
# step 10).  meas_conn_txqcd's IMPORT_CFG path loads the QCD gauge, then
# load_txqcd_field auto-measures Σ = Tr[M⁻¹]/(4V) on the stout-smeared gauge
# and fills the aux at the LEADING mean-field saddle <σ_aa> = Σ/λ²
# (NO self-consistent refinement — AUX_INIT_ITERATIONS deliberately unset, so
# this is the pure mean-field-on-QCD-gauge test the question asks for).
#
# Purpose: does the TXQCD propagator on QCD-equilibrium gauge with mean-field
# aux give the mean-field connected-pion scaling and the QCD "right answer"?
# Direct apples-to-apples vs meas_2pt/qcd_chroma_ref/conn_qcd_<cfg>.h5
# (same cfgs, same Schur-EO solver, same FB time-reversed averaging).
#
# Output: meas_2pt/txqcd_lam<λ>.0000_on_qcd_chroma_ref/conn_txqcd_<cfg>.h5
# (cfg label = chroma cfg number → 1:1 filename pairing with qcd_chroma_ref)

# Robust cd to production/: works under sbatch (SLURM_SUBMIT_DIR=submission
# dir) AND interactive `bash thisscript.sh` (where an salloc sets
# SLURM_SUBMIT_DIR to the salloc cwd, NOT production/).  Pick whichever
# location actually contains the meas binary; fail loudly otherwise.
_sd="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -x "$SLURM_SUBMIT_DIR/meas_conn_txqcd" ]; then
  cd "$SLURM_SUBMIT_DIR"
elif [ -x "$_sd/meas_conn_txqcd" ]; then
  cd "$_sd"
else
  echo "FATAL: cannot find meas_conn_txqcd (looked in SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' and script dir '$_sd'). Run from production/." >&2
  exit 1
fi
echo "[setup] running from $(pwd)"
mkdir -p slurm-logs logs
source ../env_lq2_grid.sh

# Match the proven qcd_chroma_ref OMP/UCX setup (4 concurrent GPU jobs,
# GPU-bound TXQCD multi-RHS CG; conservative thread count).
export OMP_NUM_THREADS=4
export OMPI_MCA_btl=^uct,openib
export UCX_TLS=cuda,gdr_copy,rc,rc_x,sm,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n

# TXQCD measurement solver/perf knobs (same as slurm_meas_txqcd.sh production).
export TXQCD_MULTIRHS_CG=1        # Schur EO + 24-RHS lockstep CG (fast path)
export TXQCD_MOOEE_CUBLAS=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_TIME_REVERSED=1      # chroma-style FB time-reversed averaging

LATT="${LATT:-16.16.16.48}"
# nsrc = SX³·ST.  Default 2/6 = 48 src — IDENTICAL to qcd_chroma_ref for a
# clean apples-to-apples per-config comparison.  Mean-field-aux-on-QCD-gauge
# runs ~1200 CG iters (~230 s/src ≈ ~3 h/cfg), so cover the 100 cfgs with
# several parallel jobs over days (use CFG_FROM/CFG_TO to shard, below).
SX="${MEAS_SPACE_SRC:-2}"; ST="${MEAS_TIME_SRC:-6}"
MEAS_CG_TOL="${MEAS_CG_TOL:-1e-8}"

# λ values to scan.  Override with: LAMBDAS="5 7" sbatch ...
LAMBDAS="${LAMBDAS:-5 6 7 8}"

# Output-dir suffix.  MUST be a distinct value for any non-default variant so
# it writes to its own dir (binary's txqcd_data_dir() = meas_2pt/txqcd_<lam><SUFFIX>).
MEAS_SUFFIX="${MEAS_SUFFIX:-_on_qcd_chroma_ref}"
# Optional manual Σ override for the mean-field aux (load_txqcd_field reads
# AUX_INIT).  Unset → auto-measure Σ on stout-smeared QCD gauge (default).
# SIGN-FLIP TEST: AUX_INIT=-1.5370793 imposes ⟨σ⟩=−Σ/λ² (flips the whole Δ
# insertion) — pair with MEAS_SUFFIX=_on_qcd_chroma_ref_auxneg.
AUX_INIT="${AUX_INIT:-}"

# Same cfg pool + window + cadence as slurm_meas_qcd_chroma_ref.sh.
# Shard across parallel jobs with CFG_FROM/CFG_TO (inclusive chroma cfg #),
# e.g. job A: CFG_FROM=11610 CFG_TO=11900 ; job B: CFG_FROM=11910 CFG_TO=12200
# Resume-safe skip also guards against any accidental overlap.
CHROMA_DIR=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a
CFG_FROM="${CFG_FROM:-11610}"; CFG_TO="${CFG_TO:-12600}"
CFG_LIST=()
for c in $(seq 11610 10 12600); do
  [ "$c" -ge "$CFG_FROM" ] && [ "$c" -le "$CFG_TO" ] && CFG_LIST+=("$c")
done
echo "[setup] cfg window ${CFG_FROM}..${CFG_TO} → ${#CFG_LIST[@]} cfgs; λ='${LAMBDAS}'; nsrc=$((SX*SX*SX*ST)); SUFFIX='${MEAS_SUFFIX}'; AUX_INIT='${AUX_INIT:-<auto-measure Σ>}'"

# lambda_tag(): printf "%.4f" → e.g. 5 -> lam5.0000
lam_tag () { printf "lam%.4f" "$1"; }

run_one () {
  local cfg=$1 gpu=$2 lam=$3
  local lime="${CHROMA_DIR}/cl3_16_48_b6p1_m0p2450_a_cfg_${cfg}.lime"
  local outdir="meas_2pt/txqcd_$(lam_tag "$lam")${MEAS_SUFFIX}"
  local outfile="${outdir}/conn_txqcd_${cfg}.h5"
  if [ ! -f "$lime" ]; then
    echo "[gpu $gpu λ=$lam] SKIP cfg.$cfg — lime missing: $lime"; return
  fi
  if [ -f "$outfile" ] && [ "$(stat -c%s "$outfile")" -gt 50000 ]; then
    return  # already done — resume-safe
  fi
  local logfile="logs/conn_txqcd_on_qcdref_lam${lam}_${cfg}_gpu${gpu}.log"
  echo "[gpu $gpu λ=$lam] cfg.$cfg → $outfile"
  CUDA_VISIBLE_DEVICES=$gpu \
    LAMBDA="$lam" \
    SUFFIX="$MEAS_SUFFIX" \
    ${AUX_INIT:+AUX_INIT="$AUX_INIT"} \
    IMPORT_CFG="$lime" \
    TXQCD_MULTIRHS_CG=1 TXQCD_MOOEE_CUBLAS=1 TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_PRECOMPUTE_GPU=1 TXQCD_TIME_REVERSED=1 \
    LATT=$LATT MEAS_SPACE_SRC=$SX MEAS_TIME_SRC=$ST MEAS_CG_TOL=$MEAS_CG_TOL \
    mpirun -np 1 --bind-to none ./meas_conn_txqcd "$cfg" --mpi 1.1.1.1 \
    > "$logfile" 2>&1
}

echo "=== TXQCD-on-vanilla-QCD measurements (mean-field aux) ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "  λ scan: $LAMBDAS    cfgs: ${#CFG_LIST[@]} (11610..12600 step 10)"

# λ-outer so each λ completes as a full 100-cfg ensemble; 4-GPU inner fan-out.
for lam in $LAMBDAS; do
  outdir="meas_2pt/txqcd_$(lam_tag "$lam")${MEAS_SUFFIX}"
  mkdir -p "$outdir"
  echo "----- λ=$lam  →  $outdir -----"
  date
  i=0
  while [ $i -lt ${#CFG_LIST[@]} ]; do
    pids=()
    for gpu in 0 1 2 3; do
      idx=$(( i + gpu ))
      [ $idx -lt ${#CFG_LIST[@]} ] || break
      run_one "${CFG_LIST[$idx]}" "$gpu" "$lam" &
      pids+=($!)
    done
    wait "${pids[@]}"
    i=$(( i + 4 ))
    done_count=$(ls "$outdir"/conn_txqcd_*.h5 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] λ=$lam progress: $done_count / ${#CFG_LIST[@]}"
  done
  echo "----- λ=$lam complete -----"; date
done

echo "=== all TXQCD-on-QCD-ref measurements complete ==="
date
for lam in $LAMBDAS; do
  d="meas_2pt/txqcd_$(lam_tag "$lam")${MEAS_SUFFIX}"
  echo "  λ=$lam : $(ls "$d"/conn_txqcd_*.h5 2>/dev/null | wc -l) cfgs in $d"
done
