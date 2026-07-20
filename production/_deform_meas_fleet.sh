#!/bin/bash
#SBATCH --job-name=dtx_deform_fleet
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs/dtx_deform_fleet.%j.out
# Deformed aux baryon fleet: source-centered tent, NTSRC=4 (16 solves per
# config), one config per GPU, resumable via h5-skip.  Parametrized:
#   DEFETA (0.2)     eta
#   LAM (4)          lambda -- picks cfg stream, baryon8 partner dir, output
#   DEFTRAJS         explicit traj list (default: all baryon8 partners)
#   DEFCGMAX ("")    CG_MAX override; empty = the meas binary's
#                    eta-banded default (20k/60k/100k/200k at
#                    <0.55/0.65/0.75/above -- tent-ladder-calibrated)
# Fast path: meas defaults DTXQCD_DEFORM_QUDA=1 (5.7x/iter) and
# DTXQCD_DEFORM_EO=1 at lambda>=5; MG auto-policy (off for sharp tent).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
DEFETA=${DEFETA:-0.2}
LAM=${LAM:-4}
DEFCGMAX=${DEFCGMAX:-}   # empty => meas eta-banded default
# DEFPROFILE (tent): contour profile.  Non-tent profiles get their own
# output dir (deform_<profile>_eta<tag>) -- never mix profiles in one dir;
# the un-tagged deform_eta* dirs are the legacy SHARP-TENT datasets.
DEFPROFILE=${DEFPROFILE:-tent}
# DEFNTSRC (4): wall sources per config (source-centered contour each).
# NOTE the reweighting coupling: every source offset is its OWN contour and
# needs its own ΔlogPf(t0) — high-NTSRC datasets rely on the reweighting
# factors being smooth/interpolatable in t0 (or on the ~unity-weight regime).
DEFNTSRC=${DEFNTSRC:-4}
# DEFGPUS (1): GPUs per config.  Since 2026-07-19 the MG default is the
# L1 config (BLOCK=4.4.4.4, 2.4 GB LU on the SAME GPU, h5-certified
# better than the old two-GPU layout) -- DEFGPUS=1 is correct for MG
# runs too.  (Historical: the pre-L1 22 GB dev-1 LU required pairs; the
# 2026-07-16 single-GPU e06 3-min-death cycles were that config.)
DEFGPUS=${DEFGPUS:-1}
ETATAG=$(echo "$DEFETA" | tr '.' 'p')
PROFTAG=""
if [ "$DEFPROFILE" != "tent" ]; then PROFTAG="${DEFPROFILE}_"; fi
SRCTAG=""
if [ "$DEFNTSRC" != "4" ]; then SRCTAG="_s${DEFNTSRC}"; fi
OUT=$P/meas_2pt/dtxqcd_lam${LAM}_deform_${PROFTAG}eta${ETATAG}${SRCTAG}
CFG=$P/cfgs/dtxqcd_lam${LAM}.0000_tla_fromchroma_cw
PAIR=$P/meas_2pt/dtxqcd_lam${LAM}_baryon8
mkdir -p "$OUT"
# All configs with an UNDEFORMED partner (pairing requirement): enumerate
# the baryon8 h5s.  DEFTRAJS="100 200 ..." overrides (eta-scan subsets).
if [ -n "$DEFTRAJS" ]; then
  TRAJS=($DEFTRAJS)
else
  TRAJS=()
  for f in $(ls -v $PAIR/baryon_dtxqcd_*.h5 2>/dev/null); do
    t=$(basename "$f"); t=${t#baryon_dtxqcd_}; t=${t%.h5}
    TRAJS+=("$t")
  done
fi
echo "=== deform fleet lam=$LAM eta=$DEFETA profile=$DEFPROFILE cgmax=$DEFCGMAX: ${#TRAJS[@]} paired configs ==="
if [ "$DEFGPUS" = "2" ]; then SLOTS=("0,1" "2,3"); else SLOTS=(0 1 2 3); fi
declare -a SLOT_PID SLOT_TRAJ SLOT_T0
launch() {
  local traj=$1 slot=$2
  echo "  [launch] traj=$traj -> GPUs ${SLOTS[$slot]}"
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=${SLOTS[$slot]} \
        LATT=16.16.16.48 ETA=$DEFETA PROFILE=$DEFPROFILE \
        MEAS_BARYON_NTSRC=$DEFNTSRC \
        CG_TOL=1e-7 CG_MAX=$DEFCGMAX LAMBDA_DTXQCD=$LAM \
        CFG_DIR=$CFG DATA_DIR=$OUT \
        DTXQCD_MG_NB=${DEFMGNB:-16} DTXQCD_MG_REUSE_NV=${DEFREUSE:-1} \
        OMP_NUM_THREADS=16 \
    $P/meas_baryon_deform_dtxqcd "$traj" --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$OUT/deform.$traj.log" 2>&1 &
  SLOT_PID[$slot]=$!
  SLOT_TRAJ[$slot]=$traj
  SLOT_T0[$slot]=$(date +%s)
}
# Work queue with INSTANT-DEATH requeue: a step that dies in <90 s without
# an h5 (srun/pmix launch failures -- e.g. the 2026-07-16 port-exhaustion
# cascade after a scancel on the same node -- or early aborts) is put back
# on the queue with a 300 s backoff, instead of burning the whole list.
QUEUE=("${TRAJS[@]}")
FAILSTREAK=0
REQUEUES=0
MAXREQ=$(( ${#TRAJS[@]} * 3 + 10 ))
free_slot() {   # sets FREE_G to a free slot index, or "" (runs in-shell:
  local g pid dt traj  # it mutates QUEUE/REQUEUES -- no command substitution)
  FREE_G=""
  for g in "${!SLOTS[@]}"; do
    pid=${SLOT_PID[$g]:-}
    if [ -z "$pid" ]; then FREE_G="$g"; return; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      traj=${SLOT_TRAJ[$g]}
      dt=$(( $(date +%s) - ${SLOT_T0[$g]} ))
      if [ ! -f "$OUT/baryon_deform_dtxqcd_${traj}.h5" ] && [ "$dt" -lt 90 ]; then
        QUEUE+=("$traj")
        REQUEUES=$((REQUEUES + 1))
        FAILSTREAK=$((FAILSTREAK + 1))
        echo "  [instant-death] traj=$traj after ${dt}s -- requeued (streak $FAILSTREAK, total $REQUEUES)"
        if [ "$FAILSTREAK" -ge 2 ]; then
          echo "  [backoff] sleeping 300s for node cleanup"
          sleep 300
        fi
        if [ "$REQUEUES" -gt "$MAXREQ" ]; then
          echo "FATAL: requeue budget exhausted -- node unusable"
          exit 1
        fi
      else
        FAILSTREAK=0
      fi
      SLOT_PID[$g]=""
      FREE_G="$g"
      return
    fi
  done
}
while [ ${#QUEUE[@]} -gt 0 ]; do
  traj=${QUEUE[0]}
  QUEUE=("${QUEUE[@]:1}")
  [ -f "$OUT/baryon_deform_dtxqcd_${traj}.h5" ] && continue
  placed=0
  while [ $placed -eq 0 ]; do
    free_slot
    if [ -n "$FREE_G" ]; then launch "$traj" "$FREE_G"; placed=1; else sleep 30; fi
  done
done
wait
echo "=== deform fleet done $(date) ==="
