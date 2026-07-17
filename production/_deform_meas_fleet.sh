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
#   DEFCGMAX (20000) CG_MAX (eta=0.6 sharp tent needs ~60000)
# Fast path: meas defaults DTXQCD_DEFORM_QUDA=1 (5.7x/iter) and
# DTXQCD_DEFORM_EO=1 at lambda>=5; MG auto-policy (off for sharp tent).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
DEFETA=${DEFETA:-0.2}
LAM=${LAM:-4}
DEFCGMAX=${DEFCGMAX:-20000}
# DEFPROFILE (tent): contour profile.  Non-tent profiles get their own
# output dir (deform_<profile>_eta<tag>) -- never mix profiles in one dir;
# the un-tagged deform_eta* dirs are the legacy SHARP-TENT datasets.
DEFPROFILE=${DEFPROFILE:-tent}
# DEFNTSRC (4): wall sources per config (source-centered contour each).
# NOTE the reweighting coupling: every source offset is its OWN contour and
# needs its own ΔlogPf(t0) — high-NTSRC datasets rely on the reweighting
# factors being smooth/interpolatable in t0 (or on the ~unity-weight regime).
DEFNTSRC=${DEFNTSRC:-4}
# DEFGPUS (1): GPUs per config.  MG runs (eta>=0.6) REQUIRE 2 -- the deform
# MG's dense coarse LU (~22 GB) lives on the SECOND visible device
# (CoarseDirect dev=1 in dtxqcd_deform_mg.h); with one visible GPU it lands
# on the fine-op device and the step OOMs/dies (2026-07-16 lesson: both e06
# farms cycled through configs in ~3 min deaths).
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
declare -a SLOT_PID
launch() {
  local traj=$1 slot=$2
  echo "  [launch] traj=$traj -> GPUs ${SLOTS[$slot]}"
  srun --overlap --mpi=pmix -N1 -n1 -c16 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=${SLOTS[$slot]} \
        LATT=16.16.16.48 ETA=$DEFETA PROFILE=$DEFPROFILE \
        MEAS_BARYON_NTSRC=$DEFNTSRC \
        CG_TOL=1e-7 CG_MAX=$DEFCGMAX LAMBDA_DTXQCD=$LAM \
        CFG_DIR=$CFG DATA_DIR=$OUT \
        OMP_NUM_THREADS=16 \
    $P/meas_baryon_deform_dtxqcd "$traj" --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > "$OUT/deform.$traj.log" 2>&1 &
  SLOT_PID[$gpu]=$!
}
for traj in "${TRAJS[@]}"; do
  [ -f "$OUT/baryon_deform_dtxqcd_${traj}.h5" ] && continue
  placed=0
  while [ $placed -eq 0 ]; do
    for g in "${!SLOTS[@]}"; do
      if [ -z "${SLOT_PID[$g]}" ] || ! kill -0 "${SLOT_PID[$g]}" 2>/dev/null; then
        launch "$traj" "$g"; placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 30
  done
done
wait
echo "=== deform fleet done $(date) ==="
