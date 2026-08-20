#!/bin/bash
#SBATCH --job-name=rwf_farm30
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_farm30_%j.out
# ============================================================================
# 30-CONFIG REWEIGHTING-FACTOR FARM (tent, all four segments, anchors).
# TEMPLATE -- do not launch until the one-config test closes these knobs:
#   FARM_CG_TOL  : set by the tol-1e-8 truth triangle (1310894).
#   NA1..NA4     : per-segment N_NOISE from the N=1000 sigma table (1310696).
#   (TSRC half is a separate farm script once the TSRC MG smoke prices it.)
# Work-queue over (config, segment) pairs on 4 GPUs; h5/DONE-skip resume;
# chain with: sbatch --dependency=afterany:<prev> slurm_rwf_farm30.sh
# Configs: trajs 50..1500 step 50 (30 configs, fully decorrelated at
# tau_int ~ 0.5 per 10 traj).  MG env on everywhere (inert below ETAMIN).
# Seeds: NOISE_SEED = 3000 + traj + 100000*segindex (multi-job convention).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
OUTD=$P/rwf_farm30
mkdir -p "$OUTD"
# Tolerance verdict (tolcurve 1311314, same-seed vs 1e-8 truth at [0.4,0.6]):
#   CG@1e-5 bias +0.04/-0.08 OK; CG@1e-4 bias +3.2 FATAL; MG biased +4.2
#   IDENTICALLY at 1e-4 and 1e-6 => MG path solves a perturbed operator --
#   EXCLUDED from the farm until the fork finds the mismatch.  Low-segment
#   1e-4 clemency pending rwf_truthlow (1311505).
FARM_CG_TOL=${FARM_CG_TOL:-1e-5}
NA1=${NA1:-SET_ME}; NA2=${NA2:-SET_ME}   # <- N=1000 sigma table
NA3=${NA3:-SET_ME}; NA4=${NA4:-SET_ME}
if [ "$FARM_CG_TOL" = "SET_ME" ]; then
  echo "FARM_CG_TOL / NA1..4 not set -- refusing to launch a mis-specced farm"
  exit 1
fi
SEGS=("0.0 0.2 $NA1 1" "0.2 0.4 $NA2 2" "0.4 0.6 $NA3 3" "0.6 0.7 $NA4 4")
TRAJS=$(seq 50 50 1500)
# build work list of unfinished (traj, seg) pairs
WORK=()
for t in $TRAJS; do
  for i in 0 1 2 3; do
    set -- ${SEGS[$i]}
    [ -f "$OUTD/done_tr${t}_seg$4" ] || WORK+=("$t $i")
  done
done
echo "work items remaining: ${#WORK[@]}"
idx=0
run_one () {  # run_one <gpu> <traj> <segindex>
  set -- $1 $2 ${SEGS[$3]}
  local GPU=$1 TRAJ=$2 ELO=$3 EHI=$4 NA=$5 SI=$6
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=$TRAJ \
        LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        ETA=$EHI ETA_LO=$ELO N_GL=2 N_NOISE=$NA \
        NOISE_SEED=$((3000 + TRAJ + 100000 * SI)) \
        CG_TOL=$FARM_CG_TOL \
        PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 \
    > "$OUTD/rwf_tr${TRAJ}_seg${SI}.log" 2>&1
  if grep -aq "STOCHPF traj" "$OUTD/rwf_tr${TRAJ}_seg${SI}.log"; then
    touch "$OUTD/done_tr${TRAJ}_seg${SI}"
  fi
}
# 4 sequential workers, one per GPU, pulling from the shared list via a
# FILE-based counter under an mkdir lock (subshells do NOT share shell
# variables -- the fleet script's state-loss trap).
echo 0 > "$OUTD/.next"
for g in 0 1 2 3; do
  (
    while :; do
      while ! mkdir "$OUTD/.lock" 2>/dev/null; do sleep 1; done
      i=$(cat "$OUTD/.next")
      if [ "$i" -ge "${#WORK[@]}" ]; then rmdir "$OUTD/.lock"; break; fi
      echo $((i + 1)) > "$OUTD/.next"
      rmdir "$OUTD/.lock"
      run_one $g ${WORK[$i]}
    done
  ) &
done
wait
echo "farm pass complete: $(ls $OUTD/done_* 2>/dev/null | wc -l) / $((30*4)) items"
