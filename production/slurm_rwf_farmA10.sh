#!/bin/bash
#SBATCH --job-name=rwf_farmA10
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_farmA10_%A_%a.out
# ============================================================================
# OPTION-A ANCHOR FARM (array job: 1 item = 1 GPU = 1 task, 100 items):
# 10 configs (trajs 1000..1900 step 100) x segments [0,0.2]/[0.2,0.4]/
# [0.4,0.6], tent, CG@1e-5 EVERYWHERE (uniform precision; bias 0.008-0.04
# measured, low-seg 1e-4 clemency declined for cleanliness), MG OFF
# (estimator-bias exclusion).  TARGET per config (A-ratios scaled to seg3
# N=1000, mwagman): N = 737/763/1000 -> sigma 0.62/0.69/0.82 (total 1.24).
# THIS ARRAY = LAYER 1 (half stats): N = 370/380/500 per config,
#   seg1: 1 job x N=370 (~8.2h)  seg2: 1 x N=380 (~11.8h)  seg3: 2 x N=250
# 40 items, ~540 GPU-h.  LAYER 2 (the other half) tops up additively with
# seedjob indices 11+ (edit SEGS njobs/N and the seed offset, resubmit).
# Item map: idx -> (traj, seg, seedjob); done-markers make rerun idempotent
# (resubmit the array; finished items exit in seconds).
# Seeds: NOISE_SEED = 3000 + traj + 100000*seg + 10000000*seedjob.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
OUTD=$P/rwf_farmA10
mkdir -p "$OUTD"
SEGS=("0.0 0.2 1 1 370" "0.2 0.4 2 1 380" "0.4 0.6 3 2 250")
# enumerate the full item list deterministically (traj-major, 10 per config)
ITEMS=()
for t in $(seq 1000 100 1900); do
  for s in 0 1 2; do
    set -- ${SEGS[$s]}
    for j in $(seq 1 $4); do ITEMS+=("$t $s $j"); done
  done
done
set -- ${ITEMS[$SLURM_ARRAY_TASK_ID]}
TRAJ=$1; SROW=$2; SJ=$3
set -- ${SEGS[$SROW]}
ELO=$1; EHI=$2; SI=$3; NA=$5
TAG="tr${TRAJ}_seg${SI}_j${SJ}"
if [ -f "$OUTD/done_$TAG" ]; then
  echo "item $TAG already done"
  exit 0
fi
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=$TRAJ LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=$EHI ETA_LO=$ELO N_GL=2 N_NOISE=$NA \
      NOISE_SEED=$((3000 + TRAJ + 100000 * SI + 10000000 * SJ)) \
      CG_TOL=1e-5 \
      PROFILE=tent PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
      DTXQCD_STOCHPF_DEFLATE=0 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > "$OUTD/rwf_$TAG.log" 2>&1
if grep -aq "STOCHPF traj" "$OUTD/rwf_$TAG.log"; then
  touch "$OUTD/done_$TAG"
  grep -a "STOCHPF traj" "$OUTD/rwf_$TAG.log" | tail -1
else
  echo "item $TAG INCOMPLETE (wall/death) -- resubmit array to retry"
  exit 1
fi
