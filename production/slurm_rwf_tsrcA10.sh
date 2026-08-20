#!/bin/bash
#SBATCH --job-name=rwf_tsrcA10
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrcA10_%A_%a.out
# ============================================================================
# TSRC-RATIO FARM, first seed layer (array: 1 item = 1 GPU):
# 10 configs (trajs 1000..1900 step 100) x segment-top circuits at
# eta = 0.2 / 0.4 / 0.6 (full-period quadrature ~96 nodes, TSRC_RATIO=12),
# CG@1e-5, MG OFF.  Wall-fit split:
#   eta=0.2: 1 job NT=32 (~8.5h)   eta=0.4: 1 job NT=32 (~12h)
#   eta=0.6: 2 jobs NT=16 (~21h each)
# => 4 items/config, 40 items, ~620 GPU-h.  NT top-ups are ADDITIVE (new
# seed jobs) once the per-offset error propagation is measured from this
# layer.  Seeds: 5000 + traj + 100000*si + 10000000*seedjob.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
OUTD=$P/rwf_farmA10
mkdir -p "$OUTD"
# eta si njobs nt   (PIPELINE-VALIDATION layer: eta=0.2/0.4 only -- their
# per-offset bridge reconstruction + offset-averaging measurement derives
# the production NT.  eta=0.6 TSRC HELD: node noise +-45@NT=8 is ~100x the
# eta=0.2 scale; direct quadrature may need the kink-telescope instruments.)
ARMS=("0.2 1 1 32" "0.4 2 1 32")
ITEMS=()
for t in $(seq 1000 100 1900); do
  for a in 0 1; do
    set -- ${ARMS[$a]}
    for j in $(seq 1 $3); do ITEMS+=("$t $a $j"); done
  done
done
set -- ${ITEMS[$SLURM_ARRAY_TASK_ID]}
TRAJ=$1; AROW=$2; SJ=$3
set -- ${ARMS[$AROW]}
ETA=$1; SI=$2; NT=$4
TAG="tsrc_tr${TRAJ}_e${SI}_j${SJ}"
if [ -f "$OUTD/done_$TAG" ]; then
  echo "item $TAG already done"
  exit 0
fi
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=$TRAJ LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
      ETA=$ETA N_NOISE=$NT \
      NOISE_SEED=$((5000 + TRAJ + 100000 * SI + 10000000 * SJ)) \
      CG_TOL=1e-5 \
      PROFILE=tent PROF_T0=0 \
      DTXQCD_STOCHPF_TSRC_RATIO=12 \
      DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
  $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > "$OUTD/rwf_$TAG.log" 2>&1
if grep -aq "TSRCCLOSURE" "$OUTD/rwf_$TAG.log"; then
  touch "$OUTD/done_$TAG"
  grep -a "TSRCCLOSURE" "$OUTD/rwf_$TAG.log" | tail -1
else
  echo "item $TAG INCOMPLETE -- resubmit array to retry"
  exit 1
fi
