#!/bin/bash
#SBATCH --job-name=qdef_wcg2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_wcg2.%j.out
# ============================================================================
# Rounded-tent WIDTH SCAN, production CG cost (lambda=5, eta=0.8): plain
# EO CG wall-source solve iterations vs apex rounding half-width w in
# {0,2,4,8}.  Baseline w=0: 59,931 iters (~24 min).  CGNR iters scale
# ~1/sigma_min(M~): if the census shows a big sigma_min lift, this is the
# direct production payoff — the PROFILE as the preconditioner.  The
# rounded p(t) propagates to the baked correlator grading via the shared
# prof lambda.  One solve per width (MAXSOLVES=1), deflation OFF.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

for W in 2 4 8; do
  if [ "$W" = "0" ]; then PROF=tent; else PROF=tent_rounded2; fi
  mkdir -p $P/meas_wscan2_w$W
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
        PROFILE=$PROF PROFILE_W=$W ETA=0.8 \
        CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
        DTXQCD_DEFORM_EO_DEFLATE=0 \
        DATA_DIR=$P/meas_wscan2_w$W \
    $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/wcg2_w$W.log 2>&1
  echo "== w=$W =="
  grep -aE "deform-CG.*iters=" $LOGD/wcg2_w$W.log | tail -1
done
touch $LOGD/qdef_wcg2.DONE
