#!/bin/bash
#SBATCH --job-name=qdef_defl08
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_defl08.%j.out
# ============================================================================
# The eta=0.8 deflation harvest (lambda=5): Galerkin corrections at
# PROJ=500.  Baseline 59,931 undeflated EO iters (~24 min/solve).  The
# harvest ratio here is ~29x in sqrt(kappa) (source-effective sqrt(kappa)
# ~7450 vs 16-mode-deflated ~256) — eta=0.6's thin 1.5x margin (which the
# restart penalty ate: 6.5-7.8k vs 4479) does NOT transfer; this is the
# regime the mechanism exists for.  Two solves + a correctness h5 against
# the guess-only converged reference (meas_defl_e08 from job 1301281).
# Prediction: ~1-2k iters, ~1 min/solve.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_e08g

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.8 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=2 CG_MAX=80000 \
      DTXQCD_DEFORM_EO_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_PROJ=500 \
      DATA_DIR=$P/meas_defl_e08g \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e08g.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/defl_e08g.log | tail -6
if [ -f $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 ]; then
  echo "== h5 vs guess-only converged reference =="
  python3 $P/_qdef_cmp2.py $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 \
    $P/meas_defl_e08g/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
fi
touch $LOGD/qdef_defl08.DONE
