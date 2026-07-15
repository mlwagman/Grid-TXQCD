#!/bin/bash
#SBATCH --job-name=qdef_defl08t
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_defl08t.%j.out
# ============================================================================
# eta=0.8 deflation, TIGHT basis + GUESS-ONLY (the P2 lesson, job 1301288):
# tol=1e-3 evecs are partly garbage (loose ev[15]=1.09e-3 vs tight 3.0e-4
# at eta=0.6) — guess bought ~1%, corrections+restarts stalled/regressed.
# tol=1e-5 evecs at eta=0.6: guess-only 3427/3543 vs 4479 undeflated with
# NO restart anywhere (worst case = plain CG by construction).  Here the
# headroom is ~29x in sqrt(kappa) (59,931 undeflated iters; deflated
# sqrt(kappa)~256).  PROJ=0 disables corrections.  Eigensolve ~2.7x the
# 436 s loose cost, amortized per (config, eta).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_e08t

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.8 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=2 CG_MAX=80000 \
      DTXQCD_DEFORM_EO_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
      DTXQCD_DEFORM_EO_DEFLATE_PROJ=0 \
      DATA_DIR=$P/meas_defl_e08t \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e08t.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/defl_e08t.log | tail -6
if [ -f $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 ]; then
  echo "== h5 vs guess-only converged reference =="
  python3 $P/_qdef_cmp2.py $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 \
    $P/meas_defl_e08t/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
fi
touch $LOGD/qdef_defl08t.DONE
