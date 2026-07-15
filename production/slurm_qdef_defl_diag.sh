#!/bin/bash
#SBATCH --job-name=qdef_dfdiag
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_dfdiag.%j.out
# ============================================================================
# Deflation zero-harvest diagnostic (eta=0.8, tight basis via the CACHE —
# no eigensolve): print |v_i^dag b|/|b| per mode + post-guess |r0|/|b|,
# then let the solve run only 2000 iters (CG_MAX; the count is not the
# point here) to keep the job short.  Job 1301317 measured guess harvest
# = +2 iters on 59,931: either the wall source is ~orthogonal to the
# deep modes (physics) or the basis misses a deeper mode (instrument).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_diag

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.8 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 CG_MAX=2000 \
      DTXQCD_DEFORM_EO_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
      DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 \
      DTXQCD_DEFORM_EO_DEFLATE_PROJ=0 \
      DTXQCD_DEFORM_EO_DEFLATE_DIAG=1 \
      DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/deform_evec_cache \
      DATA_DIR=$P/meas_defl_diag \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_diag.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/defl_diag.log | tail -24
touch $LOGD/qdef_dfdiag.DONE
