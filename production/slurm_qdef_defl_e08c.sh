#!/bin/bash
#SBATCH --job-name=qdef_defl08c
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_defl08c.%j.out
# ============================================================================
# eta=0.8 tight-basis deflation, CACHE-enabled + NTSRC=2 (production shape):
# first submission builds the tol=1e-5 basis and SAVES the ~1.2 GB sidecar
# (shared later by the sign-reweighting estimator — same Mpc~); a resubmit
# is the LOAD leg (eigensolve skipped, gauge fingerprint verified).  The
# t_src=24 solves exercise the frame translation with the tight basis.
# Guess-only (PROJ=0): worst case = plain CG by construction.
# ORD=1001: at eta=0.8 the wanted band spans THREE decades below the
# Chebyshev edge (2.7e-6..2.3e-3 vs lo~0.045) and ord=301 cannot amplify
# the band top enough to converge at tol=1e-5 (job 1301310: 0/16 after 11+
# restarts).  Sub-edge amplification grows exponentially with order: ~3x
# per IRL step for convergence in a few restarts.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_e08c
mkdir -p $P/deform_evec_cache

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.8 \
      MEAS_BARYON_NTSRC=2 MEAS_BARYON_MAXSOLVES=5 CG_MAX=80000 \
      DTXQCD_DEFORM_EO_DEFLATE=16 DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
      DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 \
      DTXQCD_DEFORM_EO_DEFLATE_PROJ=0 \
      DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/deform_evec_cache \
      DATA_DIR=$P/meas_defl_e08c \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_e08c.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/defl_e08c.log | tail -10
touch $LOGD/qdef_defl08c.DONE
