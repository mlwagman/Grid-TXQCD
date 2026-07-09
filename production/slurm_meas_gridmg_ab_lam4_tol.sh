#!/bin/bash
#SBATCH --job-name=meas_mg_tol_l4
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_meas_mg_tol_l4.%j.out
# ============================================================================
# Tolerance-scaling CONFIRMATION for the DTXQCD_MEAS_GRIDMG backend.
# The CG_TOL=1e-8 A/B (job 1299382) agreed to ~5e-7 rel-to-peak on every
# baryon channel — consistent with tolerance through the cubic contraction.
# This run repeats EO vs GridMG at CG_TOL=1e-10 (1 t_src, 4 solves/side).
# EXPECT: abs diffs shrink ~100x (tolerance-limited) rather than plateau
# (which would indicate a systematic backend bias).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG4=$P/cfgs/dtxqcd_lam4.0000_eo_fromchroma_cw_mds10
ABD=$P/meas_ab_lam4_tol
mkdir -p $ABD/eo $ABD/gridmg

COMMON="LATT=16.16.16.48 LAMBDA_DTXQCD=4 CFG_DIR=$CFG4 MEAS_BARYON_NTSRC=1 CG_TOL=1e-10"

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DATA_DIR=$ABD/eo \
  $P/meas_baryon_dtxqcd 120 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/tol_l4_eo.log 2>&1

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DATA_DIR=$ABD/gridmg \
      DTXQCD_MEAS_GRIDMG=1 DTXQCD_MEAS_MG_NBASIS=32 DTXQCD_MEAS_MG_GEOM=nnn \
      DTXQCD_MEAS_MG_ORD=300 DTXQCD_MEAS_MG_REFINE_MAXIT=0 DTXQCD_MEAS_MG_NEV=0 \
      DTXQCD_MEAS_MG_SETUP_PREFIX=$P/gridmg_setups/lam4_t120_fullvol_ord300 \
  $P/meas_baryon_dtxqcd 120 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/tol_l4_gridmg.log 2>&1

touch $LOGD/meas_tol_l4.DONE
echo "=== tol-scaling A/B complete $(date) ==="
