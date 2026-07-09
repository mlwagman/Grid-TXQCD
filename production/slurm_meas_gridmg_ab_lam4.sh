#!/bin/bash
#SBATCH --job-name=meas_mg_ab_l4
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_meas_mg_ab_l4.%j.out
# ============================================================================
# Correlator A/B gate for the DTXQCD_MEAS_GRIDMG backend (93bd7791):
# meas_baryon_dtxqcd on the lambda=4 EO-stream traj 120 (the Grid-MG
# campaign config), EO backend vs GridMG backend, 2 t_srcs (8 solves/side).
# Verdict: h5diff -p 1e-5 on the two baryon_dtxqcd_120.h5 outputs.
# GridMG reuses the campaign's cached nb32/nnn/ord300 setup (note GEOM=nnn
# to match the cache tag; IRL deflation runs fresh, ~20 s).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG4=$P/cfgs/dtxqcd_lam4.0000_eo_fromchroma_cw_mds10
ABD=$P/meas_ab_lam4
mkdir -p $ABD/eo $ABD/gridmg

COMMON="LATT=16.16.16.48 LAMBDA_DTXQCD=4 CFG_DIR=$CFG4 MEAS_BARYON_NTSRC=2 CG_TOL=1e-8"

# A: EO baseline (the current default backend)
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DATA_DIR=$ABD/eo \
  $P/meas_baryon_dtxqcd 120 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/ab_l4_eo.log 2>&1

# B: GridMG backend on the cached campaign setup
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DATA_DIR=$ABD/gridmg \
      DTXQCD_MEAS_GRIDMG=1 DTXQCD_MEAS_MG_NBASIS=32 DTXQCD_MEAS_MG_GEOM=nnn \
      DTXQCD_MEAS_MG_ORD=300 DTXQCD_MEAS_MG_REFINE_MAXIT=0 DTXQCD_MEAS_MG_NEV=0 \
      DTXQCD_MEAS_MG_SETUP_PREFIX=$P/gridmg_setups/lam4_t120_fullvol_ord300 \
  $P/meas_baryon_dtxqcd 120 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/ab_l4_gridmg.log 2>&1

# Verdict
if h5diff -p 1e-5 $ABD/eo/baryon_dtxqcd_120.h5 $ABD/gridmg/baryon_dtxqcd_120.h5 \
     > $LOGD/ab_l4_h5diff.log 2>&1; then
  echo "MEAS GRIDMG A/B PASSED (h5diff -p 1e-5 clean)" | tee $LOGD/ab_l4.VERDICT
else
  echo "MEAS GRIDMG A/B FAILED — see ab_l4_h5diff.log" | tee $LOGD/ab_l4.VERDICT
fi
touch $LOGD/meas_ab_l4.DONE
