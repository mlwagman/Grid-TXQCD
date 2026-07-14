#!/bin/bash
#SBATCH --job-name=qdef_eo_qc
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_eo_qc.%j.out
# Operator-level qcheck:EO at 4^4 (lambda=5 signhop stream, the §5b gate
# config).  At 16^3x48 the EO operator check is genuinely memory-starved
# (EO site caches + gemmBatched staging ~15-20 GB + managed-memory paging
# + the check's full-volume Grid applies -> Grid MM cache assert; jobs
# 1301152/84/96/98).  The operator identity is volume-independent and
# machine-precision at 4^4; the 16^3x48 correctness is certified at
# SOLUTION level by the G-sol h5 gates (PASSED 2.5e-8 / 2.9e-7).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
GRID="--grid 4.4.4.4 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_deform_qgate

for eta in 0.2 0.8; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=4.4.4.4 TRAJ=250 PROFILE=tent CFG_DIR=$CFG \
        DATA_DIR=$P/meas_deform_qgate CG_TOL=1e-7 ETA=$eta \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 CG_MAX=1 \
        DTXQCD_DEFORM_EO=1 \
    $P/meas_baryon_deform_dtxqcd 250 $GRID > $LOGD/qdef_eoqc_e${eta}.log 2>&1
  echo "== qcheck eta=$eta =="
  grep -aE "deform-qcheck" $LOGD/qdef_eoqc_e${eta}.log | tail -7
done
touch $LOGD/qdef_eo_qc.DONE
