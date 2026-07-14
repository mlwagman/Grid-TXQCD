#!/bin/bash
#SBATCH --job-name=qdef_eo
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_eo.%j.out
# ============================================================================
# EO deformed Schur operator (Mpc~): gates + lambda=5 benchmark.
#  1. qcheck + qcheck:EO at eta=0.2 and eta=0.8 (device Mpc~ vs the Schur
#     complement synthesized from the 1e-15-validated full M~ + cached
#     Mee^{-1}; adjointness of the device pair).  CG_MAX=1 self-terminates.
#  2. Timing ladder at lambda=5 (the §5b target): EO vs full solve at
#     eta={0.4,0.6,0.8}, NTSRC=1.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
COMMON="LATT=16.16.16.48 TRAJ=300 PROFILE=tent CFG_DIR=$CFG5 DATA_DIR=$P/meas_deform_qgate CG_TOL=1e-7"
mkdir -p $P/meas_deform_qgate

for eta in 0.2 0.8; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env $COMMON ETA=$eta MEAS_BARYON_NTSRC=1 CG_MAX=1 DTXQCD_DEFORM_EO=1 \
    $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_eo_gate_e${eta}.log 2>&1
  echo "== gate eta=$eta =="
  grep -aE "deform-qcheck" $LOGD/qdef_eo_gate_e${eta}.log | tail -6
done

mkdir -p $P/meas_deform_eogate_eo $P/meas_deform_eogate_full
for eta in 0.4 0.6 0.8; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env $COMMON ETA=$eta MEAS_BARYON_NTSRC=1 CG_MAX=60000 DTXQCD_DEFORM_EO=1 \
        DTXQCD_DEFORM_QUDA_CHECK=0 DATA_DIR=$P/meas_deform_eogate_eo \
    $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_eo_t_e${eta}.log 2>&1
  echo "EO   eta=$eta: $(grep -a 'deform-CG' $LOGD/qdef_eo_t_e${eta}.log | head -1)"
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env $COMMON ETA=$eta MEAS_BARYON_NTSRC=1 CG_MAX=60000 \
        DTXQCD_DEFORM_QUDA_CHECK=0 DATA_DIR=$P/meas_deform_eogate_full \
    $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_full_t_e${eta}.log 2>&1
  echo "FULL eta=$eta: $(grep -a 'deform-CG' $LOGD/qdef_full_t_e${eta}.log | head -1)"
  echo "== G-sol eta=$eta: EO vs FULL correlators =="
  python3 $P/_qdef_cmp2.py $P/meas_deform_eogate_full/baryon_deform_dtxqcd_300.h5 \
    $P/meas_deform_eogate_eo/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
done
touch $LOGD/qdef_eo.DONE
