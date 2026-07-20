#!/bin/bash
#SBATCH --job-name=recycle_gate
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/recycle_gate.%j.out
# ============================================================================
# RECYCLE solver gate at eta=0.7 (traj 30, the config the plain farm is
# measuring RIGHT NOW at 5 min/solve = the A/B reference): full NTSRC=48
# x 4 = 192-solve workload with DTXQCD_DEFORM_RECYCLE=1, output to a
# SEPARATE dir.  Verdicts from one job:
#  - iteration trajectory across solves (capture + iters prints): the
#    measured eigCG-class gain curve;
#  - h5 A/B vs the plain farm's baryon_deform h5 for the same traj
#    (init-guess deflation is unconditionally exact -- this is a
#    belt-and-braces check, threshold 1e-4 h5-compare offline);
#  - SHIFTSIGN self-diagnosis: if capture degrades after the first t0
#    steps, flip DTXQCD_RECYCLE_SHIFTSIGN and rerun (cheap).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
mkdir -p $P/meas_recycle_gate
rm -f $LOGD/recycle_gate.DONE
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 ETA=0.7 PROFILE=tent \
      LAMBDA_DTXQCD=5 CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      DATA_DIR=$P/meas_recycle_gate \
      MEAS_BARYON_NTSRC=48 \
      DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
      DTXQCD_DEFORM_RECYCLE=1 \
  $P/meas_baryon_deform_dtxqcd 30 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/recycle_gate_tr30.log 2>&1
echo "== recycle trajectory =="
grep -aE "recycle. dim" $LOGD/recycle_gate_tr30.log | head -8
grep -aE "recycle. dim" $LOGD/recycle_gate_tr30.log | tail -8
touch $LOGD/recycle_gate.DONE
