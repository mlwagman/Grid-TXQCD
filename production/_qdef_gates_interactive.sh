#!/bin/bash
# Interactive parallel run of the qdef acceptance gates (sbatch 1300692 is
# the queue-proof backup; whichever completes first decides).  3 legs, one
# GPU each, concurrent: G1 e2e eta=0.2 NTSRC=4 (~22 min), G2 twist NTSRC=1
# (~6 min), G3 untwisted NTSRC=1 reference (~6 min).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
REF=$P/meas_2pt/dtxqcd_lam4_deform_eta0p2/baryon_deform_dtxqcd_300.h5
COMMON="LATT=16.16.16.48 ETA=0.2 PROFILE=tent CFG_DIR=$CFG DTXQCD_DEFORM_QUDA=1"
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
SRUN="srun --overlap --mpi=pmix -N 1 -n 1 --gres=gpu:4 --cpu-bind=none"

mkdir -p $P/meas_deform_qgate $P/meas_deform_qtwist $P/meas_deform_qnotwist

$SRUN env CUDA_VISIBLE_DEVICES=0 $COMMON MEAS_BARYON_NTSRC=4 \
  DATA_DIR=$P/meas_deform_qgate \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_i_g1_e2e.log 2>&1 &
$SRUN env CUDA_VISIBLE_DEVICES=1 $COMMON MEAS_BARYON_NTSRC=1 GAUGE_TWIST=1 \
  DATA_DIR=$P/meas_deform_qtwist \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_i_g2_twist.log 2>&1 &
$SRUN env CUDA_VISIBLE_DEVICES=2 $COMMON MEAS_BARYON_NTSRC=1 \
  DATA_DIR=$P/meas_deform_qnotwist \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_i_g3_notwist.log 2>&1 &
wait

echo "=== G1: eta=0.2 e2e vs fleet Grid reference ==="
python3 $P/_qdef_cmp2.py $REF $P/meas_deform_qgate/baryon_deform_dtxqcd_300.h5 1e-5 \
  && G1=PASS || G1=FAIL
echo "=== G2: GAUGE_TWIST invariance ==="
python3 $P/_qdef_cmp2.py $P/meas_deform_qnotwist/baryon_deform_dtxqcd_300.h5 \
  $P/meas_deform_qtwist/baryon_deform_dtxqcd_300.h5 1e-5 \
  && G2=PASS || G2=FAIL
echo "QDEF GATES (interactive): G1=$G1 G2=$G2" | tee $LOGD/qdef_gates_i.VERDICT
touch $LOGD/qdef_gates_i.DONE
