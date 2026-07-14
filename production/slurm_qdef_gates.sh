#!/bin/bash
#SBATCH --job-name=qdef_gates
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_gates.%j.out
# ============================================================================
# Final acceptance gates for DTXQCD_DEFORM_QUDA=1 (the hop-correction QUDA
# backend for the eta-deformed solves; HANDOFF_QUDA_DEFORM_HOPWEIGHTS.md).
# Already green (interactive, 2026-07-13): operator identity vs Grid sandwich
# 1.0e-15 (eta=0.2) / 9.5e-10 (eta=0.8, the sandwich's own e^{eta p/2}
# roundoff); adjointness ~1e-15; 16/16 solves converged at iteration parity;
# 0.0389 s/iter vs 0.22 Grid = 5.7x (config: 21.6 min vs ~2 h).
# This job (interactive turnover killed the e2e h5 twice):
#   G1  eta=0.2 NTSRC=4 fleet-exact run -> h5 vs the Grid-path fleet
#       reference to 1e-5 abs-to-peak (CG_TOL=1e-7 through the contraction).
#   G2  GAUGE_TWIST invariance: NTSRC=1 twisted vs untwisted to 1e-5.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
REF=$P/meas_2pt/dtxqcd_lam4_deform_eta0p2/baryon_deform_dtxqcd_300.h5
COMMON="LATT=16.16.16.48 ETA=0.2 PROFILE=tent CFG_DIR=$CFG DTXQCD_DEFORM_QUDA=1"
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

mkdir -p $P/meas_deform_qgate $P/meas_deform_qtwist $P/meas_deform_qnotwist

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON MEAS_BARYON_NTSRC=4 DATA_DIR=$P/meas_deform_qgate \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_g1_e2e.log 2>&1
RC1=$?

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON MEAS_BARYON_NTSRC=1 GAUGE_TWIST=1 DATA_DIR=$P/meas_deform_qtwist \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_g2_twist.log 2>&1
RC2=$?

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON MEAS_BARYON_NTSRC=1 DATA_DIR=$P/meas_deform_qnotwist \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_g3_notwist.log 2>&1
RC3=$?

echo "=== run rcs: e2e=$RC1 twist=$RC2 notwist=$RC3 (134 = benign QUDA teardown) ==="

echo "=== G1: eta=0.2 e2e vs fleet Grid reference ==="
python3 $P/_qdef_cmp2.py $REF $P/meas_deform_qgate/baryon_deform_dtxqcd_300.h5 1e-5 \
  && G1=PASS || G1=FAIL

echo "=== G2: GAUGE_TWIST invariance ==="
python3 $P/_qdef_cmp2.py $P/meas_deform_qnotwist/baryon_deform_dtxqcd_300.h5 \
  $P/meas_deform_qtwist/baryon_deform_dtxqcd_300.h5 1e-5 \
  && G2=PASS || G2=FAIL

echo "QDEF GATES: G1=$G1 G2=$G2" | tee $LOGD/qdef_gates.VERDICT
touch $LOGD/qdef_gates.DONE
