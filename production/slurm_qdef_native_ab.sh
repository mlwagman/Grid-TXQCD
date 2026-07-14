#!/bin/bash
#SBATCH --job-name=qdef_native
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_native_ab.%j.out
# ============================================================================
# Native hop-correction kernel prototype: validate + benchmark.
#  1. qcheck with DTXQCD_DEFORM_NATIVE_CORR=1 (eta=0.2): the native-corr M~
#     and M~^dag vs the Grid sandwich on random vectors — one gate covers
#     basis conversion, neighbor indexing, projectors, coefficient arrays,
#     and the dagger (array-swap + projector-flip).
#  2. 300-iteration profiled pair: native vs Grid-corr per-leg phase table.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
mkdir -p $P/meas_deform_qgate
COMMON="LATT=16.16.16.48 TRAJ=300 ETA=0.2 PROFILE=tent MEAS_BARYON_NTSRC=1 CG_MAX=300 CFG_DIR=$CFG DATA_DIR=$P/meas_deform_qgate DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_PROFILE=1"
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DTXQCD_DEFORM_NATIVE_CORR=1 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_native_on.log 2>&1

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env $COMMON DTXQCD_DEFORM_QUDA_CHECK=0 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_native_off.log 2>&1

echo "=== native qcheck ==="
grep -aE "deform-qcheck|deform-native" $LOGD/qdef_native_on.log | head -8
echo "=== per-leg profile: NATIVE ==="
grep -a "deform-prof" $LOGD/qdef_native_on.log | tail -1
echo "=== per-leg profile: GRID-CORR ==="
grep -a "deform-prof" $LOGD/qdef_native_off.log | tail -1
touch $LOGD/qdef_native_ab.DONE
