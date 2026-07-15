#!/bin/bash
#SBATCH --job-name=qdef_rec32
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_rec32.%j.out
# ============================================================================
# eta=0.8 RECIPE VALIDATION on the nev=32 basis (cached by 1301347): the
# 16-mode basis missed the degenerate multiplet partners the wall source
# couples to (corr@ drops were 4th-decimal noise; PROJ corrections then
# net-HURT: 80k cap vs 59,931 plain).  With partners captured: guess +
# corrections at PROJ=1000 (~4x sqrt(kappa_complement)) target ~5-10k
# iters.  Full convergence + h5 vs the undeflated eta=0.8 reference.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG5=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
mkdir -p $P/meas_defl_rec32

srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 PROFILE=tent LAMBDA_DTXQCD=5 CFG_DIR=$CFG5 \
      CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 ETA=0.8 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=2 CG_MAX=80000 \
      DTXQCD_DEFORM_EO_DEFLATE=32 DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
      DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 \
      DTXQCD_DEFORM_EO_DEFLATE_PROJ=1000 \
      DTXQCD_DEFORM_EO_DEFLATE_DIAG=1 \
      DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/deform_evec_cache \
      DATA_DIR=$P/meas_defl_rec32 \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/defl_rec32.log 2>&1
grep -aE "deform-defl|deform-CG" $LOGD/defl_rec32.log | tail -24
if [ -f $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 ]; then
  echo "== h5 vs undeflated eta=0.8 reference =="
  python3 $P/_qdef_cmp2.py $P/meas_defl_e08/baryon_deform_dtxqcd_300.h5 \
    $P/meas_defl_rec32/baryon_deform_dtxqcd_300.h5 1e-4 | tail -1
fi
touch $LOGD/qdef_rec32.DONE
