#!/bin/bash
#SBATCH --job-name=qdef_eoflip
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_eoflip.%j.out
# lambda>=5 EO-default flip gate: BARE deform env (only LAMBDA_DTXQCD=5)
# at 16^3x48 must come up "EO Schur solve ENABLED" + the device-adjointness
# qcheck (the volume-split check: full Schur identity only <= 16^4 sites).
# CG_MAX=1 + MAXSOLVES=1 self-terminates.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
mkdir -p $P/meas_deform_qgate
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env LATT=16.16.16.48 TRAJ=300 ETA=0.6 PROFILE=tent LAMBDA_DTXQCD=5 \
      CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
      DATA_DIR=$P/meas_deform_qgate MEAS_BARYON_NTSRC=1 \
      MEAS_BARYON_MAXSOLVES=1 CG_MAX=1 \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 > $LOGD/qdef_eoflip.log 2>&1
echo "=== flip gate ==="
grep -aE "EO Schur solve ENABLED|deform-qcheck" $LOGD/qdef_eoflip.log | tail -8
touch $LOGD/qdef_eoflip.DONE
