#!/bin/bash
#SBATCH --job-name=smoking_gun
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/smoking_gun.%j.out
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/smoking_gun.DONE
for ETA in 0.4423 0.6211; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=16.16.16.48 TRAJ=300 ETA=$ETA PROF_T0=0 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
    $P/Test_dtxqcd_smoking_gun --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/smoking_gun_e$ETA.log 2>&1
  echo "== eta=$ETA =="
  grep -a "GUN" $LOGD/smoking_gun_e$ETA.log
done
touch $LOGD/smoking_gun.DONE
