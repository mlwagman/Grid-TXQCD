#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 0:20:00
#SBATCH -J isolin_smoke
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/isolin_smoke.%j.out
# ISO_LINEAR mode smoke before the chained rounding scan trusts it.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
  env LATT=4.4.4.4 CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5 \
      TRAJ=400 LAMBDA_DTXQCD=5 ISO_LINEAR=1 DERIVCHECK=1 ETA=0.4 \
      N_NOISE=8 NOISE_SEED=1600 OMP_NUM_THREADS=16 \
  $P/Test_dtxqcd_isodeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
  --shm 512 --shm-mpi 1 2>&1 | grep -aE "ISO-LINEAR|DERIVCHECK|NOT CONV"
