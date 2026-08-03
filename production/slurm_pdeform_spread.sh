#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 1:00:00
#SBATCH -J pdef_spread
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/pdef_spread.%j.out
# Config-to-config spread of the p-contour log-weight (-dS_gauss + dlogPf):
# all 8 checkpoints of the lam=5 4^4 chain at eta=0.3 and 1.0 (tent).
# sigma(log w) across configs is the ESS driver -- THE feasibility number
# the single-config pilot cannot give.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
TRAJS=$(ls "$CFG" | grep -oP 'ckpoint_lat\.\K[0-9]+' | sort -n)
for ETA in 0.3 1.0; do
  for t in $TRAJS; do
    srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
      env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=$t LAMBDA_DTXQCD=5 \
          ETA=$ETA PROFILE=tent N_GL=4 N_NOISE=16 NOISE_SEED=$((300 + t)) \
          DERIVCHECK=0 OMP_NUM_THREADS=16 \
      $P/Test_dtxqcd_pdeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 2>&1 \
      | grep -aE "PDEF-RESULT" | sed "s/^/TRAJ=$t ETA=$ETA /"
  done
done
