#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 0:30:00
#SBATCH -J pdef_pilot
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/pdef_pilot.%j.out
# p-deformation reference-estimator pilot: DERIVCHECK + eta scan on the
# lam=5 4^4 chain (traj 400).  Conditioning probe = per-node CG iters.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
for ETA in 0.3 1.0 2.0; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=400 LAMBDA_DTXQCD=5 \
        ETA=$ETA PROFILE=tent N_GL=4 N_NOISE=16 NOISE_SEED=101 \
        OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_pdeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 2>&1 | grep -aE "PDEF|NOT CONVERGED|noise"
done
