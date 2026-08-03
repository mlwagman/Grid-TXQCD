#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 2:00:00
#SBATCH -J iso_ward256
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/iso_ward256.%j.out
# High-statistics Ward-identity confirmation: uniform profile, N_NOISE=256.
# The exact statement is dlogPf == 0; the 16-noise battery gave a combined
# -1.16 +- 0.43 (2.7 sigma) -- confirm it shrinks as 1/sqrt(N) (fluke) or
# converges (structural bias hunt).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
for t in 50 400; do
  srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
    env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=$t LAMBDA_DTXQCD=5 \
        ETA=0.5 PROFILE=tent WARDCHECK=1 DERIVCHECK=0 \
        N_GL=4 N_NOISE=256 NOISE_SEED=$((900 + t)) OMP_NUM_THREADS=16 \
    $P/Test_dtxqcd_isodeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 2>&1 \
    | grep -aE "ISO-RESULT|ISO-WARDCHECK" | sed "s/^/TRAJ=$t /"
done
