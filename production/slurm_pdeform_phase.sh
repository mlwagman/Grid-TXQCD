#!/bin/bash
#SBATCH -A lqftml.lq2_gpu
#SBATCH -p lq2_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 -n 1 --cpus-per-task=16
#SBATCH -t 2:00:00
#SBATCH -J pdef_phase
#SBATCH -o /lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/pdef_phase.%j.out
# PHASE-mode (true contour, mwagman construction) spread pilot: 8 configs of
# the lam=5 4^4 chain, eta=0.3/1.0, IMP_WIDTH=1.0/0.3.  Measures the
# config-to-config spread of log|w| AND arg w -- the honest ESS drivers --
# plus conditioning (iters) for the complexified operator.  IMP_SEED fixed
# per config (500+traj) across eta, as the offline grading requires.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh > /dev/null 2>&1
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5
TRAJS=$(ls "$CFG" | grep -oP 'ckpoint_lat\.\K[0-9]+' | sort -n)
for W in 1.0 0.3; do
  for ETA in 0.3 1.0; do
    for t in $TRAJS; do
      srun --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
        env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=$t LAMBDA_DTXQCD=5 \
            PDEF_MODE=phase IMP_WIDTH=$W IMP_SEED=$((500 + t)) \
            ETA=$ETA PROFILE=tent N_GL=4 N_NOISE=16 NOISE_SEED=$((300 + t)) \
            DERIVCHECK=$([ "$t" = "50" ] && echo 1 || echo 0) \
            OMP_NUM_THREADS=16 \
        $P/Test_dtxqcd_pdeform_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
        --shm 512 --shm-mpi 1 2>&1 \
        | grep -aE "PDEF-RESULT|PDEF-DERIVCHECK|MISMATCH|NOT CONV" \
        | sed "s/^/TRAJ=$t ETA=$ETA W=$W /"
    done
  done
done
