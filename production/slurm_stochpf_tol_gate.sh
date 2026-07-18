#!/bin/bash
#SBATCH --job-name=stochpf_tol
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/stochpf_tol.%j.out
# ============================================================================
# Sampling-solve TOLERANCE ladder gate (4^4, vs the dense exact truth
# -0.0356616 - 0.110153 i at ep=0.4): CG_TOL 1e-8 / 1e-6 / 1e-5 / 1e-4,
# same seeds, N_NOISE=64, Schur estimator.  Solve error enters a
# Hutchinson sample as <w, eps_x> (quasi-random vs w => noise not bias);
# PASS = all means consistent with truth and with each other => adopt
# loose tol for sampling solves (~2x fewer iters at every node).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/stochpf_tol.DONE
for TOL in 1e-8 1e-6 1e-5 1e-4; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=4.4.4.4 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5 \
        ETA=0.8 N_GL=1 N_NOISE=64 NOISE_SEED=77 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 CG_TOL=$TOL \
    $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/stochpf_tol_$TOL.log 2>&1
  echo "== CG_TOL=$TOL =="
  grep -aE "GLNODE" $LOGD/stochpf_tol_$TOL.log | tail -1
done
touch $LOGD/stochpf_tol.DONE
