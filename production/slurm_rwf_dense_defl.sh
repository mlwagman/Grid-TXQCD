#!/bin/bash
#SBATCH --job-name=rwf_densedefl
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_densedefl.%j.out
# ============================================================================
# Section-5c design study: exact variance-vs-K curve for the deflated
# reweighting trace estimator, 4^4 dense (lambda=5 gate stream, traj 300,
# cos t0=0).  ETA here is the NODE strength ep = eta*x_q: {0.1,0.2,0.3,0.4}
# spans the production GL nodes of the measured eta=0.2/0.4 battery points.
# Decides the production K (or kills the idea if the dense mid-band
# flattens the curve) before any estimator wiring.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GPU=0
for E in 0.1 0.2 0.3 0.4; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=4.4.4.4 TRAJ=300 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_4t4_signhop_lam5 \
        ETA=$E PROFILE=cos PROF_T0=0 KMAX=512 MC_CHECK=32 \
        OMP_NUM_THREADS=8 \
    $P/Test_dtxqcd_rwf_dense_defl --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/densedefl_e$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for E in 0.1 0.2 0.3 0.4; do
  echo "== ep=$E =="
  grep -aE "GATE|sigma_min|dense-defl\]  " $LOGD/densedefl_e$E.log | head -22
done
touch $LOGD/rwf_densedefl.DONE
