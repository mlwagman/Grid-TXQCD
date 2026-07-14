#!/bin/bash
#SBATCH --job-name=qdef_rate
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_rate.%j.out
# ============================================================================
# CONVERGENCE-RATE probe of the lambda x eta interplay: run the (7x) deformed
# CGNR itself at each (lambda, eta) with a capped iteration budget and read
# kappa_eff from (iters, relres) — the production-relevant hardness number,
# no eigensolver tuning.  CG asymptotic rate: relres ~ 2 exp(-2k/sqrt(kappa))
# => sqrt(kappa_eff) ~ 2k / ln(2/relres).
# One lambda per GPU (tla streams, traj 300), eta ladder sequential.
# _Exit(3) on capped solves is EXPECTED (rate data is in the log).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"

run_ladder () {  # $1=lambda $2=gpu
  local lam=$1 gpu=$2
  local cfg=$P/cfgs/dtxqcd_lam${lam}.0000_tla_fromchroma_cw
  mkdir -p $P/meas_deform_rate_l${lam}
  for eta in 0.0 0.2 0.4 0.6 0.8; do
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$gpu LATT=16.16.16.48 TRAJ=300 ETA=$eta \
          PROFILE=tent MEAS_BARYON_NTSRC=1 CG_MAX=20000 CG_TOL=1e-7 \
          CFG_DIR=$cfg DATA_DIR=$P/meas_deform_rate_l${lam} \
          DTXQCD_DEFORM_QUDA_CHECK=0 \
      $P/meas_baryon_deform_dtxqcd 300 $GRID \
      > $LOGD/rate_l${lam}_e${eta}.log 2>&1
    echo "lam=$lam eta=$eta rc=$? $(grep -a 'deform-CG' $LOGD/rate_l${lam}_e${eta}.log | head -1)"
  done
}

# lambda=5/6 first (fast, converging), lambda=3 in parallel on its own GPU.
run_ladder 5 0 > $LOGD/rate_ladder_l5.txt 2>&1 &
run_ladder 6 1 > $LOGD/rate_ladder_l6.txt 2>&1 &
run_ladder 3 2 > $LOGD/rate_ladder_l3.txt 2>&1 &
wait

echo "=== RATE TABLE ==="
for l in 5 6 3; do cat $LOGD/rate_ladder_l$l.txt; done
touch $LOGD/qdef_rate.DONE
