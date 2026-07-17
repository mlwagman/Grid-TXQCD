#!/bin/bash
#SBATCH --job-name=rwf_defl_ab
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_defl_ab.%j.out
# ============================================================================
# THE DECISIVE production-volume variance A/B for the reweighting blocker
# (handoff 5c): 16^3x48 lambda=5 traj 300, cos t0=0, single node ep=0.4
# (ETA=0.8 N_GL=1), N_NOISE=12 per arm.  One arm per GPU:
#   S0:  SCHUR=1, no deflation            (production baseline of the new form)
#   K16: SCHUR=1 DEFLATE=16               (oblique projector; band exists HERE)
#   K48: SCHUR=1 DEFLATE=48
#   T16: SCHUR=1 DEFLATE=16 DILUTE_EOT=1  (the full stack)
# Read: GLNODE errors x sqrt(12) = per-sample sigma.  The census says
# ep=0.4 has a formed band (ev[0]=8.9e-5, ~2-20 modes below 2e-4): if the
# deep modes carry the variance, K16/K48 collapse sigma; if the bulk
# floor (~330 by sqrt(V) scaling) dominates, they do not -- either way
# this number decides the reweighting strategy.
# Eigensolve: production window defaults (LO=3e-4), CACHE dir reused.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
mkdir -p $P/defl_cache
rm -f $LOGD/rwf_defl_ab.DONE
GPU=0
for ARM in S0 K16 K48 T16; do
  case $ARM in
    S0)  EX="DTXQCD_STOCHPF_SCHUR=1" ;;
    K16) EX="DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16" ;;
    K48) EX="DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=48" ;;
    T16) EX="DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16 DTXQCD_STOCHPF_DILUTE_EOT=1" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.8 N_GL=1 N_NOISE=12 NOISE_SEED=77 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_ab_$ARM.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for ARM in S0 K16 K48 T16; do
  echo "== arm $ARM =="
  grep -aE "stochpf-defl|GLNODE" $LOGD/rwf_ab_$ARM.log | tail -2
done
touch $LOGD/rwf_defl_ab.DONE
