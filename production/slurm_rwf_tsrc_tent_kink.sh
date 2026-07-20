#!/bin/bash
#SBATCH --job-name=rwf_tsrc_kink
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrc_kink.%j.out
# ============================================================================
# TENT TSRC coverage via the KINK-AWARE LINE INTEGRAL -- head-to-head
# partner of the telescope-chain smoke (1303335) at EQUAL budget:
# 4 pilot configs, eta=0.2, 2T=96 s-nodes (2-pt GL per unit interval,
# never straddling the integer-s kink hops) x N_NOISE=12 = 1,152
# solves/config.  The ds insertion is supported on the TWO kink links
# with O(eta) coefficients (square-pulse dpp = +-2) -- the tent analog
# of the cos mode's parametric variance suppression.  Gates in-log:
# TSRC-DERIVCHECK (FD in s at s0=0.37T, valid off-integer) +
# TSRCCLOSURE sum rule.  Offline: cumulative sums at integer s =
# per-offset ratios; compare per-offset sigma vs the telescope arms.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tsrc_kink.DONE
GPU=0
for TRAJ in 300 400 500 600; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=$TRAJ \
        LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.2 N_NOISE=12 NOISE_SEED=$((1500 + TRAJ)) \
        PROFILE=tent PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_STOCHPF_TSRC_RATIO=96 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrck_tr${TRAJ}.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for TRAJ in 300 400 500 600; do
  echo "== traj $TRAJ =="
  grep -aE "TSRC-DERIVCHECK|TSRCCLOSURE" $LOGD/rwf_tsrck_tr${TRAJ}.log
done
touch $LOGD/rwf_tsrc_kink.DONE
