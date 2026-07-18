#!/bin/bash
#SBATCH --job-name=rwf_smoke02
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_smoke02.%j.out
# ============================================================================
# eta=0.2 PRODUCTION BATTERY SMOKE (the first real reweighting pilot):
# 4 configs (traj 300/400/500/600), one per GPU, each running the full
# locked stack -- comb-4 + Neyman + CG_TOL=1e-4 + MG-lite L1 + REUSE_NV
# (all defaults except the single-GPU MG geometry + ANTI=0):
#   1) ANCHOR: DlogPf over [0, 0.2], N_GL=2, N_NOISE=625 samples x 4
#      comb solves = 5,000 solves  ->  sigma_anchor ~ 1.0 predicted
#   2) TSRC-RATIO: all 48 offsets, M=12 s-nodes x N_NOISE=500 x 1 =
#      6,000 solves  ->  worst-offset sigma_D ~ 3 predicted
# Deliverable: 4 configs' complete per-offset reweighting factors +
# the measured wallclock + sigma table that calibrates the full-ensemble
# battery pricing.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_smoke02.DONE
GPU=0
for TRAJ in 300 400 500 600; do
  (
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=$TRAJ \
          LAMBDA_DTXQCD=5 \
          CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
          ETA=0.2 N_GL=2 N_NOISE=625 NOISE_SEED=$((200 + TRAJ)) \
          PROFILE=cos PROF_T0=0 DTXQCD_STOCHPF_ANTI=0 \
          DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
          DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
      $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 > $LOGD/rwf_smoke02_tr${TRAJ}_anchor.log 2>&1
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=$TRAJ \
          LAMBDA_DTXQCD=5 \
          CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
          ETA=0.2 N_NOISE=500 NOISE_SEED=$((300 + TRAJ)) \
          PROFILE=cos PROF_T0=0 \
          DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
          DTXQCD_STOCHPF_TSRC_RATIO=12 \
          DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 DTXQCD_MG_REUSE_NV=1 \
      $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 > $LOGD/rwf_smoke02_tr${TRAJ}_tsrc.log 2>&1
  ) &
  GPU=$((GPU+1))
done
wait
for TRAJ in 300 400 500 600; do
  echo "== traj $TRAJ =="
  grep -aE "STOCHPF" $LOGD/rwf_smoke02_tr${TRAJ}_anchor.log | tail -1
  grep -aE "TSRCCLOSURE" $LOGD/rwf_smoke02_tr${TRAJ}_tsrc.log | tail -1
done
touch $LOGD/rwf_smoke02.DONE
