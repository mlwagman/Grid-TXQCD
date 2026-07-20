#!/bin/bash
#SBATCH --job-name=rwf_tsrc_tent
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tsrc_tent.%j.out
# ============================================================================
# TENT TSRC-coverage smoke via the TELESCOPE CHAIN (large-rho design):
# per config (traj 300/400/500/600, one per GPU), eta=0.2 target,
# N_GL=2 proper nodes, all 48 offset steps x N_NOISE=6 common-noise
# pairs = 2x48x6x2 = 1,152 solves (~4-6 h at the eta=0.2 rates).
# Offline (RHOSTEP lines): per-step D(t0) -> cumulative per-offset
# ratios + periodic bridge; per-step sigma_diff x chain arithmetic =
# the measured tent offset-coverage cost curve.  NOTE the telescope is
# only break-even vs brute force below rho ~ 0.99 -- this smoke
# MEASURES the actual economics alongside delivering the pilot
# configs' offset structure.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tsrc_tent.DONE
GPU=0
for TRAJ in 300 400 500 600; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=$TRAJ \
        LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.2 N_GL=2 N_NOISE=6 NOISE_SEED=$((1400 + TRAJ)) \
        PROFILE=tent PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_STOCHPF_RHOCHAIN=1 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tsrct_tr${TRAJ}.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for TRAJ in 300 400 500 600; do
  echo "== traj $TRAJ =="
  grep -ac "RHOSTEP" $LOGD/rwf_tsrct_tr${TRAJ}.log
done
touch $LOGD/rwf_tsrc_tent.DONE
