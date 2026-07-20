#!/bin/bash
#SBATCH --job-name=rwf_rhot0
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_rhot0.%j.out
# ============================================================================
# Difference-telescope pricing: corr(est(t0=0), est(t0=1)) with COMMON
# noise (16^3x48 lam5 traj300 cos, undeflated Schur estimator).  If rho
# is high, DlogPf offsets can be built as one anchor + 47 cheap
# common-noise differences (Var_diff ~ 2(1-rho) Var) instead of 48 full
# estimates -- the NTSRC=48 reweighting unlock candidate.  Two node
# strengths (production-relevant): ep=0.16 (eta=0.2 top node / eta=0.4
# lower-mid) and ep=0.47 (eta=0.6 top node).  N=16 pairs = 32 solves/arm.
# Offline: correlate RHOPAIR A/B columns.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_rhot0_${RPROF:-cos}.DONE
RPROF=${RPROF:-cos}
GPU=0
for EP in ${EPS:-0.16 0.47}; do
  ETA2=$(python3 -c "print(2*$EP)")
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=$ETA2 N_GL=1 N_NOISE=16 NOISE_SEED=${RSEED:-95} \
        PROFILE=$RPROF PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=0 \
        DTXQCD_STOCHPF_RHOT0=1 \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_rho_${RPROF}_ep$EP.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for EP in ${EPS:-0.16 0.47}; do
  echo "== ep=$EP =="
  grep -ac "RHOPAIR" $LOGD/rwf_rho_${RPROF}_ep$EP.log
done
touch $LOGD/rwf_rhot0_${RPROF}.DONE
