#!/bin/bash
#SBATCH --job-name=rwf_reuse
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_reuse.%j.out
# ============================================================================
# Shared-nullspace A/B: TSRC-ratio mini-run (eta=0.4, M=4 s-nodes, N=4)
# with DTXQCD_MG_REUSE_NV=0 (fresh basis per node) vs =1 (basis built
# once at the first node, reused across the rest; probe-gated with
# full-regen fallback at contraction > 0.3).  Read per node:
# [deform-mg] null-vector timing + V-cycle contraction.  PASS = reused
# contractions stay ~0.06-class and node estimates agree within errors;
# then the battery keeps only coarsen+LU (~90 s) per node.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_reuse.DONE
GPU=0
for ARM in fresh reuse; do
  case $ARM in
    fresh) EX="DTXQCD_MG_REUSE_NV=0" ;;
    reuse) EX="DTXQCD_MG_REUSE_NV=1" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.4 N_NOISE=4 NOISE_SEED=105 \
        PROFILE=cos PROF_T0=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_TSRC_RATIO=4 \
        DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0 $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_reuse_$ARM.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for ARM in fresh reuse; do
  echo "== arm $ARM =="
  grep -aE "null vectors in|contraction|TSRCNODE|REUSING|FAILED" $LOGD/rwf_reuse_$ARM.log
done
touch $LOGD/rwf_reuse.DONE
