#!/bin/bash
#SBATCH --job-name=rwf_fmap
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_fmap.%j.out
# ============================================================================
# F(t,t') MAP at production volume (16^3x48 lam5 traj300 cos t0=0 ep=0.4):
# the definitive t-dilution scheme designer.  Per probe pair: T=48 masked
# solves; N_NOISE=2 pairs per GPU x 2 GPUs (different seeds) = 4 maps,
# 192 solves total.  Offline (analyze_fmap.py): F(t,t') = <|f|^2>/4;
# every scheme's equal-budget sigma is D * partial sums of F.  Resolves
# the C3-arm tension (measured x1.5 win vs suspicious means) with the
# exact object instead of more scheme arms.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_fmap.DONE
GPU=0
for SEED in 91 92; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.8 N_GL=1 N_NOISE=2 NOISE_SEED=$SEED \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16 \
        DTXQCD_STOCHPF_TSLICE=2 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_fmap_s$SEED.log 2>&1 &
  GPU=$((GPU+1))
done
wait
echo "== fmap lines =="
grep -ac "FMAP" $LOGD/rwf_fmap_s91.log $LOGD/rwf_fmap_s92.log
touch $LOGD/rwf_fmap.DONE
