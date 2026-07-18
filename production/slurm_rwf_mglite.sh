#!/bin/bash
#SBATCH --job-name=rwf_mglite
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_mglite.%j.out
# ============================================================================
# SINGLE-GPU MG for the reweighting solves at the top production node
# (ep=0.47, SUBCRITICAL -- the exact-LU/two-GPU design was forced by the
# supercritical band, which does not exist here).  Three 1-GPU arms,
# N_NOISE=8 Schur samples each, wallclock/solve is the metric:
#   CG :  plain CG baseline (no MG)
#   L1 :  MG, BLOCK=4.4.4.4 (N=12,288 -> 2.4 GB LU) on the SAME GPU
#         (DTXQCD_MG_LUDEV=0)
#   GM :  MG, GMRES coarse (no LU at all, DTXQCD_MG_COARSE=gmres)
# Gates: contraction probe (<1) + consistent GLNODE means.  A win here =
# MG on the fleet WITHOUT the GPU-pairs constraint (2x throughput).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_mglite.DONE
GPU=0
for ARM in CG L1 GM; do
  case $ARM in
    CG) EX="DTXQCD_DEFORM_MG=0" ;;
    L1) EX="DTXQCD_DEFORM_MG=1 DTXQCD_DEFORM_MG_ETAMIN=0.3 DTXQCD_MG_BLOCK=4.4.4.4 DTXQCD_MG_LUDEV=0" ;;
    GM) EX="DTXQCD_DEFORM_MG=1 DTXQCD_DEFORM_MG_ETAMIN=0.3 DTXQCD_MG_COARSE=gmres" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.94 N_GL=1 N_NOISE=8 NOISE_SEED=97 \
        PROFILE=cos PROF_T0=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=0 $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_mgl_$ARM.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for ARM in CG L1 GM; do
  echo "== arm $ARM =="
  grep -aE "contraction|GLNODE" $LOGD/rwf_mgl_$ARM.log | tail -2
done
touch $LOGD/rwf_mglite.DONE
