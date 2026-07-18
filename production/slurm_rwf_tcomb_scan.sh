#!/bin/bash
#SBATCH --job-name=rwf_tcomb
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tcomb.%j.out
# ============================================================================
# Companion to the block-t scan: stride COMBS + spatial control + the
# slice-covariance INSTRUMENT (16^3x48 lam5 traj300, cos t0=0, ep=0.4,
# ANTI=0, DEFLATE=16 cached, ~96 solves/arm equal budget):
#   C3:  t mod 3 comb (D=3), N_NOISE=32   -- kills |Dt|<=2 cross terms
#   C4:  t mod 4 comb (D=4), N_NOISE=24   -- kills |Dt|<=3
#   Z2:  z-parity (D=2),     N_NOISE=48   -- spatial control (predict: neutral)
#   S0i: undiluted + TSLICE, N_NOISE=96   -- logs per-t partials s_t; the
#        offline Cov(s_t,s_t') predicts EVERY t-scheme's sigma at once
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tcomb.DONE
GPU=0
for ARM in C3 C4 Z2 S0i; do
  case $ARM in
    C3)  EX="N_NOISE=32 DTXQCD_STOCHPF_DILUTE_TSTRIDE=3" ;;
    C4)  EX="N_NOISE=24 DTXQCD_STOCHPF_DILUTE_TSTRIDE=4" ;;
    Z2)  EX="N_NOISE=48 DTXQCD_STOCHPF_DILUTE_ZPAR=1" ;;
    S0i) EX="N_NOISE=96 DTXQCD_STOCHPF_TSLICE=1 DTXQCD_STOCHPF_SCHUR=1" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.8 N_GL=1 NOISE_SEED=78 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_DEFLATE=16 DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tc_$ARM.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for ARM in C3 C4 Z2 S0i; do
  echo "== arm $ARM =="
  grep -aE "GLNODE" $LOGD/rwf_tc_$ARM.log | tail -1
done
touch $LOGD/rwf_tcomb.DONE
