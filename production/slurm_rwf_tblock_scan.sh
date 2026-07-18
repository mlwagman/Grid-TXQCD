#!/bin/bash
#SBATCH --job-name=rwf_tblock
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/rwf_tblock.%j.out
# ============================================================================
# Block-t dilution scan at production volume (16^3x48 lam5 traj300, cos
# t0=0, ep=0.4).  The 4^4 t-slice win (x1.78 equal-budget) came from
# T ~ xi(S~^-1); at T=48 the candidate is contiguous blocks of ~xi
# slices.  The scan over B doubles as the integrated Dt-decay
# measurement of A = S~^-1 dS~.  EQUAL-BUDGET design: ~48 solves/arm,
# ANTI=0 (the pair buys ~5% in the Schur form), DEFLATE=16 everywhere
# (cached basis: free; keeps the production-intended exact term).
#   S0n: no dilution,  N_NOISE=96    (96 solves; the no-anti baseline)
#   B24: B=24 (D=2),   N_NOISE=48    (96)
#   B12: B=12 (D=4),   N_NOISE=24    (96)
#   B6 : B=6  (D=8),   N_NOISE=12    (96)
# Read: all arms cost 96 solves; equal-budget comparison is directly the
# GLNODE errors (same total budget), per-sample sigma = err*sqrt(N).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/rwf_tblock.DONE
GPU=0
for ARM in S0n B24 B12 B6; do
  case $ARM in
    S0n) EX="N_NOISE=96" ;;
    B24) EX="N_NOISE=48 DTXQCD_STOCHPF_DILUTE_TBLOCK=24" ;;
    B12) EX="N_NOISE=24 DTXQCD_STOCHPF_DILUTE_TBLOCK=12" ;;
    B6)  EX="N_NOISE=12 DTXQCD_STOCHPF_DILUTE_TBLOCK=6" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 \
        CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        ETA=0.8 N_GL=1 NOISE_SEED=77 \
        PROFILE=cos PROF_T0=0 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1 \
        DTXQCD_STOCHPF_SCHUR=1 DTXQCD_STOCHPF_DEFLATE=16 \
        DTXQCD_STOCHPF_ANTI=0 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/rwf_tb_$ARM.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for ARM in S0n B24 B12 B6; do
  echo "== arm $ARM =="
  grep -aE "GLNODE" $LOGD/rwf_tb_$ARM.log | tail -1
done
touch $LOGD/rwf_tblock.DONE
