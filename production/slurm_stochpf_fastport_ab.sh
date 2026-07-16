#!/bin/bash
#SBATCH --job-name=stochpf_ab
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/stochpf_ab.%j.out
# ============================================================================
# Fast-path A/B gate for the ported reweighting estimator (2026-07-16).
# 4^4 lam4 config, PROFILE=cos ETA=0.6, identical NOISE_SEED across arms:
#   R: DTXQCD_DEFORM_QUDA=0            (host CGNR reference)
#   F: DTXQCD_DEFORM_QUDA=1 EO=0       (device hop-corr full solve + qcheck)
#   E: DTXQCD_DEFORM_QUDA=1 EO=1       (EO Schur solve + qcheck)
# MG explicitly OFF (BLOCK doesn't fit 4^4; MG gets its own 16^3x48 gate).
# PASS = qcheck < 1e-10 in F/E and the three STOCHPF dlogdet values agree
# to solver tolerance.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam4.0000_4t4_signhop_lam4
for ARM in R F E; do
  case $ARM in
    R) EX="DTXQCD_DEFORM_QUDA=0" ;;
    F) EX="DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=0" ;;
    E) EX="DTXQCD_DEFORM_QUDA=1 DTXQCD_DEFORM_EO=1" ;;
  esac
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env LATT=4.4.4.4 CFG_DIR=$CFG TRAJ=250 LAMBDA_DTXQCD=4 \
        PROFILE=cos ETA=0.6 N_NOISE=4 N_GL=2 NOISE_SEED=77 \
        DTXQCD_DEFORM_MG=0 $EX \
    $P/Test_dtxqcd_eta_stoch_pf --grid 4.4.4.4 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $P/mtilde_runs/stochpf_ab_$ARM.log 2>&1
  echo "== arm $ARM =="
  grep -aE "stochpf-qcheck|STOCHPF" $P/mtilde_runs/stochpf_ab_$ARM.log | tail -2
done
touch $P/gridmg_setups/logs/stochpf_ab.DONE
