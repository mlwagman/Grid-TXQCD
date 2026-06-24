#!/bin/bash
#SBATCH --job-name=dtx_cf_l10_21
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/dtx_cf_l10_21.%j.out
# ============================================================================
# DTXQCD Nf=2+1 CHROMA-FORK consistency check at lambda=10 -- the FASTEST test
# of the missing-strange hypothesis.  Forks the thermalized chroma cl3_16_48_b6p1
# config (QCD Nf=2+1 plaq = 0.5133) and, now WITH the spectator strange
# (ADD_STRANGE=1, fast MP), runs Metropolis.  Expectation: plaq HOLDS ~0.5133
# (no descent to 0.43, which the Nf=2-only run did from the missing strange).
# If it holds, DTXQCD Nf=2+1 == QCD Nf=2+1 is confirmed directly.
#
# NO_METROP=1 force-accepts only traj 0 (absorb the aux saddle -> equilibrium
# transient on the freshly-imported gauge); the rest run real Metropolis.
# Driver count: traj_to_run = TRAJ - NO_METROP - start_traj.
# 2026-06-24: relaunched after the AUX_FLUCT_LAMBDA-defaults-to-lambda fix.  For
# lambda=10 the fix is a NO-OP (the old hardwire was 10 == lambda), so this is
# the unchanged known-good CONTROL of the fixed-init overnight set.  Fresh _fi
# suffix keeps it collision-free with the older pending _gfix job.
# Output -> cfgs/dtxqcd_lam10.0000_fromchroma_nf2p1_fi/  (fresh suffix =>
# IMPORT_CFG forks; does NOT collide with the old _fromchroma dirs).
#   submit:  sbatch production/slurm_dtxqcd_chromafork_lam10_nf2p1.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

CHROMA_CFG="${CHROMA_CFG:-/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime}"

echo "=== DTXQCD Nf=2+1 chroma-fork lambda=10 consistency check  $(date) ==="
echo "fork source: $CHROMA_CFG"
nvidia-smi --query-gpu=index,name --format=csv,noheader

IMPORT_CFG="$CHROMA_CFG" \
LAMBDA_DTXQCD=10.0 \
ADD_STRANGE=1 \
MASS_STRANGE=-0.245 \
DTXQCD_SUFFIX="_fromchroma_nf2p1_fi" \
TRAJ=41 \
NO_METROP=10 \
N_SKIP=5 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== DTXQCD Nf=2+1 chroma-fork lambda=10 exited rc=$? $(date) ==="
