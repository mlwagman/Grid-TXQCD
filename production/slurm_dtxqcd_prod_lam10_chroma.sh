#!/bin/bash
#SBATCH --job-name=dtx_p_l10c
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_p_l10c.%j.out
# ============================================================================
# DTXQCD PRODUCTION stream: lambda=10, forked from a thermalized chroma
# cl3_16_48_b6p1 config.  Canonical 16^3x48 point (MN2 MDS=10 trajL=sqrt2/4,
# csw=1.249 + stout, AUX_MULT=4, all GPU/cuBLAS + RAT_AUTO_HI autoscale on).
#
# Run control left at the driver defaults: NO_METROP unset => the driver uses
# the n_therm-based absolute warmup (force-accept until traj 100, then
# Metropolis), which is resume-safe (no re-warmup on restart).  TRAJ=2000 target.
# Resume scan runs first, so IMPORT_CFG only forks on the very first launch;
# subsequent (re)submits continue from the latest checkpoint.
# Output -> cfgs/dtxqcd_lam10.0000_fromchroma/
#   submit:  sbatch production/slurm_dtxqcd_prod_lam10_chroma.sh
# (re)submit the same script to continue; or use --dependency=afterany:<jobid>.
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

CHROMA_CFG="${CHROMA_CFG:-/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime}"

echo "=== DTXQCD production lam10 (chroma fork)  $(date) ==="
echo "fork source (first launch only): $CHROMA_CFG"
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDA_DTXQCD=10.0 \
IMPORT_CFG="$CHROMA_CFG" \
DTXQCD_SUFFIX="_fromchroma" \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== production lam10 chroma exited rc=$? $(date) ==="
