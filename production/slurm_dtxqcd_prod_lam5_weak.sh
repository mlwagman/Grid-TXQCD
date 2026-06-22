#!/bin/bash
#SBATCH --job-name=dtx_p_l5w
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_p_l5w.%j.out
# ============================================================================
# DTXQCD PRODUCTION stream: lambda=5, weak-field cold start.  Canonical 16^3x48
# point (MN2 MDS=10 trajL=sqrt2/4, csw=1.249 + stout, AUX_MULT=4, all
# GPU/cuBLAS + RAT_AUTO_HI autoscale on).  AUX_INIT_AUTO seeds the saddle;
# the n_therm=100 warmup (NO_METROP unset => driver default) thermalizes the
# gauge from weak field before Metropolis kicks in.  Resume-safe.  TRAJ=2000.
# Output -> cfgs/dtxqcd_lam5.0000_weakfield/
#   submit:  sbatch production/slurm_dtxqcd_prod_lam5_weak.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

echo "=== DTXQCD production lam5 (weak-field)  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDA_DTXQCD=5.0 \
DTXQCD_SUFFIX="_weakfield" \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== production lam5 weak exited rc=$? $(date) ==="
