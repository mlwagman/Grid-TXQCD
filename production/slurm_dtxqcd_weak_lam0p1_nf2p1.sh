#!/bin/bash
#SBATCH --job-name=dtx_w_l01_21
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/dtx_w_l01_21.%j.out
# ============================================================================
# DTXQCD Nf=2+1 WEAK-FIELD thermalization at lambda=0.1 -- totally analogous to
# the lambda=6 run (slurm_dtxqcd_weak_lam6_nf2p1.sh), the HARD small-lambda
# point.  Fast (MP) spectator strange.  Same canonical point: 16^3x48,
# m=-0.245, csw=1.249, beta=6.1, stout, MN2 MDS=10 trajL=sqrt2/4, weak-field
# wf=0.1.  At lambda=0.1 the aux fluctuations are large and the saddle init
# falls back to bisection; watch dH / Fdt for integrator stability.  By Fierz
# the equilibrium plaq is STILL ~0.5143 (lambda-independent).
# Output -> cfgs/dtxqcd_lam0.1000_weak_nf2p1/
#   submit:  sbatch production/slurm_dtxqcd_weak_lam0p1_nf2p1.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

echo "=== DTXQCD Nf=2+1 weak-field lambda=0.1 thermalization  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDA_DTXQCD=0.1 \
ADD_STRANGE=1 \
MASS_STRANGE=-0.245 \
DTXQCD_SUFFIX="_weak_nf2p1" \
TRAJ=200 \
NO_METROP=100 \
N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== DTXQCD Nf=2+1 weak-field lambda=0.1 exited rc=$? $(date) ==="
