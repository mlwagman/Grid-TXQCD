#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p1
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p1.%j.out
# lambda=0.1 NON-EO weak-field, MDS=10 + HALF trajL (sqrt2/8) -> eps=0.0177, + 8
# rational poles.  lambda=0.1 is the MOST regularized (lambda_min=31, huge aux
# <s>*=90) -> softest forces -> coarse eps + few poles should mix; CG cost is the
# only concern (kappa~1700).  Core-invariant point (the old broken-init drifted).
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.1 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.176776695296637 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds10_halfL_deg8" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p1 nonEO MDS=10 halfL deg8 exited rc=$? $(date) ==="
