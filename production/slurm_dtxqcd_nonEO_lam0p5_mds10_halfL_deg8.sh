#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p5
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p5.%j.out
# lambda=0.5 NON-EO weak-field, MDS=10 + HALF trajL (sqrt2/8) -> eps=0.0177, + 8
# rational poles.  lambda=0.5 is integrator-SOFT (huge aux regularize, lambda_min
# =1.34, Fdt~0.2-0.4 at MDS=20) -> coarser eps + fewer poles OK -> cheapest viable.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.5 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.176776695296637 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds10_halfL_deg8" \
TRAJ=105 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p5 nonEO MDS=10 halfL deg8 exited rc=$? $(date) ==="
