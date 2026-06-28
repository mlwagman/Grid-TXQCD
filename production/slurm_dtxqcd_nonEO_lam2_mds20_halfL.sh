#!/bin/bash
#SBATCH --job-name=dtx_nEO_l2_h
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l2_h.%j.out
# lambda=2 NON-EO weak-field, MDS=20 + HALF trajL (sqrt2/8) -> eps=0.0088 (= MDS=40
# at full trajL, but half the steps/traj).  lambda=2 is the stiffness peak
# (smallest lambda_min ~0.156); finest eps of the set via the trajL lever.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=2.0 USE_FULL_PF=1 MDSTEPS=20 TRAJL=0.176776695296637 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds20_halfL" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam2 nonEO MDS=20 halfL exited rc=$? $(date) ==="
