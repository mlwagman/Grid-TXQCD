#!/bin/bash
#SBATCH --job-name=dtx_nEO_l1_h
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l1_h.%j.out
# lambda=1 NON-EO weak-field, MDS=20 + HALF trajL (sqrt2/8) -> eps=0.0088.
# Replaces the chroma-fork lambda=1 (which mismatched); weak-field co-thermalizes.
# lambda=1 still near the stiffness peak (large aux <s>*=6.4) -> finest eps.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=1.0 USE_FULL_PF=1 MDSTEPS=20 TRAJL=0.176776695296637 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds20_halfL" \
TRAJ=105 NO_METROP=5 N_SKIP=2 DIAG_TRMINV_INTERVAL=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam1 nonEO MDS=20 halfL exited rc=$? $(date) ==="
