#!/bin/bash
#SBATCH --job-name=dtx_nEO_l3_30
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l3_30.%j.out
# lambda=3 NON-EO weak-field, MDS=30 (was 20, where Fdt sat at the 0.5 border).
# Full trajL=sqrt2/4 -> eps=0.0118.  Goal: Fdt comfortably <0.5, dH O(1) -> mixes.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=3.0 USE_FULL_PF=1 MDSTEPS=30 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds30" \
TRAJ=105 NO_METROP=5 N_SKIP=2 DIAG_TRMINV_INTERVAL=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam3 nonEO MDS=30 exited rc=$? $(date) ==="
