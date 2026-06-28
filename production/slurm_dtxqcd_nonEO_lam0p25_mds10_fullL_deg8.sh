#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p25f
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p25f.%j.out
# lambda=0.25 NON-EO weak-field, MDS=10 + FULL trajL (sqrt2/4) -> eps=0.0354, + 8
# rational poles.  FRESH fullL chain (separate _fullL_deg8 dir).  NOTE: fullL is
# PROVEN at lambda=0.1 but UNTESTED at 0.25, which is stiffer (closer to the
# lambda~1-2 peak; its halfL dH was ~0.2 vs 0.1's ~0.02).  WATCH the first few
# trajs' dH / Fdt max: if dH blows up, fall back to halfL (sqrt2/8) or MDS=20.
# The halfL 0.25 stream is kept running until this is confirmed clean, then
# cancel it.  Bridges lambda=0.1 (lambda_min~31) and lambda=0.5 (lambda_min~1.34).
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.25 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.353553390593274 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds10_fullL_deg8" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p25 nonEO MDS=10 fullL deg8 exited rc=$? $(date) ==="
