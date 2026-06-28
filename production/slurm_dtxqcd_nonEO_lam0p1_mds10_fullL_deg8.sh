#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p1f
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p1f.%j.out
# lambda=0.1 NON-EO weak-field, MDS=10 + FULL trajL (sqrt2/4) -> eps=0.0354, + 8
# rational poles.  FRESH fullL chain (separate _fullL_deg8 dir) replacing the
# halfL stream.  Full trajL was already tested at lambda=0.1 (job 1289222) with
# dH identical to halfL (~-0.2 thermalization transient -> <0.02): the error is
# transient-dominated, not truncation, so doubling eps barely moves dH.  Same
# MDS=10 cost/traj but 2x the MD time -> ~2x decorrelation for free.  Physics
# agrees: soft forces (M^dagM lambda_min~31 -> big eps headroom) AND a slow aux
# Gaussian (omega_aux ~ lambda -> U-turn-optimal trajL ~ 1/lambda is LONG) both
# favor the longest trajL at small lambda.  CG cost the only concern (kappa~1700).
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.1 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.353553390593274 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds10_fullL_deg8" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p1 nonEO MDS=10 fullL deg8 exited rc=$? $(date) ==="
