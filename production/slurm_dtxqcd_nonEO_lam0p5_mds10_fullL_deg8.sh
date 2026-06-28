#!/bin/bash
#SBATCH --job-name=dtx_nEO_l0p5f
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l0p5f.%j.out
# !!! SUPERSEDED / FAILED -- DO NOT RESUBMIT.  This fullL MDS=10 (eps=0.0354) blew
# up at lambda=0.5: first traj dH=11.4 (too stiff for the doubled eps; dH~eps^4).
# Replaced by slurm_dtxqcd_nonEO_lam0p5_mds20_fullL.sh (fullL MDS=20, eps-matched
# 0.0177).  Kept only as the record of where the doubled-eps fullL breaks. !!!
#
# lambda=0.5 NON-EO weak-field, MDS=10 + FULL trajL (sqrt2/4) -> eps=0.0354, + 8
# rational poles.  FRESH fullL chain (separate _fullL_deg8 dir).  STIFFEST of the
# small-lambda fullL tests: lambda=0.5 is closer to the lambda~1-2 stiffness peak
# (lambda_min~1.34), and its halfL dH was already ~0.45-0.59 (vs 0.25's ~0.2,
# 0.1's ~0.02).  Doubling eps to fullL roughly quadruples-and-up the truncation
# error, so this is where fullL is most likely to break.  WATCH the first few
# trajs' dH / Fdt max closely: if dH >> 1 or Fdt > 0.5, fall back to halfL
# (sqrt2/8) or MDS=20.  The halfL 0.5 stream stays running as the control until
# this is confirmed clean, then cancel it.
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
LAMBDA_DTXQCD=0.5 USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.353553390593274 RHMC_DEG=8 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 DTXQCD_SUFFIX="_nonEO_mds10_fullL_deg8" \
TRAJ=1500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== lam0p5 nonEO MDS=10 fullL deg8 exited rc=$? $(date) ==="
