#!/bin/bash
#SBATCH --job-name=dtx_neo_cw
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_neo_cw.%j.out
# Non-EO clean-window streams (2026-07-02): empirical EO-vs-non-EO test at
# lambda=3,4,5,6.  Motivation: the lambda=4 EO stream is unstable at MDS=10
# (LogDet Fdt max ~59 — the small-lambda EO-Schur pathology, M_ee eigenvalue
# crossings near the sign wall); non-EO has no M_ee split so it should be
# immune, and we want the wall-clock comparison before committing the window
# to either factorization.  Settings match the EO clean-window streams 1:1
# (fromchroma, MDS=10, TRAJL=sqrt(2)/4, Nf=2+1, stout) except USE_FULL_PF=1
# swaps EO Schur 1/4-root + LogDet for the single non-EO 1/4-root, solved on
# the device via non-EO Style C multishift.
# Parametrized via --export: LAM (3.0|4.0|5.0|6.0).
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader

CHROMA=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime

LAMBDA_DTXQCD=$LAM USE_FULL_PF=1 MDSTEPS=10 TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 \
IMPORT_CFG=$CHROMA DTXQCD_SUFFIX="_noneo_fromchroma_cw_mds10" \
DTXQCD_MULTISHIFT_QUDA=1 \
CG_TOL=1e-8 \
TRAJ=500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== non-EO clean-window lam=$LAM exited rc=$? $(date) ==="
