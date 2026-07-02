#!/bin/bash
#SBATCH --job-name=dtx_eo_cw
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/slurm-logs/dtx_eo_cw.%j.out
# Clean-sign-window EO production streams (2026-07-02): fromchroma fork at
# lambda in {4,5,6} — inside the measured sign-clean window (aux-prior Pf
# sign 12/12 clean at lambda>=4 on tepid+weak gauge; boundary at 3-4).
# EO = RationalEO + LogDet; MDSTEPS=20 (lambda=6 EO froze at MDS=10).
# Sign certification: per-traj g5M evals (|lambda|_min gap) is ON by default;
# a gapped series with no zero crossing certifies sign constancy.
# Parametrized via --export: LAM (4.0|5.0|6.0), TAG, MDS (default 20).
# MDS=20 is the weak-field-freeze-safe setting; MDS=10 variants test whether
# the fromchroma (equilibrium) start avoids the freeze entirely (the original
# lambda=6 EO MDS=10 freeze was observed from a WEAK-FIELD start).
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs
nvidia-smi --query-gpu=index,name --format=csv,noheader

CHROMA=/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime
MDS="${MDS:-20}"

LAMBDA_DTXQCD=$LAM MDSTEPS=$MDS TRAJL=0.353553390593274 \
ADD_STRANGE=1 MASS_STRANGE=-0.245 \
IMPORT_CFG=$CHROMA DTXQCD_SUFFIX="_eo_fromchroma_cw_mds${MDS}" \
TRAJ=500 NO_METROP=5 N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh
echo "=== eo clean-window lam=$LAM mds=$MDS exited rc=$? $(date) ==="
