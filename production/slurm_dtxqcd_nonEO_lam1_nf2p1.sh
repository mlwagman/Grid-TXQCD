#!/bin/bash
#SBATCH --job-name=dtx_nEO_l1_21
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/dtx_nEO_l1_21.%j.out
# ============================================================================
# DTXQCD Nf=2+1 NON-EO (USE_FULL_PF=1) chroma-fork at lambda=1.  Tests whether
# the FULL doubled-M48 rational action (a single x^-1/4 on M48^dag M48, LogDet
# folded in -- NO M_ee/Schur split, NO M_ee^-1) avoids the EO "cliff" that makes
# det(M_ee) near-singular at 0.5<=lambda<=2 (the EO chroma-fork at lambda=1 gave
# dH=42, Fdt->9.3).  The full M48^dag M48 is well-gapped at lambda=1 (lambda_min
# ~= 13.3), so the full action should not see the near-singular spikes.
#
# NOW WITH the AUX_FLUCT_LAMBDA fix (2026-06-24): the aux init is drawn at the
# correct equilibrium variance 1/lambda^2 = 1 per component.  The earlier non-EO
# lambda=1 smoke saw dH=-109.7 -- that was a far-from-equilibrium RELAXATION
# transient from the old hardwired init (AUX_FLUCT=10 -> variance 1/100, i.e.
# 100x too narrow at lambda=1), NOT a full-action force bug.  This run is the
# clean test: full action + correct-variance init.  Apples-to-apples with the
# EO chroma-fork failure (identical chroma start, MN2 MDS=10).
#
# NO_METROP=10 force-accepts the start->equilibrium burn-in; the rest run real
# Metropolis.  Driver count: traj_to_run = TRAJ - NO_METROP - start_traj.
# Watch dH (vs EO's 42 and the old -109.7) + Fdt (vs EO's 9.3) + CG iters.
# Output -> cfgs/dtxqcd_lam1.0000_nonEO_cf_nf2p1/
#   submit:  sbatch production/slurm_dtxqcd_nonEO_lam1_nf2p1.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

CHROMA_CFG="${CHROMA_CFG:-/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime}"

echo "=== DTXQCD Nf=2+1 NON-EO chroma-fork lambda=1  $(date) ==="
echo "fork source: $CHROMA_CFG"
nvidia-smi --query-gpu=index,name --format=csv,noheader

IMPORT_CFG="$CHROMA_CFG" \
LAMBDA_DTXQCD=1.0 \
USE_FULL_PF=1 \
ADD_STRANGE=1 \
MASS_STRANGE=-0.245 \
DTXQCD_SUFFIX="_nonEO_cf_nf2p1" \
TRAJ=41 \
NO_METROP=10 \
N_SKIP=5 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== DTXQCD Nf=2+1 NON-EO chroma-fork lambda=1 exited rc=$? $(date) ==="
