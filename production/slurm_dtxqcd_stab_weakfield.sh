#!/bin/bash
#SBATCH --job-name=dtx_stab_weak
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/dtx_stab_weak.%j.out
# ============================================================================
# DTXQCD integrator-stability test, WEAK-FIELD COLD START.
#
# Same canonical settings as the chroma-fork sibling (MN2 MDS=10 trajL=sqrt2/4,
# lambda=10, stout, AUX_INIT_AUTO, RAT_AUTO_HI, GPU/cuBLAS) but starting from a
# weak-field gauge.  NO_METROP=10 force-accepts all 10 trajectories -- from a
# cold start Metropolis would just reject the far-from-equilibrium proposals;
# this is a pure integrator-stability / thermalization-trend check.  Watch dH
# bounded (relative to H), plaq trending toward the QCD value, signPf tracked.
#
# Driver count convention (gen_dtxqcd_cfgs.cc:472): executed trajectories =
# TRAJ - NO_METROP - start_traj.  For 10 force-accepted trajectories from a
# fresh weak-field start, TRAJ = 10 + NO_METROP = 20 (TRAJ=10 NO_METROP=10 would
# run ZERO trajectories on the current binary).
# Output -> cfgs/dtxqcd_lam10.0000_weakstab/
#   submit:  sbatch production/slurm_dtxqcd_stab_weakfield.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

echo "=== DTXQCD weak-field stability test  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

DTXQCD_SUFFIX="_weakstab" \
TRAJ=20 \
NO_METROP=10 \
N_SKIP=1 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== weak-field stability test exited rc=$? $(date) ==="
