#!/bin/bash
#SBATCH --job-name=dtx_stab_fork
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/dtx_stab_fork.%j.out
# ============================================================================
# DTXQCD integrator-stability test, FORKED FROM A CHROMA CONFIG.
#
# Seeds a lambda=10 DTXQCD stream from a thermalized chroma cl3_16_48_b6p1
# gauge config (QCD plaquette) via IMPORT_CFG, then runs 10 trajectories with
# the canonical production integrator (MN2 MDS=10 trajL=sqrt2/4) and all the
# fancy init on (AUX_INIT_AUTO saddle, RAT_AUTO_HI Remez autoscale, stout,
# GPU/cuBLAS).  Goal: confirm the integrator holds the plaquette at the QCD
# value for 10 traj with bounded dH and good Metropolis acceptance.
#
# NO_METROP=1 force-accepts ONLY trajectory 0, to absorb the one-off aux
# saddle -> equilibrium transient on the freshly-imported gauge; the remaining
# trajectories run the real Metropolis test.
#
# Driver count convention (gen_dtxqcd_cfgs.cc:472): executed trajectories =
# TRAJ - NO_METROP - start_traj.  So for 10 executed trajectories from a fresh
# import with a 1-traj warmup, TRAJ = 10 + NO_METROP = 11 (NOT 10 -- TRAJ=10
# NO_METROP=1 would run only 9; TRAJ=NO_METROP would run zero).
# Output -> cfgs/dtxqcd_lam10.0000_chromafork_stab/
#   submit:  sbatch production/slurm_dtxqcd_stab_chromafork.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

CHROMA_CFG="${CHROMA_CFG:-/lustre2/nplqcd/cfgs/cl3_16_48_b6p1_m0p2450/a/cl3_16_48_b6p1_m0p2450_a_cfg_10000.lime}"
NMET="${NO_METROP:-1}"

echo "=== DTXQCD chroma-fork stability test  $(date) ==="
echo "fork source: $CHROMA_CFG"
nvidia-smi --query-gpu=index,name --format=csv,noheader

IMPORT_CFG="$CHROMA_CFG" \
DTXQCD_SUFFIX="_chromafork_stab" \
TRAJ=$(( 10 + NMET )) \
NO_METROP="$NMET" \
N_SKIP=1 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== chroma-fork stability test exited rc=$? $(date) ==="
