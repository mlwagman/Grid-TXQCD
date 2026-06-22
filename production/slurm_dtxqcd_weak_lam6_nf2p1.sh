#!/bin/bash
#SBATCH --job-name=dtx_w_l6_21
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/dtx_w_l6_21.%j.out
# ============================================================================
# DTXQCD Nf=2+1 WEAK-FIELD thermalization run at lambda=6 -- direct comparison
# against the TXQCD Nf=2+1 weak-field lambda=6 stream
# (../Grid-TXQCD/.../cfgs/txqcd_lam6.0000_weakfield_md10, plaq -> ~0.5143).
#
# KEY: DTXQCD's doubled operator is the Nf=2 LIGHT sector; the chroma reference
# plaq is Nf=2+1, so we ADD a plain-QCD Nf=1 spectator strange (ADD_STRANGE=1,
# mass_strange = m_light = -0.245) exactly as gen_txqcd_cfgs_2plus1 did.  Same
# canonical point as TXQCD: 16^3x48, m=-0.245, csw=1.249, beta=6.1, stout,
# MN2 MDS=10 trajL=sqrt2/4, weak-field cold start wf=0.1.
#
# NO_METROP=100 force-accepts the thermalization burn-in (matches the TXQCD
# weak-field head); compare per-traj plaq + scalar (sigma/s) VEV trends over the
# early trajectories.  Driver count: traj_to_run = TRAJ - NO_METROP - start.
# Output -> cfgs/dtxqcd_lam6.0000_weak_nf2p1/
#   submit:  sbatch production/slurm_dtxqcd_weak_lam6_nf2p1.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

echo "=== DTXQCD Nf=2+1 weak-field lambda=6 thermalization  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDA_DTXQCD=6.0 \
ADD_STRANGE=1 \
MASS_STRANGE=-0.245 \
DTXQCD_SUFFIX="_weak_nf2p1" \
TRAJ=200 \
NO_METROP=100 \
N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== DTXQCD Nf=2+1 weak-field lambda=6 exited rc=$? $(date) ==="
