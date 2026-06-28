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
# 2026-06-24: AUX_FLUCT_LAMBDA now defaults to lambda (was hardwired to 10), so
# the aux init is drawn at the CORRECT equilibrium variance 1/lambda^2 per
# component.  The old _weak_nf2p1 stream started ~1.7x too narrow at lambda=6
# (drawn at 1/10 instead of 1/6) -> large far-from-equilibrium relaxation
# (systematic negative dH + growing aux norm).  Fresh _fi suffix forces a clean
# fixed-init restart (a resume would skip init and keep the wrong start).
# 2026-06-25: MDS=20 (was 10).  EO lambda=6 at MDS=10 gave a systematic dH~+3.5
# (frozen chain, last 9 Metropolis all REJECTED -- the small-aux integration
# error).  dH ~ (step)^4, so MDS=20 should drop it ~16x to ~0.2 IF it's
# integration error (acceptance recovers); if +3.5 persists it's a real bias.
# FRESH _mds20 suffix (new dir) -> clean MDS=20 weak-field stream that runs
# alongside the MDS=10 _fi job (1288825) with NO checkpoint collision (cancel
# 1288825 separately to free its node).  Re-thermalizes from weak-field, so the
# MDS=20 acceptance verdict comes once it equilibrates (plaq -> ~0.51).
# Output -> cfgs/dtxqcd_lam6.0000_weak_nf2p1_mds20/
#   submit:  sbatch production/slurm_dtxqcd_weak_lam6_nf2p1.sh
# ============================================================================
mkdir -p /lustre2/nplqcd/Grid-DTXQCD/production/slurm-logs

echo "=== DTXQCD Nf=2+1 weak-field lambda=6 thermalization  $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMBDA_DTXQCD=6.0 \
MDSTEPS=20 \
ADD_STRANGE=1 \
MASS_STRANGE=-0.245 \
DTXQCD_SUFFIX="_weak_nf2p1_mds20" \
TRAJ=1500 \
NO_METROP=10 \
N_SKIP=10 \
  /lustre2/nplqcd/Grid-DTXQCD/production/run_dtxqcd_gencfgs.sh

echo "=== DTXQCD Nf=2+1 weak-field lambda=6 exited rc=$? $(date) ==="
