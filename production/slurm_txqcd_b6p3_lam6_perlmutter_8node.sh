#!/bin/bash
#SBATCH -A m3886_g                     # Adjust to your Perlmutter alloc
#SBATCH -C gpu
#SBATCH -q regular                     # `debug` for short tests, `regular` for prod
#SBATCH -t 12:00:00
#SBATCH -N 8                           # 8 nodes × 4 GPU = 32 ranks
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -J txqcd_b6p3_lam6
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#
# 8-node Perlmutter TXQCD HMC production at b6.3 48³×96, λ=6.
# Uses N-level Hasenbusch ladder (HASEN_LADDER env) — tune the masses below
# from the 16³×48 N-scan results (slurm_hasen_nscan_16x48.sh on lq2).
#
# PRE-FLIGHT CHECKLIST (do these once before the first submit):
#  1. Build Grid+QUDA:
#       cd $HOME/Grid-TXQCD
#       cp systems/Perlmutter/sourceme.sh . && source sourceme.sh
#       mkdir build && cd build
#       ../systems/Perlmutter/config-command   # check QUDA path inside
#       make -j 32
#       cd ../production && make gen_txqcd_cfgs_2plus1
#  2. Stage the chroma cfg:
#       scp lq2:/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime \
#           $PSCRATCH/cfgs/b6p3/
#       export CHROMA_CFG=$PSCRATCH/cfgs/b6p3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime
#  3. Validate build: run a 4⁴ smoke (NO_METROP=1, MDSTEPS=2) at single GPU first.

cd $SLURM_SUBMIT_DIR
source ../systems/Perlmutter/sourceme.sh
source ../env_perlmutter_grid.sh 2>/dev/null || true  # optional local env file

# ─── Perlmutter MPI tuning (from systems/Perlmutter/dwf4.slurm) ───────────────
export SLURM_CPU_BIND="cores"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_IPC_ENABLED=1
export MPICH_GPU_EAGER_REGISTER_HOST_MEM=0
export MPICH_GPU_NO_ASYNC_MEMCPY=0
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# ─── b6.3 ensemble parameters (matches chroma cl21_48_96_b6p3 cfg2000 XML) ───
export LATT=48.48.48.96
export MASS_LIGHT=-0.2416
export MASS_STRANGE=-0.2050
export CSW=1.20536588031793
export BETA=6.3
# Chroma's b6.3 u0 — affects LW gauge action rectangle coupling c1=-β/(20·u0²).
# Our default in params.h is b6.1's u0=0.832605...; MUST override for b6.3.
export U0=0.84570646270714
# Stout: ρ=0.125, n_smear=1 (matches chroma cfg2000; our params.h default).
export STOUT_RHO=0.125
export STOUT_NSMEAR=1

# ─── HMC parameters ──────────────────────────────────────────────────────────
export LAMBDA=6
export SUFFIX=_b6p3_lam6_perlmutter
export N_TRAJ=1000                     # production target
export NO_METROP=0                     # full Metropolis (set =1 for smoke)
export AUX_INIT=3.022                  # from b6.3 VEV measurement
# Chroma's MDIntegrator for cfg2000: LCM_STS_FORCE_GRAD (Force-Gradient),
# tau0=√2 (chroma units), n_steps=12.  Grid eps = chroma dt / 4
# (see HMC_CONVENTIONS.md), so Grid TRAJL=√2/4 + MDSTEPS=12 is physically
# equivalent to chroma's n_steps=12 tau0=√2.
export INTEGRATOR=ForceGradient
export MDSTEPS=12
export TRAJL=0.353553390593274         # √2/4
export GAUGE_MULT=4
export GAUGE_INNER_MULT=2
export AUX_MULT=1
# NOTE: chroma uses a sub-integrator of n_steps=2 for the anchor monomial
# (has_cancel_2flav).  Our integrator currently runs all ladder actions at
# the SAME level — anchor and ratios both at L1 outer.  For first-pass
# parity that's fine; sub-integrator optimization is a follow-up.

# ─── HASEN_LADDER (5-level, matches chroma cl21_48_96_b6p3 cfg2000 XML) ──────
# Masses extracted from the cfg2000 LIME header's chroma Monomials block:
#   has3:  ratio(-0.2416, -0.2400)   Δm = 0.0016  (very tight at light end)
#   has2:  ratio(-0.2400, -0.2320)   Δm = 0.0080
#   has1:  ratio(-0.2320, -0.2180)   Δm = 0.0140
#   has0:  ratio(-0.2180, -0.1870)   Δm = 0.0310
#   anchor: |det M(-0.1870)|          (chroma uses TwoFlavour; we use rational
#                                      because TXQCD's σ field requires it)
# Telescoping product: ratio·ratio·ratio·ratio·|det M(-0.1870)| = |det M(-0.2416)|.
export HASEN_LADDER="-0.2416,-0.2400,-0.2320,-0.2180,-0.1870"
# Override at submit time with: sbatch --export=ALL,HASEN_LADDER="..."

# ─── QUDA + MG (full Phase H stack, validated on lq2 with same flags) ────────
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export QUDA_FORCE_LIGHT=1              # MG for QCD strange RHMC
export USE_HMC_MG=1
export HMC_MG_NLEVEL=3
export HMC_MG_BLOCK_L0='4 4 4 4'
export HMC_MG_BLOCK_L1='1 2 2 2'
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
export EIG_DIAG=1
export QUDA_ENABLE_MPS=1

# ─── IMPORT_CFG: chroma cfg2000 staged to $PSCRATCH (see pre-flight #2) ──────
export IMPORT_CFG="${CHROMA_CFG:-$PSCRATCH/cfgs/b6p3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime}"

# ─── Launch ──────────────────────────────────────────────────────────────────
# Per-rank lattice 24×24×24×24 = 331776 sites.  L0-coarse 6×6×6×6 (≥4 ✓).
# Per-rank fine ≥4 in every dim ✓.  Memory: ~50 GB/rank base + ~5 GB transient.
# `--shm-mpi 1` is REQUIRED — see reference_shm_mpi_1_fix memory.
srun ./gen_txqcd_cfgs_2plus1 \
     --grid 48.48.48.96 --mpi 2.2.2.4 \
     --accelerator-threads 8 --shm 2048 --shm-mpi 1 \
     --comms-sequential

echo "=== smoke done $(date) ==="
