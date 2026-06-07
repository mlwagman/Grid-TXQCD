#!/bin/bash
#SBATCH --job-name=txqcd_b6p3_lam6
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/txqcd_b6p3_lam6.%j.out

# Production b6.3 TXQCD HMC at λ=6 on the cl21_48_96_b6p3_m0p2416_m0p2050
# ensemble.  4-node 16-rank (mpi=1.1.4.4 → per-rank 48 x 48 x 12 x 24),
# MG light Nf=2 force + multishift CG strange RHMC, IMPORT_CFG chroma seed.
#
# Setup decisions (2026-05-23):
#   * λ=6 chosen as ⅓ of the baryon-phase mass (memory:
#     project_txqcd_lambda_sweet_spot).
#   * Σ = 3.022 measured yesterday via compute_vev MG on this ensemble
#     (memory: project_b6p3_vev_trminv); passed via AUX_INIT to skip slow
#     auto-Σ measurement.
#   * 4/2/1 MULT split (GAUGE 4, GAUGE_INNER 2, AUX 1) — Fdt diagnostic
#     showed aux Fdt 9× smaller than light, so AUX_MULT=4 over-engineered.
#   * MDS=10 if dH from smoke is OK; else bump to MDS=20.  Update MDSTEPS
#     env below before submitting.
#   * USE_HMC_MG=1 + HMC_MG_NLEVEL=3 for the light force (TwoFlavour Schur
#     2-solve trick).  Strange RHMC stays plain multishift CG (no MG
#     possible per reference_mg_rhmc_landscape).
#
# Memory note: PRECOMPUTE_GPU=1 needs ~6 GB/rank for precomputed 24×24
# matrices.  Fits at 4-node (~60 GB / 80 GB A100 measured).  2-node OOMs.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

# b6.3 chroma seed (cl21_48_96_b6p3_m0p2416_m0p2050-djm-3 cfg_2000)
CHROMA_CFG=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime
N_TRAJ=${N_TRAJ:-2000}
MDSTEPS=${MDSTEPS:-10}

# b6.3 ensemble env
export LATT=48.48.48.96
export MASS_LIGHT=-0.2416
export MASS_STRANGE=-0.2050
export CSW=1.20536588031793
export BETA=6.3

# HMC env (4/2/1 mult split + smear-skip patch in TXQCDSmearedConfiguration.h)
export LAMBDA=6
export SUFFIX=_b6p3_fromchroma_mds10
export N_TRAJ=$N_TRAJ
export NO_METROP=0
export IMPORT_CFG="$CHROMA_CFG"
export AUX_INIT=3.022      # from project_b6p3_vev_trminv (measured ⟨Σ_light⟩)
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=$MDSTEPS
export TRAJL=0.353553390593274
export GAUGE_MULT=4
export GAUGE_INNER_MULT=2
export AUX_MULT=1
export HASEN_DM=0
export WEAK_FIELD_SCALE=0.1   # safety fallback; ignored when IMPORT_CFG set and ckpt exists

# QUDA + MG env (USE_HMC_MG=1 turns on light Nf=2 MG; strange stays multishift CG)
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export QUDA_FORCE_LIGHT=1
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

echo "=== b6.3 TXQCD HMC λ=6 — 4-node 16-rank, MDS=$MDSTEPS ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# 4-node 16-rank, mpi=1.1.4.4
srun --mpi=pmix --export=ALL \
     -N 4 -n 16 --cpu-bind=none \
     ./gen_txqcd_cfgs_2plus1 \
       --grid 48.48.48.96 --mpi 1.1.4.4 \
       --shm 2048 --shm-mpi 1

echo "=== b6.3 lam6 stream exited $(date) ==="
