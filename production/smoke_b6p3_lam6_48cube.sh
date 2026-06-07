#!/bin/bash
# 1-trajectory smoke of TXQCD HMC on the b6.3 48^3 x 96 ensemble at λ=6.
# Goal: measure dH at the chroma-physical mass + chosen MDS, no Metropolis
# (NO_METROP=1).  Result tells us if MDS=10 is viable for production.
#
# Setup: 2-node 8-GPU (mpi=1.1.2.4), MG on for the light Nf=2 force,
# IMPORT_CFG from chroma cfg_2000 → auto-Σ + σ=Σ/λ² init.
#
# Run from the 2-node interactive on lq2gpu05+06:
#   bash smoke_b6p3_lam6_48cube.sh

cd /lustre2/nplqcd/Grid-TXQCD/production
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

# b6.3 chroma seed
CHROMA_CFG=/lustre2/nplqcd/cfgs/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3/cl21_48_96_b6p3_m0p2416_m0p2050-djm-3_cfg_2000.lime

LOG=logs/smoke_b6p3_lam6_48cube.log
mkdir -p logs

echo "=== b6.3 48^3 x 96 λ=6 smoke (1 traj, NO_METROP=1, MDS=10) ===" | tee "$LOG"
date | tee -a "$LOG"

# b6.3 ensemble env (chroma's cl21_48_96_b6p3_m0p2416_m0p2050 params)
export LATT=48.48.48.96
export MASS_LIGHT=-0.2416
export MASS_STRANGE=-0.2050
export CSW=1.20536588031793
export BETA=6.3

# HMC env
export LAMBDA=6
export SUFFIX=_b6p3_smoke_lam6
export N_TRAJ=1
export NO_METROP=1
export IMPORT_CFG="$CHROMA_CFG"
# Skip the auto-Σ measurement (slow without MG for light-mass CG) — use the
# value measured yesterday by `compute_vev --use-mg` on this exact ensemble.
# project_b6p3_vev_trminv memory: ⟨Σ_light⟩ = 3.02213 ± 6e-5 over 4 cfgs.
export AUX_INIT=3.022
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=10
export TRAJL=0.353553390593274
export GAUGE_MULT=4
export GAUGE_INNER_MULT=2
export AUX_MULT=1
# Hasenbusch mass preconditioning: split |det M_light| into
# |det M_heavy| × |det M_light/M_heavy|.  HASEN_DM = m_heavy - m_light.
# At b6.3 light=−0.2416, HASEN_DM=0.1 sets heavy=−0.1416 → much better
# conditioned CG; ratio solve gets a smoothed source from the (V†V)^{1/4}
# numerator → expected 2-3× iter reduction at the light level.
# The existing TXQCDWilsonCloverHasenbuschAction is FD-validated on 4⁴.
export HASEN_DM=0.1

# QUDA + MG env
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
# Note (2026-05-23): TXQCD_PRECOMPUTE_GPU=1 needs ~6 GB/rank for the
# precomputed 24×24 inverse site-matrices.  Combined with MG setup +
# multishift Phi at 48^3 x 96 mpi=1.1.2.4 (per-rank 48×48×24×24),
# 8-rank/2-node split OOMs (80 GB A100).  Need ≥4 nodes (16 ranks) to fit
# at full speed — see node-count analysis at top of script.
export EIG_DIAG=1
export QUDA_ENABLE_MPS=1

# Crank QUDA multishift logging to per-iter detail for stall diagnosis.
export QUDA_VERBOSE_HMC=1

# 4-node 16-rank, mpi=2.2.2.2 → per-rank 24×24×24×48 sites.
# Same total per-rank volume as 1.1.4.4, but FATTER per-dim → ~25 % less
# halo surface area + MPI comm goes in all 4 directions (better balance).
# QUDA_VERBOSE_HMC=1 cranks multishift logging for per-iter timing diagnostics.
srun --overlap --mpi=pmix --export=ALL \
     -N 4 -n 16 --cpu-bind=none \
     ./gen_txqcd_cfgs_2plus1 \
       --grid 48.48.48.96 --mpi 2.2.2.2 \
       --shm 2048 --shm-mpi 1 \
  >> "$LOG" 2>&1

echo "=== smoke done $(date) ===" | tee -a "$LOG"

# Summary
echo ""
echo "=== Result ==="
grep -E 'dH = |compute_vev: Σ|Σ=|MG preconditioner built|MPI_ABORT|^Grid : Error' "$LOG" | head -20
