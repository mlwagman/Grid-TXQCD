#!/bin/bash
#SBATCH --job-name=b6p5_l8_1n
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/b6p5_l8_1n.%j.out

# 1-node b6.5 32^3x64 λ=8 production with FULL QUDA stack.
#
# Why 1-node + full QUDA instead of 2-node noquda?
# Multi-node TXQCD HMC at 32^3 hits a per-rank-volume QUDA pathology:
# small per-rank vol (260k sites at 2-node) makes Wilson Meooe 138 ms/call.
# Single-node has per-rank vol 524k → healthy 1.4 ms/call. See FINDINGS.md
# in overnight_2node_diag/.
#
# The prior 1-node + full QUDA OOMed at ~26 min (peak 78 GB/80 GB per GPU).
# env_lq2_grid.sh now sets QUDA_ENABLE_DEVICE_MEMORY_POOL=0 +
# QUDA_ENABLE_MANAGED_MEMORY=1, which lets QUDA spill pages to host memory.
# Tested 2026-05-25 on lq2gpu14: no OOM through MD evolution, per-call
# Meooe unchanged (1.29 ms/call at mpi=1.1.1.4, vs 1.37 at 1.1.2.2 —
# single-axis t-split has smaller total halo surface).
#
# Topology choice: mpi=1.1.1.4 (t-only split) over 1.1.2.2 for the slight
# halo-surface advantage.  Distinct cfg dir from the noquda 2-node chain.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs cfgs

# ===== b6.5 ensemble parameters (chroma-matched) =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788
export LAMBDA=8
export AUX_INIT=2.9849

# ===== HMC parameters =====
export N_TRAJ=2000
export NO_METROP=0
export INTEGRATOR=MinimumNorm2
unset LAMBDA_MN2   # use Grid's default 0.19318 (chroma cl3_32_64 outer match)
export MDSTEPS=10
export TRAJL=0.353553390593274
export GAUGE_MULT=5
export GAUGE_INNER_MULT=2
export AUX_MULT=1
export HASEN_DM=0

# ===== FULL QUDA stack (the per-rank-vol pathology doesn't fire at 1-node) =====
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=1
export TXQCD_MOOEEINV_CUBLAS=1
export TXQCD_MOOEE_CUBLAS=1
# Phase b cuBLAS Clover-inverse: cublasZgetrfBatched + getriBatched replaces
# per-site CPU Eigen.inverse() inside CloverHelpers::Instantiate.  Bit-exact
# round-trip validated 2026-05-26 on 16³×48 (||Δ||/||CT⁻¹|| ≈ 2e-16).
# Saves ~0.3-0.5 s × ~20 ImportGauge calls/traj.
export WCF_INSTANTIATE_GPU=0  # bit-exact, but +191 s/traj REGRESSION when combined with J.3 ON under srun (validated 2026-05-30; my earlier mpirun "safe" test was launcher-confounded)
# WCF_LOGDET_GPU=1: QCDLogDetCloverEOAction::S() uses cuBLAS getrfBatched and
# extracts log|det| from diag(U) instead of per-site Eigen.determinant().
# Bit-exact validated 2026-05-26 on 16³×48 (rel diff = 2.7e-14, ~machine ε).
# Saves ~3-5 sec per action eval × few per traj.
export WCF_LOGDET_GPU=1
# TXQCD_LOGDET_S_GPU=1: TXQCDLogDetCloverEOAction::S() uses cuBLAS getrfBatched
# on 24×24 site matrices (vs CPU Eigen.partialPivLu().determinant() per site),
# extracting log|det| from diag(U).  Bit-exact validated 2026-05-26 on 4⁴
# (rel diff = 1.2e-16, machine ε).  Reuses deriv_gpu's persistent device
# buffers so allocation cost is amortized.
export TXQCD_LOGDET_S_GPU=1
# WCF_LOGDET_DERIV_GPU=1: QCDLogDetCloverEOAction::deriv() runs the sigma-trace
# extraction on RB-Even (CloverTermInvEven directly), and exploits σ_νμ = -σ_μν
# antisymmetry to halve the unique sigma count (12 → 6 unique).  6 unique
# TraceIndex(Gamma·CTI_e) ops on RB-Even replace 12 full-grid ones, plus the
# explicit setCheckerboard(0, CTI_even) → Lambda construction is avoided.
# Bit-exact validated 2026-05-26 on 4⁴ (||F_cpu - F_gpu||² = 0 exactly).
export WCF_LOGDET_DERIV_GPU=1
# WCF_BUFFERS_TRANSIENT=1: free cuBLAS scratch (M_dev/Mi_dev) after each
# InstantiateGPU + compute_logdet_gpu call.  Saves ~2.4 GB/rank persistent at
# 32³×64 — necessary to avoid OOM on b6.5 1-node where peak was ~78/80 GB
# before adding these env vars.  Cost: ~50 ms cudaMalloc per ImportGauge call
# × ~22 calls/traj = ~1 s/traj overhead (0.04% of 2730 s/traj).
export WCF_BUFFERS_TRANSIENT=0
# TXQCD_LOGDET_BUILD_GPU=1 — re-enabled 2026-05-30 after the launcher-pattern
# finding: the previous "Variant C 120.9 s vs Variant A 15.3 s — 8× slowdown"
# measurement was made under `mpirun -np N --bind-to none`, which itself is 8×
# slower per CG hermop than `srun --mpi=pmix` on identical hardware (see
# [[launcher-pattern-matters]] memory).  Under srun, J.3 ON gives bit-exact
# forces and IDENTICAL hermop times to J.3 OFF (26.39 s vs 26.36 s on the same
# 1002-iter CG).  The per-call savings (eliminates pickCB 234 ms + Eigen build
# 143 ms = ~377 ms/LogDet deriv) are now free.
export TXQCD_LOGDET_BUILD_GPU=1
# TXQCD_PRECOMPUTE_BUILD_GPU=1 (Phase J.4): inside
# TXQCDWilsonCloverFermionEO::ImportFields, replace UnvectorizeAux/Clover +
# CPU PrecomputeInverses (1.45 s × 2 parities/call) with the Phase J.3 GPU
# BuildSiteMatrix kernel + cuBLAS getrf/getri.  M_dev_e_/M_dev_o_ are
# populated device-side; PackInverseToSimdGPU's scatter step then reads
# them.  Each TXQCDWilsonCloverFermionEO ctor saves ~3 s at 16³×48 b6.1
# (scales ~8× to 32³×64).  With 6-30+ ImportFields per traj → big win.
# Bit-exact MooeeInv(Mooee v)=v at 4⁴ (rel 1.9e-15 vs 2.1e-15 baseline).
export TXQCD_PRECOMPUTE_BUILD_GPU=1
# EIG_DIAG=1 causes a 37× slowdown in TXQCD-EO Meooe at 32³×64 lattice
# (1.3 → 48 ms/call), bisected 2026-05-25 on lq2gpu13 interactive 1281196.
# Mechanism unclear (eig branch only fires at traj end); 16³×48 unaffected.
# Disable on 32³×64 until root-caused.
export EIG_DIAG=0
export OMP_NUM_THREADS=8

# ===== chroma source cfg (used only if no checkpoints in cfg_dir) =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

export SUFFIX="_b6p5_lam8_mds10_1node"

LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"
echo "=== b6.5 32³×64 λ=8 MDS=10 — 1-node 4-GPU full QUDA + managed memory ==="
echo "cfg_dir=$CFG_DIR"
echo "mpi=1.1.1.4 (t-only split, lowest-halo topology)"
echo "N_TRAJ=$N_TRAJ (resume from latest ckpt if present)"
date

srun --mpi=pmix -N 1 -n 4 --cpu-bind=none \
    ./gen_txqcd_cfgs_2plus1 \
        --mpi 1.1.1.4 --shm 1024 --shm-mpi 1

echo "=== exit code $? — end $(date) ==="
echo "latest ckpts:"
ls "$CFG_DIR"/ckpoint_lat.* 2>/dev/null | sort -V | tail -5
