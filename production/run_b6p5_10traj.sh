#!/bin/bash
# 10-trajectory production-style HMC run for b6.5 32³×64 m=-0.1788 TXQCD at λ=8,
# initialized from chroma cfg_4200.  Saves cfgs every traj.
#
# Usage:   bash run_b6p5_10traj.sh [<MDS>]
#          MDS default 4 (revisit after MDS-scan locks in sweet spot).
# Output:  cfgs/txqcd_lam8.0000_b6p5_lam8_mds<MDS>_first10/
#          logs/b6p5_10traj_mds<MDS>.log
#
# Differs from test_b6p5_hmc_inverter_timing.sh in:
#   - WITH metropolis (NO_METROP=0)
#   - N_TRAJ=10 (not 1)
#   - production SUFFIX (not _test_)
#   - no rm -rf of cfg dir (preserves checkpoints across re-launches)
#   - PRECOMPUTE_GPU+cuBLAS Mooee ON (4 GPU split fits comfortably)

set -u
MDS="${1:-4}"

cd /lustre2/nplqcd/Grid-TXQCD/production
mkdir -p logs cfgs

source ../env_lq2_grid.sh

# ===== b6.5 ensemble parameters =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788
export LAMBDA=8
export AUX_INIT=2.9849                                       # Σ = vev_trminv

# ===== production HMC parameters =====
export N_TRAJ=10
export N_SKIP=1                                              # save every traj (vs prod default 10)
export NO_METROP=0                                           # WITH metropolis
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=$MDS
export TRAJL=0.353553390593274
# 2026-05-23 v6: restored MULTs to 4/4/4 matching b6.1 production ratios.
# v5 (MULT=1 MDS=4) blew up: TXQCD light multishift CG iterations 700→15000
# by MD substep 3 — integrator instability drove gauge field into ill-
# conditioned region of D†D.  Lesson: MULTs=4 is not over-engineering, it's
# necessary at this lattice with strange Nf=1 RHMC.  Per-traj ~3 hr (slow
# but stable).  In remaining 6 hr, ~2 trajs achievable, each with saved ckpt.
export GAUGE_MULT=5
export GAUGE_INNER_MULT=4
export AUX_MULT=4
export HASEN_DM=0

# ===== QUDA + TXQCD performance stack =====
# 2026-05-23: hit OOM at 50 MB push when peak GPU mem at 78 GB during strange
# MultiShiftCG.  4-GPU 32³×64 = per-rank 32×32×16×16 — tight.  Drop
# memory-heavy options to fit:
#   - TXQCD_PRECOMPUTE_GPU=0 + cuBLAS Mooee OFF (lockstep; cuBLAS Mooee
#     requires PRECOMPUTE_GPU lex table) → saves ~10 GB of per-site 24×24 mats
#   - EIG_DIAG=0 → skip Lanczos work-vector allocations
#   - --shm 512 instead of 1024 → saves 0.5 GB of stencil comms reservation
# Trade-off: ~2× slower force eval, but completes without OOM.
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
export TXQCD_PRECOMPUTE_GPU=0
export TXQCD_MOOEEINV_CUBLAS=0
export TXQCD_MOOEE_CUBLAS=0
export EIG_DIAG=0
export QUDA_ENABLE_MPS=0
export OMP_NUM_THREADS=4                                     # 4 GPUs × 4 threads = 16 (cgroup limit)

# ===== chroma source cfg =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

# ===== 4-GPU single-node run =====
NGPU=4
MPI=1.1.2.2
export SUFFIX="_b6p5_lam8_mds${MDS}_first10"
export CUDA_VISIBLE_DEVICES=0,1,2,3

LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"

LOG="logs/b6p5_10traj_mds${MDS}.log"
echo "=== b6.5 32³×64 m=-0.1788 λ=8 MDS=$MDS — N_TRAJ=10 ===" | tee -a "$LOG"
echo "cfg_dir=$CFG_DIR" | tee -a "$LOG"
echo "start=$(date)" | tee -a "$LOG"

# 4-GPU single-node mpirun with Grid+QUDA cohabitation; Grid rebuilt with
# --enable-setdevice=yes so per-rank cudaSetDevice happens before allocations.
mpirun -np "$NGPU" --bind-to none \
    ./gen_txqcd_cfgs_2plus1 --mpi "$MPI" --shm 512 --shm-mpi 0 \
    >> "$LOG" 2>&1

echo "=== exit code $? — end=$(date) ===" | tee -a "$LOG"
echo "cfgs saved:" | tee -a "$LOG"
ls "$CFG_DIR"/ckpoint_lat.* 2>/dev/null | sort -V | tee -a "$LOG"
