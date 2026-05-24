#!/bin/bash
# Inverter-timing test for new b6.5 32³×64 m=-0.1788 TXQCD ensemble at λ=8.
# Run a 1-traj NO_METROP=1 HMC against chroma cfg_4200, parametrized by GPU
# count and MDS for an integrator-sweet-spot scan.
#
# Usage:   bash test_b6p5_hmc_inverter_timing.sh <NGPU> [<MDS>]
#          NGPU in {1,2,4}, MDS default 2.
# Output:  logs/b6p5_hmc_test_<NGPU>gpu_mds<MDS>.log
#
# 2026-05-23: previous λ=21.7 guess revised to λ=8.  Single-node mpirun
# tests; rebuilt Grid with --enable-setdevice patch (see build-gpu-nvtx/
# Grid/Config.h note) so per-rank cudaSetDevice happens before any Grid
# allocations — required when Grid+QUDA cohabit on multi-GPU.

set -u
NGPU="${1:?usage: $0 <NGPU> [<MDS>]}"
MDS="${2:-2}"

cd /lustre2/nplqcd/Grid-TXQCD/production
mkdir -p logs cfgs

source ../env_lq2_grid.sh

# ===== b6.5 ensemble parameters =====
export LATT=32.32.32.64
export BETA=6.5
export CSW=1.170082389372972
export U0=0.85703554213273                                           # matches chroma cl3_32_64_b6p5 cfg4200 XML <u0>
export MASS_LIGHT=-0.1788
export MASS_STRANGE=-0.1788                                  # Nf=3 (degenerate)
export LAMBDA=8                                              # user-tuned sweet spot (revised from 21.7 2026-05-23)
export AUX_INIT=2.9849                                       # Σ = vev_trminv on b6.5

# ===== HMC short-test parameters (NO_METROP, short MDS) =====
export N_TRAJ=1
export NO_METROP=1
export INTEGRATOR=MinimumNorm2
export LAMBDA_MN2=0.1789
export MDSTEPS=$MDS                                          # via 2nd CLI arg, default 2
export TRAJL=0.353553390593274
export GAUGE_MULT=5
export GAUGE_INNER_MULT=4
export AUX_MULT=4                                            # placeholder (production at λ=21.7 will need scaling)
export HASEN_DM=0

# QUDA-accelerated TXQCD machinery (full production stack)
export QUDA_FORCE=1
export QUDA_FORCE_KERNEL=1
export TXQCD_QUDA_HYBRID=1
export TXQCD_QUDA_FULL=1
# TXQCD_PRECOMPUTE_GPU=1 holds per-site 24×24 inverses on GPU — that's
# 24*24*16 = 9 KB/site = ~10 GB for 32³×64.  Combined with other field
# allocations it pushed us to OOM at 53 GB / 80 GB on 2-GPU (per-rank vol
# 32³×32).  Disable for 2-GPU; enable for 4-GPU (per-rank halved).
# The TXQCD_MOOEE*_CUBLAS paths REQUIRE TXQCD_PRECOMPUTE_GPU=1 (lex table is
# built by the GPU pack), so they have to be toggled in lockstep.
if [ "$NGPU" -ge 4 ]; then
  export TXQCD_PRECOMPUTE_GPU=1
  export TXQCD_MOOEEINV_CUBLAS=1
  export TXQCD_MOOEE_CUBLAS=1
else
  export TXQCD_PRECOMPUTE_GPU=0
  export TXQCD_MOOEEINV_CUBLAS=0
  export TXQCD_MOOEE_CUBLAS=0
fi
export EIG_DIAG=0                                            # off for clean timing
export QUDA_ENABLE_MPS=0                                     # one process per GPU, no MPS needed
export OMP_NUM_THREADS=$((16 / NGPU))                        # 16-CPU cgroup limit

# ===== chroma source cfg =====
export IMPORT_CFG=/lustre2/nplqcd/cfgs/cl3_32_64_b6p5_m0p1788/cl3_32_64_b6p5_m0p1788_cfg_4200.lime

# ===== per-NGPU dir/MPI selection =====
case "$NGPU" in
  1) MPI=1.1.1.1; SUF=_b6p5_32cube_test_1gpu_mds${MDS};  CVD=0       ;;
  2) MPI=1.1.1.2; SUF=_b6p5_32cube_test_2gpu_mds${MDS};  CVD=0,1     ;;
  4) MPI=1.1.2.2; SUF=_b6p5_32cube_test_4gpu_mds${MDS};  CVD=0,1,2,3 ;;
  *) echo "FATAL: NGPU must be 1, 2, or 4 (got $NGPU)" >&2; exit 1 ;;
esac

export SUFFIX="$SUF"
# All ranks see all GPUs.  QUDA's initQuda calls cudaSetDevice(local_rank)
# internally; Grid (built with --enable-setdevice=no) inherits that current
# context for its subsequent cudaMalloc calls.  Pre-filtering
# CUDA_VISIBLE_DEVICES to a single GPU per rank breaks QUDA's "I need
# N_GPUS visible to bind ranks 0..N-1" check at communicator_quda.h:585.
export CUDA_VISIBLE_DEVICES="$CVD"

# Wipe stale checkpoints from previous test runs in this SUFFIX dir
LAM_TAG=$(printf "lam%.4f" "$LAMBDA")
TEST_CFG_DIR="cfgs/txqcd_${LAM_TAG}${SUFFIX}"
rm -rf "$TEST_CFG_DIR"

LOG="logs/b6p5_hmc_test_${NGPU}gpu_mds${MDS}.log"
echo "=== b6.5 32³×64 m=−0.1788 λ=${LAMBDA} — ${NGPU} GPU, mpi=$MPI MDS=$MDS CVD=$CVD ===" | tee "$LOG"
date | tee -a "$LOG"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  cfg_dir=$TEST_CFG_DIR" | tee -a "$LOG"

# Single-node multi-rank via mpirun — QUDA assigns each rank to a distinct
# local GPU (rank 0 → device 0, rank 1 → device 1, ...).
# --shm 1024: reduces stencil-comms reservation to ~1 GB (large 32³×64 needs the headroom).
mpirun -np "$NGPU" --bind-to none \
    ./gen_txqcd_cfgs_2plus1 --mpi "$MPI" --shm 1024 --shm-mpi 1 \
    >> "$LOG" 2>&1

echo "=== exit code $? — finished $(date) ===" | tee -a "$LOG"
