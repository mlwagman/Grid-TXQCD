#!/bin/bash
#SBATCH --job-name=lam6_6p5_7_5
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/lam6_6p5_7_5.%j.out

# Continue 4 chroma-pedigree TXQCD streams, one per GPU.
#
#   GPU0  λ=6    cfgs/txqcd_lam6.0000_fromchroma_md30_fork_md20        MDS=20  (fork at ckpt 160)
#   GPU1  λ=6.5  cfgs/txqcd_lam6.5000_fromchroma_md30_fork_md20        MDS=20  (fork at ckpt 180)
#   GPU2  λ=7    cfgs/txqcd_lam7.0000_fromchroma_md20_fork_t50_mds10   MDS=10
#   GPU3  λ=5    cfgs/txqcd_lam5.0000_fromchroma_md20_fork_t50_mds10   MDS=10
#
# 2026-05-22 MDS=30 → MDS=10 fork attempt FAILED.  λ=6 MDS=10 froze (Pacc=0
# over 25 traj, ⟨|dH|⟩=3.8); λ=6.5 MDS=10 mostly rejected (Pacc=0.20,
# ⟨|dH|⟩=2.5).  Naive prediction missed that the eps × 3 step (MDS 30→10)
# pushed past the integrator-stability threshold for these still-thermalizing
# streams.  2026-05-23 retry at MDS=20 (eps × 1.5):  predicted ⟨|dH|⟩ ≈ 0.75,
# Pacc ≈ 0.7, 1.5× faster than MDS=30 — safer compromise.  The failed
# _fork_md10 dirs are left in place (mostly-rejected traj 161-180/181-200,
# scientifically useless — should be deleted later).
#
# λ=6.5 MDS=30 reference continues under a separate launcher
# (slurm_nodeA_v2.sh GPU1) as the integrator cross-check.  λ=6 MDS=30 is
# parked at ckpt 160 (no further extension planned).
#
# MDS=20 fork dirs seeded with hardlinks of the latest MDS=30 ckpt + aux + RNG:
#   _fork_md20/ckpoint_lat.160         <- _md30/ckpoint_lat.160         (λ=6)
#   _fork_md20/ckpoint_lat.180         <- _md30/ckpoint_lat.180         (λ=6.5)
# HMC resumes at traj 161 / 181 under MDS=20.
#
# All four resume from latest checkpoint (gen_txqcd_cfgs_2plus1: if latest>0
# it reads the aux sidecar + RNG and IGNORES IMPORT_CFG).

# Robust cd: works under sbatch (SLURM_SUBMIT_DIR=submission dir) and an
# interactive `bash thisscript.sh` (salloc sets SLURM_SUBMIT_DIR wrong).
_sd="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -x "$SLURM_SUBMIT_DIR/gen_txqcd_cfgs_2plus1" ]; then
  cd "$SLURM_SUBMIT_DIR"
elif [ -x "$_sd/gen_txqcd_cfgs_2plus1" ]; then
  cd "$_sd"
else
  echo "FATAL: gen_txqcd_cfgs_2plus1 not found (SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' _sd='$_sd')" >&2
  exit 1
fi
echo "[setup] running from $(pwd)"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-2000}
# Resume branch ignores IMPORT_CFG when checkpoints exist; kept only as a
# safety fallback (all four dirs DO have checkpoints, so it is never used).
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

LAMS=("6"                            "6.5"                          "7"                                  "5")
SUFS=("_fromchroma_md30_fork_md20"   "_fromchroma_md30_fork_md20"   \
      "_fromchroma_md20_fork_t50_mds10" "_fromchroma_md20_fork_t50_mds10")
MDSS=("20"                           "20"                           "10"                                 "10")

echo "=== continue λ=6,6.5 (MDS=20) + λ=7,5 (MDS=10) — N_TRAJ=$N_TRAJ ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1 2 3; do
  LAM="${LAMS[$i]}"; SUF="${SUFS[$i]}"; MDS="${MDSS[$i]}"
  logfile="slurm-logs/lam${LAM}${SUF}.${SLURM_JOB_ID}.out"
  echo "[gpu $i] λ=$LAM MDS=$MDS dir=cfgs/txqcd_lam$(printf '%.4f' "$LAM")${SUF}  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i \
      LAMBDA=$LAM \
      SUFFIX="$SUF" \
      N_TRAJ=$N_TRAJ \
      IMPORT_CFG="$CHROMA_CFG" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=$MDS \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=2 \
      AUX_MULT=1 \
      `# 2026-05-23: MULT 4/4/4 → 4/2/1. Fdt diagnostic showed aux Fdt 9× smaller` \
      `# than TXQCD light at λ=5-8; AUX_MULT=4 was over-engineered. Combined with` \
      `# smear-skip patch in TXQCDSmearedConfiguration.h, ~50% wallclock reduction.` \
      HASEN_DM=0 \
      NO_METROP=0 \
      WEAK_FIELD_SCALE=0.1 \
      QUDA_FORCE=1 \
      QUDA_FORCE_KERNEL=1 \
      TXQCD_QUDA_HYBRID=1 \
      TXQCD_QUDA_FULL=1 \
      TXQCD_PRECOMPUTE_GPU=1 \
      TXQCD_MOOEEINV_CUBLAS=1 \
      TXQCD_MOOEE_CUBLAS=1 \
      EIG_DIAG=1 \
      QUDA_ENABLE_MPS=1 \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait
echo "=== all 4 streams exited ==="
date
