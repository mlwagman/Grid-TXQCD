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

# Two-stream λ=6 fresh-start recovery launcher (2 GPUs of 4 on one node, NP=1
# each, each stream gets 32 of the 64 CPU cores).  Both streams use MDS=20.
#
#   GPU0  λ=6    cfgs/txqcd_lam6.0000_weakfield_md20           MDS=20
#                FRESH weak-field cold start (WEAK_FIELD_SCALE=0.1, no IMPORT_CFG).
#   GPU1  λ=6    cfgs/txqcd_lam6.0000_fromchroma_md20          MDS=20
#                FRESH from chroma cl3_16_48_b6p1 cfg_11100 (IMPORT_CFG).
#
# NO_METROP is taken from the env, default 0 (Metropolis ON).  Submit pattern:
#   sbatch --export=ALL,NO_METROP=100  slurm_lam6_6p5_md30_lam7_5_md10.sh  # first job: 100-traj burn-in
#   sbatch --dependency=afterany:$J    slurm_lam6_6p5_md30_lam7_5_md10.sh  # dependents: NO_METROP=0 (default)
# This way the first job does the 100-traj burn-in once, and all chained
# dependents go straight to Metropolis-enforced production.
#
# 2026-05-24 post-disaster context: λ=6 plain-stream and all _fromchroma_md30
# dirs were lost.  These two parallel paths (weak-field + chroma start) race
# to recover λ=6 from scratch — whichever thermalizes first becomes the
# production λ=6 stream.  The corresponding cross-λ-fork (λ=6 from λ=6.5)
# and λ=6.5 MDS=20 cross-check have been removed from this script per the
# user's 2026-05-24 revision.
#
# All four resume from latest checkpoint (gen_txqcd_cfgs_2plus1: if latest>0
# it reads the aux sidecar + RNG and IGNORES IMPORT_CFG).

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs
# Two streams sharing a 64-CPU node: 32 OMP threads each → full node utilized.
export OMP_NUM_THREADS=32
N_TRAJ=${N_TRAJ:-2000}
NO_METROP=${NO_METROP:-0}
echo "[setup] NO_METROP=$NO_METROP (override at submit via --export=ALL,NO_METROP=...)"
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

# Per-GPU configuration arrays — both streams are λ=6 MDS=20 fresh starts.
# GPU0: λ=6 weak-field cold start (IMPORT_CFG="" forces weak-field path)
# GPU1: λ=6 fresh from chroma cfg_11100
LAMS=("6"                       "6")
SUFS=("_weakfield_md20"         "_fromchroma_md20")
MDSS=("20"                      "20")
IMPS=(""                        "$CHROMA_CFG")

echo "=== 2-stream λ=6 fresh-start recovery (weak-field + chroma), MDS=20 — N_TRAJ=$N_TRAJ ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1; do
  LAM="${LAMS[$i]}"; SUF="${SUFS[$i]}"; MDS="${MDSS[$i]}"; IMP="${IMPS[$i]}"
  logfile="slurm-logs/lam${LAM}${SUF}.${SLURM_JOB_ID}.out"
  echo "[gpu $i] λ=$LAM MDS=$MDS dir=cfgs/txqcd_lam$(printf '%.4f' "$LAM")${SUF}  log=$logfile  IMPORT=${IMP:-<weak-field>}"
  CUDA_VISIBLE_DEVICES=$i \
      LAMBDA=$LAM \
      SUFFIX="$SUF" \
      N_TRAJ=$N_TRAJ \
      IMPORT_CFG="$IMP" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=$MDS \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=2 \
      AUX_MULT=1 \
      HASEN_DM=0 \
      NO_METROP=$NO_METROP \
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
      mpirun -np 1 --map-by ppr:1:socket:PE=32 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait
echo "=== all 4 streams exited ==="
date
