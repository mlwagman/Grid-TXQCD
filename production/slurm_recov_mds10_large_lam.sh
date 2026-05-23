#!/bin/bash
#SBATCH --job-name=recov_md10L
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/recov_md10L.%j.out

# Recovery batch — MDS=10 for large λ (10, 12, 14, 16) where MDS=10 gave
# 75-100% acceptance.  Continues from existing _fromchroma_md10 dirs which
# have 4-5 cfgs each from the earlier MDS=10 batch (more thermalized than
# the brief _fromchroma_md20 batch with 1-2 cfgs each).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-2500}
RECOV_SUFFIX="_fromchroma_md10"
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

echo "=== Launching 4 TXQCD MDS=10 recov streams (large λ) ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAMS=("10" "12" "14" "16")
# 2026-05-23: per-λ MULT tuning from Fdt diagnostic.  aux Fdt scales ~linearly
# with λ; at AUX_MULT=4 we measured Fdt_aux = 0.10/0.18/0.22/0.25 at λ=7/12/14/16,
# vs TXQCD-light Fdt ≈ 0.9 (constant).  Headroom:
#   λ=10 (extrap Fdt_aux ≈ 0.15 @ MULT=4): AUX_MULT=1 → Fdt 0.6, factor 1.5× margin
#   λ=12/14/16 (Fdt 0.18-0.25 @ MULT=4):    AUX_MULT=2 → Fdt 0.36-0.50, factor 2× margin
# Combined with smear-skip patch in TXQCDSmearedConfiguration.h: expected 30-50%
# wallclock reduction per traj.
for i in 0 1 2 3; do
  LAM="${LAMS[$i]}"
  if [ "$LAM" = "10" ]; then AUX_M=1; else AUX_M=2; fi
  logfile="slurm-logs/txqcd_lam${LAM}_fromchroma.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  TXQCD λ=$LAM  AUX_MULT=$AUX_M  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i \
      LAMBDA=$LAM \
      SUFFIX="$RECOV_SUFFIX" \
      N_TRAJ=$N_TRAJ \
      IMPORT_CFG="$CHROMA_CFG" \
      INTEGRATOR=MinimumNorm2 \
      LAMBDA_MN2=0.1789 \
      MDSTEPS=10 \
      TRAJL=0.353553390593274 \
      GAUGE_MULT=4 \
      GAUGE_INNER_MULT=2 \
      AUX_MULT=$AUX_M \
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
echo "=== MDS=10 large-λ batch done ==="
date
