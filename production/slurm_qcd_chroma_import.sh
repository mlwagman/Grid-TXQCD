#!/bin/bash
#SBATCH --job-name=qcd_chroma_import
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_chroma_import.%j.out

# Chroma-import basin-stability test, complementary to slurm_qcd_basintest.sh.
# All 4 streams start from a chroma-thermalized cfg (plaq=0.5138).
# Hypothesis to test: starting in chroma's basin, do all MD settings keep
# us there? Or does NoMetrop=10 force-accept push us out of chroma's basin
# (the same way it may have pushed qcd_ref into the wrong basin from tepid)?
#
# Stream 200: MDs=7 trajL=√2/4 NoMetrop=10  — exact match to stuck qcd_ref
# Stream 201: MDs=7 trajL=√2/4 NoMetrop=0   — real Metropolis from t=0
# Stream 202: MDs=10 trajL=√2/4 NoMetrop=0  — finer step
# Stream 203: MDs=4 trajL=√2/8 NoMetrop=0   — short traj
#
# All: INTEGRATOR=MN2, λ_MN2=0.1789, GAUGE_INNER_MULT=4, IMPORT_CFG=chroma
# 11100, N_TRAJ=200 (will checkpoint every 10).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CHROMA_CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

if [[ ! -f "$CHROMA_CFG" ]]; then
  echo "ERROR: chroma cfg not found at $CHROMA_CFG"
  exit 1
fi

echo "=== Launching chroma-import streams ==="
echo "CHROMA_CFG=$CHROMA_CFG"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# (GPU_ID, STREAM_ID, SUFFIX, MDSTEPS, TRAJL, NO_METROP)
declare -a STREAMS=(
  "0  200  _chroma_mds7_nometr     7   0.353553390593274   10"
  "1  201  _chroma_mds7_metr       7   0.353553390593274    0"
  "2  202  _chroma_mds10_metr     10   0.353553390593274    0"
  "3  203  _chroma_mds4_metr       4   0.176776695296637    0"
)

for spec in "${STREAMS[@]}"; do
  read -r GPU SID SFX MDS TRJ NMET <<< "$spec"
  logfile="slurm-logs/qcd_chroma_s${SID}.${SLURM_JOB_ID}.out"
  echo "[stream $SID] GPU=$GPU SUFFIX=$SFX MDs=$MDS trajL=$TRJ NoMetrop=$NMET log=$logfile"

  CUDA_VISIBLE_DEVICES=$GPU STREAM_ID=$SID SUFFIX=$SFX \
    IMPORT_CFG=$CHROMA_CFG \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    MDSTEPS=$MDS TRAJL=$TRJ \
    GAUGE_INNER_MULT=4 \
    NO_METROP=$NMET \
    N_TRAJ=200 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All chroma-import streams exited ==="
date
