#!/bin/bash
#SBATCH --job-name=qcd_basintest
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/qcd_basintest.%j.out

# Basin test: do we reproduce the qcd_ref plaq=0.5337 stuck basin under
# different MD settings, or does the chain find chroma's plaq=0.5138?
#
# Stream 100: MDs=7 trajL=√2/4 NoMetrop=10  — exact match to stuck qcd_ref
# Stream 101: MDs=7 trajL=√2/4 NoMetrop=0   — real Metropolis from t=0
# Stream 102: MDs=10 trajL=√2/4 NoMetrop=0  — finer step
# Stream 103: MDs=4 trajL=√2/8 NoMetrop=0   — short traj
#
# All tepid start, WEAK_FIELD_SCALE=0.1, INTEGRATOR=MN2, λ_MN2=0.1789,
# GAUGE_INNER_MULT=4.  Each stream writes to its own cfg dir; we leave
# the running production qcd_ref / qcd_s* alone.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

echo "=== Launching basintest streams ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Per-stream settings: (GPU_ID, STREAM_ID, SUFFIX, MDSTEPS, TRAJL, NO_METROP)
declare -a STREAMS=(
  "0  100  _basintest_mds7_nometr   7   0.353553390593274   10"
  "1  101  _basintest_mds7_metr     7   0.353553390593274    0"
  "2  102  _basintest_mds10_metr   10   0.353553390593274    0"
  "3  103  _basintest_mds4_metr     4   0.176776695296637    0"
)

for spec in "${STREAMS[@]}"; do
  read -r GPU SID SFX MDS TRJ NMET <<< "$spec"
  logfile="slurm-logs/qcd_basintest_s${SID}.${SLURM_JOB_ID}.out"
  echo "[stream $SID] GPU=$GPU SUFFIX=$SFX MDs=$MDS trajL=$TRJ NoMetrop=$NMET log=$logfile"

  CUDA_VISIBLE_DEVICES=$GPU STREAM_ID=$SID SUFFIX=$SFX \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    MDSTEPS=$MDS TRAJL=$TRJ \
    GAUGE_INNER_MULT=4 \
    NO_METROP=$NMET \
    START_TYPE=tepid WEAK_FIELD_SCALE=0.1 \
    N_TRAJ=200 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All basintest streams exited ==="
date
