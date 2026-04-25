#!/bin/bash
#SBATCH --job-name=chroma_xc
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=slurm-logs/chroma_xc.%j.out

# 4-stream test isolating the contribution of each chroma convention to the
# MDs=7 trajL=sqrt(2) instability we see in Grid.  Reference chroma config
# uses LCM_STS_MIN_NORM_2 with lambda=0.1789, n_steps=7, multi-rate
# gauge sub-integrator with n_steps=4.  Same chroma cfg start, same RNG seed.
#
#   GPU 0: Grid default MN2 (lambda=0.1932)               — baseline
#   GPU 1: chroma's lambda=0.1789                          — does it help?
#   GPU 2: chroma's lambda + GAUGE_INNER_MULT=4            — match fully
#   GPU 3: chroma's lambda + halved trajL (MDs=14 dt=0.10) — convention test

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CHROMA_CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

run_test() {
  local gpu=$1 suffix=$2 mds=$3 lambda=$4 mult=$5 log=$6
  CUDA_VISIBLE_DEVICES=$gpu STREAM_ID=999 SUFFIX="$suffix" \
    IMPORT_CFG="$CHROMA_CFG" \
    MDSTEPS="$mds" TRAJL=1.4142135623730951 \
    INTEGRATOR=MinimumNorm2 NO_METROP=1 N_TRAJ=1 \
    START_TYPE=tepid WEAK_FIELD_SCALE=0.0 \
    LAMBDA_MN2="$lambda" GAUGE_INNER_MULT="$mult" \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1 &
}

run_test 0 _xc_grid_default 7 0.1931833275037836 4 "slurm-logs/chroma_xc_grid_default.${SLURM_JOB_ID}.out"
echo "[GPU 0] Grid default MN2 lambda=0.1932 MDs=7 mult=4"
sleep 2
run_test 1 _xc_chroma_lambda 7 0.1789 4 "slurm-logs/chroma_xc_chroma_lambda.${SLURM_JOB_ID}.out"
echo "[GPU 1] chroma lambda=0.1789 MDs=7 mult=4"
sleep 2
run_test 2 _xc_no_multirate 7 0.1789 1 "slurm-logs/chroma_xc_no_multirate.${SLURM_JOB_ID}.out"
echo "[GPU 2] chroma lambda=0.1789 MDs=7 mult=1 (single-level)"
sleep 2
run_test 3 _xc_mds14 14 0.1789 4 "slurm-logs/chroma_xc_mds14.${SLURM_JOB_ID}.out"
echo "[GPU 3] chroma lambda=0.1789 MDs=14 mult=4"
sleep 2

wait
echo "=== all streams exited ==="
date
