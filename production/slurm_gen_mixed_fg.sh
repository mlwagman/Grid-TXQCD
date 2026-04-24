#!/bin/bash
#SBATCH --job-name=mixed_fg
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/mixed_fg.%j.out

# One-node mixed: 1 QCD stream (STREAM_ID=300) + 3 TXQCD streams
# (LAMBDAS="3.5 12.0 24.0") all using the ForceGradient integrator at
# MDsteps=10, trajL=0.5 so their per-trajectory dH and accept rate can be
# compared head-to-head.  Matches the ForceGradient settings of txqcd_4s
# 1270983 / 1270995.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16

# Shared integrator settings (can override via sbatch --export=...).
: "${INTEGRATOR:=ForceGradient}"
: "${MDSTEPS:=10}"
: "${TRAJL:=0.5}"
: "${AUX_MULT:=16}"
: "${WEAK_FIELD_SCALE:=0.1}"
: "${NO_METROP:=100}"

echo "=== Launching QCD + TXQCD mixed comparison on one node ==="
echo "INTEGRATOR=$INTEGRATOR MDSTEPS=$MDSTEPS TRAJL=$TRAJL AUX_MULT=$AUX_MULT"
echo "WEAK_FIELD_SCALE=$WEAK_FIELD_SCALE NO_METROP=$NO_METROP"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Stream 0: QCD with STREAM_ID=300 (fresh cfgs/qcd_s300/)
qlog="slurm-logs/qcd_s300_fg.${SLURM_JOB_ID}.out"
echo "[stream 0] GPU=0  QCD STREAM_ID=300  log=$qlog"
CUDA_VISIBLE_DEVICES=0 STREAM_ID=300 \
    MDSTEPS="$MDSTEPS" TRAJL="$TRAJL" INTEGRATOR="$INTEGRATOR" \
    WEAK_FIELD_SCALE="$WEAK_FIELD_SCALE" NO_METROP="$NO_METROP" \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$qlog" 2>&1 &
sleep 2

# Streams 1-3: TXQCD at λ = 3.5, 12.0, 24.0
LAMBDA_ARR=(3.5 12.0 24.0)
for i in 0 1 2; do
  gpu=$((i+1))
  LAM="${LAMBDA_ARR[$i]}"
  tlog="slurm-logs/txqcd_lam${LAM}_fg.${SLURM_JOB_ID}.out"
  echo "[stream $((i+1))] GPU=$gpu  TXQCD LAMBDA=$LAM  log=$tlog"
  CUDA_VISIBLE_DEVICES=$gpu LAMBDA=$LAM \
      MDSTEPS="$MDSTEPS" TRAJL="$TRAJL" INTEGRATOR="$INTEGRATOR" \
      AUX_MULT="$AUX_MULT" WEAK_FIELD_SCALE="$WEAK_FIELD_SCALE" \
      NO_METROP="$NO_METROP" \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$tlog" 2>&1 &
  sleep 2
done

wait

echo "=== All streams exited ==="
date
