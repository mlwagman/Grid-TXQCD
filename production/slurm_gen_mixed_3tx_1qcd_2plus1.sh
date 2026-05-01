#!/bin/bash
#SBATCH --job-name=mixed_3t1q
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/mixed_3t1q.%j.out

# Three parallel TXQCD streams + one QCD reference stream on a single 4-GPU
# node.  TXQCD lambdas come from LAMBDAS env var (3 values, space-separated);
# QCD always lands on the last GPU.  All four streams use chroma physical
# match settings (MN2, MDsteps=7, trajL=sqrt(2)/4, lambda_MN2=0.1789,
# GAUGE_MULT=4 (or GAUGE_INNER_MULT=4 for QCD), AUX_MULT=2 for TXQCD).
#
# Example:
#   sbatch --export=ALL,LAMBDAS="9 12 18" slurm_gen_mixed_3tx_1qcd.sh
#
# Forwards env vars: MDSTEPS, TRAJL, INTEGRATOR, LAMBDA_MN2, GAUGE_MULT,
# AUX_MULT, WEAK_FIELD_SCALE, NO_METROP, START_TYPE, HASEN_DM, SUFFIX,
# AUX_SIGMA_L, N_TRAJ.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

LAMBDAS=${LAMBDAS:-"9 12 18"}
read -r -a LAMBDA_ARR <<< "$LAMBDAS"
N_TX=${#LAMBDA_ARR[@]}
if [ "$N_TX" -ne 3 ]; then
  echo "ERROR: expected 3 TXQCD lambdas, got $N_TX from LAMBDAS=\"$LAMBDAS\""
  exit 1
fi

echo "=== Launching 3 TXQCD + 1 QCD parallel single-GPU streams ==="
echo "TXQCD lambdas: ${LAMBDAS}"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in 0 1 2; do
  LAM="${LAMBDA_ARR[$i]}"
  logfile="slurm-logs/txqcd_lam${LAM}.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  TXQCD LAMBDA=$LAM  log=$logfile"
  CUDA_VISIBLE_DEVICES=$i LAMBDA=$LAM \
      MDSTEPS="${MDSTEPS-}" AUX_MULT="${AUX_MULT-}" GAUGE_MULT="${GAUGE_MULT-}" \
      WEAK_FIELD_SCALE="${WEAK_FIELD_SCALE-}" NO_METROP="${NO_METROP-}" \
      START_TYPE="${START_TYPE-}" HASEN_DM="${HASEN_DM-}" SUFFIX="${SUFFIX-}" \
      AUX_SIGMA_L="${AUX_SIGMA_L-}" N_TRAJ="${N_TRAJ-}" \
      INTEGRATOR="${INTEGRATOR-}" TRAJL="${TRAJL-}" LAMBDA_MN2="${LAMBDA_MN2-}" \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

# QCD reference stream on GPU 3
qcd_log="slurm-logs/qcd_ref.${SLURM_JOB_ID}.out"
echo "[stream 3] GPU=3  QCD reference  log=$qcd_log"
# gen_qcd_cfgs uses GAUGE_INNER_MULT (not GAUGE_MULT) and ignores AUX_*.
GAUGE_INNER_MULT_VAL="${GAUGE_INNER_MULT-${GAUGE_MULT-4}}"
CUDA_VISIBLE_DEVICES=3 \
    SUFFIX="${QCD_SUFFIX-_ref}" \
    MDSTEPS="${MDSTEPS-}" GAUGE_INNER_MULT="$GAUGE_INNER_MULT_VAL" \
    WEAK_FIELD_SCALE="${WEAK_FIELD_SCALE-}" NO_METROP="${NO_METROP-}" \
    START_TYPE="${START_TYPE-}" N_TRAJ="${N_TRAJ-}" \
    INTEGRATOR="${INTEGRATOR-}" TRAJL="${TRAJL-}" LAMBDA_MN2="${LAMBDA_MN2-}" \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
        ./gen_qcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$qcd_log" 2>&1 &

wait

echo "=== All four streams exited ==="
date
