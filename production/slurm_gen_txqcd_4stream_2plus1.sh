#!/bin/bash
#SBATCH --job-name=txqcd_4s
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/txqcd_4s.%j.out

# Four parallel NP=1 TXQCD HMC streams, one per GPU on an exclusive node.
# Each stream uses a different LAMBDA (auxiliary coupling) for the lambda
# scan.  Streams write to cfgs/txqcd_lam<X.XXXX>/ with RNG seeds derived
# from LAMBDA itself.
#
# LAMBDAS env var: space-separated list of 4 lambda values; one per GPU.
# Default: 1.5 2.5 2.75 3.0
# Example:
#   sbatch --export=ALL,LAMBDAS="1.5 2.5 2.75 3.0" slurm_gen_txqcd_4stream.sh
#   sbatch --export=ALL,LAMBDAS="3.25 3.5 4.0 6.0" slurm_gen_txqcd_4stream.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
LAMBDAS=${LAMBDAS:-"1.5 2.5 2.75 3.0"}
read -r -a LAMBDA_ARR <<< "$LAMBDAS"
N_STREAMS=${#LAMBDA_ARR[@]}

echo "=== Launching $N_STREAMS parallel NP=1 TXQCD HMC streams ==="
echo "LAMBDAS: ${LAMBDAS}"
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

for i in $(seq 0 $((N_STREAMS-1))); do
  LAM="${LAMBDA_ARR[$i]}"
  logfile="slurm-logs/txqcd_lam${LAM}.${SLURM_JOB_ID}.out"
  echo "[stream $i] GPU=$i  LAMBDA=$LAM  log=$logfile"
  # mpirun -np 1 --map-by ppr:1:socket:PE=16 : working binding pattern for
  # 4 parallel single-GPU streams on an lq2_gpu node (see
  # production/slurm_gen_qcd_4stream.sh and the lq2-mpi reference memory).
  # Forward tunable env vars (MDSTEPS, AUX_MULT, GAUGE_MULT, WEAK_FIELD_SCALE,
  # NO_METROP, START_TYPE) through the prefix so mpirun inherits them.  Unset
  # vars fall through to the binary's defaults.
  CUDA_VISIBLE_DEVICES=$i LAMBDA=$LAM \
      MDSTEPS="${MDSTEPS-}" AUX_MULT="${AUX_MULT-}" GAUGE_MULT="${GAUGE_MULT-}" \
      WEAK_FIELD_SCALE="${WEAK_FIELD_SCALE-}" NO_METROP="${NO_METROP-}" \
      START_TYPE="${START_TYPE-}" HASEN_DM="${HASEN_DM-}" SUFFIX="${SUFFIX-}" \
      AUX_SIGMA_L="${AUX_SIGMA_L-}" N_TRAJ="${N_TRAJ-}" \
      INTEGRATOR="${INTEGRATOR-}" TRAJL="${TRAJL-}" LAMBDA_MN2="${LAMBDA_MN2-}" \
      IMPORT_CFG="${IMPORT_CFG-}" \
      mpirun -np 1 --map-by ppr:1:socket:PE=16 \
          ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
      >"$logfile" 2>&1 &
  sleep 2
done

wait

echo "=== All streams exited ==="
date
