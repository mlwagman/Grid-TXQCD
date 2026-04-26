#!/bin/bash
#SBATCH --job-name=tx_gpuscan
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=slurm-logs/tx_gpuscan.%j.out

# TXQCD weak-scaling test at chroma physical-match MD parameters.
# Same lattice 16³×48, same chroma physical match settings, vary the GPU
# count: 1, 2, 4 GPUs.  Lattice splits along time:
#   1 GPU: --mpi 1.1.1.1, full 16³×48 per rank
#   2 GPU: --mpi 1.1.1.2, 16³×24 per rank
#   4 GPU: --mpi 1.1.1.4, 16³×12 per rank
#
# All three runs use INTEGRATOR=MinimumNorm2, LAMBDA_MN2=0.1789, MDSTEPS=7,
# TRAJL=sqrt(2)/4, GAUGE_MULT=4, AUX_MULT=2 — the chroma reference setup
# documented in production/HMC_CONVENTIONS.md.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

date
nvidia-smi --query-gpu=index,name --format=csv,noheader

run_scaling() {
  local nranks=$1 mpi=$2 cuda_devs=$3 suffix=$4 log=$5
  CUDA_VISIBLE_DEVICES=$cuda_devs LAMBDA=6.6 SUFFIX="$suffix" \
    HASEN_DM=0 AUX_SIGMA_L=-5.9 \
    MDSTEPS=7 TRAJL=0.353553390593274 \
    INTEGRATOR=MinimumNorm2 LAMBDA_MN2=0.1789 \
    GAUGE_MULT=4 AUX_MULT=2 \
    NO_METROP=0 N_TRAJ=3 \
    WEAK_FIELD_SCALE=0.05 START_TYPE=thermal \
    mpirun -np $nranks --map-by ppr:$nranks:node:PE=16 \
        ./gen_txqcd_cfgs --mpi $mpi --shm 2048 --shm-mpi 0 \
    >"$log" 2>&1
}

echo "=== 1 GPU ==="
run_scaling 1 1.1.1.1 0       _gpuscan_1   "slurm-logs/tx_gpuscan_1gpu.${SLURM_JOB_ID}.out"
echo "1 GPU done at: $(date)"

echo "=== 2 GPU ==="
run_scaling 2 1.1.1.2 0,1     _gpuscan_2   "slurm-logs/tx_gpuscan_2gpu.${SLURM_JOB_ID}.out"
echo "2 GPU done at: $(date)"

echo "=== 4 GPU ==="
run_scaling 4 1.1.1.4 0,1,2,3 _gpuscan_4   "slurm-logs/tx_gpuscan_4gpu.${SLURM_JOB_ID}.out"
echo "4 GPU done at: $(date)"

echo "=== all done ==="
date
