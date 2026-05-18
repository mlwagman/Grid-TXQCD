#!/bin/bash
#SBATCH --job-name=meas_txqcd
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_txqcd.%j.out

# TXQCD measurements (conn + disc + aux) for every config under
# cfgs/txqcd_lam<X.XXXX>/.  Lambda defaults to 0.5000 (matching params.h).
# Override with: sbatch --export=ALL,LAMBDA=0.5000 slurm_meas_txqcd.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

export OMP_NUM_THREADS=16
export OMPI_MCA_btl=^uct,openib
export UCX_TLS=cuda,gdr_copy,rc,rc_x,sm,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=16384
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_MEMTYPE_CACHE=n

# Phase M.4.b fused multi-RHS Mooee on the TXQCD Schur EO operator: ~6×
# faster vs the legacy full-volume MdagM CG, bit-equivalent to ~1e-10.
# Disable by setting TXQCD_MULTIRHS_CG=0 in the submit env if needed.
export TXQCD_MULTIRHS_CG="${TXQCD_MULTIRHS_CG:-1}"
export TXQCD_MOOEE_CUBLAS="${TXQCD_MOOEE_CUBLAS:-1}"
export TXQCD_MOOEEINV_CUBLAS="${TXQCD_MOOEEINV_CUBLAS:-1}"
export TXQCD_PRECOMPUTE_GPU="${TXQCD_PRECOMPUTE_GPU:-1}"

# Chroma-style time-reversed propagator FB averaging on baryons:
# ~√2 noise reduction at plateau (verified on lam12 cfg.60 64-src test).
# Adds 1 extra contraction per source (~5% overhead on top of fused-CG).
export TXQCD_TIME_REVERSED="${TXQCD_TIME_REVERSED:-1}"

LAMBDA="${LAMBDA:-0.5000}"
NGPU="${NGPU:-4}"

# 4-GPU multi-cfg parallelism: each cfg measurement is single-GPU
# (--mpi 1.1.1.1) and 4 cfgs run concurrently, one per GPU.  ~4× throughput
# vs the spatially-decomposed run_all_txqcd_measurements.sh on this volume.
# Set MEAS_SCRIPT=run_all_txqcd_measurements.sh to revert to single-job mode.
MEAS_SCRIPT="${MEAS_SCRIPT:-run_all_txqcd_measurements_4gpu.sh}"

LAMBDA="$LAMBDA" NGPU="$NGPU" ./"$MEAS_SCRIPT"
