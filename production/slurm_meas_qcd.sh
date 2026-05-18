#!/bin/bash
#SBATCH --job-name=meas_qcd
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/meas_qcd.%j.out

# QCD measurements (conn + disc) for every config under cfgs/qcd/.
# Skip-existing logic in run_all_qcd_measurements.sh makes this restart-safe.

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

# Phase M.3 QUDA invertMultiSrcQuda for QCD measurements: 12 RHS batched via
# QUDA's native multi-source API, ~2.7× faster than per-source CG, bit-exact.
export QCD_MULTISRC="${QCD_MULTISRC:-1}"

# Chroma-style time-reversed propagator FB averaging on baryons:
# ~√2 noise reduction at plateau (verified on TXQCD lam12 cfg.60 64-src test).
# Adds 1 extra contraction per source (~5% overhead).
export QCD_TIME_REVERSED="${QCD_TIME_REVERSED:-1}"

NGPU="${NGPU:-4}"
MEAS_SCRIPT="${MEAS_SCRIPT:-run_all_qcd_measurements_4gpu.sh}"

NGPU="$NGPU" ./"$MEAS_SCRIPT"
