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

MPI="${MPI:---mpi 1.1.1.4}"
SHM="${SHM:---shm 2048 --shm-mpi 0}"
NP="${NP:-4}"

GRID_LAUNCH="mpirun -np $NP ./select_gpu.sh" ./run_all_qcd_measurements.sh $MPI $SHM
