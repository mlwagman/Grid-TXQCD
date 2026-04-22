#!/bin/bash
#SBATCH --job-name=bench_qcd
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/bench_qcd_%x.%j.out

# Timing benchmark. Submit with:
#   sbatch --ntasks-per-node=NP --gres=gpu:NP --cpus-per-task=CPT \
#          --export=ALL,NP=NP,MPI="--mpi mx.my.mz.mt",TAG="1.1.1.4" slurm_bench_qcd.sh
#
# Uses srun --mpi=pmix (the launcher the lq2_gpu OpenMPI build supports).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs

source ../env_lq2_grid.sh

# Following NPLQCD chroma production convention: leave headroom for GPU driver.
export OMP_NUM_THREADS=8
export OMPI_MCA_btl=^uct,openib
export UCX_TLS=cuda,gdr_copy,rc,rc_x,sm,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=16384
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_MEMTYPE_CACHE=n

NP="${NP:-${SLURM_NTASKS:-1}}"
CPT="${SLURM_CPUS_PER_TASK:-16}"
MPI="${MPI:---mpi 1.1.1.1}"
SHM="${SHM:---shm 2048 --shm-mpi 0}"
TAG="${TAG:-np${NP}}"
# BIN selects the gen program (default: Nf=2+1). Examples:
#   BIN=gen_qcd_cfgs       (Nf=2+1, writes cfgs/qcd)
#   BIN=gen_qcd_nf3_cfgs   (Nf=3 degenerate, writes cfgs/qcd_nf3)
BIN="${BIN:-gen_qcd_cfgs}"

BENCH_DIR="cfgs/qcd_bench_${TAG}"
WORK="bench_work_${TAG}"
rm -rf "$BENCH_DIR" "$WORK" 2>/dev/null
mkdir -p "$BENCH_DIR" "$WORK/cfgs" "$WORK/logs"
# Point both possible cfg dirs to the same bench location so either binary works.
ln -s "$PWD/$BENCH_DIR" "$WORK/cfgs/qcd"
ln -s "$PWD/$BENCH_DIR" "$WORK/cfgs/qcd_nf3"

cd "$WORK"
echo "=== bench_qcd TAG=$TAG  NP=$NP  MPI=$MPI  OMP=$OMP_NUM_THREADS  BIN=$BIN ==="
time srun -N1 -n $NP -c $CPT --mpi=pmix ../select_gpu.sh ../$BIN $MPI $SHM 2>&1 | tee logs/${BIN}.log
