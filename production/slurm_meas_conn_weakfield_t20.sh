#!/bin/bash
#SBATCH --job-name=meas_t20
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/meas_t20.%j.out

# Meas comparison after 10 more HMC trajectories from chroma cfg (Nf=2+1, λ=7).
#   GPU 0 (run A):  IMPORT_CFG=chroma cfg → QCD-on-chroma + TXQCD-on-chroma+auto-aux
#   GPU 1 (run T10): MDs=15 ckpt 10 (anchor — was the first data point)
#   GPU 2 (run T20): MDs=15 ckpt 20 (new — after +10 HMC trajs)
#   GPU 3 (idle, or future MDs=20 ckpt 20 once it lands)

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

echo "=== launch A: chroma cfg + auto-Σ aux init ==="
CUDA_VISIBLE_DEVICES=0 LAMBDAS="7" IMPORT_CFG=$CFG \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measA_chroma_init.t20.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

echo "=== launch T10: MDs=15 ckpt 10 (anchor) ==="
CUDA_VISIBLE_DEVICES=1 LAMBDAS="7" \
    LOAD_TXQCD_CKPT=cfgs/txqcd_lam7.0000_nf2p1_l7_chromastart_mds20:10 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measT10_mds20.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

echo "=== launch T20: MDs=15 ckpt 20 (NEW — after +10 HMC trajs) ==="
CUDA_VISIBLE_DEVICES=2 LAMBDAS="7" \
    LOAD_TXQCD_CKPT=cfgs/txqcd_lam7.0000_nf2p1_l7_chromastart_mds20:20 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measT20_mds20.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

wait
echo "=== all done ==="
date
