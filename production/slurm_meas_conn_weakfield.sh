#!/bin/bash
#SBATCH --job-name=meas_conn_weakfield
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=slurm-logs/meas_conn_weakfield.%j.out

# Connected pion + nucleon correlator comparison on three setups:
#   GPU 0 (run A): IMPORT_CFG=chroma cfg → QCD-on-chroma + TXQCD-with-auto-aux-on-chroma
#   GPU 1 (run B): LOAD_TXQCD_CKPT=mds20:10 → TXQCD on evolved gauge+aux (MDs=20 chain)
#   GPU 2 (run C): LOAD_TXQCD_CKPT=mds15:10 → TXQCD on evolved gauge+aux (MDs=15 chain)
# All at λ=7, 16³×48, Nf=2+1 TXQCD operator.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

CFG=/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime

echo "=== launch A: chroma cfg + auto-Σ aux init ==="
CUDA_VISIBLE_DEVICES=0 LAMBDAS="7" IMPORT_CFG=$CFG \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measA_chroma_init.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

echo "=== launch B: TXQCD evolved (MDs=20 ckpt 10) ==="
CUDA_VISIBLE_DEVICES=1 LAMBDAS="7" \
    LOAD_TXQCD_CKPT=cfgs/txqcd_lam7.0000_nf2p1_l7_chromastart_mds20:10 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measB_txqcd_mds20_t10.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

echo "=== launch C: TXQCD evolved (MDs=15 ckpt 10) ==="
CUDA_VISIBLE_DEVICES=2 LAMBDAS="7" \
    LOAD_TXQCD_CKPT=cfgs/txqcd_lam7.0000_nf2p1_l7_chromastart_mds15:10 \
    mpirun -np 1 --map-by ppr:1:socket:PE=16 \
    ./meas_conn_weakfield_nf2 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    > slurm-logs/measC_txqcd_mds15_t10.${SLURM_JOB_ID}.out 2>&1 &
sleep 2

wait
echo "=== all done ==="
date
