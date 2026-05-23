#!/bin/bash
#SBATCH --job-name=nodeA_v2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/nodeA_v2.%j.out

# Node A successor: 4 streams of mixed type.
#  GPU 0: λ=5  MDS=10 fork continue        (cfgs/txqcd_lam5.0000_nf2p1_mds10_fork/)
#  GPU 1: λ=6.5 MDS=30 reference continue  (cfgs/txqcd_lam6.5000_fromchroma_md30/)
#  GPU 2: λ=10 MDS=10 fork continue        (cfgs/txqcd_lam10.0000_nf2p1_mds10_fork/)
#  GPU 3: λ=7  MDS=10 fork continue        (cfgs/txqcd_lam7.0000_nf2p1_mds10_fork/)
#
# 2026-05-22 swap: GPU 1 changed from Nf=3 QCD (qcd_s702_nf3_fromchroma_mds10,
# completed at ckpt 880, plaq 0.51329 = chroma ✓) to λ=6.5 MDS=30 reference.
# This reference stream remains in service while the new MDS=10 forks
# (slurm_lam6_6p5_md30_lam7_5_md10.sh GPU 0/1) take the primary production
# slots — having a parallel MDS=30 stream lets us cross-check the
# integrator-bias assumption at fixed λ.

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

N_TRAJ=${N_TRAJ:-2000}
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

echo "=== Node A v2: λ=5 fork, λ=6.5 md30 ref, λ=10 fork, λ=7 fork ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# -----------------------------------------------------------------
# GPU 0: λ=5 MDS=10 fork
# -----------------------------------------------------------------
logfile="slurm-logs/nodeA_v2_lam5.${SLURM_JOB_ID}.out"
echo "[gpu 0] λ=5 fork log=$logfile"
CUDA_VISIBLE_DEVICES=0 \
    LAMBDA=5 \
    SUFFIX="_nf2p1_mds10_fork" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

# -----------------------------------------------------------------
# GPU 1: λ=6.5 MDS=30 reference continuation.  Was QCD Nf=3 chroma-start
# (qcd_s702_nf3_fromchroma_mds10, completed at ckpt 880 with plaq 0.51329
# matching chroma 0.51365 within ~1σ — Nf=3 plaq-offset saga closed).
# Now repurposed to keep one MDS=30 stream alive at λ=6.5 as integrator
# cross-check while the new MDS=10 forks (slurm_lam6_6p5_md30_lam7_5_md10.sh
# GPU 0/1) take the primary production slots.  Same dir as before
# (cfgs/txqcd_lam6.5000_fromchroma_md30/, latest ckpt 180); resumes via
# checkpoint + aux sidecar + RNG.
logfile="slurm-logs/nodeA_v2_lam6p5_md30ref.${SLURM_JOB_ID}.out"
echo "[gpu 1] λ=6.5 MDS=30 reference log=$logfile"
CUDA_VISIBLE_DEVICES=1 \
    LAMBDA=6.5 \
    SUFFIX="_fromchroma_md30" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=30 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

# -----------------------------------------------------------------
# GPU 2: λ=10 MDS=10 fork continue (moved from old node B)
# -----------------------------------------------------------------
logfile="slurm-logs/nodeA_v2_lam10.${SLURM_JOB_ID}.out"
echo "[gpu 2] λ=10 fork log=$logfile"
CUDA_VISIBLE_DEVICES=2 \
    LAMBDA=10 \
    SUFFIX="_nf2p1_mds10_fork" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

# -----------------------------------------------------------------
# GPU 3: λ=7 MDS=10 fork continue
# -----------------------------------------------------------------
logfile="slurm-logs/nodeA_v2_lam7.${SLURM_JOB_ID}.out"
echo "[gpu 3] λ=7 fork log=$logfile"
CUDA_VISIBLE_DEVICES=3 \
    LAMBDA=7 \
    SUFFIX="_nf2p1_mds10_fork" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        ./gen_txqcd_cfgs_2plus1 --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
    >"$logfile" 2>&1 &
sleep 2

wait
echo "=== node A v2 done ==="
date
