#!/bin/bash
#SBATCH --job-name=nodeE_mix
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/nodeE_mix.%j.out

# Node E:
#  GPU 0: λ=4   chroma_md20 MDS=30 — fresh from chroma cfg_11100.lime
#         dir: cfgs/txqcd_lam4.0000_fromchroma_md20
#         (REPLACES the previous fresh-chroma MDS=10 attempt which failed
#          catastrophically — fresh-chroma-MDS=10 at λ=4 was too aggressive
#          for the large aux fluctuations ∝1/λ².  Run MDS=30 for ~50 trajs
#          to thermalize, then fork to MDS=10 fork_t50 like λ=5/7/7.5/8.)
#  GPU 1: λ=5   CONTINUE chroma fork-t50 MDS=10
#         dir: cfgs/txqcd_lam5.0000_fromchroma_md20_fork_t50_mds10
#  GPU 2: λ=6   CONTINUE chroma_md20 MDS=30
#         dir: cfgs/txqcd_lam6.0000_fromchroma_md20  (resume from cfg.50)
#  GPU 3: λ=6.5 CONTINUE chroma_md20 MDS=30
#         dir: cfgs/txqcd_lam6.5000_fromchroma_md20  (resume from cfg.70)
#
# Combines: 2 streams that broke at MDS=10 (λ=6, 6.5) — fall back to MDS=30
# 1 new test (λ=4 fresh chroma MDS=10)
# 1 working continuation (λ=5 fork-t50)

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

N_TRAJ=${N_TRAJ:-1200}
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

echo "=== Node E mixed: λ=4 fresh, λ=5 fork-t50, λ=6/6.5 MDS=30 ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

# -----------------------------------------------------------------
# GPU 0: λ=4 forked from chroma-pedigree λ=5 cfg.230 at MDS=10.  Replaces
# the failed _fromchroma_md20 MDS=30 attempt (chroma σ=0 → λ=4 σ_aa=0.042
# is too aggressive a perturbation, all trajs rejected).  Forking from
# λ=5 _fromchroma_md20_fork_t50_mds10 cfg.230 instead: that cfg is
# chroma-pedigree, deeply thermalized at plaq=0.5135, with σ≈0.245 already
# nonzero so the λ=4 chain only has to adjust σ magnitude, not build it up
# from zero.  Seed in cfgs/txqcd_lam4.0000_from_lam5_mds10/ckpoint_lat.230.
# -----------------------------------------------------------------
logfile="slurm-logs/nodeE_lam4_from_lam5.${SLURM_JOB_ID}.out"
echo "[gpu 0] λ=4 from λ=5 cfg.230 MDS=10 log=$logfile"
CUDA_VISIBLE_DEVICES=0 \
    LAMBDA=4 \
    SUFFIX="_from_lam5_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
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
# GPU 1: λ=5 chroma fork-t50 MDS=10 continue
# -----------------------------------------------------------------
logfile="slurm-logs/nodeE_lam5_fork_t50.${SLURM_JOB_ID}.out"
echo "[gpu 1] λ=5 fork-t50 MDS=10 continue log=$logfile"
CUDA_VISIBLE_DEVICES=1 \
    LAMBDA=5 \
    SUFFIX="_fromchroma_md20_fork_t50_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
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
# GPU 2: λ=6 chroma_md20_fork_t50_mds10 continue (MDS=10).  Switched from
# the slow MDS=30 _fromchroma_md20 once the MDS=10 fork was confirmed
# stable (~70 trajs at plaq=0.514, no rejections).  2.3× faster per traj
# than MDS=30 with same physics.
# -----------------------------------------------------------------
logfile="slurm-logs/nodeE_lam6_fork_t50.${SLURM_JOB_ID}.out"
echo "[gpu 2] λ=6 _fromchroma_md20_fork_t50_mds10 MDS=10 log=$logfile"
CUDA_VISIBLE_DEVICES=2 \
    LAMBDA=6 \
    SUFFIX="_fromchroma_md20_fork_t50_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
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
# GPU 3: λ=6.5 chroma_md20_fork_t50_mds10 continue (MDS=10).  Same logic
# as GPU 2.  ~70 trajs already, plaq=0.5136.
# -----------------------------------------------------------------
logfile="slurm-logs/nodeE_lam6.5_fork_t50.${SLURM_JOB_ID}.out"
echo "[gpu 3] λ=6.5 _fromchroma_md20_fork_t50_mds10 MDS=10 log=$logfile"
CUDA_VISIBLE_DEVICES=3 \
    LAMBDA=6.5 \
    SUFFIX="_fromchroma_md20_fork_t50_mds10" \
    N_TRAJ="$N_TRAJ" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    AUX_MULT=4 \
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
echo "=== node E mixed done ==="
date
