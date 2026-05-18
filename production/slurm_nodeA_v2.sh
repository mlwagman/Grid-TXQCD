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
#  GPU 1: QCD MDS=30 weak-field continue   (cfgs/qcd_s702_nf2p1_mdscan_mds30/  cfg.100→)
#  GPU 2: λ=10 MDS=10 fork continue        (cfgs/txqcd_lam10.0000_nf2p1_mds10_fork/)
#  GPU 3: λ=7  MDS=10 fork continue        (cfgs/txqcd_lam7.0000_nf2p1_mds10_fork/)
#
# Replaces fork_md10A (which had broken λ=6/6.5) + part of fork_md10B
# (λ=10 moved here; λ=8, 12 discontinued — chroma-source equivalents still
# running on 1276491/1276500).

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs
source ../env_lq2_grid.sh
export OMP_NUM_THREADS=16

N_TRAJ=${N_TRAJ:-1200}
CHROMA_CFG="/lustre2/nplqcd/agrebe/cfgs/cl3_16_48_b6p1_m0p2450/cl3_16_48_b6p1_m0p2450_a_cfg_11100.lime"

echo "=== Node A v2: λ=5 fork, QCD Nf=3 chroma test, λ=10 fork, λ=7 fork ==="
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
# GPU 1: QCD Nf=3 (Nf=1+1+1) chroma-start test.  Replaces the prior
# gen_qcd_cfgs_2plus1 fork_v2 stream, which used TwoFlavour-Schur PF for
# the light pair — the documented near-zero-mode force-amplification
# pathology (project_nf2p1_vs_nf3_basin) that biased its plaq high
# (equilibrated 0.5145 vs chroma 0.5138).  gen_qcd_cfgs.cc uses the
# chroma-exact structure: QCDLogDetCloverEOAction(FermOp,3) [=chroma
# N_FLAVOR_LOGDET_EVEN_EVEN num_flavors=3] + OneFlavour rational Schur
# [=chroma ONE_FLAVOR_EOPREC_CONSTDET_RAT].  Fresh start from the chroma
# reference cfg to test directly whether the Nf=3 structure holds 0.5138.
# New SUFFIX → its own cfg dir (does NOT touch the biased fork_v2 cfgs).
logfile="slurm-logs/nodeA_v2_qcd_nf3_chroma.${SLURM_JOB_ID}.out"
echo "[gpu 1] QCD Nf=3 chroma-start (gen_qcd_cfgs) log=$logfile"
CUDA_VISIBLE_DEVICES=1 \
    STREAM_ID=702 \
    SUFFIX="_nf3_fromchroma_mds10" \
    N_TRAJ="$N_TRAJ" \
    IMPORT_CFG="$CHROMA_CFG" \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.353553390593274 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=4 \
    HASEN_DM=0 \
    NO_METROP=0 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    mpirun -np 1 --bind-to none --map-by ppr:1:socket:PE=16 \
        -x STREAM_ID -x SUFFIX -x N_TRAJ -x IMPORT_CFG -x INTEGRATOR \
        -x LAMBDA_MN2 -x MDSTEPS -x TRAJL -x GAUGE_MULT -x GAUGE_INNER_MULT \
        -x HASEN_DM -x NO_METROP -x QUDA_FORCE -x QUDA_FORCE_KERNEL \
        -x EIG_DIAG -x QUDA_ENABLE_MPS \
        ./gen_qcd_cfgs --mpi 1.1.1.1 --shm 2048 --shm-mpi 0 \
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
echo "=== node A v2 done ==="
date
