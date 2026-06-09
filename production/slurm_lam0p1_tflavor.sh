#!/bin/bash
#SBATCH --job-name=lam0p1_tflavor
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/lam0p1_tflavor.%j.out

# λ=0.1 weak-field, flavor-t variant (TXQCD_T_FLAVOR=1 built into binary).
# Probes whether flavor-t shifts λ_crit downward, with the goal of
# simulating low λ before the t SSB locks ergodicity.  Compare against
# the existing color-t λ=0.1 baseline (txqcd_lam0.1000_lowlam_weak_md10,
# plaq=0.453, t condensate at 50-71σ — see
# memory/project_t_condensate_low_lambda.md).
#
# Same HMC recipe as the color-t low-λ chains; the binary swap is the
# only difference.  AUX_FLUCT_LAMBDA=3 (init width 1/3) keeps the t
# block tight at start; in flavor-t mode t lives in an Nf x Nf block
# with Nf=2, so the same FLUCT setting applies symmetrically.
#
# SUFFIX="_tflavor" → fresh cfg dir cfgs/txqcd_lam0.1000_tflavor/.

source /lustre2/nplqcd/Grid-TXQCD/env_lq2_grid.sh
cd "$PRODUCTION_DIR"
mkdir -p slurm-logs

export OMP_NUM_THREADS=16
N_TRAJ=${N_TRAJ:-2000}
SUFFIX="_tflavor"
NO_METROP=0
echo "[setup] NO_METROP=$NO_METROP  TXQCD_T_FLAVOR=1 (flavor-t binary)"

echo "=== Launching TXQCD λ=0.1 flavor-t stream (4 ranks × 4 GPUs, MDS=10) ==="
date
nvidia-smi --query-gpu=index,name --format=csv,noheader

LAM=0.1
logfile="slurm-logs/txqcd_lam${LAM}_tflavor.${SLURM_JOB_ID}.out"
echo "[stream] GPUs=0,1,2,3  TXQCD λ=$LAM (flavor-t)  log=$logfile"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    LAMBDA=$LAM \
    SUFFIX="$SUFFIX" \
    N_TRAJ=$N_TRAJ \
    INTEGRATOR=MinimumNorm2 \
    LAMBDA_MN2=0.1789 \
    MDSTEPS=10 \
    TRAJL=0.088388347648318 \
    RAT_LO=1e-5 \
    RAT_DEGREE=20 \
    GAUGE_MULT=4 \
    GAUGE_INNER_MULT=2 \
    AUX_MULT=1 \
    HASEN_DM=0 \
    NO_METROP=$NO_METROP \
    WEAK_FIELD_SCALE=0.1 \
    AUX_FLUCT_LAMBDA=3 \
    QUDA_FORCE=1 \
    QUDA_FORCE_KERNEL=1 \
    TXQCD_QUDA_HYBRID=1 \
    TXQCD_QUDA_FULL=1 \
    TXQCD_PRECOMPUTE_GPU=1 \
    TXQCD_MOOEEINV_CUBLAS=1 \
    TXQCD_MOOEE_CUBLAS=1 \
    EIG_DIAG=1 \
    QUDA_ENABLE_MPS=1 \
    QUDA_ENABLE_DEVICE_MEMORY_POOL=0 \
    QUDA_ENABLE_MANAGED_MEMORY=1 \
    srun --overlap --mpi=pmix -N 1 -n 4 \
         --cpus-per-task=16 --cpu-bind=none \
         --gres=gpu:4 \
        ./gen_txqcd_cfgs_2plus1_tflavor --mpi 1.1.1.4 --shm 512 --shm-mpi 1 \
        > "$logfile" 2>&1

echo "=== lam0p1_tflavor done ==="
date
