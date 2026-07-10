#!/bin/bash
#SBATCH --job-name=meas_bar_mg_2c
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/meas_bar_mg_2c.%j.out
# ============================================================================
# TWO configs/node: two independent lambda<=3 nb64 GridMG baryon measurements
# side by side (each 1.1.1.2 = 2 GPUs).  Throughput vehicle -- the bench
# proved 2x nb64 @1.1.1.2 run concurrently at ~727 s each (<1% contention);
# real wall-source baryon solves land ~360 s (job 1299828).  Config A on GPUs
# 0-1 (GPU_OFFSET=0), config B on GPUs 2-3 (GPU_OFFSET=2), both with
# QUDA_ENABLE_MPS=1 (mandatory for the multi-rank per-rank-CUDA_VISIBLE_DEVICES
# path -- see slurm_meas_baryon_gridmg_lowlam.sh for the binding+MPS rationale).
#
# Params (--export): LAM(3), TRAJ_A, TRAJ_B, NTSRC(8), CFG_DIR, DATA_DIR.
# Each config regenerates its own per-traj nb64 basis (~50 min ord1500) unless
# a matching SETUP_PREFIX_{A,B} cache exists.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
cd $P

LAM=${LAM:-3}
TRAJ_A=${TRAJ_A:?set TRAJ_A}
TRAJ_B=${TRAJ_B:?set TRAJ_B}
NTSRC=${NTSRC:-8}
CFG_DIR=${CFG_DIR:-$P/cfgs/dtxqcd_lam${LAM}.0000_noneo_fromchroma_cw_mds10}
DATA_DIR=${DATA_DIR:-$P/meas_2pt/dtxqcd_lam${LAM}.0000_gridmg}
SETUPS=$P/gridmg_setups
WRAPO=$P/srun_gpu_wrapper_offset.sh
mkdir -p $DATA_DIR

if [ "$LAM" -ge 4 ] 2>/dev/null; then
  echo "REFUSING: lambda=$LAM >= 4 uses the EO default, not GridMG." ; exit 2
fi

COMMON="LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFG_DIR DATA_DIR=$DATA_DIR MEAS_BARYON_NTSRC=$NTSRC CG_TOL=1e-8 QUDA_ENABLE_MPS=1 DTXQCD_MEAS_GRIDMG=1 DTXQCD_MEAS_MG_NBASIS=64 DTXQCD_MEAS_MG_GEOM=nn DTXQCD_MEAS_MG_ORD=1500 DTXQCD_MEAS_MG_REFINE_MAXIT=400 DTXQCD_MEAS_MG_NEV=96 COARSE_IRL_MAXIT=30"
GRID="--grid 16.16.16.48 --mpi 1.1.1.2 --shm 512 --shm-mpi 1"

echo "=== 2 configs/node  lambda=$LAM  A=traj$TRAJ_A (GPU 0-1)  B=traj$TRAJ_B (GPU 2-3) $(date) ==="

srun --overlap --mpi=pmix -N 1 -n 2 --gres=gpu:4 --cpu-bind=none \
  env GPU_OFFSET=0 $COMMON \
      DTXQCD_MEAS_MG_SETUP_PREFIX=${SETUP_PREFIX_A:-$SETUPS/measmg_lam${LAM}_t${TRAJ_A}} \
  $WRAPO ./meas_baryon_dtxqcd $TRAJ_A $GRID \
  > $SETUPS/logs/meas_2c_A_t${TRAJ_A}.log 2>&1 &

srun --overlap --mpi=pmix -N 1 -n 2 --gres=gpu:4 --cpu-bind=none \
  env GPU_OFFSET=2 $COMMON \
      DTXQCD_MEAS_MG_SETUP_PREFIX=${SETUP_PREFIX_B:-$SETUPS/measmg_lam${LAM}_t${TRAJ_B}} \
  $WRAPO ./meas_baryon_dtxqcd $TRAJ_B $GRID \
  > $SETUPS/logs/meas_2c_B_t${TRAJ_B}.log 2>&1 &

wait
echo "=== 2 configs/node lambda=$LAM A=$TRAJ_A B=$TRAJ_B complete $(date) ==="
