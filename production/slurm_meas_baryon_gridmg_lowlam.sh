#!/bin/bash
#SBATCH --job-name=meas_bar_mg_low
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/meas_bar_mg_low.%j.out
# ============================================================================
# Low-lambda aux_baryon (meas_baryon_dtxqcd) with the VALIDATED Grid-MG
# backend wired in.  This is the lambda<=3 measurement path: EO-Schur is sick
# there (M_ee cliff), so the solve runs full-volume Grid-MG (HDCG/ADEF2 +
# coarse deflation) instead — validated == EO to solve tolerance at lambda=4
# (commit 824c93dc; abs-diff/peak scaled 2.2e-7 -> 2.0e-9 with a 100x tighter
# CG_TOL, i.e. tolerance-limited, no systematic bias).
#
# lambda>=4: DO NOT use this script — use the standard invocation (EO default,
# no per-config MG setup cost, fastest wired path at lambda=4).
#
# Settings are the lambda=3 single-RHS WINNER (commit 93bd7791):
#   nb64 + nn(25-pt) geometry + powerlaw ord1500 + refine400 + nev96
#   deflation + SP smoother -> 2804 outers / 724 s per solve (vs full CG
#   70612 iters / 1711 s = 2.4x).
#
# DECOMPOSITION: 1.1.2.2 (4 GPUs, ONE config/node, domain-decomposed).
#   nb64 does NOT fit meas_baryon at 2 GPUs (job 1299507 OOM'd in solver
#   construction: the 9.4 GB basis + coarse op + the EO operator caches +
#   doubled propagator buffers exceed 2x40 GB -- note the Test_dtxqcd_grid_mg
#   BENCH fits nb64 at 1.1.1.2 because it lacks those extra meas buffers).
#   1.1.2.2 halves per-rank memory; coarse [4,4,4,12] -> local [4,4,2,6] all
#   even (SIMD-legal; 1.1.1.4 gives odd coarse-T=3 and asserts).  Because the
#   decomposition differs from the older 1.1.1.2 caches, the basis is
#   regenerated per config (the LIME setup read is decomposition-sensitive).
#
# Parametrize via --export: LAM (default 3), TRAJ (default 0), NTSRC
# (t_src count; default 8 -> 32 solves ~6.4 h; T=48 -> 192 solves ~38 h,
# split across jobs), CFG_DIR, DATA_DIR.
#
# WARNING lambda<3: nb64 adequacy is UNTESTED at lambda=2.5/2 (denser coarse
# band -> may need nb>=96 / larger nev).  Probe one cached setup there before
# a production sweep; the script runs but convergence is not guaranteed.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

P=/lustre2/nplqcd/Grid-DTXQCD/production
cd $P

LAM=${LAM:-3}
TRAJ=${TRAJ:-0}
NTSRC=${NTSRC:-8}
CFG_DIR=${CFG_DIR:-$P/cfgs/dtxqcd_lam${LAM}.0000_noneo_fromchroma_cw_mds10}
DATA_DIR=${DATA_DIR:-$P/meas_2pt/dtxqcd_lam${LAM}.0000_gridmg}
SETUPS=$P/gridmg_setups
mkdir -p $DATA_DIR

# Per-config MG setup cache (basis+coarse op keyed on this ensemble+traj).
# Protects the ~50-min nb64 basis build against a mid-run crash and is shared
# if meas_conn is run on the same traj.  ~9.4 GB/config on disk -- clean up
# after a traj is fully measured, or unset SETUP_PREFIX to regenerate.
SETUP_PREFIX=${SETUP_PREFIX:-$SETUPS/measmg_lam${LAM}_t${TRAJ}}

if [ "$LAM" -ge 4 ] 2>/dev/null; then
  echo "REFUSING: lambda=$LAM >= 4 should use the EO default (run the standard"
  echo "meas_baryon invocation WITHOUT DTXQCD_MEAS_GRIDMG).  This script is the"
  echo "lambda<=3 path." ; exit 2
fi

echo "=== meas_baryon Grid-MG  lambda=$LAM traj=$TRAJ ntsrc=$NTSRC ==="
echo "    CFG_DIR=$CFG_DIR  DATA_DIR=$DATA_DIR"

srun --mpi=pmix -N 1 -n 4 --cpu-bind=none \
  env LATT=16.16.16.48 LAMBDA_DTXQCD=$LAM CFG_DIR=$CFG_DIR DATA_DIR=$DATA_DIR \
      MEAS_BARYON_NTSRC=$NTSRC CG_TOL=1e-8 \
      DTXQCD_MEAS_GRIDMG=1 \
      DTXQCD_MEAS_MG_NBASIS=64 \
      DTXQCD_MEAS_MG_GEOM=nn \
      DTXQCD_MEAS_MG_ORD=1500 \
      DTXQCD_MEAS_MG_REFINE_MAXIT=400 \
      DTXQCD_MEAS_MG_NEV=96 \
      COARSE_IRL_MAXIT=30 \
      DTXQCD_MEAS_MG_SETUP_PREFIX=$SETUP_PREFIX \
  ./meas_baryon_dtxqcd $TRAJ --grid 16.16.16.48 --mpi 1.1.2.2 --shm 512 --shm-mpi 1

echo "=== meas_baryon Grid-MG lambda=$LAM traj=$TRAJ complete rc=$? $(date) ==="
