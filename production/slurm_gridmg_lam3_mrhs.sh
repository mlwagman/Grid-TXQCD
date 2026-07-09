#!/bin/bash
#SBATCH --job-name=gridmg_l3_mrhs
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_gridmg_l3_mrhs.%j.out
# ============================================================================
# lambda=3 mrhs-M1, 4 ranks as --mpi 1.1.2.2: the 1.1.1.4 layout asserts
# (coarse 4.4.4.12 -> local T=3, odd, can't carry SIMD; Cartesian_full.h
# rdim*simd==ldim).  1.1.2.2 -> coarse local 4.4.2.6, all even.  2-rank
# OOM'd post-coarsen (TwoLevelCGmrhs ~6 x nrhs fine fields + both coarse
# ops + basis), hence 4 ranks.
# COARSE_NEV=96: s6A verdict — 113-evec deflation is 2.8x wallclock at
# lambda=3 (2804 outers / 724 s vs 3192 / 2025 s at nev=56; the win is
# per-outer coarse-solve cost 0.63 -> 0.26 s, not the outer count).
# lambda=4 mrhs reference: 37-40 s/rhs = 1.9x over EO CG (f03dd41f).
# ============================================================================
# Self-sufficient env: sbatch submitted from a bare (post-turnover) shell
# inherits no modules -> libhdf5_cpp.so.310 load failure (job 1299199).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
BIN=/lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_grid_mg
WRAP=/lustre2/nplqcd/Grid-DTXQCD/production/srun_gpu_wrapper.sh
SETUPS=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups
CFG3=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam3.0000_noneo_fromchroma_cw_mds10

# 1.1.2.2 FAILED (job 1299205): SciDAC checksum assert reading the cached
# basis — the LIME parallel read is decomposition-sensitive (written/read
# fine at 1.1.1.2; 2D-split read fails verify).  Stay at the proven
# 1.1.1.2 and halve the batch instead: nrhs=4 (Nsimd=4, legal) halves the
# TwoLevelCGmrhs fine-field working set that OOM'd the nrhs=8 2-rank run.
# nrhs=4 plain-CG coarse solve FAILED (job 1299210, stall ~1e-3): scalar CG
# on the 5D mrhs field shares one alpha/beta across rhs — replaced with
# BlockCGrQ (commit pending); this run validates the fix.
srun --mpi=pmix -N 1 -n 2 --cpu-bind=none \
  env QUDA_ENABLE_MPS=1 FULLVOL=1 CFG_DIR=$CFG3 TRAJ=50 LAMBDA=3 \
    BLOCK=4.4.4.4 NBASIS=64 SUBSPACE=powerlaw CHEBY_ORD=1500 REFINE=1 REFINE_MAXIT=400 \
    MG_MAXIT=4000 BENCH_HERMOP=0 RUN_BASELINE=0 SMOOTH_SP=1 GEOM=nn \
    COARSE_NEV=96 COARSE_IRL_MAXIT=30 MRHS_NRHS=4 \
    SETUP_PREFIX=$SETUPS/lam3_t50_fullvol_nb64_ord1500r400 \
  $WRAP $BIN --grid 16.16.16.48 --mpi 1.1.1.2 --shm 512 --shm-mpi 1 \
  > $LOGD/m4_lam3_nb64_nrhs4_1112.log 2>&1

touch $LOGD/lam3_mrhs_sbatch.DONE
echo "=== gridmg lam3 mrhs complete $(date) ==="
