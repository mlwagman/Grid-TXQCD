#!/bin/bash
#SBATCH --job-name=l3_thruput_ab
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/l3_thruput_ab.%j.out
# ============================================================================
# lambda=3 baryon-solve THROUGHPUT: 2 configs/node (nb64, 2-GPU each) vs
# 4 configs/node fanout (nb48, 1-GPU each).  Uses Test_dtxqcd_grid_mg as the
# per-solve bench (one deflated ADEF2 solve + convergence gate; cleaner than
# a meas_baryon NTSRC loop).  All on traj 50 of the noneo lam3 stream.
#
#   Phase 1  nb48 VIABILITY: fresh-gen @1.1.1.1 (GPU 0).  Answers the crux —
#            does nb48 FIT one GPU and CONVERGE at lambda=3?  If not, the
#            4-fanout is dead and 2-configs/node nb64 wins by default.
#            Also = the solo nb48 solve time + outer count.  Saves nb48 cache.
#   Phase 2  nb64 CONTENDED: 2x [1.1.1.2, load nb64 cache] on GPUs {0,1}+{2,3}.
#   Phase 3  nb48 CONTENDED: 4x [1.1.1.1, load nb48 cache] on GPUs 0..3
#            (only if phase 1 converged).
#
# Throughput verdict = concurrency / contended-solve-time:
#   A = 2 / T_nb64_contended     B = 4 / T_nb48_contended
# Baselines: nb64 solo @1.1.1.2 = 724 s / 2804 outers (s6A).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh

P=/lustre2/nplqcd/Grid-DTXQCD/production
BIN=$P/Test_dtxqcd_grid_mg
WRAPO=$P/srun_gpu_wrapper_offset.sh
SETUPS=$P/gridmg_setups
LOGD=$SETUPS/logs
CFG3=$P/cfgs/dtxqcd_lam3.0000_noneo_fromchroma_cw_mds10
NB64=$SETUPS/lam3_t50_fullvol_nb64_ord1500r400          # existing cache
NB48=$SETUPS/lam3_t50_fullvol_nb48_ord1500r400          # generated in phase 1
G="--grid 16.16.16.48"
COMMONQ="QUDA_ENABLE_MPS=1 FULLVOL=1 CFG_DIR=$CFG3 TRAJ=50 LAMBDA=3 BLOCK=4.4.4.4 SUBSPACE=powerlaw CHEBY_ORD=1500 REFINE=1 REFINE_MAXIT=400 MG_MAXIT=4000 BENCH_HERMOP=0 RUN_BASELINE=0 SMOOTH_SP=1 GEOM=nn COARSE_NEV=96 COARSE_IRL_MAXIT=30"

echo "=== PHASE 1: nb48 viability (fresh-gen, 1 GPU) $(date) ==="
srun --overlap --mpi=pmix -N 1 -n 1 --gres=gpu:4 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 $COMMONQ NBASIS=48 SETUP_PREFIX=$NB48 \
  $BIN $G --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/tp_p1_nb48_solo.log 2>&1
P1OK=0
grep -aq 'gridmg PASSED' $LOGD/tp_p1_nb48_solo.log && P1OK=1
echo "PHASE1 nb48 solo: pass=$P1OK $(grep -a 'ADEF2 solve' $LOGD/tp_p1_nb48_solo.log | tail -1)"
grep -aiE 'out of memory|cudaMalloc' $LOGD/tp_p1_nb48_solo.log | tail -1

echo "=== PHASE 2: nb64 contended (2x 1.1.1.2) $(date) ==="
srun --overlap --mpi=pmix -N 1 -n 2 --gres=gpu:4 --cpu-bind=none \
  env GPU_OFFSET=0 $COMMONQ NBASIS=64 SETUP_PREFIX=$NB64 \
  $WRAPO $BIN $G --mpi 1.1.1.2 --shm 512 --shm-mpi 1 \
  > $LOGD/tp_p2_nb64_a.log 2>&1 &
srun --overlap --mpi=pmix -N 1 -n 2 --gres=gpu:4 --cpu-bind=none \
  env GPU_OFFSET=2 $COMMONQ NBASIS=64 SETUP_PREFIX=$NB64 \
  $WRAPO $BIN $G --mpi 1.1.1.2 --shm 512 --shm-mpi 1 \
  > $LOGD/tp_p2_nb64_b.log 2>&1 &
wait
echo "PHASE2 nb64 contended A: $(grep -a 'ADEF2 solve' $LOGD/tp_p2_nb64_a.log | tail -1)"
echo "PHASE2 nb64 contended B: $(grep -a 'ADEF2 solve' $LOGD/tp_p2_nb64_b.log | tail -1)"

if [ $P1OK -eq 1 ]; then
  echo "=== PHASE 3: nb48 contended (4x 1.1.1.1) $(date) ==="
  for g in 0 1 2 3; do
    srun --overlap --mpi=pmix -N 1 -n 1 --gres=gpu:4 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$g $COMMONQ NBASIS=48 SETUP_PREFIX=$NB48 \
      $BIN $G --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
      > $LOGD/tp_p3_nb48_g$g.log 2>&1 &
  done
  wait
  for g in 0 1 2 3; do
    echo "PHASE3 nb48 contended g$g: $(grep -a 'ADEF2 solve' $LOGD/tp_p3_nb48_g$g.log | tail -1)"
  done
else
  echo "=== PHASE 3 SKIPPED: nb48 failed phase 1 (no single-GPU nb48 path) ==="
fi

touch $LOGD/l3_thruput_ab.DONE
echo "=== throughput A/B complete $(date) ==="
