#!/bin/bash
#SBATCH --job-name=dtx_auxk_pair
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_auxk_pair.%j.out
# ============================================================================
# Aux-kernel V3b clean pair + gates.  V3a (row-hoist arrays) REGRESSED
# 8.6 -> 21.6 ms/apply: runtime-indexed local arrays forced the operands into
# local memory.  V3b = scalar operands + fully-unrolled inner loops (see the
# TU comments).  The headline perf number is the p1/p2 pair below — same job,
# same binary, same cached setup, only the env differs (coordinator
# methodology; each leg logs its variant at startup).
#   p1: BENCH 200 applies, V2  (DTXQCD_AUX_KERNEL_V3=0) + mdevice profile
#   p2: BENCH 200 applies, V3b (DTXQCD_AUX_KERNEL_V3=1) + mdevice profile
#   c1: 4^3x8 FULLVOL gates, V3b (DP gate B + SP-smoother MG)
#   c2: 16^3x48 lambda=4 cached-setup gates, V3b (gate B + baseline CG +
#       ADEF2-vs-CG)
#   c3: dH A/B 1-traj gen, V3b=1 vs V3b=0 same seed (fresh _auxk2 stage dirs;
#       the earlier _auxk dirs now hold traj-121 output and would resume)
# ============================================================================
set -u
cd /lustre2/nplqcd/Grid-DTXQCD/production
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
BIN=./Test_dtxqcd_grid_mg
SETUPS=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups
SRC=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_eo_fromchroma_cw_mds10
SRUN="srun --mpi=pmix -N 1 -n 1 --cpu-bind=none"
BENCHENV="FULLVOL=1 CFG_DIR=$SRC TRAJ=120 LAMBDA=4 BLOCK=4.4.4.4 NBASIS=32 \
BENCH_HERMOP=200 RUN_BASELINE=0 RUN_MG=0 DTXQCD_MDEVICE_PROFILE=1"

$SRUN env $BENCHENV DTXQCD_AUX_KERNEL_V3=0 \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_p1_v2_bench.log 2>&1

$SRUN env $BENCHENV DTXQCD_AUX_KERNEL_V3=1 \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_p2_v3_bench.log 2>&1

$SRUN env FULLVOL=1 BLOCK=2.2.2.2 NBASIS=24 SMOOTH_SP=1 DTXQCD_AUX_KERNEL_V3=1 \
  $BIN --grid 4.4.4.8 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_c1_4x8.log 2>&1
if ! grep -aq 'gridmg PASSED' $LOGD/auxk_c1_4x8.log; then
  echo "auxk c1 FAILED" | tee $LOGD/auxk_pair.FAILED; exit 1; fi

$SRUN env FULLVOL=1 CFG_DIR=$SRC TRAJ=120 LAMBDA=4 BLOCK=4.4.4.4 NBASIS=32 \
      SUBSPACE=powerlaw CHEBY_ORD=300 MG_MAXIT=2000 BENCH_HERMOP=0 \
      RUN_BASELINE=1 SMOOTH_SP=1 DTXQCD_AUX_KERNEL_V3=1 \
      SETUP_PREFIX=$SETUPS/lam4_t120_fullvol_ord300 \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_c2_lam4.log 2>&1
if ! grep -aq 'gridmg PASSED' $LOGD/auxk_c2_lam4.log; then
  echo "auxk c2 FAILED" | tee $LOGD/auxk_pair.FAILED; exit 1; fi

for leg in a b; do
  mkdir -p cfgs/dtxqcd_lam4.0000_auxk2_${leg}
  cp -n $SRC/ckpoint_lat.120       cfgs/dtxqcd_lam4.0000_auxk2_${leg}/ 2>/dev/null || true
  cp -n $SRC/ckpoint_lat_daux.120  cfgs/dtxqcd_lam4.0000_auxk2_${leg}/ 2>/dev/null || true
  cp -n $SRC/ckpoint_rng.120       cfgs/dtxqcd_lam4.0000_auxk2_${leg}/ 2>/dev/null || true
done
GENV="LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 USE_FULL_PF=1 TRAJ=121 NO_METROP=0"

$SRUN env $GENV DTXQCD_SUFFIX=_auxk2_a DTXQCD_AUX_KERNEL_V3=1 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_c3_a.log 2>&1

$SRUN env $GENV DTXQCD_SUFFIX=_auxk2_b DTXQCD_AUX_KERNEL_V3=0 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_c3_b.log 2>&1

touch $LOGD/auxk_pair.DONE
echo "=== auxk pair complete $(date) ==="
