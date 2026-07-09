#!/bin/bash
#SBATCH --job-name=gridmg_l3_step2
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_gridmg_l3_step2.%j.out
# ============================================================================
# lambda=3 step-2 verdicts (sbatch: interactive-node turnover kept killing
# these — allocation 1298778 died overnight 07-08 mid-solve):
#   1. s6A: COARSE_NEV=96 single-RHS knob scan.  nev sized to the measured
#      band (~112 modes below the bulk; nev=128 asks bulk evals ~3.4 that
#      Chebyshev-IRL rightly never converges).  Setup proven: 96/96 IRL,
#      113 evecs.  Baseline to beat: 3192 outers / 2025 s (nev=56).
#   2. mrhs-M1 at lambda=3: nrhs=8, 4-rank (2-rank OOM'd post-coarsen:
#      TwoLevelCGmrhs holds ~6 x nrhs fine fields + both coarse ops + basis).
#      lambda=4 reference: 37-40 s/rhs = 1.9x over EO CG (f03dd41f).
# Knob verdicts already banked: smooth16 +10% wall / mirs0.6 hits outer cap.
# ============================================================================
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
BIN=/lustre2/nplqcd/Grid-DTXQCD/production/Test_dtxqcd_grid_mg
WRAPO=/lustre2/nplqcd/Grid-DTXQCD/production/srun_gpu_wrapper_offset.sh
WRAP=/lustre2/nplqcd/Grid-DTXQCD/production/srun_gpu_wrapper.sh
SETUPS=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups
CFG3=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam3.0000_noneo_fromchroma_cw_mds10
COMMON3="QUDA_ENABLE_MPS=1 FULLVOL=1 CFG_DIR=$CFG3 TRAJ=50 LAMBDA=3 BLOCK=4.4.4.4 NBASIS=64 SUBSPACE=powerlaw CHEBY_ORD=1500 REFINE=1 REFINE_MAXIT=400 MG_MAXIT=4000 BENCH_HERMOP=0 RUN_BASELINE=0 SMOOTH_SP=1 GEOM=nn SETUP_PREFIX=$SETUPS/lam3_t50_fullvol_nb64_ord1500r400"

# 1. nev-96 single-RHS (2 ranks / GPUs 0-1)
srun --mpi=pmix -N 1 -n 2 --cpu-bind=none \
  env GPU_OFFSET=0 $COMMON3 COARSE_NEV=96 COARSE_IRL_MAXIT=30 \
  $WRAPO $BIN --grid 16.16.16.48 --mpi 1.1.1.2 --shm 512 --shm-mpi 1 \
  > $LOGD/s6A_lam3_nev96_sbatch.log 2>&1

# 2. mrhs nrhs=8 (4 ranks / 4 GPUs)
srun --mpi=pmix -N 1 -n 4 --cpu-bind=none \
  env $COMMON3 COARSE_NEV=56 MRHS_NRHS=8 \
  $WRAP $BIN --grid 16.16.16.48 --mpi 1.1.1.4 --shm 512 --shm-mpi 1 \
  > $LOGD/m2_lam3_nb64_nrhs8_sbatch.log 2>&1

touch $LOGD/lam3_step2_sbatch.DONE
echo "=== gridmg lam3 step2 complete $(date) ==="
