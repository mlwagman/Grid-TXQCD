#!/bin/bash
#SBATCH --job-name=dtx_auxk_v4pair
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_auxk_v4pair.%j.out
# ============================================================================
# Aux-kernel V4 decomposition probe: clean same-job pair vs V2 (coordinator
# methodology; V2 8.60 / V3b 12.24 / V3a 21.62 ms established occupancy as
# the limiter).  V4 = thread-per-(site, output-flavor), AoS unchanged.
# The timeboxed decider: V4 < 8.60 -> graduate (SP twin + full gates);
# else -> negative-result writeup and the kernel line closes.
#   v1: BENCH 200 applies, V2  + mdevice profile
#   v2: BENCH 200 applies, V4  + mdevice profile
#   v3: 4^3x8 FULLVOL gate B with V4 (DP correctness at machine precision)
# ============================================================================
set -u
cd /lustre2/nplqcd/Grid-DTXQCD/production
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
BIN=./Test_dtxqcd_grid_mg
SRC=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_eo_fromchroma_cw_mds10
SRUN="srun --mpi=pmix -N 1 -n 1 --cpu-bind=none"
BENCHENV="FULLVOL=1 CFG_DIR=$SRC TRAJ=120 LAMBDA=4 BLOCK=4.4.4.4 NBASIS=32 \
BENCH_HERMOP=200 RUN_BASELINE=0 RUN_MG=0 DTXQCD_MDEVICE_PROFILE=1"

$SRUN env $BENCHENV \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_v4p_v2_bench.log 2>&1

$SRUN env $BENCHENV DTXQCD_AUX_KERNEL_V4=1 \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_v4p_v4_bench.log 2>&1

$SRUN env FULLVOL=1 BLOCK=2.2.2.2 NBASIS=24 DTXQCD_AUX_KERNEL_V4=1 \
  $BIN --grid 4.4.4.8 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_v4p_gate.log 2>&1

touch $LOGD/auxk_v4pair.DONE
echo "=== auxk v4 pair complete $(date) ==="
