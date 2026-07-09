#!/bin/bash
#SBATCH --job-name=dtx_auxk_gates
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/slurm_auxk_gates.%j.out
# ============================================================================
# Aux-kernel V3 validation gates (DTXQCD_AUX_KERNEL_V3, see
# DTXQCD_AUX_FUSION_DESIGN.md).  V3 = SoA repack + row-hoisted apply;
# arithmetic sequence identical to V2 => results must be BIT-identical.
#   g1: Test_dtxqcd_grid_mg FULLVOL 4^3x8, V3=1, MG with SP smoother
#       (gate B machine precision covers the DP kernel; the MG solve's
#       ADEF2-vs-CG check covers the SP twin).
#   g2: same at 16^3x48 lambda=4 cached setup, V3=1, + BENCH_HERMOP=200 with
#       DTXQCD_MDEVICE_PROFILE=1 (the aux ms number) + baseline CG +
#       ADEF2-vs-CG.
#   g3: dH A/B — 1-traj non-EO gen from the traj-120 chromafork checkpoint,
#       V3=1 vs V3=0 same seed: dH must be BIT-identical (this is the
#       production multishift path incl. the Style C inner loop; the
#       determinism-control result from the aux-static A/B validates the
#       criterion).  Configs staged into _auxk_a / _auxk_b.
# ============================================================================
set -u
cd /lustre2/nplqcd/Grid-DTXQCD/production
LOGD=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs
BIN=./Test_dtxqcd_grid_mg
SETUPS=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups
SRC=/lustre2/nplqcd/Grid-DTXQCD/production/cfgs/dtxqcd_lam4.0000_eo_fromchroma_cw_mds10
SRUN="srun --mpi=pmix -N 1 -n 1 --cpu-bind=none"

# g1: toy-volume gates, DP+SP V3
$SRUN env FULLVOL=1 BLOCK=2.2.2.2 NBASIS=24 SMOOTH_SP=1 \
      DTXQCD_AUX_KERNEL_V3=1 \
  $BIN --grid 4.4.4.8 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_g1_4x8.log 2>&1
if ! grep -aq 'gridmg PASSED' $LOGD/auxk_g1_4x8.log; then
  echo "auxk g1 FAILED" | tee $LOGD/auxk_gates.FAILED; exit 1; fi

# g2: 16^3x48 gates + profile (cached setup; solve validates at scale)
$SRUN env FULLVOL=1 CFG_DIR=$SRC TRAJ=120 LAMBDA=4 BLOCK=4.4.4.4 NBASIS=32 \
      SUBSPACE=powerlaw CHEBY_ORD=300 MG_MAXIT=2000 BENCH_HERMOP=200 \
      RUN_BASELINE=1 SMOOTH_SP=1 DTXQCD_AUX_KERNEL_V3=1 \
      DTXQCD_MDEVICE_PROFILE=1 \
      SETUP_PREFIX=$SETUPS/lam4_t120_fullvol_ord300 \
  $BIN --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_g2_lam4.log 2>&1
if ! grep -aq 'gridmg PASSED' $LOGD/auxk_g2_lam4.log; then
  echo "auxk g2 FAILED" | tee $LOGD/auxk_gates.FAILED; exit 1; fi

# g3: dH A/B (stage checkpoint triplets; cp -n = keep existing)
for leg in a b; do
  mkdir -p cfgs/dtxqcd_lam4.0000_auxk_${leg}
  cp -n $SRC/ckpoint_lat.120       cfgs/dtxqcd_lam4.0000_auxk_${leg}/ 2>/dev/null || true
  cp -n $SRC/ckpoint_lat_daux.120  cfgs/dtxqcd_lam4.0000_auxk_${leg}/ 2>/dev/null || true
  cp -n $SRC/ckpoint_rng.120       cfgs/dtxqcd_lam4.0000_auxk_${leg}/ 2>/dev/null || true
done
GENV="LATT=16.16.16.48 LAMBDA_DTXQCD=4.0 USE_FULL_PF=1 TRAJ=121 NO_METROP=0"

$SRUN env $GENV DTXQCD_SUFFIX=_auxk_a DTXQCD_AUX_KERNEL_V3=1 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_g3_a.log 2>&1

$SRUN env $GENV DTXQCD_SUFFIX=_auxk_b DTXQCD_AUX_KERNEL_V3=0 \
  ./gen_dtxqcd_cfgs --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1 \
  > $LOGD/auxk_g3_b.log 2>&1

touch $LOGD/auxk_gates.DONE
echo "=== auxk gates complete $(date) ==="
