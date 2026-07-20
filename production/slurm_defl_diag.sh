#!/bin/bash
#SBATCH --job-name=defl_diag
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/defl_diag.%j.out
# ============================================================================
# Deflation-basis DISCRIMINATOR at eta=0.6 (the wrong-subspace finding):
#   arm A: nev=8 + VerifyDeflationBasis (DIAG=1) -- true residuals
#     ||Mpc~+Mpc~ v - lambda v|| per mode.  Large residuals = IRL
#     cluster-mixing confirmed; small = basis fine, bug is in the
#     deflated-SOLVE path (projector application / frame translation).
#   arm B: nev=24 + DIAG -- the census-proven cluster-spanning size.
#     If its ev[0] -> 1.99e-5 (census truth) AND the deflated solves
#     beat 4,480, the fix is simply "nev >= cluster size" and the
#     banded defaults return as 24/32/48 with lo unchanged.
# Fresh cache names (nev differs) -- no stale-basis reuse.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
rm -f $LOGD/defl_diag.DONE
GPU=0
for NEV in 8 24; do
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 ETA=0.6 \
        PROFILE=tent PROF_T0=0 \
        LAMBDA_DTXQCD=5 CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
        DATA_DIR=$P/meas_tentdefl \
        CG_TOL=1e-7 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=4 \
        DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
        DTXQCD_DEFORM_EO_DEFLATE=$NEV \
        DTXQCD_DEFORM_EO_DEFLATE_DIAG=1 \
        DTXQCD_DEFORM_EO_DEFLATE_CACHE="" \
    $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $LOGD/defl_diag_nev${NEV}.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for NEV in 8 24; do
  echo "== nev=$NEV =="
  grep -aE "deform-defl|Verify|residual|quda-EO-defl. iters" $LOGD/defl_diag_nev${NEV}.log | tail -8
done
touch $LOGD/defl_diag.DONE
