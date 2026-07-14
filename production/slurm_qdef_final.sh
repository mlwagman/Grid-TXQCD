#!/bin/bash
#SBATCH --job-name=qdef_final
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/qdef_final.%j.out
# ============================================================================
# Everything still outstanding for the QUDA deform port, self-contained
# (interactive turnover has now killed these twice):
#  A (GPU 0): eta=0.6,0.8 spectrum at census depth (NEV=24, lo_frac 1e-6) —
#     completes the eta ladder 0/0.2/0.4 (banked: band 3.2e-4..1.2e-3 at 0,
#     ~15% shift at 0.2, 11-mode cluster + x220 gap at 0.4) and counts the
#     cluster for the deflation-cost calculus.
#  B (GPU 1): provenance-matched Grid-path reference (current binary,
#     source-centered) NTSRC=4 -> row-wise h5 comparison vs the QUDA-path
#     output (meas_deform_qgate).  Formal G1 closure; the earlier "FAIL" was
#     a stale/partial fleet reference (rows 12/36 empty, row 24 pre-
#     source-centering; row 0 matched at 3.3e-10).
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
CFG=$P/cfgs/dtxqcd_lam4.0000_tla_fromchroma_cw
GRID="--grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1"
WRAP=$P/srun_gpu_wrapper.sh
mkdir -p $P/meas_deform_gridref

srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=0 LATT=16.16.16.48 CFG_DIR=$CFG \
      ETAS=0.6,0.8 NEV=24 IRL_LO_FRAC=1e-6 \
  $P/probe_deform_spectrum 300 $GRID > $LOGD/deform_spectrum_tail.log 2>&1 &

srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
  env CUDA_VISIBLE_DEVICES=1 LATT=16.16.16.48 TRAJ=300 ETA=0.2 PROFILE=tent \
      MEAS_BARYON_NTSRC=4 CFG_DIR=$CFG DATA_DIR=$P/meas_deform_gridref \
  $P/meas_baryon_deform_dtxqcd 300 $GRID > $LOGD/qdef_gridref2.log 2>&1 &
wait

echo "=== G1-final: QUDA path vs provenance-matched Grid reference ==="
python3 $P/_qdef_cmp2.py $P/meas_deform_gridref/baryon_deform_dtxqcd_300.h5 \
  $P/meas_deform_qgate/baryon_deform_dtxqcd_300.h5 1e-5 \
  && G1=PASS || G1=FAIL
echo "QDEF FINAL: G1=$G1" | tee $LOGD/qdef_final.VERDICT
grep -aE "spectrum. eta=0.(6|8).*(lambda_max|converged|ev\[)" \
  $LOGD/deform_spectrum_tail.log | tail -60
touch $LOGD/qdef_final.DONE
