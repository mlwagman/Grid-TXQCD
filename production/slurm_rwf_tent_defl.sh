#!/bin/bash
#SBATCH --job-name=tent_defl
#SBATCH --partition=lq2_gpu
#SBATCH --qos=normal
#SBATCH --account=nplqcd.lq2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lustre2/nplqcd/Grid-DTXQCD/production/gridmg_setups/logs/tent_defl.%j.out
# ============================================================================
# TENT PIVOT solver ladder: the certified deflated-EO solve
# (SolveTilEODeflated: 2 isolated wall modes + subcritical bulk = the
# textbook deflation case; tent-era measurement 60k -> ~1k at eta=0.8)
# exercised at eta = 0.4/0.6/0.7/0.8 (one eta per GPU), each with
# nev = 0 (baseline CG) / 8 / 16 sequentially.  Wall solves via
# meas_baryon_deform (MAXSOLVES=4 = one t_src), sharp tent, MG OFF.
# Read per arm: [deform-CG:quda-EO(-defl)] iters + the [deform-defl]
# eigensolve time; basis cached per (eta, nev) for reuse + t0
# translation in production.
# ============================================================================
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
LOGD=$P/gridmg_setups/logs
mkdir -p $P/defl_cache $P/meas_tentdefl
rm -f $LOGD/tent_defl.DONE
ETAS=(${ETAS_OVERRIDE:-0.4 0.6 0.7 0.8})
GPU=0
for ETA in "${ETAS[@]}"; do
  (
    for NEV in ${NEVS:-8 24}; do
      srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
        env CUDA_VISIBLE_DEVICES=$GPU LATT=16.16.16.48 ETA=$ETA \
            PROFILE=tent PROF_T0=0 \
            LAMBDA_DTXQCD=5 CFG_DIR=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw \
            DATA_DIR=$P/meas_tentdefl \
            CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
            MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=4 \
            DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_MG=0 \
            DTXQCD_DEFORM_EO_DEFLATE=$NEV \
            DTXQCD_DEFORM_EO_DEFLATE_ORD=1001 \
            DTXQCD_DEFORM_EO_DEFLATE_TOL=1e-5 \
            DTXQCD_DEFORM_EO_DEFLATE_LO=3e-5 \
            DTXQCD_DEFORM_EO_DEFLATE_CACHE=$P/defl_cache \
        $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
        --shm 512 --shm-mpi 1 > $LOGD/tentdefl_e${ETA}_nev${NEV}.log 2>&1
      echo "== eta=$ETA nev=$NEV =="
      grep -aE "deform-defl|deform-CG:quda-EO" $LOGD/tentdefl_e${ETA}_nev${NEV}.log | tail -3
    done
  ) &
  GPU=$((GPU+1))
done
wait
touch $LOGD/tent_defl.DONE
