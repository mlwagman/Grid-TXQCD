#!/bin/bash
# PROFILE=cos 2x2 scan on the interactive node's 4 GPUs:
# (LAM, ETA) in {4,5} x {0.6, 0.8}, one wall-source CG solve each.
# Reference points (sharp tent): lam5 eta0.6 = 4,479; lam5 eta0.8 =
# 59,931; lam4 eta0.8 caps out (kappa_eff ~ 5.5e10).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
GPU=0
for L in 4 5; do
  for E in 0.6 0.8; do
    CFG=$P/cfgs/dtxqcd_lam$L.0000_tla_fromchroma_cw
    mkdir -p $P/meas_cos_l${L}_e$E
    srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES=$GPU \
          LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=$L CFG_DIR=$CFG \
          PROFILE=cos ETA=$E \
          CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
          MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
          DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
          DATA_DIR=$P/meas_cos_l${L}_e$E \
      $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
      --shm 512 --shm-mpi 1 > $P/mtilde_runs/cos_l${L}_e$E.log 2>&1 &
    GPU=$((GPU+1))
  done
done
wait
for L in 4 5; do
  for E in 0.6 0.8; do
    echo "== lam$L eta$E =="
    grep -aE "deform-CG.*iters=|NOT CONVERGED" $P/mtilde_runs/cos_l${L}_e$E.log | tail -1
  done
done
