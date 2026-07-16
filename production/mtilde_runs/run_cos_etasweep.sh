#!/bin/bash
# PROFILE=cos eta sweep at lambda=5 across 4 GPUs: ETA in {0.2,0.3,0.4,0.5}.
# Cosine mu_max = eta*pi/2 = {0.31,0.47,0.63,0.79} — all SUBcritical
# (edge ~0.85): no walls AND no supercritical stretch, the cosine's clean
# regime.  Trade-space note: cosine keeps FULL range p_max=T/2, so
# grading exponent = eta*24; e.g. eta=0.5 -> 12, vs rounded-tent w=12 at
# eta=0.8 -> 0.8*18 = 14.4 — the (eta, profile) plane is optimized JOINTLY.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
GPU=0
for E in 0.2 0.3 0.4 0.5; do
  mkdir -p $P/meas_cos_l5_e$E
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU \
        LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        PROFILE=cos ETA=$E \
        CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
        DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
        DATA_DIR=$P/meas_cos_l5_e$E \
    $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $P/mtilde_runs/cos_l5_eta$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for E in 0.2 0.3 0.4 0.5; do
  echo "== eta=$E =="
  grep -aE "deform-CG.*iters=|NOT CONVERGED" $P/mtilde_runs/cos_l5_eta$E.log | tail -1
done
