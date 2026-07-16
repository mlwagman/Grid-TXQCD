#!/bin/bash
# Asymmetric-width scan at eta=0.4 (cheap cells), lambda=5: fix
# w_apex=12, vary trough width w_src in {0, 4, 8, 12} across 4 GPUs.
# Isolates the trough-state's rounding requirement — if it lifts at small
# w_src, we buy grading differential back (differential = eta*(24 -
# (w_apex+w_src)/2)) at fixed CG relief.  References at eta=0.4 sharp:
# EO 1,293 iters.  ETA/WA overridable.
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
CFG=$P/cfgs/dtxqcd_lam5.0000_tla_fromchroma_cw
E=${ETA:-0.4}
WA=${WA:-12}
GPU=0
for WS in 0 4 8 12; do
  mkdir -p $P/meas_asym_wa${WA}_ws${WS}_e$E
  srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none \
    env CUDA_VISIBLE_DEVICES=$GPU \
        LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=5 CFG_DIR=$CFG \
        PROFILE=tent_rounded2 PROFILE_W=$WA PROFILE_W_SRC=$WS ETA=$E \
        CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
        MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
        DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
        DATA_DIR=$P/meas_asym_wa${WA}_ws${WS}_e$E \
    $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 \
    --shm 512 --shm-mpi 1 > $P/mtilde_runs/asym_wa${WA}_ws${WS}_e$E.log 2>&1 &
  GPU=$((GPU+1))
done
wait
for WS in 0 4 8 12; do
  echo "== w_apex=$WA w_src=$WS eta=$E =="
  grep -aE "deform-CG.*iters=|NOT CONVERGED" $P/mtilde_runs/asym_wa${WA}_ws${WS}_e$E.log | tail -1
done
