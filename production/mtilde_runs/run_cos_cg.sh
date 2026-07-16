#!/bin/bash
# PROFILE=cos CG probe on the interactive node: one wall-source solve at
# LAMBDA (default 5), eta=0.8.  Compare against the tent/tent_rounded2
# w-ladder (lam5 sharp: 59,931; w=8: 8,307; w=12: 2,435; w=16: 1,178).
source /lustre2/nplqcd/Grid-DTXQCD/env_lq2_grid.sh
P=/lustre2/nplqcd/Grid-DTXQCD/production
L=${LAM:-5}
CFG=$P/cfgs/dtxqcd_lam$L.0000_tla_fromchroma_cw
mkdir -p $P/meas_cos_l$L
srun --overlap --mpi=pmix -N 1 -n 1 --cpu-bind=none --gres=gpu:1 \
  env LATT=16.16.16.48 TRAJ=300 LAMBDA_DTXQCD=$L CFG_DIR=$CFG \
      PROFILE=cos ETA=${ETA:-0.8} \
      CG_TOL=1e-7 CG_MAX=200000 DTXQCD_DEFORM_QUDA_CHECK=0 \
      MEAS_BARYON_NTSRC=1 MEAS_BARYON_MAXSOLVES=1 \
      DTXQCD_DEFORM_EO=1 DTXQCD_DEFORM_EO_DEFLATE=0 \
      DATA_DIR=$P/meas_cos_l$L \
  $P/meas_baryon_deform_dtxqcd 300 --grid 16.16.16.48 --mpi 1.1.1.1 --shm 512 --shm-mpi 1
